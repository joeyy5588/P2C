# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.

import argparse
import base64
import numpy as np
import os
import os.path as op
import random, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from oscar.utils.logger import setup_logger
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import (tsv_writer, concat_tsv_files,
        delete_tsv_files, reorder_tsv_keys)
from oscar.utils.misc import (mkdir, set_seed, 
        load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
        ScstRewardCriterion, OTRewardCriterion, evaluate_on_nocaps)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder
from oscar.datasets.rl_tsv import RLTSVDataset
from oscar.modeling.modeling_bert import BertForImageCaptioning
from CLIP import clip
from transformers.pytorch_transformers import BertTokenizer, BertConfig, BertForMaskedLM
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import nltk


class CaptionTSVDataset(Dataset):
    def __init__(self, yaml_file, tokenizer=None, add_od_labels=True,
            max_img_seq_length=50, max_seq_length=70, max_seq_a_length=40, 
            is_train=True, mask_prob=0.15, max_masked_tokens=3, **kwargs):
        """Constructor.
        Args:
            yaml file with all required data (image feature, caption, labels, etc)
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            kwargs: other arguments.
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = op.dirname(yaml_file)
        self.img_root = op.join(self.root, 'coco_image/')
        self.label_file = find_file_path_in_yaml(self.cfg['label'], self.root)
        self.feat_file = find_file_path_in_yaml(self.cfg['feature'], self.root)
        self.caption_file = find_file_path_in_yaml(self.cfg.get('caption'), self.root)

        assert op.isfile(self.feat_file)
        if add_od_labels: assert op.isfile(self.label_file)
        if is_train: assert op.isfile(self.caption_file) and tokenizer is not None

        self.label_tsv = None if not self.label_file else TSVFile(self.label_file)
        self.feat_tsv = TSVFile(self.feat_file)
        self.captions = []
        if self.caption_file and op.isfile(self.caption_file):
            with open(self.caption_file, 'r') as f:
                self.captions = json.load(f)

        self.tokenizer = tokenizer
        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length,
                max_seq_length, max_seq_a_length, mask_prob, max_masked_tokens,
                is_train=is_train)
        self.add_od_labels = add_od_labels
        self.is_train = is_train
        self.kwargs = kwargs
        self.image_keys = self.prepare_image_keys()
        self.key2index = self.prepare_image_key_to_index()
        self.key2captions = self.prepare_image_key_to_captions()

        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_valid_tsv(self):
        # based on the order of file size
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.seek(i)[0] for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.seek(i)[0] : i for i in range(tsv.num_rows())}

    def prepare_image_key_to_captions(self):
        if self.captions:
            key2captions = {key: [] for key in self.image_keys}
            for cap in self.captions:
                key2captions[cap['image_id']].append(cap['caption'])
            return key2captions

    def get_image_index(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            img_key = img_cap_pair['image_id']
            return self.key2index[img_key]
        return idx

    def get_image_key(self, idx):
        img_idx = self.get_image_index(idx)
        return self.image_keys[img_idx]

    def get_image_features(self, img_idx):
        feat_info = json.loads(self.feat_tsv.seek(img_idx)[1].replace("\'", "\""))
        # feat_info = json.loads(self.feat_tsv.seek(img_idx)[1])
        num_boxes = feat_info['num_boxes']
        features = np.frombuffer(base64.b64decode(feat_info['features']), np.float32
                ).reshape((num_boxes, -1))
        return torch.Tensor(features)

    def get_caption(self, idx):
        if self.is_train:
            img_cap_pair = self.captions[idx]
            return img_cap_pair['caption']
        return ""

    def get_od_labels(self, img_idx):
        od_labels = None
        if self.add_od_labels:
            label_info = json.loads(self.label_tsv.seek(img_idx)[1].replace("\'", "\""))
            # labels_list = list(set([l['class'] for l in label_info]))
            # od_labels = " ".join(labels_list)
            od_labels = " ".join([l['class'] for l in label_info])
        return od_labels

    def get_caption_file_in_coco_format(self):
        cap_file = op.splitext(self.caption_file)[0] + '_coco_format.json'
        return cap_file

    def get_captions_by_key(self, key):
        return self.key2captions[key]

    def __getitem__(self, idx):
        img_idx = self.get_image_index(idx)
        img_key = self.image_keys[img_idx]
        features = self.get_image_features(img_idx)
        caption = self.get_caption(idx)
        od_labels = self.get_od_labels(img_idx)
        example = self.tensorizer.tensorize_example(caption, features, text_b=od_labels)
        if op.exists(op.join(self.img_root, 'train/COCO_train2014_' + img_key.zfill(12) + '.jpg')):
            img_fn = op.join(self.img_root, 'train/COCO_train2014_' + img_key.zfill(12) + '.jpg')
        else:
            img_fn = op.join(self.img_root, 'validation/COCO_val2014_' + img_key.zfill(12) + '.jpg')
        img = Image.open(img_fn)
        img = self.transform(img)
        return img_key, example, img

    def __len__(self):
        if self.is_train:
            return len(self.captions)
        return self.get_valid_tsv().num_rows()

class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            is_train=True):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS]
            candidate_masked_idx = list(range(1, seq_a_len)) # only mask text_a
            random.shuffle(candidate_masked_idx)
            num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    from random import randint
                    i = randint(0, len(self.tokenizer.vocab))
                    self.tokenizer._convert_id_to_token(i)
                    tokens[pos] = self.tokenizer._convert_id_to_token(i)
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1 
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        segment_ids += ([pad_token_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
        else:
            padding_matrix = torch.zeros((self.max_img_seq_len - img_len,
                                          img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # prepare attention mask:
        # note that there is no attention from caption to image
        # because otherwise it will violate the triangle attention 
        # for caption as caption will have full attention on image. 
        max_len = self.max_seq_len + self.max_img_seq_len
        attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        # triangle mask for caption to caption
        attention_mask[c_start : c_end, c_start : c_end].copy_(self._triangle_mask[0 : seq_a_len, 0 : seq_a_len])
        # full attention for L-L, R-R
        attention_mask[l_start : l_end, l_start : l_end] = 1
        attention_mask[r_start : r_end, r_start : r_end] = 1
        # full attention for C-L, C-R
        attention_mask[c_start : c_end, l_start : l_end] = 1
        attention_mask[c_start : c_end, r_start : r_end] = 1
        # full attention for L-R:
        attention_mask[l_start : l_end, r_start : r_end] = 1
        attention_mask[r_start : r_end, l_start : l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids)
        return (input_ids, attention_mask, segment_ids, img_feat, masked_pos)


def build_dataset(yaml_file, tokenizer, args, is_train=True, dataset=None):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)

    if is_train:
        return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
            add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length, max_seq_a_length=args.max_seq_a_length,
            is_train=True, mask_prob=args.mask_prob, max_masked_tokens=args.max_masked_tokens)
    if args.use_cbs:
        dataset_class = CaptionTSVDatasetWithConstraints
    else:
        if dataset == 'oi':
            return RLTSVDataset(yaml_file, args=args, tokenizer=tokenizer)
        else:
            return CaptionTSVDataset(yaml_file, tokenizer=tokenizer,
                    add_od_labels=args.add_od_labels, max_img_seq_length=args.max_img_seq_length,
                    max_seq_length=50, max_seq_a_length=20,
                    is_train=True)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True, dataset=None):
    dataset = build_dataset(yaml_file, tokenizer, args, 
        is_train=(is_train and not args.scst), dataset=dataset)
    if is_train:
        shuffle = True
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, sampler=sampler,
        batch_size=images_per_gpu // 2,
        pin_memory=True,
    )
    return data_loader


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    scores = logits == labels 
    return scores


def train(args, train_dataloader, coco_dataloader, model, tokenizer):
    clip_model, preprocess = clip.load(op.join(args.data_dir, 'clip.pth.tar'))
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    clip_model.to(args.device)
    bert_model.to(args.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        clip_model = torch.nn.parallel.DistributedDataParallel(
            clip_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        bert_model = torch.nn.parallel.DistributedDataParallel(
            bert_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.per_gpu_train_batch_size * get_world_size() * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.scst:
        logger.info("  Distill training...")

    max_steps = args.max_steps
    def linear_rampup(current, rampup_length=max_steps):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)


    global_step, global_loss, global_mi_acc, global_distill_loss, global_acc = 0,  0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    eval_log = []
    best_score = 0
    clip_model.eval()
    bert_model.eval()
    oi_train_iter = iter(train_dataloader)
    coco_train_iter = iter(coco_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        for step in range(len(train_dataloader)):
            try:
                img_keys, batch, imgs, od_labels = oi_train_iter.next()
            except:
                oi_train_iter = iter(train_dataloader)
                img_keys, batch, imgs, od_labels = oi_train_iter.next()

            try:
                coco_img_keys, coco_batch, coco_imgs = coco_train_iter.next()
            except:
                coco_train_iter = iter(coco_dataloader)
                coco_img_keys, coco_batch, coco_imgs = coco_train_iter.next()

            if batch[0].size()!= coco_batch[0].size():
                continue

            coco_batch = tuple(t.to(args.device) for t in coco_batch)
            batch = tuple(t.to(args.device) for t in batch)
            
            model.train()
            inputs = {'input_ids': coco_batch[0], 'attention_mask': coco_batch[1],
                'token_type_ids': coco_batch[2], 'img_feats': coco_batch[3], 
                'masked_pos': coco_batch[4], 'masked_ids': coco_batch[5]
            }
            outputs = model(**inputs)
            xe_loss, logits = outputs[:2]
            masked_ids = inputs['masked_ids']
            masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

            cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
                tokenizer.convert_tokens_to_ids([tokenizer.cls_token, 
                tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]
            )
            inputs = {'is_decode': True,
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'img_feats': batch[3],
                'masked_pos': batch[4],
                'do_sample': False,
                'bos_token_id': cls_token_id,
                'pad_token_id': pad_token_id,
                'eos_token_ids': [sep_token_id],
                'mask_token_id': mask_token_id,
                # for adding od labels
                'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                # hyperparameters of beam search
                'max_length': args.max_gen_length,
                'num_beams': args.sc_beam_size,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "length_penalty": args.length_penalty,
                "num_return_sequences": 1,
                "num_keep_best": 1,
            }

            def _ids_to_captions(all_ids):
                captions = []
                for ids in all_ids:
                    c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                    captions.append(c)
                return captions

            def _tokens_to_sent(tokens):
                pretok_sent = ""
                for tok in tokens:
                    if tok.startswith("##"):
                        pretok_sent += tok[2:]
                    else:
                        pretok_sent += " " + tok
                pretok_sent = pretok_sent[1:]

                return pretok_sent

            _triangle_mask = torch.tril(torch.ones((20, 20), dtype=torch.long))
            def process_new_batch(gen_cap, batch):
                bert_input_id_list = []
                bert_attn_mask_list = []
                masked_pos_list = []
                segment_id_list = []
                input_id_list = []
                attn_mask_list = []
                masked_ids_list = []
                img_feats = batch[3]
                for c in range(len(gen_cap)):
                    nltk_tokens = nltk.word_tokenize(gen_cap[c])
                    bert_tokens = tokenizer.tokenize(gen_cap[c])
                    real_idx = []
                    if len(bert_tokens) > 18:
                        bert_tokens = bert_tokens[:18]
                        pretok_sent = _tokens_to_sent(bert_tokens)
                        nltk_tokens = nltk.word_tokenize(pretok_sent)
                    for wordidx, t in enumerate(bert_tokens):
                        if t[:2] == "##":
                            real_idx[-1].append(wordidx+1)
                        else:
                            real_idx.append([wordidx+1])

                    pos_tags = nltk.pos_tag(nltk_tokens)

                    # Only mask verbs
                    # pos_mask = [1 if 'VB' in x[1] else 0 for x in pos_tags]
                    # masked_idx = []
                    # for i in range(len(pos_mask)):
                    #     if pos_mask[i] == 1:
                    #         masked_idx.extend(real_idx[i])
                    # if len(masked_idx) > 3:
                    #     masked_idx = masked_idx[:3]

                    max_num_masked = 2
                    # pos_mask = [1 if ('NN' not in x[1] and 'DT' not in x[1] and '.' not in x[1]) else 0 for x in pos_tags]
                    pos_mask = [1 if (('VB' in x[1]) and (x[0]!='is') and (x[0])!='are') else 0 for x in pos_tags]
                    candidate_masked_idx = []
                    for i in range(len(pos_mask)):
                        if pos_mask[i] == 1:
                            candidate_masked_idx.extend(real_idx[i])

                    random.shuffle(candidate_masked_idx)
                    if len(candidate_masked_idx) < max_num_masked:
                        masked_idx = candidate_masked_idx
                    else:
                        masked_idx = candidate_masked_idx[:max_num_masked]

                    bert_tokens = [tokenizer.cls_token] + bert_tokens + [tokenizer.sep_token]
                    masked_token = [bert_tokens[i] for i in masked_idx]
                    masked_pos = torch.zeros(50, dtype=torch.int)
                    for pos in masked_idx:
                        bert_tokens[pos] = tokenizer.mask_token
                    masked_pos[masked_idx] = 1
                    if len(masked_idx) < max_num_masked:
                        masked_token = masked_token + ([tokenizer.pad_token] *
                                (max_num_masked - len(masked_idx)))

                    masked_ids = tokenizer.convert_tokens_to_ids(masked_token)

                    bert_attn_mask = torch.zeros(20, dtype=torch.long)
                    bert_attn_mask[:len(bert_tokens)] = 1
                    bert_input_id_len = len(bert_tokens)
                    bert_tokens = bert_tokens +  ([tokenizer.pad_token] * (20-bert_input_id_len))
                    bert_input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
                    
                    segment_ids = [0] * 20
                    segment_ids += batch[2][c][20:].cpu().tolist()

                    input_ids = bert_input_ids + batch[0][c][20:].cpu().tolist()
                    attn_mask = batch[1][c].cpu()
                    attn_mask[:20, :20] = 0
                    attn_mask[bert_input_id_len:20] = 0
                    attn_mask[:bert_input_id_len, :bert_input_id_len].copy_(_triangle_mask[:bert_input_id_len,:bert_input_id_len])

                    input_ids = torch.tensor(input_ids, dtype=torch.long)
                    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
                    masked_ids = torch.tensor(masked_ids, dtype=torch.long)
                    bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.long)

                    bert_input_id_list.append(bert_input_ids)
                    bert_attn_mask_list.append(bert_attn_mask)
                    masked_pos_list.append(masked_pos)
                    segment_id_list.append(segment_ids)
                    input_id_list.append(input_ids)
                    attn_mask_list.append(attn_mask)
                    masked_ids_list.append(masked_ids)

                bert_input_id_list = torch.stack(bert_input_id_list)
                bert_attn_mask_list = torch.stack(bert_attn_mask_list)
                masked_pos_list = torch.stack(masked_pos_list)
                segment_id_list = torch.stack(segment_id_list)
                input_id_list = torch.stack(input_id_list)
                attn_mask_list = torch.stack(attn_mask_list)
                masked_ids_list = torch.stack(masked_ids_list)

                return (input_id_list, attn_mask_list, segment_id_list, img_feats, masked_pos_list, masked_ids_list), \
                (bert_input_id_list, bert_attn_mask_list)
                
            model.eval()
            with torch.no_grad():
                greedy_res_raw, _ = model(**inputs)
                greedy_res_raw.squeeze_(1)  # batch_size * max_len
            greedy_res = _ids_to_captions(greedy_res_raw)
            batch_size= len(greedy_res)

            new_batch, bert_batch = process_new_batch(greedy_res, batch)

            if new_batch[5].sum() == 0:
                print("no verbs batch")
                continue

            model.train()
            new_batch = tuple(t.to(args.device) for t in new_batch)
            bert_batch = tuple(t.to(args.device) for t in bert_batch)

            bert_inputs = {'input_ids': bert_batch[0], 'attention_mask': bert_batch[1]}
            with torch.no_grad():
                bert_logits = bert_model(**bert_inputs)[0]
                bert_res_ids = torch.argmax(bert_logits, dim=-1)
                bert_res_raw = greedy_res_raw.detach().clone()
                bert_res_raw[new_batch[4][:,:20]==1] = bert_res_ids[new_batch[4][:,:20]==1]
                teacher_logits = bert_logits[new_batch[4][:,:20]==1, :]
                teacher_ids = torch.argmax(teacher_logits, dim=-1)
            bert_res = _ids_to_captions(bert_res_raw)

            inputs = {'input_ids': new_batch[0], 'attention_mask': new_batch[1],
                    'token_type_ids': new_batch[2], 'img_feats': new_batch[3], 
                    'masked_pos': new_batch[4], 'masked_ids': new_batch[5]
            }
            outputs = model(**inputs)
            unlabeled_xe_loss, student_logits = outputs[:2]

            def pretty_print_result(distill_mask, distill_diff, greedy_res, bert_res):
                for i in range(len(greedy_res)):
                    if distill_mask[i] > 0:
                        print(img_keys[i], distill_diff[i].item(), greedy_res[i], bert_res[i])

            distill_loss_fct = nn.KLDivLoss(reduction='none')
            ce_loss_fct = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
            distill_mask, distill_diff, mi_acc = mi_criterion(clip_model, greedy_res, bert_res, imgs, args.device)
            pretty_print_result(distill_mask, distill_diff, greedy_res, bert_res)
            # distill_diff = distill_diff.unsqueeze(1).repeat(1,20)
            # distill_diff = distill_diff[new_batch[4][:,:20]==1].detach()
            distill_mask = distill_mask.to(teacher_ids.dtype).unsqueeze(1).repeat(1,20)
            distill_mask = distill_mask[new_batch[4][:,:20]==1]
            teacher_ids = teacher_ids * distill_mask
            distill_loss = ce_loss_fct(student_logits, teacher_ids) / teacher_ids.size(0)
            # distill_loss = distill_loss_fct(F.log_softmax(student_logits/args.distill_temperature, dim=-1), \
            # F.softmax(teacher_logits/args.distill_temperature, dim=-1)) * (args.distill_temperature)**2
            # distill_loss = torch.sum(distill_loss, dim=-1)
            # distill_diff = distill_diff.to(distill_loss.dtype)
            # distill_diff = torch.clamp(distill_diff, min=0)
            # distill_loss = distill_loss * distill_diff
            # distill_loss = torch.sum(distill_loss) / batch_size
            distill_lambda = linear_rampup(step)
            distill_lambda = 1.0
            loss = (distill_lambda * distill_loss + args.mlm_lambda * xe_loss)/2

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            global_loss += xe_loss.item()
            global_distill_loss += distill_loss.item()
            global_mi_acc += mi_acc
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, xe_loss: {:.4f} ({:.4f}), " \
                        "distill_loss: {:.4f} ({:.4f}), Score: {:.4f} ({:.4f}), MI: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], xe_loss, global_loss / global_step, distill_loss.item(), global_distill_loss/global_step,
                        batch_acc, global_acc / global_step, mi_acc, global_mi_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        evaluate_file = evaluate(args, val_dataset, model, tokenizer,
                                checkpoint_dir)
                        with open(evaluate_file, 'r') as f:
                            res = json.load(f)
                        best_score = max(best_score, res['CIDEr'])
                        res['epoch'] = epoch
                        res['global_step'] = step
                        res['best_CIDEr'] = best_score
                        eval_log.append(res)
                        with open(args.output_dir + '/eval_logs.json', 'w') as f:
                            json.dump(eval_log, f)
    return checkpoint_dir


def scst_train_iter(args, model, clip_model, ot_criterion, scst_criterion, batch, tokenizer, imgs, od_labels, coco_dataloader, coco_img_keys):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, 
        tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]
    )
    inputs = {'is_decode': True,
        'input_ids': batch[0], 'attention_mask': batch[1],
        'token_type_ids': batch[2], 'img_feats': batch[3],
        'masked_pos': batch[4],
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.sc_beam_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": 1,
        "num_keep_best": 1,
    }

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    if args.sc_baseline_type == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res_raw, _ = model(**inputs)
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
        greedy_res = _ids_to_captions(greedy_res_raw)
    else:
        greedy_res = None

    mi_lambda = args.mi_lambda
    scst_lambda = args.scst_lambda

    model.train()
    inputs['do_sample'] = True
    inputs['num_return_sequences'] = args.sc_train_sample_n
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False
    sample_res = _ids_to_captions(sample_res_raw)

    ot_scores = ot_criterion(greedy_res[:len(greedy_res)//2], sample_res[:len(sample_res)//2], od_labels)
    ot_scores = np.append(ot_scores, np.zeros(len(ot_scores)))
    ot_acc, sep_acc = ot_criterion.get_score()

    if mi_lambda > 0:
        mi_scores, mi_acc = mi_criterion(clip_model, greedy_res, sample_res, imgs, args.device)
    else:
        mi_scores, mi_acc = 0, 0

    gt_res = [coco_dataloader.dataset.get_captions_by_key(k) for k in coco_img_keys]
    scst_scores = scst_criterion(gt_res, greedy_res[len(greedy_res)//2:], sample_res[len(sample_res)//2:])
    scst_scores = np.append(np.zeros(len(scst_scores)), scst_scores)
    sc_acc = scst_criterion.get_score()

    final_reward = ot_scores + mi_lambda * mi_scores + scst_scores * scst_lambda
    final_reward = torch.as_tensor(final_reward, device=sample_logprobs.device, dtype=torch.float)
    loss = - sample_logprobs * final_reward
    loss = loss.mean()

    return loss, sc_acc, mi_acc, sep_acc

def mi_criterion(model, greedy_res, bert_res, imgs, device):
    batch_size = len(greedy_res)
    current_rank = dist.get_rank()
    world_batch_size = batch_size * dist.get_world_size()

    imgs = imgs.to(device)
    greedy_text_inputs = clip.tokenize(greedy_res).to(device)
    bert_text_inputs = clip.tokenize(bert_res).to(device)

    with torch.no_grad():

        logit_scale, image_features, greedy_text_features = model(imgs, greedy_text_inputs)
        logit_scale, image_features, bert_text_features = model(imgs, bert_text_inputs)

        img_feature_list = [torch.ones_like(image_features) for _ in range(dist.get_world_size())]
        dist.all_gather(img_feature_list, image_features)
        image_features = torch.cat(img_feature_list)

        greedy_txt_feature_list = [torch.ones_like(greedy_text_features) for _ in range(dist.get_world_size())]
        dist.all_gather(greedy_txt_feature_list, greedy_text_features)
        greedy_text_features = torch.cat(greedy_txt_feature_list)

        bert_txt_feature_list = [torch.ones_like(bert_text_features) for _ in range(dist.get_world_size())]
        dist.all_gather(bert_txt_feature_list, bert_text_features)
        bert_text_features = torch.cat(bert_txt_feature_list)

        all_text_features = torch.cat([greedy_text_features, bert_text_features], dim=0)

        logits_per_image = logit_scale * image_features @ all_text_features.t()
        logits_per_text = logit_scale * all_text_features@ image_features.t()

        greedy_img_logits = logits_per_image[:, :world_batch_size].detach().clone()
        bert_img_logits = logits_per_image[:, :world_batch_size].detach().clone()
        diag_ind = np.diag_indices(world_batch_size)
        bert_img_logits[diag_ind] = torch.diag(logits_per_image[:, world_batch_size:])
        greedy_txt_logits = logits_per_text[:world_batch_size]
        bert_txt_logits = logits_per_text[world_batch_size:]

        greedy_img_probs = greedy_img_logits.softmax(dim=-1)
        bert_img_probs = bert_img_logits.softmax(dim=-1)
        greedy_txt_probs = greedy_txt_logits.softmax(dim=-1)
        bert_txt_probs = bert_txt_logits.softmax(dim=-1)

        greedy_img_diag = torch.diag(greedy_img_probs)
        greedy_text_diag = torch.diag(greedy_txt_probs)

        bert_img_diag = torch.diag(bert_img_probs)
        bert_text_diag = torch.diag(bert_txt_probs)

        bert_total_diag = (bert_img_diag + bert_text_diag)/2
        greedy_total_diag = (greedy_img_diag + greedy_text_diag)/2

        mi_threshold = 0.1
        distill_mask = bert_total_diag > greedy_total_diag + mi_threshold
        distill_diff = bert_total_diag - greedy_total_diag
        distill_diff = distill_diff[batch_size*current_rank:batch_size*(current_rank+1)]
        distill_mask = distill_mask[batch_size*current_rank:batch_size*(current_rank+1)]

        batch_acc = greedy_total_diag.mean().item()

        return distill_mask, distill_diff, batch_acc

def compute_mi(model, img, txt, img_features=None):
    batch_size = txt.size(0)
    current_rank = dist.get_rank()

    with torch.no_grad():
        if img_features is not None:
            image_features = img_features
            logit_scale, _, text_features = model(img,txt)
        else:
            logit_scale, image_features, text_features = model(img,txt)

        if img_features is None:
            img_feature_list = [torch.ones_like(image_features) for _ in range(dist.get_world_size())]
            dist.all_gather(img_feature_list, image_features)
            image_features = torch.cat(img_feature_list)

        txt_feature_list = [torch.ones_like(text_features) for _ in range(dist.get_world_size())]
        dist.all_gather(txt_feature_list, text_features)
        text_features = torch.cat(txt_feature_list)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        img_probs = logits_per_image.softmax(dim=-1)
        txt_probs = logits_per_text.softmax(dim=-1)

        img_diag = torch.diag(img_probs)
        text_diag = torch.diag(txt_probs)

        total_diag = (img_diag + text_diag)/2
        score = total_diag
        score = score[batch_size*current_rank:batch_size*(current_rank+1)]


        if img_features is not None:
            return score, text_features
        else:
            return score, image_features, text_features

def compute_similarity(model, img, txt, img_features=None):
    with torch.no_grad():
        if img_features is not None:
            image_features = img_features
            logit_scale, _, text_features = model(img,txt)
        else:
            logit_scale, image_features, text_features = model(img,txt)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        img_diag = torch.diag(logits_per_image)
        text_diag = torch.diag(logits_per_text)

        total_diag = (img_diag + text_diag)/2
        score = total_diag

        if img_features is not None:
            return score
        else:
            return score, image_features

def get_predict_file(output_dir, yaml_file, args):
    cc = ['pred']
    # make sure it works with/without / in end of the path.
    data = op.basename(op.join(args.data_dir, '')[:-1])
    split = op.basename(yaml_file)
    assert split.endswith('.yaml')
    split = split[:-5]
    cc.append(data)
    cc.append(split)
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.add_od_labels:
        cc.append('odlabels')
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.use_cbs:
        cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    fpath = op.splitext(predict_file)[0]
    return fpath + '.eval.json'


def get_evaluate_method(predict_file):
    if 'nocaps' in op.basename(predict_file):
        return 'nocaps'
    else:
        return 'coco'


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir,
            val_dataloader.dataset.yaml_file, args)
    if op.isfile(predict_file):
        logger.info('Skip predict. {} already exists'.format(predict_file))
    else:
        test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        torch.distributed.barrier()

    evaluate_file = get_evaluate_file(predict_file)
    if op.isfile(evaluate_file):
        logger.info('Skip evaluation. {} already exists'.format(evaluate_file))
        return evaluate_file

    if is_main_process():
        eval_method = get_evaluate_method(predict_file)
        if eval_method == 'coco':
            gt_file = val_dataloader.dataset.get_caption_file_in_coco_format()
            result = evaluate_on_coco_caption(predict_file, gt_file, outfile=evaluate_file)
        else:
            split = 'val' if 'val' in op.basename(val_dataloader.dataset.yaml_file) else 'test'
            result = evaluate_on_nocaps(split, predict_file, 
                        data_dir=args.data_dir, evaluate_file=evaluate_file)
        logger.info("evaluation result: {}".format(str(result)))

    if get_world_size() > 1:
        torch.distributed.barrier()
    return evaluate_file


def test(args, test_dataloader, model, tokenizer, predict_file):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, 
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(), 
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    inputs_param = {'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }
    if args.use_cbs:
        inputs_param.update({'use_cbs': True,
            'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
        })
    def gen_rows():
        time_meter = 0

        with torch.no_grad():
            for step, (img_keys, batch) in enumerate(tqdm(test_dataloader, total=len(test_dataloader))):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                }
                if args.use_cbs:
                    inputs.update({
                        'fsm': batch[5],
                        'num_constraints': batch[6],
                    })
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step+1)))

    tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        torch.distributed.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    if world_size > 1:
        torch.distributed.barrier()


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))


    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--coco_dir", default='datasets/coco_caption', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--coco_yaml", default='train.yaml', type=str, required=False, 
                        help="yaml file for training.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False, 
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False, 
                        help="yaml file used for validation during training.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_seq_a_length", default=40, type=int, 
                        help="The maximum sequence length for caption.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true', 
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true', 
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float, 
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float, 
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int, 
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=40, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=5,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--sc_beam_size', type=int, default=1,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=20,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument("--distill_temperature", default=2.0, type=float, help="distill_temperature.")
    parser.add_argument('--mlm_lambda', default=1.0, type=float, help="mlm_lambda.")
    parser.add_argument("--chunk_start_id", default=-1, type=int,
                        help="Image Chunk Start ID")
    parser.add_argument("--chunk_end_id", default=-1, type=int,
                        help="Image Chunk End ID")
    parser.add_argument(
        "--use_gtlabels",
        type=int, default=1,
        help="use groundtruth labels for text b or not"
    )
    parser.add_argument('--use_oi', action='store_true',
                        help='Use OpenImages dataset')
    parser.add_argument("--use_b", type=int, default=1, help="use_b")
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    args.num_gpus = get_world_size()
    args.distributed = args.num_gpus > 1
    args.device = torch.device('cuda')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        assert args.model_name_or_path is not None
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                args.model_name_or_path, num_labels=args.num_labels, finetuning_task='image_captioning')
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after
        model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer,
            args.distributed, is_train=True, dataset='oi')
        coco_dataloader = make_data_loader(args, op.join(args.coco_dir,args.coco_yaml), tokenizer, 
            args.distributed, is_train=True, dataset='coco')
        val_dataloader = None
        if args.evaluate_during_training:
            val_dataloader = make_data_loader(args, args.val_yaml, tokenizer,
                args.distributed, is_train=False)
        last_checkpoint = train(args, train_dataloader, coco_dataloader, model, tokenizer)

        # test the last checkpoint after training
        if args.do_test:
            logger.info("Evaluate on dataset: " + args.test_yaml)
            test_dataloader = make_data_loader(args, args.test_yaml, 
                tokenizer, args.distributed, is_train=False)
            evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    # inference and evaluation
    elif args.do_test or args.do_eval:
        logger.info("Evaluate on dataset: " + args.test_yaml)
        test_dataloader = make_data_loader(args, args.test_yaml,
            tokenizer, args.distributed, is_train=False)

        if not args.do_eval:
            predict_file = get_predict_file(checkpoint, test_dataloader.dataset.yaml_file, args)
            test(args, test_dataloader, model, tokenizer, predict_file)
            logger.info("Prediction results saved to: {}".format(predict_file))
        else:
            evaluate_file = evaluate(args, test_dataloader, model, tokenizer,
                    checkpoint)
            logger.info("Evaluation results saved to: {}".format(evaluate_file))

if __name__ == "__main__":
    main()
