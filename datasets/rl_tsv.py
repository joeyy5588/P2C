import os
import time
import json
import logging
import random
import glob
import base64
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
from oscar.utils.misc import load_from_yaml_file
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=50, 
            max_seq_a_length=20, is_train=False):
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
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):

        tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)

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


class RLTSVDataset(Dataset):
    def __init__(self, yaml_file, args=None, tokenizer=None, seq_len=35,
                 encoding="utf-8", corpus_lines=None, on_memory=True,
                 **kwargs):
        self.cfg = load_from_yaml_file(yaml_file)
        self.root = os.path.dirname(yaml_file)
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_tsvfile = TSVFile(os.path.join(self.root, self.cfg['corpus_file']))
        self.tensorizer = CaptionTensorizer(tokenizer)

        self.datasets_names = self.cfg['corpus'].split('_')
        self.datasets_with_splits = ['googlecc', 'sbu', 'oi', 'objects365', 'tagoi']
        self.datasets_with_onesplit = ['coco', 'flickr30k', 'gqa']
        logging.info('Datasets: {}'.format(','.join(self.datasets_names)))
        self.image_label_path = self.cfg['image_label_path']
        for key, val in self.image_label_path.items():
            # get the absolute path
            if key in self.datasets_names:
                self.image_label_path[key] = os.path.join(self.root, val)
        self.image_feature_path = self.cfg['image_feature_path']
        self.image_file_name = 'features.tsv'
        if args.data_dir is not None:
            for key, val in self.image_feature_path.items():
                # get the absolute path
                if key in self.datasets_names:
                    self.image_feature_path[key] = os.path.join(args.data_dir,
                                                                val)
                else:
                    logging.info("Data {} with path {} is not used in the "
                                 "training.".format(key, val))
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_img = '' # to avoid random sentence from same image
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.image_data_prefix = os.path.join(args.data_dir, 'OpenImages/train')

        self.args = args

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        self.chunk_list = None
        if 0 <= args.chunk_start_id <= args.chunk_end_id and args.chunk_end_id >= 0:
            self.chunk_list = [str(c_i) for c_i in range(args.chunk_start_id,
                                                    args.chunk_end_id)]
            logging.info('Chunk list: {}'.format(','.join(self.chunk_list)))

        # load image tags and features
        t_start = time.time()
        self.img_label_file = None
        self.img_qa_file = None
        self.img_label_offset_map = None
        self.img_qa_offset_map = None
        self.img_feature_file = None
        self.img_feat_offset_map = None
        self.load_img_labels()
        self.load_img_tsv_features()
        t_end = time.time()
        logging.info('Info: loading img features using {} secs'
                     .format(t_end - t_start))

        # load samples into memory
        if on_memory:
            self.all_docs = []
            self.all_qa_docs = []
            self.imgid2labels = {}
            self.corpus_lines = 0
            max_tokens = 0
            for line_no in tqdm(range(len(self.corpus_tsvfile))):
                doc = []
                row = self.corpus_tsvfile.seek(line_no)
                img_info = row[0].split('_')
                label_info = row[1].split('_')
                assert img_info[0] == label_info[
                    0], "Dataset names for image and label do not match!"
                dataset_name = label_info[0]
                if dataset_name == 'cc':
                    dataset_name = 'googlecc'

                if dataset_name not in self.datasets_names:
                    continue

                if dataset_name in self.datasets_with_splits:
                    chunk_id = img_info[-2]
                    if self.chunk_list is not None and chunk_id not in self.chunk_list:
                        continue
                    else:
                        img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
                else:
                    img_feat_offset_map = self.img_feat_offset_map[dataset_name]
                assert img_info[-1] in img_feat_offset_map, "{}: Image id {} cannot be found in image feature imageid_to_index file!".format(row[0], img_info[-1])

                # append id info
                doc.append('%s|%s' % (row[0], row[1]))
                # append text_a info
                self.corpus_lines = self.corpus_lines + 1
                sample = {"doc_id": len(self.all_docs), "line": len(doc)}
                self.sample_to_doc.append(sample)
                assert len(row[2]) != 0, "Text_a is empty in {} : {}"\
                    .format(dataset_name, row[0])
                doc.append(row[2])
                # append text_b info
                self.corpus_lines = self.corpus_lines + 1
                label_id = label_info[-1]
                if 'qa' in label_info:
                    assert img_info[-1] == label_info[
                        -2], "Image ids for image and qa do not match!"
                    label_line_no = self.img_qa_offset_map[dataset_name][label_id]
                    rowb = self.img_qa_file[dataset_name].seek(label_line_no)
                else:
                    assert img_info[-1] == label_info[
                        -1], "Image ids for image and label do not match!"
                    label_line_no = self.img_label_offset_map[dataset_name][label_id]
                    rowb = self.img_label_file[dataset_name].seek(label_line_no)
                assert label_id == rowb[0]
                results = json.loads(rowb[1])
                if 'qa' not in label_info: # more intuitively, should be if 'qa' not in label_info:
                    objects = results['objects']
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": results["image_h"], "image_w": results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert results["image_h"] == self.imgid2labels[row[0]][
                            "image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert results["image_w"] == self.imgid2labels[row[0]][
                            "image_w"], "Image_w does not match in image {}!".format(row[0])
                    if args.use_gtlabels and 'gt_objects' in results:
                        # use ground-truth tags for text_b
                        textb = ' '.join([cur_d['class'] for cur_d in results["gt_objects"]])
                    else:
                        textb = ' '.join([cur_d['class'] for cur_d in objects])
                else:
                    tag_label_line_no = self.img_label_offset_map[dataset_name][img_info[-1]]
                    tag_rowb = self.img_label_file[dataset_name].seek(tag_label_line_no)
                    tag_results = json.loads(tag_rowb[1])
                    if row[0] not in self.imgid2labels:
                        self.imgid2labels[row[0]] = {
                            "image_h": tag_results["image_h"], "image_w": tag_results["image_w"],
                            "boxes": None
                        }
                    else:
                        assert tag_results["image_h"] == self.imgid2labels[row[0]][
                            "image_h"], "Image_h does not match in image {}!".format(row[0])
                        assert tag_results["image_w"] == self.imgid2labels[row[0]][
                            "image_w"], "Image_w does not match in image {}!".format(row[0])
                    textb = ' '.join(results['labels'])
                assert len(textb) != 0, "Text_b is empty in {} : {}".format(dataset_name, row[1])
                doc.append(textb)

                # add to all_docs
                max_tokens = max(max_tokens, len(doc[1].split(' '))
                                 + len(doc[2].split(' ')))
                if 'qa' in label_info:
                    self.all_qa_docs.append({"doc":doc, "doc_id": len(self.all_docs)})
                self.all_docs.append(doc)

            self.num_docs = len(self.all_docs)
            logging.info("Max_tokens: {}".format(max_tokens))
        # load samples later lazily from disk
        else:
            raise ValueError("on_memory = False Not supported yet!")

        logging.info(
            "Total docs - Corpus_lines: {}-{}".format(self.num_docs,
                                                      self.corpus_lines))
        logging.info(
            "Total QA docs - Corpus_lines: {}".format(len(self.all_qa_docs))
        )

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        return self.corpus_lines - self.num_docs

    def get_img_info(self, idx):
        sample = self.sample_to_doc[idx]
        # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
        img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
        imgid2labels = self.imgid2labels[img_id]
        return {"height": imgid2labels["image_h"], "width": imgid2labels["image_w"]}

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                raise ValueError("on_memory = False Not supported yet!")

        img_id, t1, t2 = self.get_corpus_line(item)
        if t2 == "":
            t2 = "unknown"
        # get image feature
        img_feat = self.get_img_feature(img_id)
        example = self.tensorizer.tensorize_example(img_feat=img_feat, text_b=t2)
        img_key = img_id.split('_')[-1]
        img = Image.open(os.path.join(self.image_data_prefix, img_key+'.jpg'))
        img = self.transform(img)
        # return cur_tensors
        return img_key, example, img, t2

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        assert item < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[item]
            # img_id = self.all_docs[sample["doc_id"]][0].strip() # original
            img_id = self.all_docs[sample["doc_id"]][0].strip().split('|')[0]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            self.current_img = img_id

            assert t1 != ""
            if self.args.use_b or 'qa' in self.all_docs[sample["doc_id"]][0].split('_'):
                assert t2 != ""
            else:
                t2 = ""
            return img_id, t1, t2
        else:
            raise ValueError("on_memory = False Not supported yet!")

    # tsv image labels
    def load_img_labels(self):
        self.check_img_label_file()
        self.check_img_label_offset_map()

    def check_img_label_file(self):
        if self.img_label_file is None:
            self.img_label_file = {}
            self.img_qa_file = {}
            for dataset_name in self.datasets_names:
                img_label_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'predictions_gt.tsv')
                img_qa_file_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_fileB.tsv')
                t_s = time.time()
                self.img_label_file[dataset_name] = TSVFile(img_label_file_path)
                if os.path.exists(img_qa_file_path):
                    self.img_qa_file[dataset_name] = TSVFile(img_qa_file_path)
                t_e = time.time()
                logging.info(
                    "Open image label file {}, time: {}".format(
                        img_label_file_path, (t_e - t_s)))

    def check_img_label_offset_map(self):
        if self.img_label_offset_map is None:
            self.img_label_offset_map = {}
            self.img_qa_offset_map = {}
            for dataset_name in self.datasets_names:
                img_label_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'imageid2idx.json')
                img_qa_offset_map_path = os.path.join(
                    self.image_label_path[dataset_name], 'QA_qaid2idx.json')
                t_s = time.time()
                self.img_label_offset_map[dataset_name] = json.load(
                    open(img_label_offset_map_path))
                if os.path.exists(img_qa_offset_map_path):
                    self.img_qa_offset_map[dataset_name] = json.load(
                        open(img_qa_offset_map_path))
                t_e = time.time()
                logging.info(
                    "Load img label offset map: {}, time: {}".format(
                        img_label_offset_map_path, (t_e - t_s)))

    def get_img_labels(self, image_id):
        """ decode the image labels: read the image label from the img_label.tsv """
        self.check_img_label_file()
        self.check_img_label_offset_map()

        if image_id in self.img_label_offset_map:
            img_offset = self.img_label_offset_map[image_id]

            self.img_label_file.seek(img_offset, 0)
            arr = [s.strip() for s in
                   self.img_label_file.readline().split('\t')]
            eles = json.loads(arr[1])
            labels = eles['labels']
            return labels

        return None

    # tsv feature loading
    def load_img_tsv_features(self):
        self.check_img_feature_file()
        self.check_img_feature_offset_map()

    def check_img_feature_file(self):
        if self.img_feature_file is None:
            # self.img_feature_file = [] # original
            self.img_feature_file = {}
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading dataset {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    self.img_feature_file[dataset_name] = {}
                    self.img_feat_offset_map[dataset_name] = {}
                    chunk_list = []
                    if self.chunk_list is not None:
                        chunk_list = self.chunk_list
                        chunk_file_list = []
                        for chunk_fp_id in chunk_list:
                            chunk_file_list.append(
                                os.path.join(self.image_feature_path[dataset_name], chunk_fp_id, self.image_file_name)
                            )
                        if dataset_name == 'googlecc':
                            for i, (chunk_fp_id, chunk_fp) in enumerate(zip(chunk_list, chunk_file_list)):
                                assert os.path.exists(chunk_file_list[i]), "Chunk file {} does not exists!".format(chunk_fp)
                    else:
                        chunk_file_list = glob.glob(
                            self.image_feature_path[dataset_name] + "/*/{}".format(self.image_file_name)
                        )
                        for chunk_fp in chunk_file_list:
                            chunk_fp_id = chunk_fp.split('/')[-2]
                            chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feature_file[dataset_name][chunk_fp_id] = TSVFile(chunk_fp)
                        chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                        assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_offsetmap, 'r'))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                    logging.info(
                        "Image chunk info: {}".format('\n'.join(chunk_file_list))
                    )
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = os.path.join(self.image_feature_path[dataset_name], self.image_file_name)
                    self.img_feature_file[dataset_name] = TSVFile(chunk_fp)
                    chunk_offsetmap = os.path.join(os.path.dirname(chunk_fp), 'imageid2idx.json')
                    assert os.path.isfile(chunk_offsetmap), "Imageid2idx file {} does not exists!".format(chunk_offsetmap)
                    self.img_feat_offset_map[dataset_name] = json.load(open(chunk_offsetmap, 'r'))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError("Not supported dataset: {}".format(dataset_name))

    def check_img_feature_offset_map(self):
        """ load the image feature offset map """
        if self.img_feat_offset_map is None:
            self.img_feat_offset_map = {}
            for dataset_name in self.datasets_names:
                logging.info("* Loading imageid2idx_map {}".format(dataset_name))
                if dataset_name in self.datasets_with_splits:
                    chunk_list = []
                    chunk_file_list = glob.glob(
                        self.image_feature_path[
                            dataset_name] + "/*/imageid2idx.json"
                    )
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        chunk_list.append(chunk_fp_id)
                    logging.info(
                        "* Load Image Chunks {}".format(len(chunk_list)))

                    t_s_total = time.time()
                    for chunk_fp in chunk_file_list:
                        chunk_fp_id = chunk_fp.split('/')[-2]
                        t_s = time.time()
                        self.img_feat_offset_map[dataset_name][
                            chunk_fp_id] = json.load(open(chunk_fp))
                        t_e = time.time()
                        logging.info(
                            "Open image chunk {}, time: {}".format(
                                chunk_fp_id, (t_e - t_s)))
                    t_e_total = time.time()
                    logging.info(
                        "Open total {} image chunks, time: {}".format(
                            len(chunk_list), (t_e_total - t_s_total)))
                elif dataset_name in self.datasets_with_onesplit:
                    t_s = time.time()
                    chunk_fp = self.image_feature_path[
                                   dataset_name] + "/imageid2idx.json"
                    self.img_feat_offset_map[dataset_name] = json.load(
                        open(chunk_fp))
                    t_e = time.time()
                    logging.info(
                        "Open dataset {}, time: {}".format(
                            chunk_fp, (t_e - t_s)))
                else:
                    raise ValueError(
                        "Not supported dataset: {}".format(dataset_name))

    def get_img_feature(self, image_id):
        """ decode the image feature: read the image feature from the right chunk id """
        self.check_img_feature_file()
        self.check_img_feature_offset_map()
        img_infos = image_id.split('_')
        dataset_name = img_infos[0]
        if dataset_name == 'cc':
            dataset_name = 'googlecc'
        img_id = img_infos[-1]
        if dataset_name in self.datasets_with_splits:
            chunk_id = img_infos[-2]
            img_feat_offset_map = self.img_feat_offset_map[dataset_name][chunk_id]
            img_feature_file = self.img_feature_file[dataset_name][chunk_id]
        else:
            img_feat_offset_map = self.img_feat_offset_map[dataset_name]
            img_feature_file = self.img_feature_file[dataset_name]
        if img_id in img_feat_offset_map:
            img_offset = img_feat_offset_map[img_id]

            arr = img_feature_file.seek(img_offset)
            num_boxes = int(arr[1])
            feat = np.frombuffer(base64.b64decode(arr[-1]),
                                 dtype=np.float32).reshape(
                (num_boxes, self.args.img_feature_dim))
            feat = torch.from_numpy(feat)
            return feat

        return None


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None,
                 lm_labels=None, img_id=None, is_img_match=None,
                 img_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model

        self.img_id = img_id
        self.is_img_match = is_img_match
        self.img_label = img_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids, img_feat_len, is_next=None,
                 is_img_match=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.img_feat_len = img_feat_len
        self.is_img_match = is_img_match


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logging.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(
                        token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def convert_tokenb_to_features(args, tokens_b, max_seq_length, tokenizer,
                                img_feat_len):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param args: parameter settings
    :param img_feat_len: lens of actual img features
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    if len(tokens_b) > max_seq_length - 1:
        tokens_b = tokens_b[:(max_seq_length - 1)]

    tokens_b, t2_label = random_word(tokens_b, tokenizer)

    # concatenate lm labels and account for SEP
    lm_label_ids = t2_label + [-1]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []

    if tokens_b:
        assert len(tokens_b) > 0
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # image features
    if args.max_img_seq_length > 0:
        if img_feat_len > args.max_img_seq_length:
            input_mask = input_mask + [1] * img_feat_len
        else:
            input_mask = input_mask + [1] * img_feat_len
            pad_img_feat_len = args.max_img_seq_length - img_feat_len
            input_mask = input_mask + ([0] * pad_img_feat_len)

    lm_label_ids = lm_label_ids + [-1] * args.max_img_seq_length

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             img_feat_len=img_feat_len)
    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()