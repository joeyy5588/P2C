python -m torch.distributed.launch --nproc_per_node=8 run_vivo_pretrain.py \
    --use_b 1 \
    --max_grad_norm 10.0 \
    --gradient_accumulation_steps 1 \
    --use_img_layernorm 1 \
    --output_dir output/vivo \
    --bert_model bert \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --learning_rate 5e-05 \
    --warmup_steps 0 \
    --do_train \
    --max_seq_length 15 \
    --on_memory \
    --max_img_seq_length 50 \
    --img_feature_dim 2054 \
    --drop_out 0.1 \
    --train_batch_size 1024 \
    --ckpt_period 10000 \
    --max_iters 160000 \
    --log_period 100 \
    --dataset_file data/oi.yaml \
    --data_dir YOUR_PATH_TO_OPENIMAGES_DATASET \
    --textb_sample_mode 1 \
    --texta_false_prob 0.25 \
    --use_gtlabels 1

