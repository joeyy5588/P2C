python -m torch.distributed.launch --nproc_per_node=2 run_captioning.py \
    --eval_model_dir CHECKPOINT_DIR \
    --data_dir PATH_TO_NOCAPS_DIR \
    --do_eval \
    --do_test \
    --num_beams 5 \
    --use_cbs \
    --max_gen_length 20 \
    --per_gpu_eval_batch_size 1 
