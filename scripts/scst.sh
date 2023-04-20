python -m torch.distributed.launch --nproc_per_node=1 run_captioning.py \
    --model_name_or_path YOUR_CHECKPOINT_FROM_XE \
    --data_dir PATH_TO_COCO_DIR \
    --do_lower_case \
    --do_train \
    --add_od_labels \
    --scheduler constant \
    --learning_rate 2e-6 \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs 5 \
    --tie_weights \
    --freeze_embedding \
    --scst \
    --output_dir output/scst/ \
    --logging_steps 100 \
    --save_steps 5000 \

