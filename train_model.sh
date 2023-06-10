#!bin/bash
save_dir=$1

mkdir -p $save_dir
cp train_model.sh $save_dir/train_model.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_model.py \
    --data_dir ../original_data/data \
    --train_split train_1M --val_split dev  -s SRC -t TGT \
    --pretrained_encoder_name "facebook/mbart-large-50-many-to-one-mmt" \
    --total_epochs 10 \
    --max_len 210 \
    --save_dir $save_dir \
    --lr 1e-5 \
    --warmup_steps 200 \
    --grad_accum_steps 8 \
    --total_save_limit 5 \
    --logging_dir ./logs \
    --logging_steps 100 \
    --num_workers 16 \
    --clip_norm 1.0 \
    --batch_size 20 2>&1 | tee $save_dir.log
    
