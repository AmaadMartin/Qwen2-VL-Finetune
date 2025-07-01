#!/bin/bash

DIR=$(pwd)

module load anaconda3
module load cuda
module load nvhpc
module load gcc

conda activate qwen2

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed ./finetune/Qwen2-VL-Finetune/src/training/train.py \
    --use_liger False \
    --deepspeed "./finetune/Qwen2-VL-Finetune/scripts/zero2_fp8.json" \
    --model_id $MODEL_NAME \
    --data_path ./data/json_data/lazy_ReVL_all_tasks_100000.json \
    --image_folder ./data/images \
    --remove_unused_columns False \
    --lora_enable True \
    --tune_merger True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir ./finetune/Qwen2-VL-Finetune/output/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4