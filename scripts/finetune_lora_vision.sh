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

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

deepspeed ./finetune/Qwen2-VL-Finetune/src/training/train.py \
    --use_liger False \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed "./finetune/Qwen2-VL-Finetune/scripts/zero3_offload.json" \
    --model_id $MODEL_NAME \
    --data_path ./data/json_data/lazy_ReVL_all_tasks_100000.json \
    --image_folder ./data/images \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir ./finetune/Qwen2-VL-Finetune/output/lora_vision_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
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
    --dataloader_num_workers 4 \
    --bits 4