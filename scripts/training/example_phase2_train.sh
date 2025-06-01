#!/bin/bash
# Example training script demonstrating Phase 2 features

# Example 1: Basic training with profiling (no DeepSpeed)
echo "Example 1: Basic training with profiling"
python train.py \
    --config configs.yaml \
    --parallel_mode sequential \
    --profile_steps 5 \
    --logging_steps 1 \
    --max_train_steps 10

# Example 2: Training with parallel agents
echo "Example 2: Training with parallel agents"
python train.py \
    --config configs.yaml \
    --parallel_mode parallel \
    --profile_steps 5 \
    --max_train_steps 10

# Example 3: DeepSpeed training with single GPU
echo "Example 3: DeepSpeed training (single GPU)"
accelerate launch --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_config_zero2.json \
    train.py \
    --config configs.yaml \
    --use_deepspeed true \
    --parallel_mode sequential

# Example 4: Multi-GPU DeepSpeed training with parallel agents
echo "Example 4: Multi-GPU DeepSpeed training with parallel agents"
accelerate launch --config_file accelerate_deepspeed_config.yaml \
    train.py \
    --config configs.yaml \
    --use_deepspeed true \
    --parallel_mode parallel \
    --profile_steps 5

# Example 5: Full Gemma-2-9B training with K=4 agents
echo "Example 5: Full Gemma-2-9B training"
accelerate launch --config_file accelerate_deepspeed_config.yaml \
    train.py \
    --model_name "google/gemma-2-9b" \
    --num_agents 4 \
    --use_deepspeed true \
    --parallel_mode parallel \
    --mixed_precision bf16 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 768 \
    --output_dir "ci_llm_gemma2_9b_K4_deepspeed" 