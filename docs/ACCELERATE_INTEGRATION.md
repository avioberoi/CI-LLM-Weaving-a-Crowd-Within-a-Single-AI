# CI-LLM Accelerate Integration Guide

## Overview

This document explains the HuggingFace Accelerate integration in the CI-LLM training framework, including distributed training support, robust checkpointing, and optimized data loading.

## Key Improvements

### 1. **HuggingFace Accelerate Integration**

The training script now uses Accelerate to handle:
- **Device Placement**: Automatic handling of CPU/GPU placement
- **Distributed Training**: Support for single-node multi-GPU training via DDP
- **Mixed Precision**: Optional fp16/bf16 training for faster computation
- **Gradient Accumulation**: Proper handling across distributed setups

#### Usage:
```bash
# Single GPU
python train.py
# or
accelerate launch --num_processes=1 train.py

# Multi-GPU (single node)
accelerate launch --num_processes=4 train.py  # for 4 GPUs
```

### 2. **Robust Checkpointing & Resume**

The framework now supports complete training state saving and resumption:

#### Saving Checkpoints:
- Automatic checkpoints saved every `save_steps` (configured in `configs.yaml`)
- Saves to `{output_dir}/checkpoint_step_{global_step}/`
- Final checkpoint saved to `{output_dir}/final_checkpoint/`

#### What's Saved:
- All K agent adapter weights (via custom metadata)
- Optimizer state
- Learning rate scheduler state
- Training progress (epoch, global_step)
- Random states for reproducibility
- Tokenizer
- Training configuration

#### Resuming Training:
```yaml
# In configs.yaml:
resume_from_checkpoint: "ci_llm_output/checkpoint_step_100"
```
Or via command line:
```bash
python train.py --resume_from_checkpoint path/to/checkpoint
```

### 3. **Optimized Data Loading**

- **Multi-worker Loading**: `num_workers` configurable (default: 4)
- **Pin Memory**: Automatically enabled for CUDA to speed up GPU transfer
- **Efficient Batching**: Maintained through Accelerate's DataLoader wrapper

### 4. **Structured Logging**

- Console and file logging (`{output_dir}/training.log`)
- Distributed-aware logging (only main process logs)
- Metrics logged via `accelerator.log()` for integration with W&B/TensorBoard
- Configurable logging frequency via `logging_steps`

### 5. **Mixed Precision Training**

Enable mixed precision for faster training and lower memory usage:
```yaml
# In configs.yaml:
mixed_precision: "bf16"  # Options: "no", "fp16", "bf16"
```

## Configuration Parameters

New/updated parameters in `configs.yaml`:

```yaml
# Checkpointing
resume_from_checkpoint: null     # Path to checkpoint directory
save_steps: 100                  # Save checkpoint every N steps

# Data Loading
num_workers: 4                   # DataLoader worker processes

# Training
mixed_precision: "no"            # Mixed precision training
weight_decay: 0.01              # AdamW weight decay
logging_steps: 10               # Log metrics every N steps
seed: 42                        # Random seed for reproducibility
```

## Multi-GPU Training Setup

For multi-GPU training on a single node:

1. **Update SLURM script**:
```bash
#SBATCH --gres=gpu:4  # Request 4 GPUs
```

2. **Launch with Accelerate**:
```bash
accelerate launch --num_processes=4 train.py
```

3. **Effective batch size** = `batch_size * num_gpus * gradient_accumulation_steps`

## Testing & Validation

Run the comprehensive dry-run check:
```bash
python dry_run_sanity_check.py
```

This tests:
- Environment setup
- Model initialization
- Accelerate integration
- Checkpointing save/load
- Diversity loss calculation
- Generation with aggregator

## Implementation Details

### Custom Checkpoint Handling

Due to the multi-adapter PEFT model structure, we maintain custom checkpoint metadata:
- `ci_llm_meta_config.json`: LoRA configuration and K value
- `agent{i}_weights.pth`: Individual adapter weights

This ensures compatibility with both Accelerate's checkpoint system and the existing CILLMModel loading logic.

### SWD Representation Extraction

The hidden state extraction for SWD loss is now modular:
- Default: Last token representation
- Options: Mean pooling, first token
- Configurable via future updates

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `mixed_precision: "bf16"`
- Reduce `max_seq_length`

### Checkpoint Loading Issues
- Ensure checkpoint path exists
- Check `ci_llm_meta_config.json` is present
- Verify K value matches between checkpoint and config

### Multi-GPU Issues
- Ensure all GPUs are visible: `nvidia-smi`
- Check NCCL backend is available
- Verify no port conflicts for distributed training

## Future Enhancements

- Multi-node distributed training support
- Integration with experiment tracking (W&B, TensorBoard)
- Automatic mixed precision tuning
- Dynamic batching for variable length sequences 