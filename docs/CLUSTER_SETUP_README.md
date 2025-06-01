# CI-LLM Cluster Setup and Execution Guide

This guide provides step-by-step instructions for setting up and running the CI-LLM project on the RCC cluster at UChicago.

## Prerequisites
- Access to RCC cluster with GPU allocation
- HF_TOKEN set in your ~/.bashrc for accessing Gemma models
- Sufficient storage space in /project/jevans/avi (>100GB recommended)

## Setup Process

### 1. Initial Setup - Create Virtual Environment

First, connect to the cluster and navigate to the project directory:
```bash
ssh <username>@midway3.rcc.uchicago.edu
cd /project/jevans/avi/course-project-avi-oberoi
```

Run the setup script to create the virtual environment:
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

This script will:
- Create a virtual environment at `/project/jevans/avi/ci_llm_venv`
- Install PyTorch 2.0.1 with CUDA 11.7 support
- Install all required dependencies
- Create activation scripts for future use

### 2. Download Models and Datasets

After setting up the environment, download the required models and datasets:
```bash
# Activate the environment
source /project/jevans/avi/activate_ci_llm.sh

# Run the download script
python download_assets.py
```

This will download:
- Gemma-2 models (2B, 9B)
- Gemma-3 models (1B-it, 4B-it)
- GSM8K dataset

**Note**: This step requires internet access, so run it on the login node before submitting jobs.

### 3. Run Sanity Check

Before running full training, verify the setup:
```bash
sbatch test_sanity.sbatch
```

Check the output:
```bash
tail -f ci_llm_sanity_*.out
```

The sanity check will:
- Verify GPU availability
- Test model initialization
- Check Accelerate integration
- Test checkpointing
- Verify diversity loss calculation
- Test generation with aggregator

### 4. Run Short Test Training

Once sanity checks pass, run a short training test:
```bash
sbatch short_test.sbatch
```

This uses:
- Gemma-2-2b model (smaller for testing)
- 2 agents
- 50 training steps
- Reduced sequence length

Monitor progress:
```bash
tail -f ci_llm_short_test_*.out
```

### 5. Run Full Gemma-3 Training

After successful testing, launch the full training:
```bash
sbatch train_gemma3.sbatch
```

This configuration:
- Uses Gemma-3-4b-it model
- 4 CI-LLM agents
- DeepSpeed ZeRO-2 optimization
- 4 GPUs with DDP
- Mixed precision (bf16)
- Full GSM8K dataset

Monitor training:
```bash
tail -f ci_llm_gemma3_*.out
```

## Configuration Files

### configs.yaml
Main configuration file with all hyperparameters. Key settings:
- `model_name`: Which model to use
- `num_agents`: Number of CI-LLM agents (K)
- `parallel_mode`: "sequential" or "parallel"
- `use_deepspeed`: Enable DeepSpeed optimization

### accelerate_deepspeed_config.yaml
Accelerate configuration for DeepSpeed integration:
- Uses ZeRO Stage 2
- Mixed precision bf16
- 4 processes (GPUs)

### ds_config_zero2.json
DeepSpeed configuration:
- ZeRO-2 optimization
- Gradient clipping
- Automatic batch size handling

## Output Structure

Training outputs are saved in:
```
ci_llm_output/
├── gemma3_run_<job_id>/
│   ├── checkpoint_step_200/
│   ├── checkpoint_step_400/
│   ├── final_checkpoint/
│   └── training.log
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Training progress
- Individual adapter weights
- Configuration metadata

## Monitoring and Debugging

### Check GPU Usage
```bash
squeue -u $USER  # Check job status
scontrol show job <job_id>  # Detailed job info
```

### Common Issues

1. **Out of Memory**
   - Reduce batch_size
   - Increase gradient_accumulation_steps
   - Enable gradient checkpointing

2. **Models Not Found**
   - Ensure HF_TOKEN is set
   - Check if download_assets.py completed successfully
   - Verify HF_HOME path in environment

3. **CUDA Errors**
   - Check CUDA_VISIBLE_DEVICES
   - Verify GPU allocation in SLURM script
   - Ensure cuda/11.7 module is loaded

## Resume Training

To resume from a checkpoint:
```yaml
# In configs.yaml:
resume_from_checkpoint: "ci_llm_output/gemma3_run_123456/checkpoint_step_400"
```

Then resubmit the job:
```bash
sbatch train_gemma3.sbatch
```

## Evaluation

After training completes, evaluation runs automatically. For manual evaluation:
```bash
python eval_gsm8k.py \
    --model_name "google/gemma-3-4b-it" \
    --trained_peft_checkpoint_dir "ci_llm_output/gemma3_run_<job_id>/final_checkpoint" \
    --output_file "eval_results.jsonl"
```

## Advanced Usage

### Different Model Sizes
To use larger Gemma-3 models, update download_assets.py to include:
- `google/gemma-3-12b-it`
- `google/gemma-3-27b-it`

Then adjust SLURM resources accordingly:
- More memory (512G+)
- More GPUs (8 for 27B model)
- Longer time limits

### Multi-Node Training
For multi-node setup, modify accelerate_deepspeed_config.yaml:
```yaml
num_machines: 2
machine_rank: 0  # 0 for first node, 1 for second
```

## Support

For RCC-specific issues: help@rcc.uchicago.edu
For CI-LLM issues: Check the project repository 