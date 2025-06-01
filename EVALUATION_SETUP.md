# CI-LLM GSM8K Evaluation Setup

This document explains how to evaluate your trained CI-LLM model on the GSM8K dataset.

## Files Created

### 1. Configuration Files
- **`configs_eval.yaml`**: Evaluation configuration that matches your training setup
  - Points to your final checkpoint: `ci_llm_gemma2_2b_K4/final_checkpoint`
  - Configures generation settings for GSM8K
  - Sets up 4 agents (K=4) as per your training

### 2. Batch Files
- **`eval_gsm8k.sbatch`**: Main evaluation job (4 hours, 1 GPU, 64GB RAM, ssd-gpu partition)
- **`eval_gsm8k_debug.sbatch`**: Debug evaluation job (30 minutes, 1 GPU, 64GB RAM, gpu partition)
- **`test_checkpoint.sbatch`**: Quick test to verify checkpoint loading (30 minutes, 1 GPU, 32GB RAM)

### 3. Scripts
- **`eval_gsm8k.py`**: Main evaluation script (already existed, but fixed parameter mismatch)
- **`test_checkpoint_loading.py`**: Quick test script to verify checkpoint works
- **`run_evaluation.sh`**: Helper script to submit jobs easily

## How to Run Evaluation

### Option 1: Using the Helper Script (Recommended)
```bash
cd /project/jevans/avi/course-project-avi-oberoi
./run_evaluation.sh
```

This script will:
1. Check that all files exist
2. Offer you four options:
   - Quick checkpoint test only
   - Debug evaluation (100 examples)
   - Full evaluation only  
   - Both debug and full (debug first, then evaluation)

### Option 2: Manual Submission

#### Test the checkpoint first (recommended):
```bash
sbatch test_checkpoint.sbatch
```

#### Run debug evaluation (100 examples):
```bash
sbatch eval_gsm8k_debug.sbatch
```

#### Run full evaluation:
```bash
sbatch eval_gsm8k.sbatch
```

#### Run debug then full with dependency:
```bash
DEBUG_JOB=$(sbatch eval_gsm8k_debug.sbatch | grep -o '[0-9]*')
sbatch --dependency=afterok:$DEBUG_JOB eval_gsm8k.sbatch
```

## Monitoring Jobs

- **Check job status**: `squeue -u $USER`
- **View output**: `tail -f ci_llm_eval_gsm8k_<job_id>.out`
- **View errors**: `tail -f ci_llm_eval_gsm8k_<job_id>.err`
- **Cancel job**: `scancel <job_id>`

## Expected Output

### Checkpoint Test
- Should load the model successfully
- Test a simple forward pass
- Confirm 4 agents are loaded

### Full Evaluation
- Processes all 1,319 GSM8K test examples
- Saves detailed results to `eval_results/gsm8k_results.jsonl`
- Reports final accuracy percentage
- Should take 2-4 hours depending on generation length

## Results

Results will be saved in:
- **`eval_results/gsm8k_results.jsonl`**: Detailed per-example results
- Each line contains: question, gold answer, generated answer, prediction, correctness

## Potential Issues Fixed

1. **Parameter name mismatch**: Fixed `trained_peft_checkpoint_dir` vs `trained_checkpoint_dir`
2. **Tokenizer loading**: Added fallback for tokenizer loading issues
3. **Error handling**: Added comprehensive error handling for model loading
4. **Resource allocation**: Appropriately sized GPU/memory requirements for evaluation
5. **Checkpoint verification**: Added upfront verification that checkpoint exists

## Resource Requirements

- **Test job**: 30 minutes, 1 GPU, 32GB RAM
- **Evaluation job**: 4 hours, 1 GPU, 64GB RAM
- Both jobs use the `ssd-gpu` partition with `ssd` account/QOS

## Next Steps After Evaluation

1. Check the final accuracy in the job output
2. Analyze detailed results in `eval_results/gsm8k_results.jsonl`
3. Compare with baseline performance
4. Look for common error patterns in the generated solutions 