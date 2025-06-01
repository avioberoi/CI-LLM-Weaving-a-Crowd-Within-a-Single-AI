#!/bin/bash



if [ ! -f "eval_gsm8k.py" ]; then
    echo "Error"
    exit 1
fi

if [ ! -d "ci_llm_gemma2_2b_K4/final_checkpoint" ]; then
    echo "Error: Final checkpoint directory not found!"
    echo "Expected: ci_llm_gemma2_2b_K4/final_checkpoint"
    echo "Please verify your training completed successfully."
    exit 1
fi

echo "✓ Checkpoint directory found"

# Check if config files exist
if [ ! -f "configs_eval.yaml" ]; then
    echo "Error: configs_eval.yaml not found"
    exit 1
fi


echo ""
echo "Available evaluation options:"
echo "1. Quick checkpoint test (recommended first)"
echo "2. Debug evaluation (100 examples, ~20 min)"
echo "3. Full GSM8K evaluation (1,319 examples, ~4 hours)"
echo "4. Both debug and full (debug first, then full eval)"
echo ""

read -p "Choose option (1/2/3/4): " choice

case $choice in
    1)
        echo "Submitting checkpoint test job..."
        job_id=$(sbatch test_checkpoint.sbatch | grep -o '[0-9]*')
        echo "Test job submitted with ID: $job_id"
        echo "Monitor with: squeue -u $USER"
        echo "View output with: tail -f ci_llm_test_checkpoint_${job_id}.out"
        ;;
    2)
        echo "Submitting debug evaluation job (100 examples)..."
        job_id=$(sbatch eval_gsm8k_debug.sbatch | grep -o '[0-9]*')
        echo "Debug evaluation job submitted with ID: $job_id"
        echo "Monitor with: squeue -u $USER"
        echo "View output with: tail -f ci_llm_eval_debug_${job_id}.out"
        ;;
    3)
        echo "Submitting full evaluation job..."
        job_id=$(sbatch eval_gsm8k.sbatch | grep -o '[0-9]*')
        echo "Evaluation job submitted with ID: $job_id"
        echo "Monitor with: squeue -u $USER"
        echo "View output with: tail -f ci_llm_eval_gsm8k_${job_id}.out"
        ;;
    4)
        echo "Submitting debug evaluation first..."
        debug_job_id=$(sbatch eval_gsm8k_debug.sbatch | grep -o '[0-9]*')
        echo "Debug job submitted with ID: $debug_job_id"
        echo ""
        echo "Submitting full evaluation job (will wait for debug to complete)..."
        eval_job_id=$(sbatch --dependency=afterok:$debug_job_id eval_gsm8k.sbatch | grep -o '[0-9]*')
        echo "Full evaluation job submitted with ID: $eval_job_id (depends on $debug_job_id)"
        echo ""
        echo "Monitor with: squeue -u $USER"
        echo "Debug output: tail -f ci_llm_eval_debug_${debug_job_id}.out"
        echo "Full eval output: tail -f ci_llm_eval_gsm8k_${eval_job_id}.out"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Jobs submitted successfully!"
echo ""
