#!/usr/bin/env bash
# monitor_and_rerun.sh: submit full_train.sbatch, then check every 10 minutes

# Move to script directory (project root)
cd "$(dirname "$0")"

submit_job() {
  output=$(sbatch full_train.sbatch)
  echo "$output"
  JOB_ID=$(echo "$output" | awk '{print $4}')
  echo "Submitted batch job $JOB_ID"
}

# Initial submission
submit_job

# Monitor loop
while true; do
  # Wait 10 minutes
  sleep 600
  # Check if job is still in queue
  if squeue -j "$JOB_ID" > /dev/null 2>&1; then
    echo "Job $JOB_ID still running at $(date)"
    continue
  fi
  # Job has finished; inspect error file
  ERR_FILE="ci_llm_full_train_${JOB_ID}.err"
  if [ -f "$ERR_FILE" ] && [ -s "$ERR_FILE" ]; then
    echo "Job $JOB_ID failed at $(date) — $ERR_FILE has content. Resubmitting..."
    submit_job
    continue
  else
    echo "Job $JOB_ID completed successfully at $(date)."
    break
  fi
done 