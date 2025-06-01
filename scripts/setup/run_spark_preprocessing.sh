#!/bin/bash
#SBATCH --job-name=ci_llm_spark_preprocess
#SBATCH --account=pi-jevans
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/spark_preprocess_%j.out
#SBATCH --error=slurm/spark_preprocess_%j.err

# ==============================================================================
# CI-LLM Apache Spark Data Preprocessing Job
# ==============================================================================

echo "====================================="
echo "CI-LLM Spark Data Preprocessing Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "====================================="

# Load required modules
echo "Loading modules..."
module load python/anaconda-2022.05
module load spark

# Activate CI-LLM environment
echo "Activating CI-LLM environment..."
source activate ci_llm_venv

# Set environment variables
export HF_HOME=/project/jevans/avi/hf_cache
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  PYTHONPATH=$PYTHONPATH"

# Configuration parameters
CONFIG_FILE="configs/full_train.yaml"
OVERRIDE_CONFIG=""
LOCAL_SPARK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --override_config)
            OVERRIDE_CONFIG="$2"
            shift 2
            ;;
        --local_spark)
            LOCAL_SPARK=true
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Main config: $CONFIG_FILE"
echo "  Override config: $OVERRIDE_CONFIG"
echo "  Local Spark: $LOCAL_SPARK"

# Spark configuration for cluster
SPARK_DRIVER_MEMORY="8g"
SPARK_EXECUTOR_MEMORY="4g"
SPARK_EXECUTOR_CORES="4"
SPARK_EXECUTOR_INSTANCES="8"

echo "====================================="
echo "Starting Spark preprocessing..."
echo "====================================="

# Function to run preprocessing for a specific dataset type
run_preprocessing() {
    local dataset_type=$1
    echo "Processing dataset type: $dataset_type"
    
    if [ "$LOCAL_SPARK" = true ]; then
        # Run with local Spark (for testing)
        python preprocess_spark.py \
            --config "$CONFIG_FILE" \
            ${OVERRIDE_CONFIG:+--override_config "$OVERRIDE_CONFIG"} \
            --dataset_type "$dataset_type" \
            --local_spark
    else
        # Run with cluster Spark
        spark-submit \
            --master yarn \
            --deploy-mode cluster \
            --driver-memory "$SPARK_DRIVER_MEMORY" \
            --executor-memory "$SPARK_EXECUTOR_MEMORY" \
            --executor-cores "$SPARK_EXECUTOR_CORES" \
            --num-executors "$SPARK_EXECUTOR_INSTANCES" \
            --py-files utils/ \
            --conf spark.sql.adaptive.enabled=true \
            --conf spark.sql.adaptive.coalescePartitions.enabled=true \
            --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
            preprocess_spark.py \
            --config "$CONFIG_FILE" \
            ${OVERRIDE_CONFIG:+--override_config "$OVERRIDE_CONFIG"} \
            --dataset_type "$dataset_type"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $dataset_type dataset"
    else
        echo "✗ Failed to process $dataset_type dataset"
        return 1
    fi
}

# Process training dataset
echo "Processing training data..."
run_preprocessing "train"
if [ $? -ne 0 ]; then
    echo "Training data preprocessing failed. Exiting."
    exit 1
fi

# Process evaluation dataset
echo "Processing evaluation data..."
run_preprocessing "eval"
if [ $? -ne 0 ]; then
    echo "Evaluation data preprocessing failed. Exiting."
    exit 1
fi

echo "====================================="
echo "Spark preprocessing completed!"
echo "End time: $(date)"
echo "====================================="

# Verify output files
echo "Verifying output files..."
if [ -d "data/processed/gsm8k_train.parquet" ]; then
    echo "✓ Training data Parquet files created"
    echo "  Files: $(ls -la data/processed/gsm8k_train.parquet/ | wc -l) files"
else
    echo "✗ Training data Parquet files not found"
fi

if [ -d "data/processed/gsm8k_test.parquet" ]; then
    echo "✓ Evaluation data Parquet files created" 
    echo "  Files: $(ls -la data/processed/gsm8k_test.parquet/ | wc -l) files"
else
    echo "✗ Evaluation data Parquet files not found"
fi

echo "Preprocessing job completed successfully!" 