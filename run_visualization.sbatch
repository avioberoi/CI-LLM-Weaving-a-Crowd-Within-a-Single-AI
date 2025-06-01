#!/bin/bash

#SBATCH --job-name=dask_viz
#SBATCH --output=logs/dask_viz_%j.out
#SBATCH --error=logs/dask_viz_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Anaconda
module load anaconda/2023.09

# Create and activate the conda environment if it doesn't exist
if ! conda env list | grep -q "dask_viz_env"; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
fi

# Activate the environment
source activate dask_viz_env

# Run the visualization script
python dask_eda_visualization.py

# Deactivate the environment
conda deactivate 