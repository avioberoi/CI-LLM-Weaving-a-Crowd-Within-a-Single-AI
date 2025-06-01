#!/bin/bash
# Setup script for CI-LLM environment on RCC cluster

set -e  # Exit on error

echo "==================================="
echo "CI-LLM Environment Setup Script"
echo "==================================="

# Load required modules
echo "Loading required modules..."
module load python/anaconda-2022.05
module load cuda/11.7

# Set project directory
PROJECT_DIR="/project/jevans/avi"
VENV_DIR="$PROJECT_DIR/ci_llm_venv"
REPO_DIR="$PROJECT_DIR/course-project-avi-oberoi"

# Create virtual environment
echo "Creating virtual environment at $VENV_DIR..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

python -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip and essential tools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.7 support
echo "Installing PyTorch with CUDA 11.7 support..."
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install other requirements
echo "Installing requirements from requirements.txt..."
cd "$REPO_DIR"
# Remove PyTorch lines from requirements since we already installed them
grep -v "torch==" requirements.txt | grep -v "torchvision==" | grep -v "torchaudio==" > requirements_no_torch.txt
pip install -r requirements_no_torch.txt
rm requirements_no_torch.txt

# Set up environment variables
echo "Setting up environment variables..."
cat > "$VENV_DIR/bin/setup_env.sh" << 'EOF'
#!/bin/bash
# Environment variables for CI-LLM

# HuggingFace cache directories
export HF_HOME="/project/jevans/avi/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Use HF_HUB_OFFLINE=1 in job scripts for offline cluster runs
# CUDA settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Python path
export PYTHONPATH="/project/jevans/avi/course-project-avi-oberoi:${PYTHONPATH}"

echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  PYTHONPATH=$PYTHONPATH"
EOF

chmod +x "$VENV_DIR/bin/setup_env.sh"

# Create activation wrapper
echo "Creating activation wrapper..."
cat > "$PROJECT_DIR/activate_ci_llm.sh" << 'EOF'
#!/bin/bash
# Activation script for CI-LLM environment

# Load modules
module load python/anaconda-2022.05
module load cuda/11.7

# Activate virtual environment
source /project/jevans/avi/ci_llm_venv/bin/activate

# Set environment variables
source /project/jevans/avi/ci_llm_venv/bin/setup_env.sh

echo "CI-LLM environment activated!"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
EOF

chmod +x "$PROJECT_DIR/activate_ci_llm.sh"

# Test installation
echo "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source $PROJECT_DIR/activate_ci_llm.sh"
echo ""
echo "Next steps:"
echo "1. Download models and datasets: python $REPO_DIR/download_assets.py"
echo "2. Run sanity check: sbatch $REPO_DIR/test_sanity.sbatch"
echo "3. Submit training job: sbatch $REPO_DIR/train_gemma3.sbatch" 