# CI-LLM: Weaving a "Crowd Within" a Single AI
## Collective Intelligence-Informed Large Language Models

> **A Technical Journey in Modeling Nuanced Societal Discourse Through Multi-Agent AI**  
> *Avi Oberoi and Abenezer*  
> *University of Chicago*  
> *2024*

---

## Abstract

This repository details the design, implementation, and scalable computing strategies employed in the Collective Intelligence Large Language Model (CI-LLM) project. CI-LLM addresses the challenge of modeling nuanced and complex societal discourse by internalizing principles of collective intelligence—diversity, independence, and aggregation—within a single, parameter-efficient LLM. We document our architectural considerations, iterative development process, and integration of advanced techniques like QLoRA, Hugging Face Accelerate, and DeepSpeed for efficient and scalable training on high-performance computing clusters.

---

## Table of Contents

1. [The Social Science Problem](#the-social-science-problem)
2. [The Vision: CI-LLM Architecture](#the-vision-ci-llm-architecture)
3. [Scalable Computing Justification](#scalable-computing-justification)
4. [Technical Implementation](#technical-implementation)
5. [Project Structure](#project-structure)
6. [Results and Analysis](#results-and-analysis)
7. [Quick Start](#quick-start)
8. [Usage Instructions](#usage-instructions)
9. [Hardware Requirements](#hardware-requirements)
10. [Implementation Journey](#implementation-journey)

---

## The Social Science Problem

### The Challenge: Moving Beyond Single-Perspective AI

Large Language Models (LLMs) like the Gemma series have demonstrated remarkable capabilities in processing and generating human language, but they typically operate as a single, albeit powerful, "mind." This creates fundamental limitations when tackling problems demanding true multi-perspective understanding or robust reasoning in the face of ambiguity:

- **Brittleness and Idiosyncrasies**: A single model may have specific failure modes or "blind spots"
- **Entrenched Biases**: Reflecting dominant narratives from training data without surfacing alternative, valid viewpoints
- **Difficulty in Representing True Nuance**: While models can be prompted for different personas, their core reasoning pathways may converge or struggle to balance genuinely conflicting information

### The Social Science Question

Can we create an LLM that doesn't just mimic different perspectives when prompted, but actually fosters a form of **internal collective intelligence**? Can we build an AI that benefits from the "wisdom of crowds" by having multiple, somewhat independent, diverse "agents" or reasoning pathways within its own structure?

This question is central to understanding how artificial systems can better model the complex, multi-faceted nature of human discourse and decision-making processes that characterize real-world social phenomena.

---

## The Vision: CI-LLM Architecture

Our core hypothesis is that by explicitly designing an LLM architecture that embodies key principles of human collective intelligence—namely **Independence, Diversity, and Aggregation**—we can create models that are more robust, generate more nuanced outputs, and potentially generalize better.

### High-Level Architecture

The CI-LLM framework consists of:

#### 1. **Shared Backbone LLM**
- **Current Implementation**: Google Gemma-2-2B (configurable to other Gemma-2 variants)
- **Quantization**: 4-bit quantization via QLoRA for memory efficiency
- **Status**: Backbone weights remain frozen during CI-specific training to conserve resources

#### 2. **K Lightweight "Agent" Personas**
- **Implementation**: K=4 distinct "agents" (configurable)
- **Architecture**: Each agent realized as Low-Rank Adaptation (LoRA) weights
- **Efficiency**: QLoRA quantizes the base model to 4-bits, training only small LoRA adapter layers per agent
- **Current Settings**: 
  - LoRA rank: 16
  - LoRA alpha: 32
  - LoRA dropout: 0.05
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

#### 3. **Mechanisms for Independence & Diversity**

**Stochasticity in Agent Updates:**
- **DropConnect Implementation**: Custom masking on LoRA weights during forward/backward passes
- **Current Setting**: `dropconnect_p: 0.05` (enabled for diversity experimentation)
- **Mechanism**: Random binary masks applied to LoRA A and B weight matrices during training

**Explicit Diversity Regularization:**
- **Method**: Sliced Wasserstein Distance (SWD) between agent latent representations
- **Implementation**: `SlicedWassersteinDiversityRegularizer` with 64 random projections
- **Loss Function**: `L_total = (1/K) * Σ L_k(task) - λ * R_SW(diversity)`
- **Lambda Weight**: `λ_sw = 0.01`

#### 4. **Aggregation Mechanism**
- **Method**: Dirichlet-weighted Bayesian ensemble
- **Current Implementation**: Simple averaging (expected value of symmetric Dirichlet prior)
- **Formula**: `ĝ(x) = (1/K) * Σ g_k(x)`
- **Future Enhancement**: Learnable Dirichlet concentration parameters

---

## Scalable Computing Justification

### The Computational Challenge

Training CI-LLM presents significant computational challenges that necessitate high-performance computing infrastructure across both data preprocessing and model training phases:

#### 1. **Model Scale Complexity**
- **Base Model**: Gemma-2-2B has 2 billion parameters (currently deployed)
- **Multi-Agent Training**: K=4 agents with independent LoRA adapters
- **Memory Requirements**: Even with 4-bit quantization, multi-agent training requires substantial VRAM
- **Training Time**: Diversity regularization and ensemble training significantly increase computational overhead

#### 2. **Data Processing Scale Requirements**
- **Dataset Size**: GSM8K contains 7,473 training samples and 1,319 test samples requiring tokenization
- **Distributed Preprocessing**: Apache Spark implementation for parallel tokenization across cluster nodes
- **Memory-Intensive Operations**: Tokenization and label masking for instruction-following format
- **Cross-Modal Operations**: Integration between Hugging Face datasets and distributed Spark processing

#### 3. **Why Scalable Computing is Essential**

**Memory Constraints:**
- Standard GPUs cannot accommodate the memory requirements for multi-agent training
- DeepSpeed ZeRO-2 optimization enables sharding of optimizer states and gradients across multiple GPUs
- Enables training of larger Gemma-2 variants with K=4+ agents on 40GB A100s
- **Distributed Data Processing**: Apache Spark distributes tokenization workload across cluster nodes to handle large-scale datasets

**Training Efficiency:**
- Distributed training across multiple GPUs reduces training time from weeks to days
- Gradient accumulation and mixed precision training optimize memory usage
- Parallel agent processing (Phase 2 implementation) reduces adapter switching overhead
- **Preprocessing Pipeline**: Spark-based preprocessing eliminates bottlenecks in data loading and tokenization

**Scalability for Research:**
- Ability to experiment with larger K values (more agents)
- Support for larger base models (Gemma-2-9B and beyond)
- Multi-node training capabilities for future scaling
- **Dataset Scalability**: Spark preprocessing scales to datasets with millions of samples without memory constraints

#### 4. **Large-Scale Computing Methods Employed**

**Apache Spark Distributed Preprocessing:**
- **Cluster-Wide Tokenization**: Distributed User-Defined Functions (UDFs) for parallel tokenization across executor nodes
- **Memory Optimization**: Lazy evaluation and DataFrame operations prevent memory overflow on large datasets
- **Format Standardization**: Converts Hugging Face datasets to Parquet format for efficient training data loading
- **Schema Enforcement**: Structured data types ensure consistent tokenization output (`input_ids`, `attention_mask`, `labels`)
- **Fault Tolerance**: Spark's RDD lineage provides automatic recovery from node failures during preprocessing

**DeepSpeed ZeRO-2 Integration:**
- Optimizer state and gradient sharding across GPUs
- Dramatic VRAM reduction per GPU
- CPU offloading for additional memory savings

**Hugging Face Accelerate:**
- Seamless distributed training abstraction
- Automatic device placement and data parallel processing
- Mixed precision training (bf16) for improved efficiency

**Parallel Agent Architecture:**
- K separate PeftModel instances sharing frozen base model weights
- Memory-efficient weight sharing through PyTorch tensor storage
- Reduced adapter switching overhead

---

## Technical Implementation

### Core Components

#### 1. **Distributed Data Preprocessing** (`preprocess_spark.py`)
- **Apache Spark Integration**: Distributed tokenization using PySpark for cluster-wide data processing
- **Hugging Face Integration**: Seamless loading from `load_from_disk` with automatic schema inference
- **UDF-Based Tokenization**: Custom User-Defined Functions for parallel tokenization with `AutoTokenizer`
- **Format Conversion**: Converts datasets to optimized Parquet format for efficient training data loading
- **Memory Management**: Lazy evaluation and DataFrame operations prevent memory overflow on executor nodes
- **Configuration Support**: YAML-driven configuration for tokenizer settings, sequence lengths, and output paths

#### 2. **Model Architecture** (`src/models/gemma_backbone.py`)
- **Class**: `CILLMModel` - CI-LLM Model with K independent LoRA adapter heads
- **Modes**: Supports both `sequential` (original) and `parallel` (Phase 2) execution modes
- **Features**: QLoRA 4-bit quantization, DropConnect mechanism, comprehensive checkpointing

#### 3. **Diversity Regularization** (`src/losses/sliced_w2.py`)
- **Class**: `SlicedWassersteinDiversityRegularizer`
- **Method**: Computes Sliced Wasserstein Distance between agent representations
- **Details**: Pairwise SWD computation, configurable projections (64 default), last token representation

#### 4. **Bayesian Aggregation** (`src/aggregator/dirichlet_bayesian.py`)
- **Class**: `DirichletAggregator`
- **Implementation**: Symmetric Dirichlet prior with uniform expected weights
- **Support**: Both sampled and deterministic averaging, handles logits and probabilities

#### 5. **Training Pipeline** (`src/training/train.py`)
- **Phase 1**: Accelerate integration, robust checkpointing, structured logging
- **Phase 2**: DeepSpeed ZeRO-2, parallel agent forward pass, performance profiling

---

## Project Structure

```
ci-llm/
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   └── gemma_backbone.py     # CI-LLM implementation
│   ├── losses/                   # Loss functions
│   │   └── sliced_w2.py          # Sliced Wasserstein regularization
│   ├── aggregator/               # Aggregation mechanisms
│   │   └── dirichlet_bayesian.py # Dirichlet ensemble aggregator
│   ├── training/                 # Training scripts
│   │   └── train.py              # Main training loop
│   └── evaluation/               # Evaluation scripts
│       └── eval_gsm8k.py         # GSM8K evaluation
├── configs/                      # Configuration files
│   ├── default.yaml              # Main configuration (Gemma-2-2B)
│   ├── eval.yaml                 # Evaluation configuration
│   ├── full_train.yaml           # Alternative config (Gemma-2-2B)
│   └── deepspeed/                # DeepSpeed configurations
├── scripts/                      # Utility scripts
│   ├── setup/                    # Environment setup
│   ├── training/                 # Training utilities
│   └── evaluation/               # Evaluation utilities
├── tests/                        # Test files
│   └── dry_run_sanity_check.py   # Comprehensive system test
├── docs/                         # Documentation
├── data/                         # Data storage
│   ├── hf_datasets/              # Hugging Face datasets
│   └── processed/                # Spark-processed Parquet files
├── utils/                        # Common utilities
│   ├── __init__.py               # Package initialization
│   └── general_utils.py          # Configuration and utility functions
├── preprocess_spark.py           # Apache Spark distributed preprocessing
└── environment.yml               # Conda environment
```

---

## Results and Analysis

### Training Convergence Failure Hypothesis

Our evaluation on GSM8K revealed **0% accuracy across 20 test samples**, indicating complete model failure. Analysis suggests this stems from **multi-objective loss function divergence** rather than architectural flaws.

**Primary Issues Identified:**

**Conflicting Optimization Objectives**: The dual-objective loss function combines cross-entropy loss (pushing agents toward correct answers) with Sliced Wasserstein diversity regularization (λ_sw = 0.01, 64 projections) that actively **penalizes agent agreement**. This creates opposing forces preventing coherent mathematical reasoning.

**Insufficient Training for Complex Architecture**: The multi-agent system with K=4 independent LoRA adapters plus diversity constraints creates a significantly larger parameter space requiring extensive training. Standard fine-tuning durations prove inadequate for this complexity.

**Gradient Interference**: Aggregated gradients from 4 diverse agents may **cancel each other out** or create conflicting optimization directions, preventing meaningful learning of mathematical reasoning patterns.

**Evidence from Evaluation Results**: 
- Complete absence of GSM8K "####" format indicates failure to learn basic task structure
- Generation of whitespace and repetitive characters suggests model learned trivial loss minimization rather than mathematical reasoning
- Uniform failure pattern across all samples indicates systematic non-convergence rather than content-specific issues

**Conclusion**: The 0% accuracy reflects incomplete convergence of a fundamentally complex multi-objective optimization problem where diversity regularization prevented coherent mathematical reasoning despite apparent training loss reduction.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd course-project-avi-oberoi-clean

# Create conda environment
conda env create -f environment.yml
conda activate ci-llm

# Install additional requirements
pip install -r scripts/setup/requirements.txt
```

### 2. System Validation

```bash
# Run comprehensive system test
python tests/dry_run_sanity_check.py
```

### 3. Training

```bash
# Basic training
python src/training/train.py --config configs/full_train.yaml

# Distributed training with DeepSpeed
accelerate launch --config_file configs/accelerate_deepspeed_config.yaml \
    src/training/train.py --config configs/full_train.yaml
```

### 4. Evaluation

```bash
# Evaluate trained model
python src/evaluation/eval_gsm8k.py --config configs/eval.yaml
```

---

## Usage Instructions

### Configuration Management

The system uses YAML configuration files for all hyperparameters:

**Key Configuration Files:**
- `configs/full_train.yaml`: Main training configuration (Gemma-2-2B, K=4)
- `configs/default.yaml`: Alternative configuration (Gemma-2-based)
- `configs/eval.yaml`: Evaluation-specific settings

**Important Parameters:**
```yaml
# Model Configuration
model_name: "google/gemma-2-2b"
num_agents: 4
parallel_mode: "parallel"  # or "sequential"

# Diversity Regularization
lambda_sw: 0.01
swd_num_projections: 64
dropconnect_p: 0.05

# Training Optimization
use_deepspeed: true
mixed_precision: "bf16"
gradient_accumulation_steps: 8

# Data Processing (Spark Configuration)
dataset_name: "data/hf_datasets/gsm8k_main"
processed_data_path_train: "data/processed/gsm8k_train.parquet"
processed_data_path_eval: "data/processed/gsm8k_test.parquet"
```

### Data Preprocessing Workflow

#### 1. **Distributed Preprocessing with Apache Spark**
```bash
# Local Spark preprocessing (development)
python preprocess_spark.py --config configs/full_train.yaml \
    --dataset_type train --local_spark

# Cluster Spark preprocessing (production)
spark-submit --py-files utils/ preprocess_spark.py \
    --config configs/full_train.yaml --dataset_type train

# SLURM cluster deployment (recommended for large datasets)
sbatch scripts/setup/run_spark_preprocessing.sh --config configs/full_train.yaml

# Process both train and test splits
python preprocess_spark.py --config configs/full_train.yaml --dataset_type train
python preprocess_spark.py --config configs/full_train.yaml --dataset_type eval
```

#### 2. **Configuration for Spark Processing**
The preprocessing pipeline requires additional configuration parameters:
```yaml
# Spark-specific configuration
spark:
  app_name_preprocess: "CI-LLM Data Preprocessing"
  driver_memory: "4g"

# Data paths
dataset_name: "data/hf_datasets/gsm8k_main"  # Input: HF dataset path
processed_data_path_train: "data/processed/gsm8k_train.parquet"
processed_data_path_eval: "data/processed/gsm8k_test.parquet"
```

### Training Workflow

#### 1. **Complete Pipeline (Preprocessing + Training)**
```bash
# Step 1: Distributed data preprocessing
python preprocess_spark.py --config configs/full_train.yaml --dataset_type train
python preprocess_spark.py --config configs/full_train.yaml --dataset_type eval

# Step 2: Multi-GPU training with preprocessed data
accelerate launch --config_file configs/accelerate_deepspeed_config.yaml \
    src/training/train.py --config configs/full_train.yaml
```

#### 2. **Single-Node Training**
```bash
# Standard training
python src/training/train.py --config configs/full_train.yaml

# Resume from checkpoint
python src/training/train.py --config configs/full_train.yaml \
    --resume_from_checkpoint path/to/checkpoint
```

#### 3. **Multi-GPU Training with DeepSpeed**
```bash
# Configure accelerate (first time only)
accelerate config

# Launch training
accelerate launch --config_file configs/accelerate_deepspeed_config.yaml \
    src/training/train.py --config configs/full_train.yaml
```

#### 4. **Monitoring Training**
```bash
# Monitor training progress
./scripts/training/monitor_and_rerun.sh
```

### Evaluation Workflow

#### 1. **Configure Evaluation**
Edit `configs/eval.yaml`:
```yaml
# Evaluation Configuration
trained_peft_checkpoint_dir: "path/to/trained/checkpoint"
max_new_tokens: 384
debug_mode: false  # Set to true for quick testing
```

#### 2. **Run Evaluation**
```bash
python src/evaluation/eval_gsm8k.py --config configs/eval.yaml
```

---

## Hardware Requirements

### Minimum Requirements
- **GPU**: 1x RTX 3090 (24GB VRAM)
- **Model**: Gemma-2-2B with K=2 agents
- **Use Case**: Development and small-scale testing

### Recommended Setup
- **GPU**: 4x A100 (40GB VRAM each)
- **Model**: Gemma-2-2B with K=4 agents
- **Features**: Full DeepSpeed ZeRO-2, parallel agents
- **Use Case**: Full training runs

### Optimal Configuration
- **GPU**: 4x A100 (80GB VRAM each)
- **Model**: Gemma-2-9B with K=4+ agents
- **Features**: All optimizations, large batch sizes
- **Use Case**: Production research, large-scale experiments

### Multi-Node Scaling
```bash
# Example multi-node launch (2 nodes)
accelerate launch --num_machines 2 --machine_rank 0 \
    --main_process_ip $MASTER_ADDR \
    src/training/train.py --config configs/full_train.yaml
```

---

## Implementation Journey

### Phase 1: Foundation (Accelerate Integration)

**Achievements:**
- Robust distributed training with HuggingFace Accelerate
- Comprehensive checkpointing system supporting multi-adapter PEFT models
- Optimized data loading and tokenization pipeline
- Structured logging and monitoring

**Key Challenges Overcome:**
- Multi-adapter PEFT checkpointing complexity
- Device placement and memory management
- Label masking for GSM8K dataset format

### Phase 2: Optimization (DeepSpeed & Parallelization)

**Achievements:**
- DeepSpeed ZeRO-2 integration for memory optimization
- Parallel agent forward pass implementation
- Performance profiling and monitoring
- Multi-node training preparation

**Technical Innovations:**
- K separate PeftModel instances sharing frozen base model
- Memory-efficient weight sharing through PyTorch tensor storage
- Hybrid checkpointing (Accelerate + custom adapter management)

**Performance Impact:**
- Dramatic VRAM reduction enabling larger models
- Reduced adapter switching overhead
- Foundation for true parallel agent execution

### Current Status & Future Work

**Phase 3 Roadmap:**
- Learnable Dirichlet aggregation weights
- True multi-GPU agent parallelization
- Advanced diversity regularization techniques
- Multi-node distributed training validation
- **Loss Function Rebalancing**: Addressing convergence issues identified in current results
- **Enhanced Spark Integration**: Support for larger datasets and streaming preprocessing

---

## Key Features Summary

- **Multi-Agent Architecture**: K independent QLoRA adapters with shared backbone
- **Diversity Regularization**: Sliced-Wasserstein distance for agent diversity
- **Bayesian Aggregation**: Dirichlet-weighted token-level prediction fusion
- **Distributed Data Preprocessing**: Apache Spark-based parallel tokenization and format conversion
- **Memory Optimization**: DeepSpeed ZeRO-2 for large model training
- **Distributed Training**: Accelerate integration for multi-GPU setups
- **Parallel Processing**: Prototype parallel agent execution
- **Robust Checkpointing**: Resume training from any point
- **Comprehensive Testing**: Full system validation and profiling
- **Scalable Data Pipeline**: Fault-tolerant preprocessing with automatic schema inference

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ci-llm-2024,
  title={CI-LLM: Weaving a "Crowd Within" a Single AI - Collective Intelligence-Informed Large Language Models},
  author={Oberoi, Avi and Abenezer},
  year={2024},
  institution={University of Chicago},
  url={https://github.com/macs30200-s23/course-project-avi-oberoi}
}
```

---

## References

- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*
- Hong, L., & Page, S. E. (2004). Groups of diverse problem solvers can outperform groups of high-ability problem solvers. *PNAS*
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*
- Page, S. (2008). *The Difference: How the Power of Diversity Creates Better Groups*
- Surowiecki, J. (2005). *The Wisdom of Crowds*
- Team, G., et al. (2024). Gemma 2 Technical Report

---

*For detailed technical documentation, see `docs/README.md` and related documentation files.* 