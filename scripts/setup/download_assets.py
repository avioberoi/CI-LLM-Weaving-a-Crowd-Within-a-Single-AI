from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import sys

# --- Configuration ---
models_to_download = {
    # Gemma-2 models (existing)
    # "gemma-2-2b": "google/gemma-2-2b",  # Disabled for full training
    # "gemma-2-9b": "google/gemma-2-9b",  # For full training
    
    # Gemma-3 models (new, smaller ones for testing)
    # "gemma-3-1b-it": "google/gemma-3-1b-it",  # 1B instruct model
    # "gemma-3-4b-it": "google/gemma-3-4b-it",  # Disabled for full training
    
    # Uncomment for larger models when ready
    # "gemma-3-12b-it": "google/gemma-3-12b-it",  # 12B instruct model
    # "gemma-3-27b-it": "google/gemma-3-27b-it",  # 27B instruct model
    "gemma-3-12b-it": "google/gemma-3-12b-it",  # 12B instruct model for full training
}

datasets_to_download = [
    {"name": "gsm8k", "subset": "main"}
]

# Use the project directory
project_base_dir = "/project/jevans/avi"
hf_cache_dir = "/project/jevans/avi/hf_cache"

# --- Environment Setup ---
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")

print(f"HF_HOME is set to: {os.environ['HF_HOME']}")
print(f"TRANSFORMERS_CACHE is set to: {os.environ['TRANSFORMERS_CACHE']}")
print(f"HF_DATASETS_CACHE is set to: {os.environ['HF_DATASETS_CACHE']}")

# Ensure cache directories exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Check if HF token is available
if "HF_TOKEN" not in os.environ:
    print("WARNING: HF_TOKEN not found in environment. You may not be able to download gated models.")
    print("Please ensure HF_TOKEN is set in your ~/.bashrc")
    sys.exit(1)

# --- Download Models & Tokenizers ---
for model_key, model_name_on_hub in models_to_download.items():
    print(f"\n{'='*60}")
    print(f"Processing: {model_name_on_hub} ({model_key})")
    print(f"{'='*60}")

    try:
        print(f"Downloading/caching tokenizer for {model_name_on_hub}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_on_hub,
            use_auth_token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True
        )
        print(f"✓ Tokenizer for {model_name_on_hub} successfully cached.")

        print(f"Downloading/caching model {model_name_on_hub}...")
        print("This may take several minutes for larger models...")
        
        # For large models, we only download the config and weights, not load into memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name_on_hub,
            use_auth_token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto"  # Let it choose appropriate dtype
        )
        print(f"✓ Model {model_name_on_hub} successfully cached.")
        
        # Clear model from memory
        del model

    except Exception as e:
        print(f"✗ Error processing {model_name_on_hub}: {e}")
        print("Please ensure:")
        print("  1. You have accepted terms for gated models on HuggingFace")
        print("  2. Your HF_TOKEN is valid and has appropriate permissions")
        print("  3. You have sufficient disk space in the cache directory")

# --- Download Datasets ---
for dataset_info in datasets_to_download:
    dataset_name = dataset_info["name"]
    dataset_subset = dataset_info.get("subset")
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name} (subset: {dataset_subset or 'default'})")
    print(f"{'='*60}")
    
    try:
        # This will download to HF_DATASETS_CACHE if not already present
        dataset = load_dataset(dataset_name, dataset_subset)
        print(f"✓ Dataset '{dataset_name}' (subset: {dataset_subset}) successfully cached.")
        
        # Print dataset info
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"✗ Error downloading dataset {dataset_name}: {e}")

print(f"\n{'='*60}")
print("Pre-download Summary")
print(f"{'='*60}")
print(f"Cache directory: {hf_cache_dir}")
print(f"Models cached: {', '.join(models_to_download.keys())}")
print(f"Datasets cached: {', '.join([d['name'] for d in datasets_to_download])}")
print("\nIMPORTANT: In your SLURM scripts, ensure you set:")
print(f"  export HF_HOME={hf_cache_dir}")
print(f"  export TRANSFORMERS_CACHE={os.path.join(hf_cache_dir, 'transformers')}")
print(f"  export HF_DATASETS_CACHE={os.path.join(hf_cache_dir, 'datasets')}")
print("  export HF_HUB_OFFLINE=1  # For offline mode on compute nodes")
print("\nYour scripts should use 'local_files_only=True' in from_pretrained calls.")