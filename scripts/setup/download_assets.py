from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import sys

# --- Configuration ---
models_to_download = {
    "gemma-2-2b": "google/gemma-2-2b",  
}

datasets_to_download = [
    {"name": "gsm8k", "subset": "main"}
]

project_base_dir = "/project/jevans/avi"
hf_cache_dir = "/project/jevans/avi/hf_cache"

os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")



# Ensure cache directories exist
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

# Check if HF token is available
if "HF_TOKEN" not in os.environ:
    print("HF_TOKEN not found in environment")
    sys.exit(1)

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
        print(f"Tokenizer for {model_name_on_hub} successfully cached.")

        print(f"Downloading/caching model {model_name_on_hub}...")        
        # For large models, we only download the config and weights, not load into memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name_on_hub,
            use_auth_token=os.environ.get("HF_TOKEN"),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto"  # Let it choose appropriate dtype
        )
        print(f"Model {model_name_on_hub} successfully cached.")
        
        del model

    except Exception as e:
        print(f"Error processing {model_name_on_hub}: {e}")

for dataset_info in datasets_to_download:
    dataset_name = dataset_info["name"]
    dataset_subset = dataset_info.get("subset")
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name} (subset: {dataset_subset or 'default'})")
    print(f"{'='*60}")
    
    try:
        dataset = load_dataset(dataset_name, dataset_subset)
        print(f"Dataset '{dataset_name}' (subset: {dataset_subset}) successfully cached.")
        
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"Error downloading dataset {dataset_name}: {e}")

