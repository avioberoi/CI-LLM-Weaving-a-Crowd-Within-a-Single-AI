#!/usr/bin/env python3
"""
Simple test script to verify that the trained checkpoint can be loaded correctly.
This helps catch any issues before running the full evaluation.
"""

import os
import yaml
import torch
from models.gemma_backbone import CILLMModel

def load_config(config_path="configs_eval.yaml"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Evaluation config {config_path} not found")

def main():
    print("Loading configuration...")
    config = load_config()
    
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if checkpoint directory exists
    checkpoint_dir = config['trained_peft_checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    print(f"Checkpoint directory exists: {checkpoint_dir}")
    
    # List checkpoint contents
    print("Checkpoint contents:")
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"  {item}: {size / (1024**2):.1f} MB")
        else:
            print(f"  {item}/")
    
    print("\nInitializing CILLMModel...")
    try:
        ci_model = CILLMModel(
            model_name=config['model_name'],
            K=config['num_agents'],
            lora_rank=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=0.0,
            dropconnect_p=0.0,
            target_modules=config['lora_target_modules'],
            initialize_adapters_rand=False,
            use_gradient_checkpointing=False,
            trained_checkpoint_dir=config['trained_peft_checkpoint_dir']
        ).to(device)
        
        ci_model.eval()
        print("✓ CILLMModel loaded successfully!")
        print(f"Model has {ci_model.K} agents")
        
    except Exception as e:
        print(f"✗ Error loading CILLMModel: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a simple forward pass
    print("\nTesting forward pass...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'], 
            trust_remote_code=True,
            local_files_only=True  # Force local loading only, no internet access
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_input = "What is 2 + 2?"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            agent_logits_list, _ = ci_model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        print(f"✓ Forward pass successful!")
        print(f"Got logits from {len(agent_logits_list)} agents")
        print(f"Logits shape: {agent_logits_list[0].shape}")
        
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed! The checkpoint can be loaded successfully.")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 