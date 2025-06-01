#!/usr/bin/env python3
"""
Test checkpoint_step_600 on a single GSM8K sample
"""

import torch
from transformers import AutoTokenizer, GenerationConfig
from models.gemma_backbone import CILLMModel
from aggregator.dirichlet_bayesian import DirichletAggregator
from datasets import load_dataset

def main():
    print("=== Testing Checkpoint Step 600 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    model_name = "google/gemma-2-2b" 
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    
    # Load model from checkpoint 600
    print("\n2. Loading CI-LLM model from checkpoint step 600...")
    checkpoint_path = "/project/jevans/avi/course-project-avi-oberoi/ci_llm_gemma2_2b_K4/checkpoint_step_600"
    
    # Initialize aggregator
    aggregator = DirichletAggregator(tau=1.0, device=device)
    
    # Initialize model
    model = CILLMModel(
        model_name=model_name,
        aggregator=aggregator,
        use_quantization=True,
        local_files_only=True,
        tokenizer=tokenizer
    )
    
    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    print(f"✓ Model loaded from {checkpoint_path}")
    
    # Load a single GSM8K sample
    print("\n3. Loading GSM8K test sample...")
    dataset = load_dataset("gsm8k", "main", split="test[:1]")
    sample = dataset[0]
    
    question = sample["question"]
    correct_answer = sample["answer"]
    
    print(f"Question: {question}")
    print(f"Expected answer: {correct_answer}")
    
    # Test generation
    print("\n4. Testing generation...")
    
    # Format prompt
    prompt = f"Question: {question}\nAnswer:"
    
    print(f"Prompt: {repr(prompt)}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print(f"Input tokens: {inputs['input_ids'].tolist()}")
    print(f"Input decoded: {repr(tokenizer.decode(inputs['input_ids'][0]))}")
    
    # Generate
    with torch.no_grad():
        # Test with different generation configs
        configs = [
            {"max_new_tokens": 100, "do_sample": False, "temperature": None},  # Greedy
            {"max_new_tokens": 100, "do_sample": True, "temperature": 0.7, "top_p": 0.9},  # Sampling
            {"max_new_tokens": 100, "do_sample": True, "temperature": 1.0, "top_k": 50},  # Different sampling
        ]
        
        for i, config in enumerate(configs):
            print(f"\n--- Generation Config {i+1}: {config} ---")
            
            gen_config = GenerationConfig(
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **config
            )
            
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config
            )
            
            # Decode generated tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]  # Remove input tokens
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Generated tokens: {generated_tokens.tolist()}")
            print(f"Generated text: {repr(generated_text)}")
            print(f"Generated text (clean): {generated_text}")
            
            if generated_text.strip():
                print(f"✓ Non-empty generation!")
            else:
                print(f"✗ Empty or whitespace-only generation")

if __name__ == "__main__":
    main() 