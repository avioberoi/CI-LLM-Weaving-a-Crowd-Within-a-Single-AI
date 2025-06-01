#!/usr/bin/env python3
"""
Test script to check if the base Gemma model (without adapters) works correctly.
This will help isolate whether the issue is with:
1. The base model
2. The CI-LLM generation code
3. The trained adapters
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def main():
    print("=== Base Model Test ===")
    
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
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    print(f"✓ Tokenizer loaded")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    print(f"  eos_token_id: {tokenizer.eos_token_id}")
    print(f"  bos_token_id: {tokenizer.bos_token_id}")
    print(f"  unk_token_id: {tokenizer.unk_token_id}")
    
    # Load base model
    print("\n2. Loading base Gemma model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        model.eval()
        print("✓ Base model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load base model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test simple prompt
    print("\n3. Testing generation...")
    test_prompt = "What is 2 + 2?"
    print(f"Test prompt: '{test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {inputs.input_ids}")
    print(f"Input text decoded: '{tokenizer.decode(inputs.input_ids[0])}'")
    
    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=20,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate with base model
    print("\n4. Generating with base model...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )
        
        print(f"Generated tokens: {generated_ids}")
        print(f"Generated sequence length: {generated_ids.shape}")
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated text: '{generated_text}'")
        
        # Check what was actually generated (excluding input)
        input_length = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][input_length:]
        print(f"New tokens only: {new_tokens}")
        
        if len(new_tokens) > 0:
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"New text only: '{new_text}'")
            
            # Check what each token is
            print("\nToken-by-token analysis:")
            for i, token_id in enumerate(new_tokens):
                token_text = tokenizer.decode([token_id])
                print(f"  Token {i}: {token_id} -> '{token_text}' (repr: {repr(token_text)})")
                
            # Check if it's generating only newlines like CI-LLM
            if all(token_id == 109 for token_id in new_tokens):
                print("\n⚠️  WARNING: Base model is also generating only newlines!")
                print("   This suggests the issue might be with tokenizer or environment.")
            else:
                print("\n✓ Base model generates proper tokens (not just newlines)")
                print("  This suggests the issue is with CI-LLM adapters or generation code.")
        else:
            print("No new tokens generated!")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Base Model Test Complete ===")

if __name__ == "__main__":
    main() 