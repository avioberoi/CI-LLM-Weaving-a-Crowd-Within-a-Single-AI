#!/usr/bin/env python3
"""
Debug script to investigate CI-LLM generation issues
"""

import torch
from transformers import AutoTokenizer, GenerationConfig
from models.gemma_backbone import CILLMModel
from aggregator.dirichlet_bayesian import DirichletAggregator

def main():
    print("=== CI-LLM Generation Debug Script ===")
    
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
    
    # Load model
    print("\n2. Loading CI-LLM model...")
    checkpoint_dir = "ci_llm_gemma2_2b_K4/final_checkpoint"
    try:
        ci_model = CILLMModel(
            model_name=model_name,
            K=4,  # Should match the checkpoint
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.0,  # No dropout during eval
            dropconnect_p=0.0,  # No dropconnect during eval
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            initialize_adapters_rand=False,
            use_gradient_checkpointing=False,
            trained_checkpoint_dir=checkpoint_dir
        ).to(device)
        ci_model.eval()
        print("✓ CI-LLM model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load CI-LLM model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load aggregator
    print("\n3. Loading aggregator...")
    aggregator = DirichletAggregator(
        num_agents=ci_model.K,
        alpha_val=1.0,
        input_is_logits=True
    ).to(device)
    aggregator.eval()
    print("✓ Aggregator loaded")
    
    # Test simple prompt
    print("\n4. Testing generation...")
    test_prompt = "What is 2 + 2?"
    print(f"Test prompt: '{test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {inputs.input_ids}")
    print(f"Input text decoded: '{tokenizer.decode(inputs.input_ids[0])}'")
    
    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=20,  # Small number for debugging
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate
    print("\n5. Generating...")
    try:
        with torch.no_grad():
            generated_ids = ci_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                aggregator=aggregator
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
        else:
            print("No new tokens generated!")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    main() 