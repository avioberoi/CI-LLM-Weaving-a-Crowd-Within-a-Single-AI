import os
import shutil
import torch
from torch import nn, optim
from transformers import AutoTokenizer, GenerationConfig
import json

from models.gemma_backbone import CILLMModel
from aggregator.dirichlet_bayesian import DirichletAggregator
from losses.sliced_w2 import SlicedWassersteinDiversityRegularizer
from peft import PeftModel, get_peft_model_state_dict

# --- Minimal Configuration for Dry Run ---
DRY_RUN_CONFIG = {
    "model_name": "google/gemma-2-2b", # Smaller Gemma model for faster loading/testing
    "num_agents": 2,                  # K: Reduced number of agents
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.0,              # Disable PEFT dropout for simplicity in dry run
    "lora_target_modules": ["q_proj", "v_proj"], # Fewer target modules
    "initialize_adapters_rand": True, # Test this initialization path
    "use_gradient_checkpointing": True, # Disable for CPU or simpler GPU runs

    "dropconnect_p": 0.1,             # Test DropConnect mechanism (even if effect isn't measured)
    
    "lambda_sw": 0.01,
    "swd_num_projections": 10,        # Fewer projections for speed

    "batch_size": 1,                  # Minimal batch size
    "seq_length": 32,                 # Short sequence length
    "lr": 1e-5,

    "temp_output_dir": "dry_run_temp_output", # For testing save/load
    "max_new_tokens_generate": 10,

    # These would normally come from a dataset for tokenize_and_format
    "dummy_vocab_size": 32000 # Gemma's vocab size (approx for tokenizer)
}

def get_dummy_batch(batch_size, seq_length, vocab_size, device):
    """Creates a dummy batch of input_ids, attention_mask, and labels."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    # For causal LM, labels are input_ids shifted, with padding/prompt masked
    # For simplicity in dry run, let's make labels same as input_ids,
    # and assume CrossEntropyLoss will handle ignore_index for padding if any.
    # In a real scenario, proper label creation is vital.
    labels = input_ids.clone() 
    # Let's imagine first few tokens are prompt and should be ignored
    if seq_length > 5:
        labels[:, :5] = -100 
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main_dry_run():
    print("--- Starting CI-LLM Dry Run Sanity Check ---")
    config = DRY_RUN_CONFIG
    
    # Attempt to use CUDA if available, otherwise CPU
    # Note: QLoRA (BitsAndBytes) is primarily for CUDA. CPU execution might be very slow or fail
    # if certain CUDA-specific operations are present in the quantization.
    # For a true CPU dry run of logic *around* the model, one might mock the model itself.
    # But to test QLoRA setup, a GPU is needed. We'll proceed assuming a GPU is available for QLoRA parts.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and "gemma" in config["model_name"]: # QLoRA needs CUDA
        print("WARNING: CUDA not available. QLoRA components for Gemma may fail or be extremely slow on CPU.")
        print("Attempting to run on CPU, but this is not the intended environment for QLoRA.")
        # Override device_map for CPU run attempt if model tries to force GPU
        # This is tricky as BitsAndBytesConfig is CUDA-centric.
        # For a CPU-only dry run of *logic*, consider a non-quantized small model.
        # For this script, we'll let it try, and it might fail at model loading if no CUDA.

    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    print("\n--- 1. Initializing Tokenizer ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer '{config['model_name']}' loaded. Pad token ID: {tokenizer.pad_token_id}")
        DRY_RUN_CONFIG["dummy_vocab_size"] = tokenizer.vocab_size # Update with actual vocab size
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return

    # 2. Initialize CILLMModel
    print("\n--- 2. Initializing CILLMModel ---")
    try:
        ci_model = CILLMModel(
            model_name=config['model_name'],
            K=config['num_agents'],
            lora_rank=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            dropconnect_p=config['dropconnect_p'],
            target_modules=config['lora_target_modules'],
            initialize_adapters_rand=config['initialize_adapters_rand'],
            use_gradient_checkpointing=config['use_gradient_checkpointing'],
            # trained_peft_checkpoint_dir=None # For initial creation
        ).to(device)
        ci_model.train() # Set to train mode to test DropConnect
        print(f"CILLMModel initialized with K={ci_model.K} agents.")
        # Check parameters to be trained
        trainable_params = sum(p.numel() for p in ci_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in ci_model.parameters())
        print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
        print(f"Total parameters: {total_params}")

    except Exception as e:
        print(f"Error initializing CILLMModel: {e}")
        if "CUDA" in str(e).upper():
            print("This might be due to running QLoRA/BitsAndBytes without a CUDA-enabled GPU.")
        return

    # 3. Initialize Aggregator and Loss Functions
    print("\n--- 3. Initializing Aggregator and Loss Functions ---")
    try:
        aggregator = DirichletAggregator(
            num_agents=ci_model.K, # Use K from the model instance
            alpha_val=1.0, 
            input_is_logits=True
        ).to(device)
        aggregator.eval() # Aggregator is usually not trained
        print("DirichletAggregator initialized.")

        diversity_loss_fn = SlicedWassersteinDiversityRegularizer(
            num_projections=config['swd_num_projections']
        ).to(device)
        print("SlicedWassersteinDiversityRegularizer initialized.")
        
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        print("CrossEntropyLoss initialized.")
    except Exception as e:
        print(f"Error initializing aggregator/losses: {e}")
        return

    # 4. Optimizer
    print("\n--- 4. Initializing Optimizer ---")
    try:
        optimizer = optim.AdamW(ci_model.parameters(), lr=config['lr'])
        print("AdamW Optimizer initialized.")
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return

    # 5. Training Step Sanity Check
    print("\n--- 5. Training Step Sanity Check ---")
    try:
        dummy_batch = get_dummy_batch(
            config['batch_size'], 
            config['seq_length'], 
            DRY_RUN_CONFIG["dummy_vocab_size"], # Use actual vocab size
            device
        )
        input_ids = dummy_batch["input_ids"]
        attention_mask = dummy_batch["attention_mask"]
        labels = dummy_batch["labels"]

        print(f"Dummy input_ids shape: {input_ids.shape}")

        # Forward pass
        # CILLMModel.forward returns (list_of_agent_logits, list_of_agent_last_hidden_states)
        list_of_agent_logits, list_of_agent_hidden_states = ci_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states_flag=True # Need hidden states for SWD
        )
        print(f"Forward pass successful. Number of agent logit sets: {len(list_of_agent_logits)}")
        if list_of_agent_logits:
            print(f"Shape of logits for first agent: {list_of_agent_logits[0].shape}")
        if list_of_agent_hidden_states and list_of_agent_hidden_states[0] is not None:
            print(f"Shape of hidden states for first agent: {list_of_agent_hidden_states[0].shape}")
        
        # Calculate Per-Agent Task Loss
        task_losses = []
        for k_logits in list_of_agent_logits:
            loss_k = ce_loss_fn(k_logits.reshape(-1, k_logits.size(-1)), labels.reshape(-1))
            task_losses.append(loss_k)
        avg_task_loss = torch.stack(task_losses).mean()
        print(f"Average Per-Agent Task Loss (CE): {avg_task_loss.item():.4f}")

        # Calculate Diversity Loss (SWD)
        loss_swd = torch.tensor(0.0, device=device)
        if config['num_agents'] > 1 and list_of_agent_hidden_states and all(hs is not None for hs in list_of_agent_hidden_states):
            # Process hidden states for SWD (e.g., take last token's hidden state)
            processed_reprs_for_swd = []
            for hidden_state_seq in list_of_agent_hidden_states: # [batch, seq_len, hidden_dim]
                last_token_repr = hidden_state_seq[:, -1, :] # [batch_size, hidden_dim]
                processed_reprs_for_swd.append(last_token_repr)
            
            if len(processed_reprs_for_swd) == ci_model.K:
                R_sw = diversity_loss_fn(processed_reprs_for_swd)
                loss_swd = R_sw 
                print(f"Diversity Loss (R_sw): {loss_swd.item():.4f}")
            else:
                print("Warning: Could not compute SWD due to mismatch in representations.")
        else:
            print("SWD loss skipped (num_agents <= 1 or hidden states not available).")

        # Combined Loss
        total_loss = avg_task_loss - config['lambda_sw'] * loss_swd
        print(f"Total Combined Loss: {total_loss.item():.4f}")

        # Backward pass and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        print("Backward pass successful.")
        optimizer.step()
        print("Optimizer step successful.")
        print("Training step sanity check complete.")

    except Exception as e:
        print(f"Error during training step sanity check: {e}")
        import traceback
        traceback.print_exc()
        # return # Optionally stop if training step fails

    # 6. Inference/Generation Sanity Check
    print("\n--- 6. Inference/Generation Sanity Check ---")
    try:
        ci_model.eval() # Set to eval mode (disables DropConnect, dropout)
        
        dummy_prompt_text = "What is the capital of France?"
        # Ensure tokenizer.pad_token_id is set before tokenizing for generate
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
            
        eval_inputs = tokenizer(dummy_prompt_text, return_tensors="pt").to(device)
        
        print(f"Dummy prompt: '{dummy_prompt_text}'")
        print(f"Tokenized prompt shape: {eval_inputs.input_ids.shape}")

        gen_config = GenerationConfig(
            max_new_tokens=config['max_new_tokens_generate'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # Greedy for simplicity
        )

        with torch.no_grad():
            generated_ids = ci_model.generate(
                input_ids=eval_inputs["input_ids"],
                attention_mask=eval_inputs["attention_mask"],
                generation_config=gen_config,
                aggregator=aggregator
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated IDs shape: {generated_ids.shape}")
        print(f"Generated text (dummy): '{generated_text}'")
        print("Generation sanity check complete.")

    except Exception as e:
        print(f"Error during generation sanity check: {e}")
        import traceback
        traceback.print_exc()
        # return

    # 7. Saving/Loading Adapters Sanity Check (Manual Adapter Weight Save/Load)
    print("\n--- 7. Saving/Loading Adapters Sanity Check (Manual State Dict Strategy) ---")
    temp_save_dir = config['temp_output_dir']
    try:
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir) 
        os.makedirs(temp_save_dir, exist_ok=True)

        print(f"Attempting to save individual PEFT adapter weights to: {temp_save_dir}")
        
        if ci_model.K > 0 and isinstance(ci_model.peft_model, PeftModel):
            # 1. Save common LoraConfig parameters
            common_lora_params_to_save = {
                'r': ci_model.lora_rank_arg,
                'lora_alpha': ci_model.lora_alpha_arg,
                'lora_dropout': ci_model.lora_dropout_arg,
                'target_modules': ci_model.target_modules_arg,
                'bias': "none", 
                'task_type': "CAUSAL_LM",
                'K_agents': ci_model.K 
            }
            common_config_save_path = os.path.join(temp_save_dir, "ci_llm_meta_config.json")
            with open(common_config_save_path, 'w') as f:
                json.dump(common_lora_params_to_save, f, indent=4)
            print(f"Saved CI-LLM meta config to {common_config_save_path}")

            # 2. Save each adapter's state dictionary
            for i in range(ci_model.K):
                adapter_name = f"agent{i}"
                adapter_weights_save_path = os.path.join(temp_save_dir, f"{adapter_name}_weights.pth")
                try:
                    adapter_state_dict = get_peft_model_state_dict(ci_model.peft_model, adapter_name=adapter_name)
                    torch.save(adapter_state_dict, adapter_weights_save_path)
                    print(f"Saved weights for adapter '{adapter_name}' to {adapter_weights_save_path}")
                except Exception as e:
                    print(f"Error saving weights for adapter {adapter_name}: {e}")
                    raise # Critical for test
            
            print(f"Contents of {temp_save_dir} after save: {os.listdir(temp_save_dir)}")
        else:
            print("K=0 or peft_model is not a PeftModel instance. Skipping adapter saving.")


        # Attempt to load the model using the new CILLMModel.__init__ logic
        print(f"Attempting to re-load CILLMModel using trained adapters from: {temp_save_dir}")
        loaded_ci_model = CILLMModel(
            model_name=config['model_name'], 
            K=config['num_agents'], # This K will be used by __init__ and potentially updated from meta_config
            # Pass original LoRA params as defaults in case meta_config is missing (though it shouldn't be)
            lora_rank=config['lora_rank'], 
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['lora_target_modules'],
            initialize_adapters_rand=False, 
            use_gradient_checkpointing=False,
            trained_checkpoint_dir=temp_save_dir, 
            load_adapters_trainable=False 
        ).to(device)
        loaded_ci_model.eval() # Set to eval mode
        print(f"CILLMModel re-loaded. Effective K: {loaded_ci_model.K}. Adapters in peft_config: {list(loaded_ci_model.peft_model.peft_config.keys()) if hasattr(loaded_ci_model.peft_model, 'peft_config') else 'N/A (Base Model likely)'}")
        
        # Optional: a quick forward pass with the loaded model
        print("Performing a quick forward pass with the re-loaded model...")
        dummy_batch_reloaded = get_dummy_batch(
            config['batch_size'], 
            config['seq_length'], 
            DRY_RUN_CONFIG["dummy_vocab_size"], # Ensure this uses the updated vocab size
            device
        )
        with torch.no_grad():
            reloaded_logits_list, _ = loaded_ci_model(
                input_ids=dummy_batch_reloaded["input_ids"],
                attention_mask=dummy_batch_reloaded["attention_mask"],
                output_hidden_states_flag=False
            )
        if reloaded_logits_list and len(reloaded_logits_list) > 0 and reloaded_logits_list[0] is not None:
            print(f"Forward pass with re-loaded model successful. Number of logit sets: {len(reloaded_logits_list)}. First agent logits shape: {reloaded_logits_list[0].shape}")
        elif reloaded_logits_list and len(reloaded_logits_list) > 0 and reloaded_logits_list[0] is None:
            print(f"Forward pass with re-loaded model returned {len(reloaded_logits_list)} logit sets, but the first is None.")
        elif not reloaded_logits_list:
            print("Forward pass with re-loaded model produced an empty list of logits.")
        else: # K=0 case, loaded_ci_model.peft_model is the base model
            print("Forward pass with re-loaded model (K=0, base model) - attempting direct call if K=0 in CILLMModel.")
            if loaded_ci_model.K == 0:
                base_model_output = loaded_ci_model.peft_model(input_ids=dummy_batch_reloaded["input_ids"], attention_mask=dummy_batch_reloaded["attention_mask"])
                print(f"Base model (K=0) output logits shape: {base_model_output.logits.shape}")


    except Exception as e:
        print(f"Error during saving/loading sanity check: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
            print(f"Cleaned up temporary directory: {temp_save_dir}")


    print("\n--- CI-LLM Dry Run Sanity Check Finished ---")

if __name__ == "__main__":
    main_dry_run()