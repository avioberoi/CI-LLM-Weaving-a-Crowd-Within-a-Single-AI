import os
import json # For saving/loading common LoraConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    PeftModel,
    get_peft_model_state_dict, # For saving individual adapter weights
    set_peft_model_state_dict  # For loading individual adapter weights
)
from peft.tuners.lora import LoraLayer # To access LoRA specific layers

class CILLMModel(nn.Module):
    """
    CI-LLM Model wraps a backbone LLM with K independent LoRA adapter heads.
    Can be initialized from scratch or by loading a pre-trained PEFT model with adapters.
    Stochastic DropConnect on LoRA weights is applied during training for agent independence,
    with careful backup and restoration of original weights.
    
    Supports two modes:
    - 'sequential': Original mode where adapters are switched sequentially
    - 'parallel': New mode where K PeftModel instances allow parallel execution
    """
    def __init__(self, 
                 model_name: str, 
                 K: int, 
                 lora_rank: int = 8,
                 lora_alpha: int = 16, 
                 lora_dropout: float = 0.05, 
                 dropconnect_p: float = 0.0, # Probability of dropping weights in LoRA A/B
                 target_modules: list[str] = None,
                 initialize_adapters_rand: bool = False, # Only used if NOT loading from checkpoint
                 use_gradient_checkpointing: bool = True,
                 trained_checkpoint_dir: str = None,  # Path to load pre-trained PEFT model
                 load_adapters_trainable: bool = False,
                 parallel_mode: str = "sequential"):  # New parameter for parallel execution
        """
        Initialize the CI-LLM model.

        Parameters:
        - model_name (str): HuggingFace model name for the base backbone.
        - K (int): Expected number of agent heads (LoRA adapters).
                   If loading from checkpoint, K might be inferred or verified.
        - ... (other LoRA params) ...
        - trained_peft_checkpoint_dir (str, optional): Path to a directory containing a saved PEFT model
                                                       (adapters and config). If provided, loads this PEFT model.
                                                       Otherwise, creates a new PEFT model with K adapters.
        - parallel_mode (str): "sequential" or "parallel" - determines execution strategy
        """
        super().__init__()
        
        self.model_name = model_name
        self.lora_rank_arg = lora_rank 
        self.lora_alpha_arg = lora_alpha
        self.lora_dropout_arg = lora_dropout
        if target_modules is None:
            self.target_modules_arg = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            self.target_modules_arg = target_modules
        
        self.dropconnect_p = dropconnect_p
        self.initialize_adapters_rand_on_create = initialize_adapters_rand
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.K_arg = K 
        self.parallel_mode = parallel_mode

        # --- 1. Load and Prepare Base Backbone Model ---
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load model weights in 4-bit on CPU to avoid DTensor wrapper for LoRA injection
        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config, 
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True
        )
        
        base_model_for_peft = prepare_model_for_kbit_training(
            base_model_for_peft, use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        if self.use_gradient_checkpointing :
            if hasattr(base_model_for_peft, "gradient_checkpointing_enable"):
                base_model_for_peft.gradient_checkpointing_enable()
            if hasattr(base_model_for_peft, "config"):
                 base_model_for_peft.config.use_cache = False

        # --- 2. Determine LoraConfig Parameters for PEFT Shell Construction ---
        effective_lora_rank = self.lora_rank_arg
        effective_lora_alpha = self.lora_alpha_arg
        effective_lora_dropout = self.lora_dropout_arg
        effective_target_modules = self.target_modules_arg
        effective_K = self.K_arg

        if trained_checkpoint_dir and os.path.isdir(trained_checkpoint_dir):
            meta_config_path = os.path.join(trained_checkpoint_dir, "ci_llm_meta_config.json")
            if os.path.exists(meta_config_path):
                try:
                    with open(meta_config_path, 'r') as f:
                        loaded_meta_config = json.load(f)
                        print(f"Loaded CI-LLM meta config from {meta_config_path}")
                        effective_lora_rank = loaded_meta_config.get('r', effective_lora_rank)
                        effective_lora_alpha = loaded_meta_config.get('lora_alpha', effective_lora_alpha)
                        effective_lora_dropout = loaded_meta_config.get('lora_dropout', effective_lora_dropout)
                        effective_target_modules = loaded_meta_config.get('target_modules', effective_target_modules)
                        effective_K = loaded_meta_config.get('K_agents', effective_K) 
                except Exception as e:
                    print(f"Warning: Could not load or parse 'ci_llm_meta_config.json': {e}. Using init Lora params.")
            else:
                print(f"Warning: 'ci_llm_meta_config.json' not found in {trained_checkpoint_dir}. "
                      "Using LORA parameters from __init__ args to structure PEFT model.")
        
        self.K = effective_K # K defines the structure to build
        if self.K == 0 and not trained_checkpoint_dir :
             raise ValueError("K must be > 0 when creating a new CILLMModel without a checkpoint to load that might define K.")
        if self.K == 0 and trained_checkpoint_dir: # Loading a K=0 model (base model only)
            print("K=0 specified or loaded from meta_config, CILLMModel will use the base model directly.")
            self.peft_model = base_model_for_peft
            self.peft_model.train(load_adapters_trainable) # Set mode
            return # Skip adapter creation and loading for K=0

        self.common_lora_config_runtime = LoraConfig(
            r=effective_lora_rank,
            lora_alpha=effective_lora_alpha,
            lora_dropout=effective_lora_dropout,
            target_modules=effective_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.base_model = base_model_for_peft  # Keep reference to base model
        
        # --- 3. Initialize PEFT Model(s) based on parallel_mode ---
        if self.parallel_mode == "parallel":
            # Create K separate PeftModel instances
            self.agent_peft_models = nn.ModuleList()
            
            for i in range(self.K):
                adapter_name = f"agent{i}"
                
                # Create PeftModel with single adapter
                # IMPORTANT: All PeftModels share the same base_model instance
                # PyTorch should share the underlying parameter storage
                agent_model = get_peft_model(
                    self.base_model,  # Shared base model
                    self.common_lora_config_runtime, 
                    adapter_name=adapter_name
                )
                
                self.agent_peft_models.append(agent_model)
                print(f"Created PeftModel instance for {adapter_name}")
            
            print(f"Initialized {self.K} parallel PeftModel instances sharing base model")
            
            # For compatibility with existing code paths
            self.peft_model = self.agent_peft_models[0]  # Reference to first agent model
            
        else:  # sequential mode (original implementation)
            # --- 3. Initialize PEFT Model Shell (Structure) ---
            self.peft_model = get_peft_model(base_model_for_peft, self.common_lora_config_runtime, adapter_name="agent0")
            for i in range(1, self.K):
                adapter_name = f"agent{i}"
                self.peft_model.add_adapter(adapter_name, self.common_lora_config_runtime)
            print(f"Initialized PEFT model structure for K={self.K} adapters: {list(self.peft_model.peft_config.keys())}")

        # --- 4. Load Trained Adapter Weights OR Initialize New Ones ---
        if trained_checkpoint_dir and os.path.isdir(trained_checkpoint_dir):
            print(f"Attempting to load trained adapter weights for {self.K} agents from: {trained_checkpoint_dir}")
            loaded_adapter_count = 0
            
            if self.parallel_mode == "parallel":
                # Load weights for each parallel model
                for i in range(self.K):
                    adapter_name = f"agent{i}"
                    adapter_weights_path = os.path.join(trained_checkpoint_dir, f"{adapter_name}_weights.pth")
                    
                    if os.path.exists(adapter_weights_path):
                        try:
                            print(f"Loading weights for adapter '{adapter_name}' from {adapter_weights_path}...")
                            adapter_state_dict = torch.load(adapter_weights_path, map_location='cpu')
                            set_peft_model_state_dict(
                                self.agent_peft_models[i], 
                                adapter_state_dict, 
                                adapter_name=adapter_name
                            )
                            print(f"Successfully loaded weights for adapter '{adapter_name}'.")
                            loaded_adapter_count += 1
                        except Exception as e:
                            print(f"Error loading weights for adapter '{adapter_name}' from {adapter_weights_path}: {e}")
                    else:
                        print(f"Warning: Adapter weights file for '{adapter_name}' ({adapter_weights_path}) not found.")
            else:
                # Original loading logic for sequential mode
                for i in range(self.K):
                    adapter_name = f"agent{i}"
                    adapter_weights_path = os.path.join(trained_checkpoint_dir, f"{adapter_name}_weights.pth") # Or .safetensors
                    
                    if os.path.exists(adapter_weights_path):
                        try:
                            print(f"Loading weights for adapter '{adapter_name}' from {adapter_weights_path}...")
                            adapter_state_dict = torch.load(adapter_weights_path, map_location='cpu') # Load to CPU first
                            set_peft_model_state_dict(self.peft_model, adapter_state_dict, adapter_name=adapter_name)
                            print(f"Successfully loaded weights for adapter '{adapter_name}'.")
                            loaded_adapter_count +=1
                        except Exception as e:
                            print(f"Error loading weights for adapter '{adapter_name}' from {adapter_weights_path}: {e}")
                    else:
                        print(f"Warning: Adapter weights file for '{adapter_name}' ({adapter_weights_path}) not found. Weights not loaded for this agent.")
            
            if loaded_adapter_count != self.K:
                print(f"Warning: Expected to load weights for {self.K} agents, but only {loaded_adapter_count} were successfully loaded.")
            if loaded_adapter_count == 0 and self.K > 0:
                print(f"Warning: No adapter weights successfully loaded from {trained_checkpoint_dir}. "
                      "Adapters will be as initialized by PEFT or by _randomly_initialize_all_adapters if flag is set.")
                if self.initialize_adapters_rand_on_create: 
                    self._randomly_initialize_all_adapters()
        
        elif self.initialize_adapters_rand_on_create and self.K > 0: 
            self._randomly_initialize_all_adapters()
        
        # Set overall PeftModel train/eval mode
        if self.parallel_mode == "parallel":
            for agent_model in self.agent_peft_models:
                agent_model.train(load_adapters_trainable if trained_checkpoint_dir else True)
        else:
            self.peft_model.train(load_adapters_trainable if trained_checkpoint_dir else True)

    def _randomly_initialize_all_adapters(self):
        """
        Re-initializes LoRA A and B matrices for all 'agentX' adapters with N(0, 0.02).
        """
        print("Applying custom random initialization to 'agentX' adapter weights...")
        
        if self.parallel_mode == "parallel":
            # Initialize each parallel model's adapter
            for i, agent_model in enumerate(self.agent_peft_models):
                adapter_name = f"agent{i}"
                for module in agent_model.base_model.modules():
                    if isinstance(module, LoraLayer):
                        if adapter_name in module.lora_A and module.lora_A[adapter_name] is not None:
                            nn.init.normal_(module.lora_A[adapter_name].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_A[adapter_name], 'bias') and module.lora_A[adapter_name].bias is not None:
                                nn.init.zeros_(module.lora_A[adapter_name].bias)
                                
                        if adapter_name in module.lora_B and module.lora_B[adapter_name] is not None:
                            nn.init.normal_(module.lora_B[adapter_name].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_B[adapter_name], 'bias') and module.lora_B[adapter_name].bias is not None:
                                nn.init.zeros_(module.lora_B[adapter_name].bias)
        else:
            # Original sequential mode logic
            for i in range(self.K):
                adapter_name_to_init = f"agent{i}"
                # Check if this adapter configuration actually exists in the peft_model
                if adapter_name_to_init not in self.peft_model.peft_config:
                    print(f"Warning: Adapter config for '{adapter_name_to_init}' not found. Skipping initialization.")
                    continue

                for module in self.peft_model.base_model.modules():
                    if isinstance(module, LoraLayer):
                        if adapter_name_to_init in module.lora_A and module.lora_A[adapter_name_to_init] is not None:
                            # print(f"Initializing lora_A for {adapter_name_to_init} in {module}")
                            nn.init.normal_(module.lora_A[adapter_name_to_init].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_A[adapter_name_to_init], 'bias') and module.lora_A[adapter_name_to_init].bias is not None:
                                nn.init.zeros_(module.lora_A[adapter_name_to_init].bias)
                                
                        if adapter_name_to_init in module.lora_B and module.lora_B[adapter_name_to_init] is not None:
                            # print(f"Initializing lora_B for {adapter_name_to_init} in {module}")
                            nn.init.normal_(module.lora_B[adapter_name_to_init].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_B[adapter_name_to_init], 'bias') and module.lora_B[adapter_name_to_init].bias is not None:
                                nn.init.zeros_(module.lora_B[adapter_name_to_init].bias)
        
        print("Custom random initialization for 'agentX' adapters complete.")


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, 
                output_hidden_states_flag: bool = False, **kwargs):
        """
        Forward pass that computes outputs for all K agents independently.
        Applies DropConnect to LoRA weights during training if self.dropconnect_p > 0.
        
        In parallel mode, executes all agent forward passes simultaneously.
        In sequential mode, uses the original adapter switching approach.
        """
        all_agent_logits = []
        all_agent_last_hidden_states = []
        
        current_model_kwargs = kwargs.copy()
        current_model_kwargs['output_hidden_states'] = output_hidden_states_flag
        
        if self.parallel_mode == "parallel":
            # Parallel execution of all agents
            # Note: PyTorch will handle parallelization automatically if models are on same device
            # For true parallelism across GPUs, would need to place different agents on different devices
            
            # Handle DropConnect for parallel mode
            original_weights_backup = {}
            
            if self.training and self.dropconnect_p > 0:
                # Apply DropConnect to all agent models
                for i, agent_model in enumerate(self.agent_peft_models):
                    adapter_name = f"agent{i}"
                    for module_path, module in agent_model.base_model.named_modules():
                        if isinstance(module, LoraLayer) and adapter_name in module.lora_A:
                            # Backup and apply DropConnect to lora_A
                            lora_A_layer = module.lora_A[adapter_name]
                            if lora_A_layer is not None:
                                backup_key_A = (i, module_path, adapter_name, 'A')
                                original_weights_backup[backup_key_A] = lora_A_layer.weight.data.clone()
                                weight_A_data = lora_A_layer.weight.data
                                mask_A = (torch.rand_like(weight_A_data) > self.dropconnect_p).float()
                                lora_A_layer.weight.data = weight_A_data * mask_A

                            # Backup and apply DropConnect to lora_B
                            lora_B_layer = module.lora_B[adapter_name]
                            if lora_B_layer is not None:
                                backup_key_B = (i, module_path, adapter_name, 'B')
                                original_weights_backup[backup_key_B] = lora_B_layer.weight.data.clone()
                                weight_B_data = lora_B_layer.weight.data
                                mask_B = (torch.rand_like(weight_B_data) > self.dropconnect_p).float()
                                lora_B_layer.weight.data = weight_B_data * mask_B
            
            # Execute forward passes
            # TODO: For true parallel execution across GPUs, could use torch.nn.parallel
            # For now, this executes sequentially but avoids adapter switching overhead
            for i, agent_model in enumerate(self.agent_peft_models):
                out = agent_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    **current_model_kwargs
                )
                
                all_agent_logits.append(out.logits)
                if output_hidden_states_flag and hasattr(out, 'hidden_states') and out.hidden_states is not None:
                    all_agent_last_hidden_states.append(out.hidden_states[-1])
                elif output_hidden_states_flag:
                    all_agent_last_hidden_states.append(None)
            
            # Restore original weights after DropConnect
            if self.training and self.dropconnect_p > 0:
                for (agent_idx, m_path, ad_name, matrix_type), orig_weight in original_weights_backup.items():
                    agent_model = self.agent_peft_models[agent_idx]
                    try:
                        target_module = dict(agent_model.base_model.named_modules())[m_path]
                        if isinstance(target_module, LoraLayer):
                            if matrix_type == 'A' and ad_name in target_module.lora_A:
                                target_module.lora_A[ad_name].weight.data = orig_weight
                            elif matrix_type == 'B' and ad_name in target_module.lora_B:
                                target_module.lora_B[ad_name].weight.data = orig_weight
                    except KeyError:
                        print(f"Warning: Module path '{m_path}' not found during DropConnect restoration.")
                
                original_weights_backup.clear()
            
        else:  # Sequential mode (original implementation)
            # original_weights_backup is re-initialized for each forward call to avoid state leakage
            # between calls if not all restorations happened correctly in a previous call (defensive).
            original_weights_backup = {}

            for i in range(self.K): # self.K should be reliable now
                adapter_name = f"agent{i}"
                
                # Ensure the adapter we are trying to set is actually part of the model
                if adapter_name not in self.peft_model.peft_config:
                    # This might happen if K was set higher than number of loaded/created adapters
                    # print(f"Warning: Adapter '{adapter_name}' not found in peft_model. Skipping agent {i}.")
                    # Append Nones or handle appropriately if an agent is missing
                    # For now, let's assume if K is correct, all agents exist.
                    # If K was dynamically adjusted downwards when loading, this loop will be correct.
                    # If K is still too high, this will be an issue.
                    # A more robust solution might be to iterate `for adapter_name in self.peft_model.peft_config.keys()`
                    # if those keys are guaranteed to be `agent0, agent1...`
                    # Or, ensure self.K is always correct after __init__.
                    continue # Skip this agent if its adapter doesn't exist

                self.peft_model.set_adapter(adapter_name)

                if self.training and self.dropconnect_p > 0:
                    for module_path, module in self.peft_model.base_model.named_modules():
                        if isinstance(module, LoraLayer) and adapter_name in module.lora_A: # Check current adapter
                            lora_A_layer = module.lora_A[adapter_name]
                            if lora_A_layer is not None:
                                backup_key_A = (module_path, adapter_name, 'A')
                                original_weights_backup[backup_key_A] = lora_A_layer.weight.data.clone()
                                weight_A_data = lora_A_layer.weight.data
                                mask_A = (torch.rand_like(weight_A_data) > self.dropconnect_p).float()
                                lora_A_layer.weight.data = weight_A_data * mask_A

                            lora_B_layer = module.lora_B[adapter_name]
                            if lora_B_layer is not None:
                                backup_key_B = (module_path, adapter_name, 'B')
                                original_weights_backup[backup_key_B] = lora_B_layer.weight.data.clone()
                                weight_B_data = lora_B_layer.weight.data
                                mask_B = (torch.rand_like(weight_B_data) > self.dropconnect_p).float()
                                lora_B_layer.weight.data = weight_B_data * mask_B
                
                out = self.peft_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    **current_model_kwargs
                )
                
                all_agent_logits.append(out.logits)
                if output_hidden_states_flag and hasattr(out, 'hidden_states') and out.hidden_states is not None:
                    all_agent_last_hidden_states.append(out.hidden_states[-1])
                elif output_hidden_states_flag:
                    all_agent_last_hidden_states.append(None)

                if self.training and self.dropconnect_p > 0:
                    keys_to_remove_from_backup = []
                    for (m_path, ad_name_backup, matrix_type), orig_weight in original_weights_backup.items():
                        if ad_name_backup == adapter_name: # Only restore weights for the current agent
                            try:
                                target_module = dict(self.peft_model.base_model.named_modules())[m_path]
                                if isinstance(target_module, LoraLayer):
                                    if matrix_type == 'A' and ad_name_backup in target_module.lora_A:
                                        target_module.lora_A[ad_name_backup].weight.data = orig_weight
                                        keys_to_remove_from_backup.append((m_path, ad_name_backup, matrix_type))
                                    elif matrix_type == 'B' and ad_name_backup in target_module.lora_B:
                                        target_module.lora_B[ad_name_backup].weight.data = orig_weight
                                        keys_to_remove_from_backup.append((m_path, ad_name_backup, matrix_type))
                            except KeyError:
                                print(f"Warning: Module path '{m_path}' not found during DropConnect restoration for agent '{ad_name_backup}'. This should not happen.")
                    
                    for key_to_remove in keys_to_remove_from_backup:
                        del original_weights_backup[key_to_remove]
                
            if self.training and self.dropconnect_p > 0 and len(original_weights_backup) > 0:
                # print(f"Warning: `original_weights_backup` not empty after forward pass. Clearing. Size: {len(original_weights_backup)}")
                original_weights_backup.clear()

        return all_agent_logits, all_agent_last_hidden_states

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, 
                 generation_config: GenerationConfig = None,
                 aggregator = None,
                 **kwargs):
        """
        Custom generate method for CI-LLM.
        Works with both sequential and parallel modes.
        """
        if aggregator is None:
            raise ValueError("Aggregator module must be provided for CI-LLM generation.")
        
        # Get reference to appropriate model for config access
        reference_model = self.agent_peft_models[0] if self.parallel_mode == "parallel" else self.peft_model
        
        if not hasattr(reference_model, 'config') or reference_model.config is None:
             raise ValueError("reference_model.config is not available. Cannot determine pad_token_id or eos_token_id.")


        # Resolve generation_config
        # (This ensures that max_new_tokens etc. from kwargs are used if generation_config is also from reference_model)
        temp_gen_config = GenerationConfig() # Start with all HuggingFace defaults
        if hasattr(reference_model, 'generation_config') and reference_model.generation_config is not None:
            # Update from reference_model's config
            for key, value in reference_model.generation_config.to_dict().items():
                 if hasattr(temp_gen_config, key): setattr(temp_gen_config, key, value)
        if generation_config is not None: # Passed config overrides
            for key, value in generation_config.to_dict().items():
                 if hasattr(temp_gen_config, key): setattr(temp_gen_config, key, value)
        for key, value in kwargs.items(): # Direct kwargs override all
            if hasattr(temp_gen_config, key):
                setattr(temp_gen_config, key, value)
        
        effective_generation_config = temp_gen_config

        # Ensure pad_token_id and eos_token_id are set in the effective_generation_config
        # Use model's config for defaults if not present in generation_config
        if effective_generation_config.pad_token_id is None:
            effective_generation_config.pad_token_id = reference_model.config.pad_token_id
        if effective_generation_config.eos_token_id is None:
            effective_generation_config.eos_token_id = reference_model.config.eos_token_id
        
        # Critical fallback: if pad_token_id is STILL None, use eos_token_id
        if effective_generation_config.pad_token_id is None and effective_generation_config.eos_token_id is not None:
            # print("Warning: generate config pad_token_id is None after checks, using eos_token_id.")
            effective_generation_config.pad_token_id = effective_generation_config.eos_token_id

        if effective_generation_config.eos_token_id is None:
            # This could be problematic if the model relies on EOS for stopping.
            print("Warning: eos_token_id is not set in generation_config or model.config. Generation might not stop correctly.")


        current_input_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(current_input_ids)

        batch_size = current_input_ids.shape[0]
        
        for _ in range(effective_generation_config.max_new_tokens):
            model_inputs = {"input_ids": current_input_ids, "attention_mask": attention_mask}
            
            # Use the unified forward method (works for both sequential and parallel modes)
            agent_logits_list, _ = self.forward(
                **model_inputs,
                output_hidden_states_flag=False
            ) 
            
            next_token_logits_k_list = [k_logits[:, -1, :] for k_logits in agent_logits_list]
            aggregated_next_token_dist = aggregator(next_token_logits_k_list, sample_weights=False)
            
            if effective_generation_config.do_sample:
                # Add sampling logic here (e.g., multinomial sampling, top-k, top-p)
                # For now, let's stick to greedy if do_sample is False, or error if True but not implemented
                # This is a placeholder for more complex sampling
                probs = F.softmax(aggregated_next_token_dist, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) #.squeeze(-1)
            else: # Greedy
                next_token = torch.argmax(aggregated_next_token_dist, dim=-1).unsqueeze(-1)

            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=attention_mask.device)], dim=-1)

            if effective_generation_config.eos_token_id is not None:
                # Squeeze next_token for comparison if it's [batch_size, 1]
                if (next_token.squeeze(-1) == effective_generation_config.eos_token_id).all():
                    break
        
        return current_input_ids