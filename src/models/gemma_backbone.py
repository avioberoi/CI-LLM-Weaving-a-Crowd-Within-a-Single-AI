import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import parallel_apply
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
from peft.tuners.lora import LoraLayer

class CILLMModel(nn.Module):
    """CILLMModel is a PyTorch module that implements a Causal Inference LLM (CILLM) model
    with K agents, each represented by a LoRA adapter. It supports parallel and sequential
    modes of operation, allowing for flexible training and inference configurations."""
    def __init__(
        self,
        model_name: str,
        K: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        dropconnect_p: float = 0.0,
        target_modules: list[str] = None,
        initialize_adapters_rand: bool = False,
        use_gradient_checkpointing: bool = True,
        trained_checkpoint_dir: str = None,
        load_adapters_trainable: bool = False,
        parallel_mode: str = "sequential",
    ):
        super().__init__()
        self.model_name = model_name
        self.lora_rank_arg = lora_rank
        self.lora_alpha_arg = lora_alpha
        self.lora_dropout_arg = lora_dropout
        self.target_modules_arg = target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        self.dropconnect_p = dropconnect_p
        self.initialize_adapters_rand_on_create = initialize_adapters_rand
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.K = K
        self.parallel_mode = parallel_mode

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model_for_peft = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager",
            local_files_only=True,
        )
        base_model_for_peft = prepare_model_for_kbit_training(
            base_model_for_peft, use_gradient_checkpointing=self.use_gradient_checkpointing
        )
        if self.use_gradient_checkpointing:
            if hasattr(base_model_for_peft, "gradient_checkpointing_enable"):
                base_model_for_peft.gradient_checkpointing_enable()
            if hasattr(base_model_for_peft, "config"):
                base_model_for_peft.config.use_cache = False

        effective_lora_rank = self.lora_rank_arg
        effective_lora_alpha = self.lora_alpha_arg
        effective_lora_dropout = self.lora_dropout_arg
        effective_target_modules = self.target_modules_arg
        effective_K = self.K

        if trained_checkpoint_dir and os.path.isdir(trained_checkpoint_dir):
            meta_config_path = os.path.join(trained_checkpoint_dir, "ci_llm_meta_config.json")
            if os.path.exists(meta_config_path):
                try:
                    with open(meta_config_path, "r") as f:
                        loaded_meta_config = json.load(f)
                        effective_lora_rank = loaded_meta_config.get("r", effective_lora_rank)
                        effective_lora_alpha = loaded_meta_config.get("lora_alpha", effective_lora_alpha)
                        effective_lora_dropout = loaded_meta_config.get("lora_dropout", effective_lora_dropout)
                        effective_target_modules = loaded_meta_config.get("target_modules", effective_target_modules)
                        effective_K = loaded_meta_config.get("K_agents", effective_K)
                except Exception:
                    pass

        self.K = effective_K
        if self.K == 0 and not trained_checkpoint_dir:
            raise ValueError("K must be > 0 when creating a new CILLMModel without a checkpoint.")
        if self.K == 0 and trained_checkpoint_dir:
            self.peft_model = base_model_for_peft
            self.peft_model.train(load_adapters_trainable)
            return

        self.common_lora_config_runtime = LoraConfig(
            r=effective_lora_rank,
            lora_alpha=effective_lora_alpha,
            lora_dropout=effective_lora_dropout,
            target_modules=effective_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.base_model = base_model_for_peft

        if self.parallel_mode == "parallel":
            self.agent_peft_models = nn.ModuleList()
            for i in range(self.K):
                adapter_name = f"agent{i}"
                agent_model = get_peft_model(
                    self.base_model,
                    self.common_lora_config_runtime,
                    adapter_name=adapter_name,
                )
                self.agent_peft_models.append(agent_model)

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs found, but parallel_mode='parallel' was requested.")
            self.agent_devices = []
            for i, agent_model in enumerate(self.agent_peft_models):
                device = torch.device(f"cuda:{i % num_gpus}")
                self.agent_peft_models[i] = agent_model.to(device)
                self.agent_devices.append(device)
            self.peft_model = self.agent_peft_models[0]
        else:
            self.peft_model = get_peft_model(
                base_model_for_peft,
                self.common_lora_config_runtime,
                adapter_name="agent0",
            )
            for i in range(1, self.K):
                adapter_name = f"agent{i}"
                self.peft_model.add_adapter(adapter_name, self.common_lora_config_runtime)

        if trained_checkpoint_dir and os.path.isdir(trained_checkpoint_dir):
            loaded_adapter_count = 0
            if self.parallel_mode == "parallel":
                for i in range(self.K):
                    adapter_name = f"agent{i}"
                    adapter_weights_path = os.path.join(
                        trained_checkpoint_dir, f"{adapter_name}_weights.pth"
                    )
                    if os.path.exists(adapter_weights_path):
                        try:
                            adapter_state_dict = torch.load(adapter_weights_path, map_location="cpu")
                            set_peft_model_state_dict(
                                self.agent_peft_models[i], adapter_state_dict, adapter_name=adapter_name
                            )
                            loaded_adapter_count += 1
                        except Exception:
                            pass
            else:
                for i in range(self.K):
                    adapter_name = f"agent{i}"
                    adapter_weights_path = os.path.join(
                        trained_checkpoint_dir, f"{adapter_name}_weights.pth"
                    )
                    if os.path.exists(adapter_weights_path):
                        try:
                            adapter_state_dict = torch.load(adapter_weights_path, map_location="cpu")
                            set_peft_model_state_dict(self.peft_model, adapter_state_dict, adapter_name=adapter_name)
                            loaded_adapter_count += 1
                        except Exception:
                            pass
            if loaded_adapter_count == 0 and self.K > 0 and self.initialize_adapters_rand_on_create:
                self._randomly_initialize_all_adapters()
        elif self.initialize_adapters_rand_on_create and self.K > 0:
            self._randomly_initialize_all_adapters()

        if self.parallel_mode == "parallel":
            for agent_model in self.agent_peft_models:
                agent_model.train(load_adapters_trainable if trained_checkpoint_dir else True)
        else:
            self.peft_model.train(load_adapters_trainable if trained_checkpoint_dir else True)

    def _randomly_initialize_all_adapters(self):
        """Randomly initializes the LoRA adapters for all agents."""
        if self.parallel_mode == "parallel":
            for i, agent_model in enumerate(self.agent_peft_models):
                adapter_name = f"agent{i}"
                for module in agent_model.base_model.modules():
                    if isinstance(module, LoraLayer):
                        if adapter_name in module.lora_A and module.lora_A[adapter_name] is not None:
                            nn.init.normal_(module.lora_A[adapter_name].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_A[adapter_name], "bias") and module.lora_A[adapter_name].bias is not None:
                                nn.init.zeros_(module.lora_A[adapter_name].bias)
                        if adapter_name in module.lora_B and module.lora_B[adapter_name] is not None:
                            nn.init.normal_(module.lora_B[adapter_name].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_B[adapter_name], "bias") and module.lora_B[adapter_name].bias is not None:
                                nn.init.zeros_(module.lora_B[adapter_name].bias)
        else:
            for i in range(self.K):
                adapter_name_to_init = f"agent{i}"
                if adapter_name_to_init not in self.peft_model.peft_config:
                    continue
                for module in self.peft_model.base_model.modules():
                    if isinstance(module, LoraLayer):
                        if adapter_name_to_init in module.lora_A and module.lora_A[adapter_name_to_init] is not None:
                            nn.init.normal_(module.lora_A[adapter_name_to_init].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_A[adapter_name_to_init], "bias") and module.lora_A[adapter_name_to_init].bias is not None:
                                nn.init.zeros_(module.lora_A[adapter_name_to_init].bias)
                        if adapter_name_to_init in module.lora_B and module.lora_B[adapter_name_to_init] is not None:
                            nn.init.normal_(module.lora_B[adapter_name_to_init].weight, mean=0.0, std=0.02)
                            if hasattr(module.lora_B[adapter_name_to_init], "bias") and module.lora_B[adapter_name_to_init].bias is not None:
                                nn.init.zeros_(module.lora_B[adapter_name_to_init].bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_hidden_states_flag: bool = False,
        **kwargs
    ):
        current_model_kwargs = {"output_hidden_states": output_hidden_states_flag, **kwargs}
        all_agent_logits = []
        all_agent_last_hidden_states = []

        if self.parallel_mode == "parallel":
            if self.training and self.dropconnect_p > 0:
                for idx, agent_model in enumerate(self.agent_peft_models):
                    for module_path, module in agent_model.base_model.named_modules():
                        if isinstance(module, LoraLayer) and f"agent{idx}" in module.lora_A:
                            A = module.lora_A[f"agent{idx}"]
                            B = module.lora_B[f"agent{idx}"]
                            if A is not None:
                                mask_A = (torch.rand_like(A.weight) > self.dropconnect_p).float().to(A.weight.device)
                                A.weight.data *= mask_A
                            if B is not None:
                                mask_B = (torch.rand_like(B.weight) > self.dropconnect_p).float().to(B.weight.device)
                                B.weight.data *= mask_B

            modules = []
            inputs_list = []
            kwargs_list = []

            for idx, agent_model in enumerate(self.agent_peft_models):
                dev = self.agent_devices[idx]
                modules.append(agent_model)
                inputs_list.append((input_ids.to(dev), attention_mask.to(dev)))
                kwargs_list.append(current_model_kwargs.copy())

            outputs = parallel_apply(modules, inputs_list, kwargs_tup=kwargs_list)

            for out in outputs:
                all_agent_logits.append(out.logits.cpu())
                if output_hidden_states_flag and out.hidden_states is not None:
                    all_agent_last_hidden_states.append(out.hidden_states[-1].cpu())
                elif output_hidden_states_flag:
                    all_agent_last_hidden_states.append(None)

            return all_agent_logits, all_agent_last_hidden_states

        else:
            all_agent_logits = []
            all_agent_last_hidden_states = []
            original_weights_backup = {}

            for i in range(self.K):
                adapter_name = f"agent{i}"
                if adapter_name not in self.peft_model.peft_config:
                    continue
                self.peft_model.set_adapter(adapter_name)

                if self.training and self.dropconnect_p > 0:
                    for module_path, module in self.peft_model.base_model.named_modules():
                        if isinstance(module, LoraLayer) and adapter_name in module.lora_A:
                            lora_A_layer = module.lora_A[adapter_name]
                            if lora_A_layer is not None:
                                backup_key_A = (module_path, adapter_name, "A")
                                original_weights_backup[backup_key_A] = lora_A_layer.weight.data.clone()
                                mask_A = (torch.rand_like(lora_A_layer.weight) > self.dropconnect_p).float()
                                lora_A_layer.weight.data *= mask_A
                            lora_B_layer = module.lora_B[adapter_name]
                            if lora_B_layer is not None:
                                backup_key_B = (module_path, adapter_name, "B")
                                original_weights_backup[backup_key_B] = lora_B_layer.weight.data.clone()
                                mask_B = (torch.rand_like(lora_B_layer.weight) > self.dropconnect_p).float()
                                lora_B_layer.weight.data *= mask_B

                out = self.peft_model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states_flag
                )
                all_agent_logits.append(out.logits)
                if output_hidden_states_flag and out.hidden_states is not None:
                    all_agent_last_hidden_states.append(out.hidden_states[-1])
                elif output_hidden_states_flag:
                    all_agent_last_hidden_states.append(None)

                if self.training and self.dropconnect_p > 0:
                    keys_to_remove = []
                    for (m_path, ad_name, matrix_type), orig_weight in original_weights_backup.items():
                        if ad_name == adapter_name:
                            try:
                                target_module = dict(self.peft_model.base_model.named_modules())[m_path]
                                if isinstance(target_module, LoraLayer):
                                    if matrix_type == "A" and ad_name in target_module.lora_A:
                                        target_module.lora_A[ad_name].weight.data = orig_weight
                                        keys_to_remove.append((m_path, ad_name, matrix_type))
                                    elif matrix_type == "B" and ad_name in target_module.lora_B:
                                        target_module.lora_B[ad_name].weight.data = orig_weight
                                        keys_to_remove.append((m_path, ad_name, matrix_type))
                            except KeyError:
                                pass
                    for key in keys_to_remove:
                        del original_weights_backup[key]

            return all_agent_logits, all_agent_last_hidden_states

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        aggregator=None,
        **kwargs
    ):
        if aggregator is None:
            raise ValueError("Aggregator must be provided.")

        reference_model = (
            self.agent_peft_models[0] if self.parallel_mode == "parallel" else self.peft_model
        )
        if not hasattr(reference_model, "config") or reference_model.config is None:
            raise ValueError("reference_model.config is required.")

        temp_gen_config = GenerationConfig()
        if hasattr(reference_model, "generation_config") and reference_model.generation_config is not None:
            for key, value in reference_model.generation_config.to_dict().items():
                if hasattr(temp_gen_config, key):
                    setattr(temp_gen_config, key, value)
        if generation_config is not None:
            for key, value in generation_config.to_dict().items():
                if hasattr(temp_gen_config, key):
                    setattr(temp_gen_config, key, value)
        for key, value in kwargs.items():
            if hasattr(temp_gen_config, key):
                setattr(temp_gen_config, key, value)

        effective_generation_config = temp_gen_config
        if effective_generation_config.pad_token_id is None:
            effective_generation_config.pad_token_id = reference_model.config.pad_token_id
        if effective_generation_config.eos_token_id is None:
            effective_generation_config.eos_token_id = reference_model.config.eos_token_id
        if effective_generation_config.pad_token_id is None and effective_generation_config.eos_token_id is not None:
            effective_generation_config.pad_token_id = effective_generation_config.eos_token_id
        if effective_generation_config.eos_token_id is None:
            print("Warning: eos_token_id not set; generation might not stop.")

        current_input_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(current_input_ids)

        batch_size = current_input_ids.shape[0]
        for _ in range(effective_generation_config.max_new_tokens):
            model_inputs = {"input_ids": current_input_ids, "attention_mask": attention_mask}
            agent_logits_list, _ = self.forward(
                **model_inputs, output_hidden_states_flag=False
            )
            next_token_logits_k_list = [k_logits[:, -1, :] for k_logits in agent_logits_list]
            aggregated_next_token_dist = aggregator(next_token_logits_k_list, sample_weights=False)
            if effective_generation_config.do_sample:
                probs = F.softmax(aggregated_next_token_dist, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(aggregated_next_token_dist, dim=-1).unsqueeze(-1)

            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=attention_mask.device)],
                dim=-1,
            )
            if effective_generation_config.eos_token_id is not None:
                if (next_token.squeeze(-1) == effective_generation_config.eos_token_id).all():
                    break

        return current_input_ids
