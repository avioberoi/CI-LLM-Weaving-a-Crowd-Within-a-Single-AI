import os
import yaml
import json
import logging
import time
import argparse
from typing import Dict, Optional, List, Tuple, Union
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler, GenerationConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DeepSpeedPlugin
from peft import get_peft_model_state_dict

from models.gemma_backbone import CILLMModel
from losses.sliced_w2 import SlicedWassersteinDiversityRegularizer
from aggregator.dirichlet_bayesian import DirichletAggregator


# Default configuration dictionary
DEFAULT_CONFIG = {
    "model_name": "google/gemma-2-9b",
    "num_agents": 3,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": None,
    "dropconnect_p": 0.0,
    "initialize_adapters_rand": False,
    "use_gradient_checkpointing": True,
    "epochs": 1,
    "batch_size": 1,
    "lr": 5e-5,
    "lambda_sw": 0.01,
    "swd_num_projections": 50,
    "max_seq_length": 512,
    "dataset_name": "gsm8k",
    "dataset_subset": "main",
    "output_dir": "ci_llm_output",
    "save_steps": 100,
    "gradient_accumulation_steps": 4,
    "max_train_steps": 200,
    "debug_subset_size": 16,
    "num_workers": 4,  # Added for DataLoader optimization
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "num_warmup_steps_ratio": 0.0,
    "logging_steps": 10,
    "seed": 42,
    "resume_from_checkpoint": None,  # Path to checkpoint directory
    "mixed_precision": "no",  # Options: "no", "fp16", "bf16"
    "use_deepspeed": False,  # Enable DeepSpeed integration
    "deepspeed_config_file": None,  # Path to DeepSpeed config JSON
    "parallel_mode": "sequential",  # Options: "sequential", "parallel"
    "profile_steps": None,  # Number of steps to profile (e.g., 5)
    "log_memory_usage": True,  # Log GPU memory usage
}


class PerformanceProfiler:
    """Simple performance profiler for tracking timing and memory usage."""
    
    def __init__(self, enabled: bool = True, logger: Optional[logging.Logger] = None):
        self.enabled = enabled
        self.logger = logger
        self.timers = {}
        self.start_times = {}
        
    def start(self, name: str):
        if self.enabled:
            self.start_times[name] = time.time()
            
    def end(self, name: str):
        if self.enabled and name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(elapsed)
            del self.start_times[name]
            
    def log_summary(self, step: int):
        if self.enabled and self.logger:
            self.logger.info(f"\n--- Performance Summary at Step {step} ---")
            for name, times in self.timers.items():
                avg_time = sum(times) / len(times)
                self.logger.info(f"{name}: {avg_time:.4f}s (avg over {len(times)} calls)")
                
    def reset(self):
        self.timers = {}
        self.start_times = {}


def log_memory_usage(accelerator: Accelerator, logger: logging.Logger, prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available() and accelerator.is_main_process:
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9  # GB
        logger.info(
            f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, "
            f"Max Allocated: {max_allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def setup_logging(accelerator: Accelerator, output_dir: str) -> logging.Logger:
    """
    Set up logging for training with both console and file outputs.
    
    Args:
        accelerator: HuggingFace Accelerator instance
        output_dir: Directory for log files
        
    Returns:
        Logger instance configured for distributed training
    """
    logger = get_logger(__name__)
    
    # Only log on main process to avoid duplicate logs
    if accelerator.is_main_process:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.logger.addHandler(file_handler)
        
    logger.info(f"Logging initialized. Output directory: {output_dir}")
    return logger


def load_config(config_path: str = "configs.yaml") -> Dict:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Merge with defaults to ensure all keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    print(f"Warning: {config_path} not found. Using default config values.")
    return DEFAULT_CONFIG


def tokenize_and_format_gsm8k(
    batch_of_samples: List[Dict[str, str]], 
    tokenizer: AutoTokenizer, 
    max_seq_length: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and format GSM8K data samples for training.
    
    Each sample contains a "question" and "answer" field. The model should
    learn to generate the answer given the question. Labels are masked for
    the question portion.
    
    Args:
        batch_of_samples: List of dictionaries with "question" and "answer" keys
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length for truncation
        
    Returns:
        Dictionary with "input_ids", "attention_mask", and "labels" tensors
    """
    # Extract questions and answers
    questions = [sample["question"] for sample in batch_of_samples]
    answers = [sample["answer"] for sample in batch_of_samples] 
    
    # Combine question and answer with space separator
    full_texts = [q + " " + a for q, a in zip(questions, answers)]
    
    # Tokenize the full texts
    tokenized_inputs = tokenizer(
        full_texts,
        padding="max_length", 
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    labels = input_ids.clone()

    # Mask labels for the question portion
    for i in range(len(questions)):
        # Tokenize question + space to find masking length
        prompt_for_masking = questions[i] + " "
        
        # Tokenize without special tokens to get raw token count
        q_token_ids = tokenizer(questions[i], add_special_tokens=False)['input_ids']
        space_token_ids = tokenizer(" ", add_special_tokens=False)['input_ids']
        
        # Calculate tokens to mask
        len_q_and_space_tokens = len(q_token_ids) + len(space_token_ids)
        
        # Account for BOS token if present
        start_index_for_masking = 0
        if tokenizer.bos_token_id is not None and input_ids[i, 0] == tokenizer.bos_token_id:
            start_index_for_masking = 1
            
        end_index_for_masking = start_index_for_masking + len_q_and_space_tokens
        
        # Apply masking
        actual_mask_end_index = min(end_index_for_masking, labels.size(1))
        if start_index_for_masking < actual_mask_end_index:
            labels[i, start_index_for_masking:actual_mask_end_index] = -100

        # Mask padding tokens
        if tokenizer.pad_token_id is not None:
            labels[i, attention_mask[i] == 0] = -100
            
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def extract_representations_for_swd(
    agent_hidden_states: List[torch.Tensor],
    strategy: str = "last_token"
) -> List[torch.Tensor]:
    """
    Extract representations from agent hidden states for SWD calculation.
    
    Args:
        agent_hidden_states: List of hidden state tensors [batch, seq_len, hidden_dim]
        strategy: Strategy for extracting representations ("last_token", "mean_pool", "first_token")
        
    Returns:
        List of processed representation tensors [batch, hidden_dim]
    """
    processed_reprs = []
    
    for hidden_states in agent_hidden_states:
        if hidden_states is None:
            continue
            
        if strategy == "last_token":
            # Use the last token's representation
            repr_tensor = hidden_states[:, -1, :]
        elif strategy == "mean_pool":
            # Mean pool across sequence length
            repr_tensor = hidden_states.mean(dim=1)
        elif strategy == "first_token":
            # Use the first token's representation (e.g., CLS-like)
            repr_tensor = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown representation extraction strategy: {strategy}")
            
        processed_reprs.append(repr_tensor)
        
    return processed_reprs


def save_custom_checkpoint_metadata(output_dir: str, config: Dict, model: CILLMModel) -> None:
    """
    Save custom metadata for CI-LLM checkpointing.
    
    This saves the ci_llm_meta_config.json and individual adapter weights
    that might be needed if Accelerate doesn't handle multi-adapter PEFT models directly.
    
    Args:
        output_dir: Directory to save metadata
        config: Training configuration
        model: CILLMModel instance
    """
    # Save CI-LLM meta config
    meta_config = {
        'r': model.lora_rank_arg,
        'lora_alpha': model.lora_alpha_arg,
        'lora_dropout': model.lora_dropout_arg,
        'target_modules': model.target_modules_arg,
        'bias': "none",
        'task_type': "CAUSAL_LM",
        'K_agents': model.K,
        'parallel_mode': getattr(model, 'parallel_mode', 'sequential')  # Save the mode used
    }
    
    meta_config_path = os.path.join(output_dir, "ci_llm_meta_config.json")
    with open(meta_config_path, 'w') as f:
        json.dump(meta_config, f, indent=4)
    
    # Save individual adapter weights if needed
    if model.K > 0:
        if hasattr(model, 'parallel_mode') and model.parallel_mode == 'parallel':
            # Save weights from parallel models
            for i in range(model.K):
                adapter_name = f"agent{i}"
                adapter_weights_path = os.path.join(output_dir, f"{adapter_name}_weights.pth")
                try:
                    adapter_state_dict = get_peft_model_state_dict(
                        model.agent_peft_models[i], 
                        adapter_name=adapter_name
                    )
                    torch.save(adapter_state_dict, adapter_weights_path)
                except Exception as e:
                    print(f"Warning: Could not save adapter {adapter_name}: {e}")
        elif hasattr(model, 'peft_model'):
            # Original sequential mode saving
            for i in range(model.K):
                adapter_name = f"agent{i}"
                adapter_weights_path = os.path.join(output_dir, f"{adapter_name}_weights.pth")
                try:
                    adapter_state_dict = get_peft_model_state_dict(
                        model.peft_model, 
                        adapter_name=adapter_name
                    )
                    torch.save(adapter_state_dict, adapter_weights_path)
                except Exception as e:
                    print(f"Warning: Could not save adapter {adapter_name}: {e}")


def main():
    """Main training function with Accelerate and DeepSpeed integration."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CI-LLM Training")
    parser.add_argument("--config", type=str, default="configs.yaml", help="Path to YAML configuration file")
    args, _ = parser.parse_known_args()
    # Load configuration
    config = load_config(args.config)
    
    # Setup DeepSpeed plugin if enabled
    deepspeed_plugin = None
    if config.get('use_deepspeed', False):
        if config.get('deepspeed_config_file'):
            # Load DeepSpeed config from file
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=config['deepspeed_config_file'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                gradient_clipping=1.0,
                zero_stage=2,  # Default to ZeRO-2
            )
        else:
            # Use programmatic DeepSpeed configuration
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                gradient_clipping=1.0,
                offload_optimizer_device="none",
                offload_param_device="none",
                zero3_init_flag=False,
                zero3_save_16bit_model=False,
            )
    
    # Initialize accelerator with mixed precision and DeepSpeed if specified
    # Use separate accumulation: DS plugin handles accumulation when deepspeed is enabled
    grad_accum_steps = 1 if config.get('use_deepspeed', False) else config['gradient_accumulation_steps']
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision=config.get('mixed_precision', 'no'),
        log_with=None,  # Can be configured later for W&B, TensorBoard, etc.
        deepspeed_plugin=deepspeed_plugin,
    )
    
    # Set up logging
    logger = setup_logging(accelerator, config['output_dir'])
    
    # Initialize performance profiler
    profiler = PerformanceProfiler(
        enabled=config.get('profile_steps') is not None,
        logger=logger
    )
    
    # Set random seed for reproducibility
    set_seed(config['seed'])
    
    # Log configuration
    if accelerator.is_main_process:
        logger.info("Training configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        
        if config.get('use_deepspeed'):
            logger.info("DeepSpeed is enabled")
            if deepspeed_plugin:
                logger.info(f"  Zero Stage: {deepspeed_plugin.zero_stage}")
        
        # Log initial memory usage
        if config.get('log_memory_usage'):
            log_memory_usage(accelerator, logger, "[Initial] ")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], 
        trust_remote_code=True, 
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load dataset
    logger.info(f"Loading dataset: {config['dataset_name']}, subset: {config['dataset_subset']}")
    dataset = load_dataset(config['dataset_name'], config['dataset_subset'])
    train_data = dataset["train"]

    if config.get("debug_subset_size"):
        logger.info(f"Using debug subset of {config['debug_subset_size']} samples")
        train_data = Subset(train_data, range(config['debug_subset_size']))

    # Initialize model
    logger.info(f"Initializing CILLMModel with {config['model_name']} and K={config['num_agents']} agents")
    logger.info(f"Parallel mode: {config.get('parallel_mode', 'sequential')}")
    
    # Check if resuming from checkpoint
    resume_checkpoint_dir = config.get('resume_from_checkpoint')
    if resume_checkpoint_dir and os.path.exists(resume_checkpoint_dir):
        logger.info(f"Will resume from checkpoint: {resume_checkpoint_dir}")
        # The trained_checkpoint_dir parameter in CILLMModel will handle loading adapter weights
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
            trained_checkpoint_dir=resume_checkpoint_dir,
            load_adapters_trainable=True,
            parallel_mode=config.get('parallel_mode', 'sequential')
        )
    else:
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
            parallel_mode=config.get('parallel_mode', 'sequential')
        )
    
    ci_model.train()
    logger.info("CILLMModel initialized")
    
    # Log memory after model initialization
    if config.get('log_memory_usage') and accelerator.is_main_process:
        log_memory_usage(accelerator, logger, "[After Model Init] ")

    # Initialize loss functions
    cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    if config['num_agents'] > 1 and config['lambda_sw'] > 0:
        diversity_loss_fn = SlicedWassersteinDiversityRegularizer(
            num_projections=config['swd_num_projections']
        )
        logger.info("SlicedWassersteinDiversityRegularizer initialized")
    else:
        diversity_loss_fn = None
        logger.info("Diversity loss (SWD) disabled")

    # Ensure learning rate is a float
    config['lr'] = float(config['lr'])

    # Initialize optimizer
    optimizer = optim.AdamW(
        ci_model.parameters(), 
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    logger.info("Optimizer initialized")

    # Create DataLoader with optimizations
    collate_wrapper = lambda batch: tokenize_and_format_gsm8k(
        batch, tokenizer, config['max_seq_length']
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_wrapper,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    logger.info(f"DataLoader initialized with {config.get('num_workers', 4)} workers")

    # Calculate training steps
    num_training_steps = config.get("max_train_steps") or (
        len(train_loader) // config['gradient_accumulation_steps'] * config['epochs']
    )
    
    # Initialize learning rate scheduler
    num_warmup_steps = 0
    if config.get('num_warmup_steps_ratio') and config['num_warmup_steps_ratio'] > 0:
        num_warmup_steps = int(num_training_steps * config['num_warmup_steps_ratio'])
    
    lr_scheduler = get_scheduler(
        name=config.get('lr_scheduler_type', 'linear'),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"LR Scheduler initialized for {num_training_steps} training steps")

    # Prepare everything with accelerator
    ci_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        ci_model, optimizer, train_loader, lr_scheduler
    )
    
    # Move diversity loss to device if needed
    if diversity_loss_fn is not None:
        diversity_loss_fn = diversity_loss_fn.to(accelerator.device)

    # Load checkpoint if resuming
    starting_epoch = 0
    global_step = 0
    
    if resume_checkpoint_dir and os.path.exists(resume_checkpoint_dir):
        try:
            logger.info(f"Loading training state from {resume_checkpoint_dir}")
            accelerator.load_state(resume_checkpoint_dir)
            
            # Load additional training progress info
            progress_file = os.path.join(resume_checkpoint_dir, "training_progress.json")
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    starting_epoch = progress.get('epoch', 0)
                    global_step = progress.get('global_step', 0)
                    logger.info(f"Resuming from epoch {starting_epoch}, step {global_step}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting from scratch")

    # Training loop
    logger.info("Starting training loop...")
    
    # Setup profiling if enabled
    profile_steps = config.get('profile_steps')
    profiling_active = False
    
    for epoch in range(starting_epoch, config['epochs']):
        ci_model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_swd_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            # Skip steps if resuming
            if epoch == starting_epoch and global_step > 0:
                steps_to_skip = global_step % (len(train_loader) // config['gradient_accumulation_steps'])
                if step < steps_to_skip * config['gradient_accumulation_steps']:
                    continue
            
            # Enable profiling for specified steps
            if profile_steps and global_step < profile_steps:
                profiling_active = True
            else:
                profiling_active = False
            
            # Backprop & optimizer—branch between DeepSpeed and standard DDP
            if config.get('use_deepspeed', False):
                # DeepSpeed: plugin handles gradient accumulation & sync
                if profiling_active:
                    profiler.start('data_transfer')
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                if profiling_active:
                    profiler.end('data_transfer')
                    profiler.start('forward_pass')

                # Forward pass
                list_of_agent_logits, list_of_agent_hidden_states = ci_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states_flag=(diversity_loss_fn is not None),
                )

                if profiling_active:
                    profiler.end('forward_pass')
                    profiler.start('loss_calculation')

                # Task + diversity loss
                task_losses = [
                    cross_entropy_loss_fn(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                    for logits in list_of_agent_logits
                ]
                avg_task_loss = torch.stack(task_losses).mean() if task_losses else torch.tensor(0.0)
                loss_swd = (
                    diversity_loss_fn(
                        extract_representations_for_swd(
                            list_of_agent_hidden_states, strategy='last_token'
                        )
                    )
                    if diversity_loss_fn and list_of_agent_hidden_states
                    else torch.tensor(0.0, device=avg_task_loss.device)
                )
                total_loss = avg_task_loss - config['lambda_sw'] * loss_swd

                if profiling_active:
                    profiler.end('loss_calculation')
                    profiler.start('backward_pass')

                # Backward
                accelerator.backward(total_loss)

                if profiling_active:
                    profiler.end('backward_pass')
                    profiler.start('optimizer_step')

                # Step only when sync_gradients (end of accumulation cycle)
                # if accelerator.sync_gradients:
                # if global_step % config['gradient_accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()

                if profiling_active:
                    profiler.end('optimizer_step')

                # Update epoch metrics
                epoch_loss += total_loss.item()
                epoch_task_loss += avg_task_loss.item()
                epoch_swd_loss += loss_swd.item()
            else:
                # DDP: use accelerate.accumulate to handle multi-step accumulation
                with accelerator.accumulate(ci_model):
                    if profiling_active:
                        profiler.start('data_transfer')
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']

                    if profiling_active:
                        profiler.end('data_transfer')
                        profiler.start('forward_pass')

                    list_of_agent_logits, list_of_agent_hidden_states = ci_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states_flag=(diversity_loss_fn is not None),
                    )

                    if profiling_active:
                        profiler.end('forward_pass')
                        profiler.start('loss_calculation')

                    task_losses = [
                        cross_entropy_loss_fn(
                            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                        )
                        for logits in list_of_agent_logits
                    ]
                    avg_task_loss = torch.stack(task_losses).mean() if task_losses else torch.tensor(0.0)
                    loss_swd = (
                        diversity_loss_fn(
                            extract_representations_for_swd(
                                list_of_agent_hidden_states, strategy='last_token'
                            )
                        )
                        if diversity_loss_fn and list_of_agent_hidden_states
                        else torch.tensor(0.0, device=avg_task_loss.device)
                    )
                    total_loss = avg_task_loss - config['lambda_sw'] * loss_swd

                    if profiling_active:
                        profiler.end('loss_calculation')
                        profiler.start('backward_pass')

                    accelerator.backward(total_loss)
                    if profiling_active:
                        profiler.end('backward_pass')
                        profiler.start('optimizer_step')

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if profiling_active:
                        profiler.end('optimizer_step')

                    epoch_loss += total_loss.item()
                    epoch_task_loss += avg_task_loss.item()
                    epoch_swd_loss += loss_swd.item()
            
            # Increment global step after accumulation
            if accelerator.sync_gradients:
                global_step += 1
                
                # Log profiling summary if we've finished profiling
                if profile_steps and global_step == profile_steps:
                    profiler.log_summary(global_step)
                    profiler.reset()
                
                # Log memory usage periodically
                if config.get('log_memory_usage') and global_step % config.get('logging_steps', 10) == 0:
                    if accelerator.is_main_process:
                        log_memory_usage(accelerator, logger, f"[Step {global_step}] ")
                
                # Logging
                if global_step % config.get('logging_steps', 10) == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    metrics = {
                        "train/total_loss": total_loss.item(),
                        "train/task_loss": avg_task_loss.item(),
                        "train/swd_loss": loss_swd.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step
                    }
                    accelerator.log(metrics, step=global_step)
                    
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}: "
                            f"Total Loss: {total_loss.item():.4f}, "
                            f"Task Loss: {avg_task_loss.item():.4f}, "
                            f"SWD Loss: {loss_swd.item():.4f}, "
                            f"LR: {current_lr:.2e}"
                        )
                
                # Save checkpoint
                if global_step % config['save_steps'] == 0:
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(
                            config['output_dir'], 
                            f"checkpoint_step_{global_step}"
                        )
                        logger.info(f"Saving checkpoint to {checkpoint_dir}")
                        
                        # Save accelerator state
                        accelerator.save_state(checkpoint_dir)
                        
                        # Access underlying model for custom checkpointing when using DDP/FSDP
                        unwrapped_model = accelerator.unwrap_model(ci_model)
                        save_custom_checkpoint_metadata(checkpoint_dir, config, unwrapped_model)
                        
                        # Save training progress
                        progress = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'config': config
                        }
                        with open(os.path.join(checkpoint_dir, "training_progress.json"), 'w') as f:
                            json.dump(progress, f, indent=4)
                        
                        # Save tokenizer
                        tokenizer.save_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
                
                # Check if reached max steps
                if config.get("max_train_steps") and global_step >= config["max_train_steps"]:
                    break
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_task_loss = epoch_task_loss / len(train_loader)
        avg_epoch_swd_loss = epoch_swd_loss / len(train_loader)
        
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch+1} completed. "
                f"Avg Loss: {avg_epoch_loss:.4f}, "
                f"Avg Task Loss: {avg_epoch_task_loss:.4f}, "
                f"Avg SWD Loss: {avg_epoch_swd_loss:.4f}"
            )
        
        # Check early stopping
        if config.get("max_train_steps") and global_step >= config["max_train_steps"]:
            logger.info(f"Reached max_train_steps ({config['max_train_steps']})")
            break
        
        accelerator.wait_for_everyone() # Wait for all processes to complete

    # Final save
    if accelerator.is_main_process:
        final_checkpoint_dir = os.path.join(config['output_dir'], "final_checkpoint")
        logger.info(f"Training complete. Saving final model to {final_checkpoint_dir}")
        
        accelerator.save_state(final_checkpoint_dir)
        unwrapped_model = accelerator.unwrap_model(ci_model)
        save_custom_checkpoint_metadata(final_checkpoint_dir, config, unwrapped_model)
        
        # Save final training progress
        progress = {
            'epoch': config['epochs'],
            'global_step': global_step,
            'config': config,
            'completed': True
        }
        with open(os.path.join(final_checkpoint_dir, "training_progress.json"), 'w') as f:
            json.dump(progress, f, indent=4)
        
        tokenizer.save_pretrained(os.path.join(final_checkpoint_dir, "tokenizer"))
        
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()