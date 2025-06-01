import os
import yaml
import json
import logging
import time
import argparse
from typing import Dict, Optional, List
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler, GenerationConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DeepSpeedPlugin

from src.models.gemma_backbone import CILLMModel
from src.losses.sliced_w2 import SlicedWassersteinDiversityRegularizer
from src.aggregator.dirichlet_bayesian import DirichletAggregator

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
    "num_workers": 4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "num_warmup_steps_ratio": 0.0,
    "logging_steps": 10,
    "seed": 42,
    "resume_from_checkpoint": None,
    "mixed_precision": "no",
    "use_deepspeed": False,
    "deepspeed_config_file": None,
    "parallel_mode": "parallel",
    "profile_steps": None,
    "log_memory_usage": True,
}

class PerformanceProfiler:
    """Track timing and memory usage."""
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
            self.timers.setdefault(name, []).append(elapsed)
            del self.start_times[name]
    def log_summary(self, step: int):
        if self.enabled and self.logger:
            self.logger.info(f"\n--- Perf Summary at Step {step} ---")
            for name, times in self.timers.items():
                avg = sum(times) / len(times)
                self.logger.info(f"{name}: {avg:.4f}s over {len(times)} calls")
    def reset(self):
        self.timers.clear()
        self.start_times.clear()

def log_memory_usage(accelerator: Accelerator, logger: logging.Logger, prefix: str = ""):
    """Log GPU memory."""
    if torch.cuda.is_available() and accelerator.is_main_process:
        a = torch.cuda.memory_allocated() / 1e9
        m = torch.cuda.max_memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        logger.info(f"{prefix}GPU Mem - Allocated: {a:.2f}GB, Max: {m:.2f}GB, Reserved: {r:.2f}GB")

def setup_logging(accelerator: Accelerator, output_dir: str) -> logging.Logger:
    """Initialize logger."""
    logger = get_logger(__name__)
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fmt)
        logger.logger.addHandler(fh)
    logger.info(f"Logging to {output_dir}")
    return logger

def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """Load YAML or defaults."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    print(f"Warning: {config_path} not found, using defaults")
    return DEFAULT_CONFIG

def tokenize_and_format_gsm8k(batch, tokenizer, max_seq_length):
    """Tokenize GSM8K."""
    qs = [s["question"] for s in batch]
    ans = [s["answer"] for s in batch]
    texts = [q + " " + a for q, a in zip(qs, ans)]
    toks = tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length, return_tensors="pt")
    input_ids = toks["input_ids"]
    attention_mask = toks["attention_mask"]
    labels = input_ids.clone()
    for i, q in enumerate(qs):
        q_ids = tokenizer(q, add_special_tokens=False)["input_ids"]
        sp_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        mask_len = len(q_ids) + len(sp_ids)
        start = 1 if tokenizer.bos_token_id and input_ids[i, 0] == tokenizer.bos_token_id else 0
        end = min(start + mask_len, labels.size(1))
        if start < end:
            labels[i, start:end] = -100
        if tokenizer.pad_token_id is not None:
            labels[i, attention_mask[i] == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def extract_representations_for_swd(agent_hidden_states: List[torch.Tensor], strategy: str = "last_token"):
    """Get agent representations."""
    reprs = []
    for hs in agent_hidden_states:
        if hs is None:
            continue
        if strategy == "last_token":
            r = hs[:, -1, :]
        elif strategy == "mean_pool":
            r = hs.mean(dim=1)
        elif strategy == "first_token":
            r = hs[:, 0, :]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        reprs.append(r)
    return reprs

def save_custom_checkpoint_metadata(output_dir: str, config: Dict, model: CILLMModel):
    """Save CI‐LLM metadata and adapter weights."""
    meta = {
        'r': model.lora_rank_arg,
        'lora_alpha': model.lora_alpha_arg,
        'lora_dropout': model.lora_dropout_arg,
        'target_modules': model.target_modules_arg,
        'bias': "none",
        'task_type': "CAUSAL_LM",
        'K_agents': model.K,
        'parallel_mode': getattr(model, 'parallel_mode', 'sequential')
    }
    with open(os.path.join(output_dir, "ci_llm_meta_config.json"), 'w') as f:
        json.dump(meta, f, indent=4)
    if model.K > 0:
        if getattr(model, 'parallel_mode', None) == 'parallel':
            for i in range(model.K):
                name = f"agent{i}"
                path = os.path.join(output_dir, f"{name}_weights.pth")
                try:
                    sd = get_peft_model_state_dict(model.agent_peft_models[i], adapter_name=name)
                    torch.save(sd, path)
                except:
                    pass
        else:
            for i in range(model.K):
                name = f"agent{i}"
                path = os.path.join(output_dir, f"{name}_weights.pth")
                try:
                    sd = get_peft_model_state_dict(model.peft_model, adapter_name=name)
                    torch.save(sd, path)
                except:
                    pass

def main():
    """Training with Accelerate."""
    parser = argparse.ArgumentParser(description="CI-LLM Training")
    parser.add_argument("--config", type=str, default="configs.yaml")
    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    deepspeed_plugin = None
    if config.get('use_deepspeed', False):
        if config.get('deepspeed_config_file'):
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=config['deepspeed_config_file'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                gradient_clipping=1.0,
                zero_stage=2,
            )
        else:
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                gradient_clipping=1.0,
                offload_optimizer_device="none",
                offload_param_device="none",
                zero3_init_flag=False,
                zero3_save_16bit_model=False,
            )

    grad_accum = 1 if config.get('use_deepspeed', False) else config['gradient_accumulation_steps']
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=config.get('mixed_precision', 'no'),
        log_with=None,
        deepspeed_plugin=deepspeed_plugin,
    )
    logger = setup_logging(accelerator, config['output_dir'])
    profiler = PerformanceProfiler(enabled=config.get('profile_steps') is not None, logger=logger)
    set_seed(config['seed'])

    if accelerator.is_main_process:
        logger.info("Config:")
        for k, v in config.items():
            logger.info(f"  {k}: {v}")
        if config.get('use_deepspeed'):
            logger.info("DeepSpeed enabled")
        if config.get('log_memory_usage'):
            log_memory_usage(accelerator, logger, "[Initial] ")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    logger.info(f"Loading dataset: {config['dataset_name']}/{config['dataset_subset']}")
    ds = load_dataset(config['dataset_name'], config['dataset_subset'])
    train_data = ds["train"]
    if config.get("debug_subset_size"):
        logger.info(f"Using debug subset of {config['debug_subset_size']} samples")
        train_data = Subset(train_data, range(config['debug_subset_size']))

    logger.info(f"Initializing CILLMModel with {config['model_name']} and K={config['num_agents']}")
    mode = config.get('parallel_mode', 'sequential')
    resume_dir = config.get('resume_from_checkpoint')
    if resume_dir and os.path.exists(resume_dir):
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
            trained_checkpoint_dir=resume_dir,
            load_adapters_trainable=True,
            parallel_mode=mode
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
            parallel_mode=mode
        )
    ci_model.train()
    logger.info("CILLMModel ready")
    if config.get('log_memory_usage') and accelerator.is_main_process:
        log_memory_usage(accelerator, logger, "[After Model Init] ")

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    if config['num_agents'] > 1 and config['lambda_sw'] > 0:
        diversity_fn = SlicedWassersteinDiversityRegularizer(num_projections=config['swd_num_projections'])
        logger.info("SWD regularizer ready")
    else:
        diversity_fn = None
        logger.info("SWD disabled")

    lt = float(config['lr'])
    optimizer = optim.AdamW(ci_model.parameters(), lr=lt, weight_decay=config.get('weight_decay', 0.01))
    logger.info("Optimizer ready")

    collate_fn = lambda batch: tokenize_and_format_gsm8k(batch, tokenizer, config['max_seq_length'])
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    logger.info(f"DataLoader initialized with {config.get('num_workers', 4)} workers")

    total_steps = config.get("max_train_steps") or (
        len(train_loader) // config['gradient_accumulation_steps'] * config['epochs']
    )
    num_warmup = 0
    if config.get('num_warmup_steps_ratio', 0) > 0:
        num_warmup = int(total_steps * config['num_warmup_steps_ratio'])
    lr_scheduler = get_scheduler(
        name=config.get('lr_scheduler_type', 'linear'),
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps
    )
    logger.info(f"Scheduler ready for {total_steps} steps")

    ci_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        ci_model, optimizer, train_loader, lr_scheduler
    )
    if diversity_fn is not None:
        diversity_fn = diversity_fn.to(accelerator.device)

    start_epoch = 0
    global_step = 0
    if resume_dir and os.path.exists(resume_dir):
        try:
            logger.info(f"Loading state from {resume_dir}")
            accelerator.load_state(resume_dir)
            pf = os.path.join(resume_dir, "training_progress.json")
            if os.path.exists(pf):
                with open(pf, 'r') as f:
                    prog = json.load(f)
                    start_epoch = prog.get('epoch', 0)
                    global_step = prog.get('global_step', 0)
                    logger.info(f"Resuming epoch {start_epoch}, step {global_step}")
        except Exception:
            logger.info("Could not load checkpoint, starting fresh")

    logger.info("Starting training")
    profile_steps = config.get('profile_steps')
    profiling = False

    for epoch in range(start_epoch, config['epochs']):
        ci_model.train()
        e_loss = 0.0
        e_task = 0.0
        e_swd = 0.0

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and global_step > 0:
                skip = global_step % (len(train_loader) // config['gradient_accumulation_steps'])
                if step < skip * config['gradient_accumulation_steps']:
                    continue

            profiling = profile_steps and global_step < profile_steps

            if config.get('use_deepspeed', False):
                if profiling: profiler.start('data')
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                if profiling:
                    profiler.end('data')
                    profiler.start('forward')
                logits_list, hidden_list = ci_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states_flag=(diversity_fn is not None),
                )
                if profiling:
                    profiler.end('forward')
                    profiler.start('loss')
                losses = [
                    loss_fn(l.view(-1, l.size(-1)), labels.view(-1))
                    for l in logits_list
                ]
                avg_task = torch.stack(losses).mean() if losses else torch.tensor(0.0)
                loss_swd = (
                    diversity_fn(extract_representations_for_swd(hidden_list, strategy='last_token'))
                    if diversity_fn and hidden_list else torch.tensor(0.0, device=avg_task.device)
                )
                total = avg_task - config['lambda_sw'] * loss_swd
                if profiling:
                    profiler.end('loss')
                    profiler.start('backward')
                accelerator.backward(total)
                if profiling:
                    profiler.end('backward')
                    profiler.start('step')
                optimizer.step()
                lr_scheduler.step()
                if profiling:
                    profiler.end('step')

                e_loss += total.item()
                e_task += avg_task.item()
                e_swd += loss_swd.item()

            else:
                with accelerator.accumulate(ci_model):
                    if profiling: profiler.start('data')
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    if profiling:
                        profiler.end('data')
                        profiler.start('forward')
                    logits_list, hidden_list = ci_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states_flag=(diversity_fn is not None),
                    )
                    if profiling:
                        profiler.end('forward')
                        profiler.start('loss')
                    losses = [
                        loss_fn(l.view(-1, l.size(-1)), labels.view(-1))
                        for l in logits_list
                    ]
                    avg_task = torch.stack(losses).mean() if losses else torch.tensor(0.0)
                    loss_swd = (
                        diversity_fn(extract_representations_for_swd(hidden_list, strategy='last_token'))
                        if diversity_fn and hidden_list else torch.tensor(0.0, device=avg_task.device)
                    )
                    total = avg_task - config['lambda_sw'] * loss_swd
                    if profiling:
                        profiler.end('loss')
                        profiler.start('backward')
                    accelerator.backward(total)
                    if profiling:
                        profiler.end('backward')
                        profiler.start('step')
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if profiling:
                        profiler.end('step')

                    e_loss += total.item()
                    e_task += avg_task.item()
                    e_swd += loss_swd.item()

            if accelerator.sync_gradients:
                global_step += 1
                if profile_steps and global_step == profile_steps:
                    profiler.log_summary(global_step)
                    profiler.reset()
                if config.get('log_memory_usage') and global_step % config.get('logging_steps', 10) == 0:
                    if accelerator.is_main_process:
                        log_memory_usage(accelerator, logger, f"[Step {global_step}] ")
                if global_step % config.get('logging_steps', 10) == 0:
                    curr_lr = optimizer.param_groups[0]['lr']
                    mets = {
                        "train/total_loss": total.item(),
                        "train/task_loss": avg_task.item(),
                        "train/swd_loss": loss_swd.item(),
                        "train/learning_rate": curr_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step
                    }
                    accelerator.log(mets, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}: Total {total.item():.4f}, Task {avg_task.item():.4f}, "
                            f"SWD {loss_swd.item():.4f}, LR {curr_lr:.2e}"
                        )
                if global_step % config['save_steps'] == 0:
                    if accelerator.is_main_process:
                        ckpt = os.path.join(config['output_dir'], f"checkpoint_step_{global_step}")
                        logger.info(f"Saving to {ckpt}")
                        accelerator.save_state(ckpt)
                        um = accelerator.unwrap_model(ci_model)
                        save_custom_checkpoint_metadata(ckpt, config, um)
                        prog = {'epoch': epoch, 'global_step': global_step, 'config': config}
                        with open(os.path.join(ckpt, "training_progress.json"), 'w') as f:
                            json.dump(prog, f, indent=4)
                        tokenizer.save_pretrained(os.path.join(ckpt, "tokenizer"))
                if config.get("max_train_steps") and global_step >= config["max_train_steps"]:
                    break

        ae = e_loss / len(train_loader)
        at = e_task / len(train_loader)
        aswd = e_swd / len(train_loader)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} done. AvgLoss: {ae:.4f}, Task {at:.4f}, SWD {aswd:.4f}")
        if config.get("max_train_steps") and global_step >= config["max_train_steps"]:
            logger.info(f"Reached max_train_steps ({config['max_train_steps']})")
            break
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(config['output_dir'], "final_checkpoint")
        logger.info(f"Saving final to {final_dir}")
        accelerator.save_state(final_dir)
        um = accelerator.unwrap_model(ci_model)
        save_custom_checkpoint_metadata(final_dir, config, um)
        final_prog = {'epoch': config['epochs'], 'global_step': global_step, 'config': config, 'completed': True}
        with open(os.path.join(final_dir, "training_progress.json"), 'w') as f:
            json.dump(final_prog, f, indent=4)
        tokenizer.save_pretrained(os.path.join(final_dir, "tokenizer"))
        logger.info("Training done")

if __name__ == "__main__":
    main()
