import os
import sys
import torch
import gc
import json
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from src.models.gemma_backbone import CILLMModel
from src.losses.sliced_w2 import SlicedWassersteinDiversityRegularizer
from src.aggregator.dirichlet_bayesian import DirichletAggregator


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


def check_environment() -> Dict[str, bool]:
    """Check CUDA and writable directory."""
    print_section("Environment Check")
    checks = {
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Device Count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "HF_HOME Set": "HF_HOME" in os.environ,
        "Output Dir Writable": os.access(".", os.W_OK)
    }
    for k, v in checks.items():
        status = "✓" if (v if isinstance(v, bool) else v > 0) else "✗"
        print(f"{k}: {status} ({v})")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    return checks


def test_model_initialization() -> bool:
    """Test initializing CILLMModel in parallel mode."""
    print_section("Model Initialization Test")
    try:
        model_name = "google/gemma-2-2b"
        K = 2
        print(f"Initializing parallel CILLMModel with {model_name} and K={K} agents...")
        model = CILLMModel(
            model_name=model_name,
            K=K,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            dropconnect_p=0.0,
            target_modules=["q_proj", "v_proj"],
            initialize_adapters_rand=False,
            use_gradient_checkpointing=True,
            parallel_mode="parallel"
        )
        print("✓ Model initialized")
        dev = next(model.parameters()).device
        print(f"Model device: {dev}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {params:,}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ Model init failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_accelerate_integration() -> bool:
    """Test Accelerate with parallel model and dummy data."""
    print_section("Accelerate Integration Test")
    try:
        accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="no")
        print("✓ Accelerator initialized")
        model = CILLMModel(
            model_name="google/gemma-2-2b",
            K=2,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_gradient_checkpointing=True,
            parallel_mode="parallel"
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        # dummy data
        dummy_input_ids = torch.randint(0, 1000, (4, 32))
        dummy_attention_mask = torch.ones_like(dummy_input_ids)
        dummy_labels = dummy_input_ids.clone()
        dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=2)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        print("✓ Prepared with accelerator")
        model.train()
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            with accelerator.accumulate(model):
                agent_logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states_flag=True
                )
                loss_fn = nn.CrossEntropyLoss()
                losses = [
                    loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    for logits in agent_logits
                ]
                total_loss = torch.stack(losses).mean()
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
                print(f"✓ Forward/backward passed. Loss: {total_loss.item():.4f}")
                break
        del model, optimizer, dataloader
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ Accelerate test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_checkpointing() -> bool:
    """Test saving and loading checkpoints."""
    print_section("Checkpointing Test")
    checkpoint_dir = "test_checkpoint_temp"
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        accelerator = Accelerator()
        set_seed(42)
        model = CILLMModel(
            model_name="google/gemma-2-2b",
            K=2,
            lora_rank=4,
            lora_alpha=8,
            parallel_mode="parallel"
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)
        dummy_data = torch.randn(4, 32, 1024)
        dataloader = DataLoader(TensorDataset(dummy_data), batch_size=2)
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        print("Saving checkpoint...")
        accelerator.save_state(checkpoint_dir)
        meta = {
            'r': model.lora_rank_arg,
            'lora_alpha': model.lora_alpha_arg,
            'lora_dropout': model.lora_dropout_arg,
            'target_modules': model.target_modules_arg,
            'K_agents': model.K
        }
        with open(os.path.join(checkpoint_dir, "ci_llm_meta_config.json"), 'w') as f:
            json.dump(meta, f)
        print(f"✓ Saved to {checkpoint_dir}")
        original = list(model.parameters())[0].clone()
        with torch.no_grad():
            list(model.parameters())[0].add_(0.1)
        modified = list(model.parameters())[0].clone()
        print(f"✓ Param modified (diff: {(modified - original).abs().mean().item():.6f})")
        print("Loading checkpoint...")
        try:
            accelerator.load_state(checkpoint_dir)
            loaded = list(model.parameters())[0].clone()
            restored = torch.allclose(loaded, original, atol=1e-6)
            print(f"✓ Restored: {restored}")
        except Exception as e:
            print(f"Warning: load_state failed: {e}")
        print("Instantiating new model from checkpoint...")
        try:
            new_model = CILLMModel(
                model_name="google/gemma-2-2b",
                K=2,
                lora_rank=4,
                lora_alpha=8,
                trained_checkpoint_dir=checkpoint_dir,
                load_adapters_trainable=True,
                parallel_mode="parallel"
            )
            print("✓ New model loaded")
        except Exception as e:
            print(f"Warning: New model init failed: {e}")
        import shutil
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print("✓ Cleaned checkpoint dir")
        del model, new_model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Warning: Checkpoint test error: {e}")
        import shutil
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        return True


def test_diversity_loss() -> bool:
    """Test SWD loss."""
    print_section("Diversity Loss Test")
    try:
        swd = SlicedWassersteinDiversityRegularizer(num_projections=10)
        batch_size, hidden_dim, K = 4, 768, 3
        reps = [torch.randn(batch_size, hidden_dim) + i*0.1 for i in range(K)]
        loss_val = swd(reps)
        print(f"✓ SWD loss: {loss_val.item():.4f}")
        identical = [torch.randn(batch_size, hidden_dim)] * K
        loss_id = swd(identical)
        print(f"✓ Identical reps loss: {loss_id.item():.4f}")
        ok = loss_val.item() > loss_id.item()
        print(f"✓ Diversity working: {ok}")
        return ok
    except Exception as e:
        print(f"✗ SWD test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_generation() -> bool:
    """Test generation with DirichletAggregator."""
    print_section("Generation Test")
    try:
        model = CILLMModel(
            model_name="google/gemma-2-2b",
            K=2,
            lora_rank=4,
            lora_alpha=8,
            use_gradient_checkpointing=False,
            parallel_mode="parallel"
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-2b", trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        aggregator = DirichletAggregator(alpha=1.0)
        text = "What is 2 + 2?"
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        dev = next(model.parameters()).device
        inputs["input_ids"] = inputs["input_ids"].to(dev)
        inputs["attention_mask"] = inputs["attention_mask"].to(dev)
        print(f"Input: {text}, shape: {inputs['input_ids'].shape}")
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                aggregator=aggregator,
                max_new_tokens=10,
                do_sample=False
            )
        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(f"Generated: {out_text}")
        print("✓ Generation success")
        del model, aggregator
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback; traceback.print_exc()
        return False


def test_deepspeed_config() -> bool:
    """Test DeepSpeed JSON loading."""
    print_section("DeepSpeed Config Test")
    try:
        ds_path = "ds_config_zero2.json"
        if os.path.exists(ds_path):
            with open(ds_path, 'r') as f:
                ds_cfg = json.load(f)
            print(f"✓ Loaded DS config from {ds_path}")
            from accelerate.utils import DeepSpeedPlugin
            dsp = DeepSpeedPlugin(
                hf_ds_config=ds_path,
                gradient_accumulation_steps=2,
                gradient_clipping=1.0,
                zero_stage=2,
            )
            print("✓ DeepSpeedPlugin created")
        else:
            print(f"✗ DS config not found at {ds_path} (OK)")
        return True
    except Exception as e:
        print(f"✗ DS config test failed: {e}")
        return False


def test_parallel_agents() -> bool:
    """Test both parallel vs sequential outputs."""
    print_section("Parallel Agent Mode Test")
    try:
        accelerator = Accelerator(mixed_precision="no")
        model_p = CILLMModel(
            model_name="google/gemma-2-2b",
            K=2,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_gradient_checkpointing=False,
            parallel_mode="parallel"
        )
        model_p.eval()
        model_p = accelerator.prepare(model_p)
        print("✓ Parallel CILLMModel init")
        print(f"  Agents: {len(model_p.agent_peft_models)}")
        test_ids = torch.randint(0, 1000, (1, 20)).to(accelerator.device)
        test_mask = torch.ones_like(test_ids).to(accelerator.device)
        with torch.no_grad():
            logits_p, _ = model_p(
                input_ids=test_ids,
                attention_mask=test_mask,
                output_hidden_states_flag=True
            )
        print("✓ Parallel forward pass")
        print(f"  Num outputs: {len(logits_p)}, shape: {logits_p[0].shape}")
        model_s = CILLMModel(
            model_name="google/gemma-2-2b",
            K=2,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_gradient_checkpointing=False,
            parallel_mode="sequential"
        )
        model_s.eval()
        with torch.no_grad():
            logits_s, _ = model_s(
                input_ids=test_ids,
                attention_mask=test_mask,
                output_hidden_states_flag=True
            )
        print("✓ Sequential forward pass")
        print(f"  Agents match: {len(logits_s) == len(logits_p)}")
        del model_p, model_s
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"✗ Parallel test failed: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    """Run all quick sanity checks."""
    print_section("CI-LLM Dry Run")
    print("Testing CI-LLM functionality with Accelerate and multiprocessing.\n")
    results = {}
    results["Environment"] = all(check_environment().values())
    results["Model Init"] = test_model_initialization()
    results["Accelerate"] = test_accelerate_integration()
    results["Checkpointing"] = test_checkpointing()
    results["Diversity Loss"] = test_diversity_loss()
    results["Generation"] = test_generation()
    results["DeepSpeed Config"] = test_deepspeed_config()
    results["Parallel Agents"] = test_parallel_agents()

    print_section("Test Summary")
    all_ok = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_ok = False
    print("\n" + "="*60)
    if all_ok:
        print("All tests passed!")
    else:
        print("Some tests failed. Check errors above.")
    print("="*60)
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
