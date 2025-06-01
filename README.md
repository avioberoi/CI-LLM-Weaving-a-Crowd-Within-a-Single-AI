# **Collective-Intelligence-Informed LLM**
 
> Multi-agent reasoning on GSM-8K with Google **Gemma-3 4B** in 4-bit precision,QLoRA adapters, Sliced-Wasserstein diversity, and a Dirichlet aggregator.
> 
> We instantiate the classic pillars of collective intelligence **diversity, independence, aggregation** inside a single LLM.  
> A frozen **Gemma-3 4B** backbone is paired with **K** independent QLoRA adapters whose weights are stochastic-masked (DropConnect) to encourage representational independence.
> A **Sliced-Wasserstein²** regulariser maximises latent diversity across agents, while a **Dirichlet Bayesian aggregator** fuses their token-level predictions.
> Fine-tuned on **GSM-8K**, CI-LLM seeks the same error-cancelling and bias-mitigating benefits observed in human “wise crowds,” but fully algorithmic and GPU-tractable.
---

## Directory tree
```
├── aggregator/
│ └── dirichlet_bayesian.py # Dirichlet-Bayesian aggregator
├── configs.yaml # All tunable hyper-parameters
├── data/ # HF datasets cached here
├── eval_gsm8k.py # Evaluation script (exact-match accuracy)
├── losses/
│ └── sliced_w2.py # Sliced-Wasserstein regularisers
├── models/
│ └── gemma_backbone.py # CI-LLM backbone (QLoRA agents + DropConnect)
├── train.py # Main training loop
├── mock_dryrun.py # Synthetic sanity-check pipeline
└── README.md
```
---

## Quick start (A100 / 48 CPU cores)

```bash
git clone https://github.com/macs30200-s23/course-project-avi-oberoi.git
cd course-project-avi-oberoi
conda env create -f environment.yml
conda activate ci-llm

# Dry-run sanity check (random data, CPU)
python mock_dryrun.py

# Fine-tune on GSM-8K (4 GPUs, A100)
torchrun --nproc_per_node=4 train.py --config configs.yaml

# Evaluate
python eval_gsm8k.py --config configs.yaml
```
---

## Implementation Highlights

| Component                 | File                       | Notes                                                                                           |
| ------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------- |
| **Backbone + Agents**     | `models/gemma_backbone.py` | Loads Gemma-3 in 4-bit, attaches **K** LoRA adapters, masks weights with DropConnect per agent. |
| **Diversity Regulariser** | `losses/sliced_w2.py`      | Three variants (logits, final hidden, arbitrary layer). Differentiable, GPU-friendly.           |
| **Aggregator**            | `aggregator/dirichlet_bayesian.py`  | Samples Dirichlet weights per batch, produces weighted token-level distribution.                |
| **Training Loop**         | `train.py`                 | CE-loss on aggregated logits + λ × SW². Saves each adapter & full checkpoint.                   |
| **Evaluation**            | `eval_gsm8k.py`            | Generates CoT and computes exact-match accuracy on GSM-8K test split.                           |
| **Mock dry-run**          | `mock_dryrun.py`     | Random tensors only – validates shapes & forward passes without GPU.                            |


Hardware profiles available on Midway3
| Partition | #Nodes | GPUs | GPU type | VRAM/node |
| --------- | ------ | ---- | -------- | --------- |
| gpu       | 1      | 4    | **A100** | 384 GB    |
| gpu       | 5      | 4    | V100     | 192 GB    |
| gpu       | 5      | 4    | RTX6000  | 192 GB    |

---
