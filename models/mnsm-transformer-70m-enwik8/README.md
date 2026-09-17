---
license: mit
language: en
library_name: pytorch
tags:
  - sequence-modeling
  - transformer
  - byte-level
  - baseline
  - structural-comparison
datasets:
  - enwik8
metrics:
  - perplexity
pipeline_tag: text-generation
---

# Transformer 70M (enwik8 byte-level) — structural-comparison baseline

A 70M-parameter byte-level Transformer language model trained on enwik8 for
structural comparison with [Memory-NLS](https://huggingface.co/qvr0/mnsm-memnls-70m-enwik8)
at matched architectural shape.

This model exists for **structural differentiation, not benchmark competition**.
It is included in the
[`qrv0/mnsm`](https://github.com/qrv0/mnsm)
repository as the contrast against which the Memory-NLS architecture's
structural anti-collapse property is empirically demonstrated.

## What this model exhibits

During the 50,000-step training run, this model:

1. **Reached a low validation minimum** (val_ppl 2.54 at step 22,500, 45% of training)
2. **Catastrophically collapsed** at step 28,000–34,000: validation perplexity spiked from 3.10 to 27.17 (an 8.8× degradation in 5,000 steps)
3. **Recovered partially** through the remaining steps but never returned to its pre-crash minimum
4. **Ended at val_ppl 4.87** — worse than its mid-training minimum and worse than the matched-shape Memory-NLS model (val_ppl 4.27)

The collapse is consistent with the structural-realist prediction:
architectures without explicit anti-collapse mechanism are vulnerable to
catastrophic loss of representational capacity during sustained training.
Engineering patches (skip connections, layer normalization, gradient
clipping, learning rate scheduling) defer this failure but do not remove it.

See [`results/08-optimization-collapse-empirical.md`](https://github.com/qrv0/mnsm/blob/main/results/08-optimization-collapse-empirical.md)
for the full structural finding.

## Architecture

| Property | Value |
|---|---|
| Parameters | 71,863,296 |
| `d_model` | 768 |
| `n_layers` | 10 |
| `n_heads` | 12 |
| `ffn_mult` | 4 |
| `max_seq_len` | 1024 |
| `vocab_size` | 256 (byte-level) |

Standard pre-norm Transformer with multi-head causal self-attention and
feedforward MLP blocks. No rotary positional embeddings, RMSNorm, SwiGLU,
or other modern attention engineering — kept architecturally parallel to
the Memory-NLS comparison.

## Training

Identical infrastructure to Memory-NLS:

- **Dataset**: enwik8 (~100MB Wikipedia byte stream)
- **Steps**: 50,000
- **Sequence length**: 1024
- **Batch size**: 8
- **Optimizer**: AdamW, β=(0.9, 0.95), weight decay 0.01
- **Learning rate**: cosine schedule 3e-4 → 3e-5, 500 warmup steps
- **Precision**: bfloat16 mixed
- **Hardware**: NVIDIA RTX 4060 Laptop GPU
- **Wall time**: 3.2 hours
- **Random seed**: 42

## Usage

```python
import json
import importlib.util
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "qvr0/mnsm-transformer-70m-enwik8"

config_path = hf_hub_download(REPO, "config.json")
weights_path = hf_hub_download(REPO, "model.safetensors")
modeling_path = hf_hub_download(REPO, "modeling.py")

spec = importlib.util.spec_from_file_location("modeling", modeling_path)
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)

with open(config_path) as f:
    config_dict = json.load(f)

model = modeling.TransformerLanguageModel(modeling.TransformerConfig(**config_dict))
state = load_file(weights_path)
model.load_state_dict(state)
model.eval()

prompt = "The history of "
input_ids = torch.tensor([list(prompt.encode("utf-8"))])
out = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=40)
print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))
```

## Final evaluation

| Metric | Value |
|---|---|
| Final validation perplexity | 4.87 |
| Min validation perplexity | 2.54 (at step 22,500, 45% of training, pre-crash) |
| Final train loss | 1.5121 |
| Final val loss | 1.5825 |
| Catastrophic collapse | Step 28,000–34,000, peak val_ppl 27.17 |

## Citation

```bibtex
@misc{mnsm,
  title  = {Memory-Nonlinear State Models: A Memory-Augmented Nonlinear Schrödinger
            Field Equation with State Space Model Correspondence},
  author = {qrv0},
  year   = {2026},
  url    = {https://github.com/qrv0/mnsm},
  note   = {Three structural principles, one equation, seven cross-domain instantiations.}
}
```

## Related

- Full repository: https://github.com/qrv0/mnsm
- Companion Memory-NLS model: https://huggingface.co/qvr0/mnsm-memnls-70m-enwik8
- Structural finding documentation: https://github.com/qrv0/mnsm/blob/main/results/08-optimization-collapse-empirical.md
- License: MIT (code) + CC BY 4.0 (documentation)
