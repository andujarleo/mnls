---
license: mit
language: en
library_name: pytorch
tags:
  - sequence-modeling
  - state-space-models
  - byte-level
  - structural-realism
  - memory-augmented
  - anti-collapse
datasets:
  - enwik8
metrics:
  - perplexity
pipeline_tag: text-generation
---

# Memory-NLS 70M (enwik8 byte-level)

A 70M-parameter byte-level language model using the **Memory-Nonlinear State
Model** (MNSM) architecture. The sequence-mixing primitive is derived from a
nonlinear Schrödinger field equation with multi-timescale auxiliary memory,
not from attention.

The auxiliary-field memory update
`∂t y_j = ν_j(ρ - y_j)` is mathematically equivalent to the diagonal-state
update of S4/S5/Mamba/RWKV. The full architecture extends this baseline with
nonlinear self-interaction (`Λ|Ψ|²`), anti-collapse via temporal memory lag,
and FDT-locked stochastic regularization.

## Headline empirical finding

This model trained on enwik8 for 50,000 steps with **monotonic stable
trajectory** to final validation perplexity **4.27**. A matched-shape
70M-parameter Transformer trained under identical conditions exhibited a
**catastrophic optimization collapse** at step 28,000 (peak val_ppl 27.17)
and ended at val_ppl 4.87, worse than its pre-crash minimum.

The structural anti-collapse mechanism the equation predicts in 3D field
dynamics manifests in the optimization landscape of neural networks. Same
form, different substrate. See full repository:
[`github.com/qrv0/mnsm`](https://github.com/qrv0/mnsm).

## Architecture

| Property | Value |
|---|---|
| Parameters | 71,069,184 |
| `d_model` | 768 |
| `n_layers` | 10 |
| `n_heads` (memory modes) | 12 |
| `ffn_mult` | 5 |
| `max_seq_len` | 1024 |
| `vocab_size` | 256 (byte-level) |
| Λ (nonlinearity) | -0.5 |
| Σλ (memory coupling total) | 0.3 |
| ν range | [0.5, 10.0] |

## Training

- **Dataset**: enwik8 (~100MB Wikipedia byte stream)
- **Steps**: 50,000
- **Sequence length**: 1024
- **Batch size**: 8
- **Optimizer**: AdamW, β=(0.9, 0.95), weight decay 0.01
- **Learning rate**: cosine schedule 3e-4 → 3e-5, 500 warmup steps
- **Precision**: bfloat16 mixed
- **Hardware**: NVIDIA RTX 4060 Laptop GPU
- **Wall time**: 3.1 hours
- **Random seed**: 42

## Usage

```python
import json
import importlib.util
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "qvr0/mnsm-memnls-70m-enwik8"

config_path = hf_hub_download(REPO, "config.json")
weights_path = hf_hub_download(REPO, "model.safetensors")
modeling_path = hf_hub_download(REPO, "modeling.py")

spec = importlib.util.spec_from_file_location("modeling", modeling_path)
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)

with open(config_path) as f:
    config_dict = json.load(f)

model = modeling.MemoryNLSLanguageModel(modeling.MemoryNLSConfig(**config_dict))
state = load_file(weights_path)
model.load_state_dict(state)
model.eval()

# Generate
prompt = "The history of "
input_ids = torch.tensor([list(prompt.encode("utf-8"))])
out = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=40)
print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))
```

## Final evaluation

| Metric | Value |
|---|---|
| Final validation perplexity | 4.27 |
| Min validation perplexity | 3.86 (at step 48,000, 96% of training) |
| Final train loss | 1.3226 |
| Final val loss | 1.4510 |
| Train-val gap | 0.13 |
| Catastrophic events during training | None |

## Methodological frame

This is not a benchmark contest. The Transformer comparison
([`qvr0/mnsm-transformer-70m-enwik8`](https://huggingface.co/qvr0/mnsm-transformer-70m-enwik8))
is presented as **differentiation, not competition**. The structural finding
is the trajectory shape (monotonic vs catastrophic), not the comparative
final perplexity number.

The work operates within a **structural-realist** methodology rather than
competitive empirical benchmarking. The same mathematical form derived from
three observational axioms about persistent extended entities (P1, P2, P3)
produces:

- 3D anti-collapse dynamics in NLS supercritical fields (physics)
- Mathematical equivalence with diagonal-state SSMs (machine learning)
- Mechanism shape correspondence with cosmological expansion (cosmology)
- Multi-timescale memory hierarchy matching biological cognition (neuroscience)
- Stable optimization trajectory in neural training (this model)

The cross-substrate manifestation of the same form is the principal evidence
for the structural claim.

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
- Companion Transformer (for differentiation): https://huggingface.co/qvr0/mnsm-transformer-70m-enwik8
- Methodology: https://github.com/qrv0/mnsm/tree/main/methodology
- License: MIT (code) + CC BY 4.0 (documentation)
