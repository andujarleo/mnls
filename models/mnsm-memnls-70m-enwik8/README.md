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
[`github.com/andujarleo/mnls`](https://github.com/andujarleo/mnls).

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

Use the [offline inference example](../../docs/pretrained.md#cpu-inference-using-local-files)
from the repository root. It loads the local configuration and weights with the
preserved implementation, including the tied embedding weights.

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
([`mnsm-transformer-70m-enwik8`](../mnsm-transformer-70m-enwik8/))
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
  author = {Leonardo Andujar},
  year   = {2026},
  url    = {https://github.com/andujarleo/mnls},
  note   = {Three structural principles, one equation, seven cross-domain instantiations.}
}
```

## Related

- Full repository: https://github.com/andujarleo/mnls
- [Companion Transformer](../mnsm-transformer-70m-enwik8/)
- [Reproduction guide](../../docs/reproduction.md)
- License: MIT (code) + CC BY 4.0 (documentation)
