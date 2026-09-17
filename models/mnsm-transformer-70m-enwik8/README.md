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
structural comparison with [Memory-NLS](../mnsm-memnls-70m-enwik8/)
at matched architectural shape.

This model exists for **structural differentiation, not benchmark competition**.
It is included in the
[`andujarleo/mnls`](https://github.com/andujarleo/mnls)
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

See [Final generation comparison](../../docs/final-generation.md)
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

Use the [offline inference example](../../docs/pretrained.md#cpu-inference-using-local-files)
from the repository root. It loads the local configuration and weights with the
preserved implementation, including the tied embedding weights.

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
  author = {Leonardo Andujar},
  year   = {2026},
  url    = {https://github.com/andujarleo/mnls},
  note   = {Three structural principles, one equation, seven cross-domain instantiations.}
}
```

## Related

- Full repository: https://github.com/andujarleo/mnls
- [Companion Memory-NLS model](../mnsm-memnls-70m-enwik8/)
- [Final generation comparison](../../docs/final-generation.md)
- License: MIT (code) + CC BY 4.0 (documentation)
