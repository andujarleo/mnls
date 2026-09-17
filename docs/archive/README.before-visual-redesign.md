<!-- Archived presentation; model links updated to local copies. -->
# Memory-Nonlinear State Models

**Memory-NLS: auxiliary-field memory in 3D field dynamics and neural sequence models.**

This repository contains a focused extraction from a larger research project:
a 3D numerical solver, a PyTorch language model, experiment scripts, and recorded
training trajectories. The neural model combines causal memory convolution with
a cubic interaction. The experiments examine long-horizon training dynamics
alongside a Transformer implementation.

The original implementations, experiment parameters, histories, log, and figures
are preserved byte for byte. Publication tooling provides documentation and
checks around that reference implementation.

[Quick start](#quick-start) · [Recorded results](#recorded-results) ·
[Architecture](docs/architecture.md) · [Pretrained models](docs/pretrained.md) · [Reproduction guide](docs/reproduction.md) ·
[Contributing](CONTRIBUTING.md)

## Recorded results

![Recorded validation perplexity on enwik8](assets/scale_up_val_ppl.png)

The supplied enwik8 run contains 50,000 training steps per model, with roughly
71 million parameters each. Memory-NLS trends downward toward a plateau, with
fluctuations; the Transformer reaches a lower minimum and later undergoes a
sharp increase in validation perplexity before partially recovering.

| Quantity | Memory-NLS | Transformer |
|---|---:|---:|
| Parameters | 71,069,184 | 71,863,296 |
| Final validation perplexity | 4.27 | 4.87 |
| Minimum validation perplexity | 3.86 | 2.54 |
| Step at minimum | 48,000 | 22,500 |

Source: the supplied [Memory-NLS history](outputs/scale_up/memnls/history.json)
and [Transformer history](outputs/scale_up/xformer/history.json).
Perplexity is `exp(val_loss)` at the recorded evaluations. These values describe
this run; they are not averages over repeated runs. The original research frames
the trajectory difference as optimization-dynamics anti-collapse. Its original
presentation is preserved in [the archived README](docs/archive/README.original.md).

The repository also includes 50,000-step TinyShakespeare histories:

| Model | Final validation perplexity | Minimum | Step at minimum |
|---|---:|---:|---:|
| Memory-NLS | 6.93 | 6.72 | 44,000 |
| Transformer | 206.25 | 4.64 | 5,000 |

## Quick start

### Inspect the recorded results — no GPU or dependencies needed

From the repository root, with Python 3.10 or newer:

```bash
python3 tools/verify_reference.py
python3 -B -m unittest discover -s tests -v
```

The first command checks SHA-256 fingerprints of 29 preserved files and computes
the result tables from the saved histories. It does not start training, download
data, or write into `outputs/`. The tests also verify detection of modified and
missing files.

### Run small neural checks on CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests/runtime -v
```

These checks exercise causal FFT convolution, a small model's forward/backward
pass, and causality. They do not reproduce the 50,000-step training results.
See the [environment notes](docs/reproduction.md#environments) for the tested
local versions and their distinction from the original CUDA environment.

## Repository map

| Path | Contents |
|---|---|
| `implementation/neural/` | Memory layer, language model, Transformer, training and generation |
| `implementation/physics/` | 3D solver, memory kernels, observables and backend selection |
| `experiments/neural/` | TinyShakespeare and enwik8 scripts; bundled TinyShakespeare text |
| `experiments/physics/` | Anti-collapse and Bravais-sweep scripts |
| `outputs/` | Four recorded neural histories and the enwik8 run log |
| `assets/` | Original training figures |
| `models/` | Two complete pretrained snapshots, configurations and checksums |
| `docs/` | Architecture, execution instructions and preservation record |
| `tools/`, `tests/` | Offline integrity checks and small CPU checks |

## Running experiments

The experiment scripts retain their original output paths. **Use a separate
copy for new runs** so their writes cannot replace the supplied histories.
Full commands, data requirements and known extraction issues are documented in
the [reproduction guide](docs/reproduction.md).

The original runs used an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM. That machine
is no longer available. Final weights are preserved locally in `models/`, copied from:
[Memory-NLS](../../models/mnsm-memnls-70m-enwik8/) and
[Transformer](../../models/mnsm-transformer-70m-enwik8/).
The [pretrained guide](docs/pretrained.md) supplies source revisions, checksum
verification, offline CPU loading and Git LFS instructions. A complete original environment lock and training state are
not included, so bitwise training replay is not established by this release.

## Project status

This is a research code release with preserved reference experiments. Changes to
model mathematics, numerical methods, training settings or random-number handling
must be treated as new experiments; see [CONTRIBUTING.md](CONTRIBUTING.md).

The published model cards state MIT for code and CC BY 4.0 for documentation.
License files still need to be restored into this local extraction from the
authoritative project. Dataset source information is in the reproduction guide.
