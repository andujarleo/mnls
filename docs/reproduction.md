# Reproduction and preservation

## Included reference

The reference manifest covers 20 Python files, the bundled TinyShakespeare text,
four neural history JSON files, one enwik8 run log, two PNG figures, and a copy of
the original README. It records the bytes supplied for publication preparation.
The live README is editable; the original remains in `archive/README.original.md`.

Weights and their upstream metadata are now preserved locally in `models/`, with
a separate checksum manifest: see [pretrained models](pretrained.md).
No intermediate checkpoints, enwik8 download, original dependency lock, physics
output arrays, or figure-generation script were present in the local extraction.
The figures are preserved as supplied; tables can be recomputed from the JSON.

## Three distinct checks

| Check | Requirements | What it establishes |
|---|---|---|
| Reference integrity and result tables | Python standard library | Files match the supplied extraction; saved losses produce the displayed perplexities |
| Neural runtime checks | PyTorch on CPU | Causal convolution and a tiny model execute with finite gradients |
| Historical training replay | Data, suitable hardware, original environment and random state | Requires a separate run; not established by the first two checks |

```bash
python3 tools/verify_reference.py
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s tests/runtime -v
```

The runtime suite is separate so the first two commands need no machine-learning
dependencies. None of these commands starts training or downloads data.

## Environments

The original README reports Arch Linux, CUDA 12.x and an NVIDIA RTX 4060 Laptop
GPU. The author identifies it as the 8 GB model and reports that the machine is
no longer available. Exact Python, PyTorch, NumPy, CuPy, CUDA patch and driver
versions were not recorded in the supplied files.

Local publication checks passed on macOS ARM64 with Python 3.14.7, PyTorch 2.13.0
and NumPy 2.5.1, using CPU. These are the verification machine's versions, not
recovered versions from the historical run. Full training and physics experiments
were not rerun during this preparation.

`requirements.txt` lists runtime dependencies without inventing a historical
version lock. A fresh resolution may select different versions; record the
versions actually used for a new experiment. The preserved package initializer
imports `torch.amp.GradScaler`, so its old suggestion of `torch>=2.0` alone is not
a verified compatibility range. Use the runtime checks to verify an installation.

CuPy is optional for physics on GPU and must match the CUDA installation. It is
omitted from the portable requirements. `MEMNLS_FORCE_CPU=1` selects NumPy for
physics. Neural scripts select CUDA when available and otherwise CPU; they do
not select Apple's MPS backend.

## New runs without overwriting the reference

The historical scripts write to fixed locations under their own repository root.
Changing the working directory alone does **not** redirect their outputs.
Create a separate working copy first. From the reference repository root:

```bash
python3 - <<'PY'
from pathlib import Path
import shutil
import tempfile

source = Path.cwd()
destination = Path(tempfile.mkdtemp(prefix="mnls-run-")) / "mnls"
shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".DS_Store"
))
print(destination)
PY
```

Change into the printed directory and use the Python environment prepared above.
The copy initially contains the old outputs; these may be replaced there by the
new run. Label the new run and retain its command, environment and resulting files.
This temporary-directory example is disposable: save new results to durable
storage if they should be kept. A durable working copy elsewhere also works.

### Neural experiments

Run only the desired experiment, from the separate copy:

```bash
python experiments/neural/verify_training_infra.py
python experiments/neural/train_tinyshakespeare.py
python experiments/neural/compare_architectures.py
python experiments/neural/long_training_dynamics.py
python experiments/neural/scale_up_dynamics.py
```

The first three use short TinyShakespeare training configurations. The last two
run 50,000 steps per model. These are full experiments, not quick checks.
The supplied scale-up histories record about 3.11 hours for Memory-NLS and 3.21
hours for Transformer; times depend on hardware and environment.

The scale-up script uses byte tokens, vocabulary 256, sequence length 1,024,
batch size 8, width 768 and 10 layers. It splits enwik8 into the first 90 million
bytes for training and the next 5 million for validation; the final 5 million
are not evaluated by this script. Validation samples 16 batches per evaluation.

### Physics experiments

```bash
python experiments/physics/reproduce_3d_anti_collapse.py
python experiments/physics/reproduce_3d_bravais_sweep.py
```

Or use `python experiments/physics/reproduce_all.py` to run both sequentially.
For CPU, prefix the chosen command with `MEMNLS_FORCE_CPU=1`. The original grids
and durations remain unchanged and can be expensive on CPU.

## Data sources

TinyShakespeare is bundled at `experiments/neural/tinyshakespeare.txt` and
fingerprinted in the manifest. Its download code names the
[char-rnn TinyShakespeare source](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The scale-up script downloads [enwik8](https://mattmahoney.net/dc/enwik8.zip)
and extracts it locally; it does not verify a source checksum.
These are the upstream locations recorded by the scripts. Dataset redistribution
terms are separate from the project's code license.

## Preserved implementation notes

These observations document the supplied code. Publication preparation does not
patch them or retroactively change the recorded results.

- The dedicated long-training and scale-up scripts do not set an explicit seed.
  The shared helper sets a seed after model construction. The model cards report
  seed 42, but a complete original random state and deterministic-backend setup
  are not supplied. The archived README's bitwise-reproduction statement is not
  a guarantee established by this extraction.
- Validation uses randomly sampled batches. The Memory-NLS scale-up history
  contains 43 increases between adjacent evaluations. Its broad downward trend
  is not strict monotonicity.
- The Memory-NLS residual path is documented in `architecture.md`. Changing it
  changes the model and requires a separate experiment.
- `implementation/physics/sanity.py` imports absent historical module names
  `solver3d`, `kernels3d` and `observables3d`. That legacy entry point is not part
  of the new runtime checks. The main physics package uses the existing names.
- Historical docstrings refer to `CLAUDE.md`, `equation/`, `interfaces/` and
  material outside this extraction. These references remain for provenance;
  the local architecture page describes the included implementation.
- Scale-up checkpoints save weights and a step number without optimizer or RNG
  state. The script has no resume path. Published final weights enable inference,
  but do not reconstruct the original training trajectory.

## Publication record

Preparation added documentation, a dependency list, an integrity manifest,
an offline verifier, preservation tests and CPU neural checks. Listed reference
files were verified unchanged. The original README was copied before replacement.
No experiment configuration, result, model or solver was edited.

The model cards link to `qrv0/mnsm` on GitHub and state MIT for code and CC BY 4.0
for documentation. License files are absent from this local extraction and must
be restored from the authoritative project. The repository uses Git LFS to track
the two complete weight snapshots; see the pretrained guide for clone and checksum
verification instructions.
