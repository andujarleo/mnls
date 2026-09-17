# Preserved pretrained models

Both public model repositories have been downloaded into `models/`, including
their weights, configurations, original model cards, modeling code and attributes:

| Model | Repository | Pinned revision |
|---|---|---|
| Memory-NLS | [qrv0/mnsm-memnls-70m-enwik8](https://huggingface.co/qrv0/mnsm-memnls-70m-enwik8) | `6133a700e8ec2de6bbf3ab5148894979aab7a2ab` |
| Transformer | [qrv0/mnsm-transformer-70m-enwik8](https://huggingface.co/qrv0/mnsm-transformer-70m-enwik8) | `f34e1381653a711ead27228c5f9555c56241998a` |

These revision IDs were read from the public Hub API during publication preparation.
The published configurations match the corresponding local scale-up configurations.
The namespace is **qrv0**; the `qvr0` spelling in the model-card usage examples
is a typo. Both model cards declare MIT metadata and state MIT for code and
CC BY 4.0 for documentation in their related information.

## Verify the local copies

The two weight files total 571,757,304 bytes (about 572 MB). Their SHA-256 hashes
and byte sizes match the upstream LFS metadata at the pinned revisions. The four
auxiliary files per model were checked against the upstream Git blob hashes.
Source IDs, revisions and hashes are recorded in [manifest.json](../models/manifest.json).

From the repository root:

```bash
shasum -a 256 -c models/SHA256SUMS
```

The model folders are complete local copies and can be used without access to
the original Hugging Face account. Preserve this folder in your own backups too.
The original remote READMEs retain their historical links and namespace typo;
the working example below uses the local files.

## CPU inference using local files

From this repository root, with its runtime dependencies installed:

```bash
python -m pip install -r requirements-inference.txt
```

The example uses the preserved local implementation and needs no network after
installing dependencies. It does not execute the downloaded `modeling.py`.
CPU inference needs additional RAM and runs independently of CUDA.

```python
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from implementation.neural import (
    MemoryNLSConfig, MemoryNLSLanguageModel,
    TransformerConfig, TransformerLanguageModel,
)

models = {
    "memnls": (
        "mnsm-memnls-70m-enwik8",
        MemoryNLSConfig, MemoryNLSLanguageModel,
    ),
    "transformer": (
        "mnsm-transformer-70m-enwik8",
        TransformerConfig, TransformerLanguageModel,
    ),
}
name, config_class, model_class = models["memnls"]
folder = Path("models") / name
with (folder / "config.json").open() as handle:
    config = config_class(**json.load(handle))
model = model_class(config)
state = load_file(folder / "model.safetensors", device="cpu")
# The architecture ties these two weights; some safetensors exports store one.
if "lm_head.weight" not in state and "tok_emb.weight" in state:
    state["lm_head.weight"] = state["tok_emb.weight"]
if "tok_emb.weight" not in state and "lm_head.weight" in state:
    state["tok_emb.weight"] = state["lm_head.weight"]
model.load_state_dict(state, strict=True)
model.eval()
prompt = torch.tensor([list("The history of ".encode("utf-8"))])
with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=40, temperature=0)
print(bytes(output[0].tolist()).decode("utf-8", errors="replace"))
```

Change the selection to `models["transformer"]` for the other model. Greedy
decoding (`temperature=0`) avoids sampling randomness in this example; it is
not the sampling setting used for the historical milestone outputs.

Both full weight files were loaded into the preserved local implementations with
`strict=True`. CPU checks verified finite logits and generated eight new greedy
tokens per model. This verifies weight/code compatibility and local inference;
it is not a new measurement of the historical validation perplexities.

Loading final weights is distinct from replaying training. Neither model repository
supplies the original optimizer and RNG state needed to resume that trajectory.

## Store the weights on GitHub

The root `.gitattributes` and the preserved model attributes mark `.safetensors`
for Git LFS. GitHub blocks individual files above 100 MiB in ordinary Git;
[Git LFS or release assets](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
support distributing larger files. The LFS route keeps the files at their model
paths in a normal checkout, with pointers in Git and full content in LFS storage.

Install Git LFS before cloning or adding model files. Run `git lfs install --local`
inside the checkout when configuring it for development. Commit the attributes
alongside the model files and verify `git lfs ls-files` lists both weights before
pushing. An ordinary commit of the raw weights will exceed GitHub's file limit.

After publication, validate a fresh clone with `git lfs pull` and the checksum
command above. Keep the complete local copies; a Git pointer alone is not a
backup of the weights.
