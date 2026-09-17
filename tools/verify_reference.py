"""Verify preserved bytes and summarize recorded runs without importing models."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "docs/archive/reference-manifest.json").read_text())
    if manifest.get("format") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty reference manifest")
    failures = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            failures.append(f"INVALID PATH: {name}")
        elif not path.is_file():
            failures.append(f"MISSING: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"CHANGED: {name}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"PASS: {len(manifest['files'])} reference files match SHA-256.\n")
    print("Recorded validation perplexity (exp of saved validation loss):")
    print("| Run | Model | Final | Minimum | Step at minimum |")
    print("|---|---|---:|---:|---:|")
    for run in ("long_training", "scale_up"):
        for model in ("memnls", "xformer"):
            history = json.loads((root / "outputs" / run / model / "history.json").read_text())
            losses, steps = history["val_loss"], history["val_step"]
            if not losses or len(losses) != len(steps) or not all(math.isfinite(v) for v in losses):
                raise ValueError(f"Invalid validation history: {run}/{model}")
            best = min(range(len(losses)), key=losses.__getitem__)
            print(f"| {run} | {model} | {math.exp(losses[-1]):.2f} | "
                  f"{math.exp(losses[best]):.2f} | {steps[best]} |")
    print("\nThis checks the supplied files; it does not rerun or certify training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
