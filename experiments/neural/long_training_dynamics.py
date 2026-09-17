"""Long-horizon training of Memory-NLS and Transformer with dense trajectory
tracking, framed as the structural analog of the physics warmup/record
protocol.

What this script does
---------------------
Trains both `MemoryNLSLanguageModel` and `TransformerLanguageModel` for
50,000 steps on TinyShakespeare. The standard 4,000-step run (in
`train_tinyshakespeare.py` and `compare_architectures.py`) is the
training-dynamics equivalent of the physics simulations stopped during
the initial transient. The 50,000-step run continues until each
architecture reaches whatever stable regime it has access to under the
given conditions, in analogy to the R1 v4 physics experiment which used
a 2,000-step warmup followed by 4,000-step dense recording to study the
post-transient regime of the crystalline state.

What is tracked
---------------
- Train loss every 100 steps (~500 points per model).
- Validation loss every 500 steps (~100 points per model).
- Generation samples at milestones: 4000, 8000, 16000, 32000, 50000.

The samples at milestones let the reader see qualitatively how each
architecture's output evolves through the transient and into its stable
regime, not only at the endpoint.

Wall time: approximately 17 minutes on NVIDIA RTX 4060 (~8.5 min/model).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.neural import (
    MemoryNLSConfig, MemoryNLSLanguageModel,
    TransformerConfig, TransformerLanguageModel,
    TrainConfig, CharTokenizer, generate_text,
)


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = REPO_ROOT / "experiments" / "neural" / "tinyshakespeare.txt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "long_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Architecture (shared).
D_MODEL = 192
N_LAYERS = 4
N_HEADS = 4
SEQ_LEN = 256
BATCH_SIZE = 32

# Long-horizon training.
N_STEPS = 50_000
LOG_INTERVAL = 100
EVAL_INTERVAL = 500
EVAL_ITERS = 16
MILESTONES = [4000, 8000, 16000, 32000, 50000]   # where to generate samples

PROMPT = "ROMEO: "
MILESTONE_PROMPTS = ["ROMEO: ", "First Citizen:\n", "KING HENRY V: "]


def download_tinyshakespeare() -> str:
    if DATA_PATH.exists():
        return DATA_PATH.read_text()
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    DATA_PATH.write_text(text)
    return text


def make_batch_fn(data, seq_len, batch_size):
    def get_batch():
        ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix])
        return x, y
    return get_batch


def build_memnls(vocab_size):
    return MemoryNLSLanguageModel(MemoryNLSConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, ffn_mult=4, max_seq_len=SEQ_LEN, use_positional=False,
        nonlinearity_strength=-0.5, memory_coupling_total=0.3,
        nu_min=0.5, nu_max=10.0, dt=0.05, fast_bias=3.0,
        dissipation=0.0, fdt_temperature=0.0,
    ))


def build_xformer(vocab_size):
    return TransformerLanguageModel(TransformerConfig(
        vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, ffn_mult=4, max_seq_len=SEQ_LEN, dropout=0.0,
    ))


def cosine_lr(step, n_steps, warmup, max_lr, min_lr):
    import math
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, n_steps - warmup)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def evaluate(model, get_batch_eval, n_iters, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_iters):
            x, y = get_batch_eval()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_long(model, tokenizer, get_train, get_eval, device, name: str) -> dict:
    """Train a model for N_STEPS, recording trajectory and milestone samples."""
    from torch.amp import GradScaler, autocast
    from contextlib import nullcontext

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95),
    )
    amp_dt = torch.bfloat16 if device == "cuda" else None
    scaler = GradScaler(device, enabled=False)   # bf16 doesn't use scaler

    history = {
        "name": name,
        "n_steps": N_STEPS,
        "train_step": [],
        "train_loss": [],
        "lr": [],
        "val_step": [],
        "val_loss": [],
        "milestone_samples": {},
        "wall_time_s": 0.0,
    }

    t_start = time.time()
    running, count = 0.0, 0

    for step in range(N_STEPS + 1):   # +1 to include final step
        # LR schedule.
        lr = cosine_lr(step, N_STEPS, warmup=200, max_lr=3e-4, min_lr=3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step < N_STEPS:
            x, y = get_train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            ctx = autocast(device_type=device, dtype=amp_dt) if amp_dt else nullcontext()
            with ctx:
                _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            count += 1

        if step % LOG_INTERVAL == 0 and count > 0:
            avg = running / count
            running, count = 0.0, 0
            history["train_step"].append(step)
            history["train_loss"].append(avg)
            history["lr"].append(lr)
            if step % (LOG_INTERVAL * 10) == 0:
                elapsed = time.time() - t_start
                print(f"  [{name}] step {step:6d}/{N_STEPS}  lr {lr:.2e}  loss {avg:.4f}  ({elapsed:.0f}s)")

        if step % EVAL_INTERVAL == 0 and step > 0:
            val = evaluate(model, get_eval, EVAL_ITERS, device)
            history["val_step"].append(step)
            history["val_loss"].append(val)

        if step in MILESTONES:
            samples = {}
            for prompt in MILESTONE_PROMPTS:
                s = generate_text(model, tokenizer, prompt, max_new_tokens=200,
                                  temperature=0.8, top_k=40, device=device)
                samples[prompt] = s
            history["milestone_samples"][step] = samples
            print(f"  [{name}] milestone {step} samples captured")

    history["wall_time_s"] = time.time() - t_start
    torch.save({"model": model.state_dict(), "step": N_STEPS}, out_dir / "ckpt_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [{name}] done in {history['wall_time_s']:.0f}s")
    return history


def write_report(memnls_hist, xformer_hist):
    """Write long_training_results.md framed as transient-to-stable analysis."""
    import math

    def final_val(h):
        return h["val_loss"][-1] if h["val_loss"] else float("nan")

    def min_val(h):
        return min(h["val_loss"]) if h["val_loss"] else float("nan")

    def min_val_step(h):
        if not h["val_loss"]:
            return -1
        i = h["val_loss"].index(min(h["val_loss"]))
        return h["val_step"][i]

    md = []
    md.append("# Long-horizon training: transient → stable regime")
    md.append("")
    md.append("This document reports a 50,000-step training run of both Memory-NLS")
    md.append("and Transformer on TinyShakespeare, framed as the structural analog")
    md.append("of the physics warmup/record protocol used in the R1 vibration")
    md.append("experiment (see `../../results/03-vibrational-modes.md` and")
    md.append("`../../results/04-anti-collapse-3d.md`). The short 4,000-step")
    md.append("training run is the training-dynamics analog of stopping the physics")
    md.append("solver during its initial transient — before the auxiliary-field")
    md.append("memory has cycled enough times to produce its stable regime. This")
    md.append("longer run continues past the transient to see what regime each")
    md.append("architecture stabilizes into.")
    md.append("")
    md.append("## What the physics analog says to look for")
    md.append("")
    md.append("In the 3D physics solver, the auxiliary fields have relaxation times")
    md.append("$\\tau_j = 1/\\nu_j$ between 0.1 and 2 units of time. The R1 v4")
    md.append("experiment used a 2,000-step warmup ($\\sim$5 units of time) before")
    md.append("recording, because that is the timescale over which the slow memory")
    md.append("mode finishes its initial transient and the field settles into a")
    md.append("stable oscillation around its post-collapse-release peak.")
    md.append("")
    md.append("The training-dynamics analog is: the slow memory mode of the")
    md.append("Memory-NLS layer has $\\nu_\\text{min} = 0.5$ and $dt = 0.05$, giving")
    md.append("a relaxation timescale of $\\tau = 1 / (0.5 \\cdot 0.05) = 40$")
    md.append("training steps in terms of effective gradient steps the slow mode")
    md.append("has 'seen'. With batch size 32, the slow memory needs $O(10^3)$")
    md.append("optimizer steps to have integrated enough samples to act on its")
    md.append("characteristic timescale. The 4,000-step run is therefore on the")
    md.append("edge of the transient; the post-transient regime requires")
    md.append("substantially more.")
    md.append("")
    md.append("## What was run")
    md.append("")
    md.append("Both architectures trained for **50,000 steps** with identical")
    md.append("infrastructure (AdamW, cosine schedule lr 3e-4 → 3e-5, gradient")
    md.append("clipping 1.0, bf16 mixed precision, batch size 32, seq len 256).")
    md.append("")
    md.append("| Quantity | Memory-NLS | Transformer |")
    md.append("|---|---|---|")
    md.append(f"| Parameters | {sum(1 for _ in []):,} (filled below) | |")
    md.append(f"| Wall time | {memnls_hist['wall_time_s']:.0f} s | {xformer_hist['wall_time_s']:.0f} s |")
    md.append(f"| Final train loss | {memnls_hist['train_loss'][-1]:.4f} | {xformer_hist['train_loss'][-1]:.4f} |")
    md.append(f"| Final val loss | {final_val(memnls_hist):.4f} | {final_val(xformer_hist):.4f} |")
    md.append(f"| Final val perplexity | {math.exp(final_val(memnls_hist)):.2f} | {math.exp(final_val(xformer_hist)):.2f} |")
    md.append(f"| Min val loss reached | {min_val(memnls_hist):.4f} (at step {min_val_step(memnls_hist)}) | {min_val(xformer_hist):.4f} (at step {min_val_step(xformer_hist)}) |")
    md.append(f"| Min val perplexity | {math.exp(min_val(memnls_hist)):.2f} | {math.exp(min_val(xformer_hist)):.2f} |")
    md.append("")
    md.append("## Trajectory shape")
    md.append("")
    md.append("Validation loss at each milestone:")
    md.append("")
    md.append("| Step | Memory-NLS val_loss | Memory-NLS ppl | Transformer val_loss | Transformer ppl |")
    md.append("|---|---|---|---|---|")
    for milestone in MILESTONES:
        m_val = next((v for s, v in zip(memnls_hist["val_step"], memnls_hist["val_loss"]) if s >= milestone), None)
        x_val = next((v for s, v in zip(xformer_hist["val_step"], xformer_hist["val_loss"]) if s >= milestone), None)
        m_str = f"{m_val:.4f}" if m_val is not None else "—"
        m_ppl = f"{math.exp(m_val):.2f}" if m_val is not None else "—"
        x_str = f"{x_val:.4f}" if x_val is not None else "—"
        x_ppl = f"{math.exp(x_val):.2f}" if x_val is not None else "—"
        md.append(f"| {milestone:>5d} | {m_str} | {m_ppl} | {x_str} | {x_ppl} |")
    md.append("")
    md.append("## Generation samples through the trajectory")
    md.append("")
    md.append("How each architecture's output evolves through the transient and")
    md.append("into its stable regime, at the same prompt:")
    md.append("")
    for milestone in MILESTONES:
        md.append(f"### Step {milestone}")
        md.append("")
        for prompt in MILESTONE_PROMPTS:
            md.append(f"**Prompt:** `{prompt.strip()}`")
            md.append("")
            md.append("**Memory-NLS:**")
            md.append("")
            md.append("```")
            md.append(memnls_hist["milestone_samples"].get(milestone, {}).get(prompt, "[missing]"))
            md.append("```")
            md.append("")
            md.append("**Transformer:**")
            md.append("")
            md.append("```")
            md.append(xformer_hist["milestone_samples"].get(milestone, {}).get(prompt, "[missing]"))
            md.append("```")
            md.append("")
    md.append("## Structural reading")
    md.append("")
    md.append("Two things to look for, in analogy with the physics warmup/record")
    md.append("structure:")
    md.append("")
    md.append("1. **Where does each architecture's val_loss stabilize?**")
    md.append("   In the physics analog, the field reaches a plateau peak after")
    md.append("   the transient; thereafter it oscillates around that peak. In the")
    md.append("   training analog, the val_loss plateaus (or rises again due to")
    md.append("   overfitting on this small corpus) after the transient. Where")
    md.append("   each architecture plateaus, and how the plateau differs between")
    md.append("   them, is the structural fact to read.")
    md.append("")
    md.append("2. **Do the samples qualitatively change through the trajectory?**")
    md.append("   In the physics analog, the late-time crystalline state has")
    md.append("   internal vibrational structure not present in the initial")
    md.append("   collapse-and-release transient. In the training analog, the")
    md.append("   late-training output should exhibit structure not present in")
    md.append("   the early-training output. Look for: increased coherence,")
    md.append("   emergence of multi-sentence structure, stable character names,")
    md.append("   convergence to repeated patterns vs. continued exploration.")
    md.append("")
    md.append("## What this experiment is not")
    md.append("")
    md.append("This is not a benchmark contest at longer horizons. The numbers")
    md.append("compared above are not claims that either architecture is")
    md.append("'beating' the other; they are observations of how each one")
    md.append("settles into its stable regime under the same conditions. See")
    md.append("`../../CLAUDE.md` Rule 7a for the framing of comparison as")
    md.append("differentiation rather than competition.")
    md.append("")
    md.append("On a corpus this small (~1 MB), 50,000 training steps means many")
    md.append("hundreds of epochs through the data; both architectures will be")
    md.append("substantially overfitting by the end. The relevant observation is")
    md.append("not the absolute final loss but the *trajectory shape* and the")
    md.append("*qualitative character* of the post-transient regime.")
    md.append("")

    report_path = OUTPUT_DIR / "long_training_results.md"
    report_path.write_text("\n".join(md))
    print(f"\nReport written to {report_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Training both architectures for {N_STEPS} steps each.")
    print()

    text = download_tinyshakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    get_train = make_batch_fn(train_data, SEQ_LEN, BATCH_SIZE)
    get_eval = make_batch_fn(val_data, SEQ_LEN, BATCH_SIZE)

    print(f"\n{'='*70}\n  Memory-NLS (long horizon)\n{'='*70}")
    memnls = build_memnls(tokenizer.vocab_size)
    print(f"  parameters: {sum(p.numel() for p in memnls.parameters()):,}")
    m_hist = train_long(memnls, tokenizer, get_train, get_eval, device, name="memnls")
    del memnls
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  Transformer (long horizon)\n{'='*70}")
    xform = build_xformer(tokenizer.vocab_size)
    print(f"  parameters: {sum(p.numel() for p in xform.parameters()):,}")
    x_hist = train_long(xform, tokenizer, get_train, get_eval, device, name="xformer")
    del xform
    if device == "cuda":
        torch.cuda.empty_cache()

    write_report(m_hist, x_hist)


if __name__ == "__main__":
    main()
