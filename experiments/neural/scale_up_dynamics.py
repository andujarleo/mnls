"""Scale-up of the long-horizon training experiment to ~70M parameters
on enwik8, framed as the physics warmup/record analog at a substantially
larger scale.

What this script does
---------------------
Trains both `MemoryNLSLanguageModel` and `TransformerLanguageModel` for
50,000 steps on enwik8 at ~70M parameters each, sequence length 1024.
The shorter, smaller-scale run (`long_training_dynamics.py`, 50k steps
at 1.5M params on TinyShakespeare) established the existence of the
post-transient stabilization regime as a structural difference between
the two architectures. This run continues the observation at ~50× more
parameters and ~100× more data, to see how the structural regime each
architecture stabilizes into transforms (or does not) under those
conditions.

This is not a benchmark contest. The question being asked is: does the
structural regime — Memory-NLS plateauing at a generalization point,
Transformer collapsing onto the training distribution — persist at this
scale, transform in some specific way, or dissolve? Any of those
outcomes is information about the structure. See `../../CLAUDE.md`
section "Intelligence-as-structure, not intelligence-as-scale".

Wall time: approximately 5.6 hours on NVIDIA RTX 4060 Laptop (~2.8h/model).
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
import zipfile
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.neural import (
    MemoryNLSConfig, MemoryNLSLanguageModel,
    TransformerConfig, TransformerLanguageModel,
)


class ByteTokenizer:
    """Byte-level tokenizer. Vocab size 256, identity mapping byte ↔ id.

    This is the canonical enwik8 char-level setup: read the file as raw
    bytes and use byte values 0–255 as token IDs directly. Avoids the
    huge Unicode vocabulary that would result from treating enwik8 as
    a sequence of Unicode characters.
    """

    vocab_size = 256

    def encode(self, text):
        if isinstance(text, str):
            return list(text.encode("utf-8"))
        return list(text)

    def decode(self, ids):
        try:
            return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
        except Exception:
            return bytes(int(i) for i in ids).decode("latin-1", errors="replace")


@torch.no_grad()
def generate_bytes(model, tokenizer, prompt, max_new_tokens=300,
                   temperature=0.8, top_k=40, device="cuda"):
    """Sample text from a byte-tokenized model."""
    model.eval()
    ids = tokenizer.encode(prompt) or [0]
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(inp, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = REPO_ROOT / "experiments" / "neural" / "enwik8"
DATA_ZIP = REPO_ROOT / "experiments" / "neural" / "enwik8.zip"
OUTPUT_DIR = REPO_ROOT / "outputs" / "scale_up"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Architecture (shared shape; ~70M each).
D_MODEL = 768
N_LAYERS = 10
N_HEADS = 12
SEQ_LEN = 1024
BATCH_SIZE = 8

# Training horizon and tracking.
N_STEPS = 50_000
LOG_INTERVAL = 100
EVAL_INTERVAL = 500
EVAL_ITERS = 16
CHECKPOINT_INTERVAL = 10_000
MILESTONES = [2000, 4000, 8000, 16000, 32000, 50000]

# enwik8 is XML-tagged Wikipedia text; prompts that pull on the corpus's
# distribution (no Shakespeare line headers here).
MILESTONE_PROMPTS = [
    "The history of ",
    "In the year ",
    "<page>\n  <title>",
]


def load_enwik8_bytes() -> bytes:
    """Load enwik8 as raw bytes. Downloads on first call."""
    if DATA_PATH.exists():
        return DATA_PATH.read_bytes()
    if not DATA_ZIP.exists():
        url = "https://mattmahoney.net/dc/enwik8.zip"
        print(f"  downloading enwik8 from {url} (~36 MB)…")
        urllib.request.urlretrieve(url, DATA_ZIP)
    with zipfile.ZipFile(DATA_ZIP) as z:
        with z.open("enwik8") as f:
            data = f.read()
    DATA_PATH.write_bytes(data)
    return data


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
        n_heads=N_HEADS, ffn_mult=5, max_seq_len=SEQ_LEN, use_positional=False,
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
            with autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_long(model, tokenizer, get_train, get_eval, device, name: str) -> dict:
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95),
    )

    history = {
        "name": name,
        "n_steps": N_STEPS,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "param_count": sum(p.numel() for p in model.parameters()),
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

    for step in range(N_STEPS + 1):
        lr = cosine_lr(step, N_STEPS, warmup=500, max_lr=3e-4, min_lr=3e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step < N_STEPS:
            x, y = get_train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type=device, dtype=torch.bfloat16):
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
            if step % (LOG_INTERVAL * 5) == 0:
                elapsed = time.time() - t_start
                eta = elapsed / max(1, step) * (N_STEPS - step) if step > 0 else 0
                print(f"  [{name}] step {step:6d}/{N_STEPS}  lr {lr:.2e}  loss {avg:.4f}  "
                      f"({elapsed:.0f}s, eta {eta/60:.0f}m)")

        if step % EVAL_INTERVAL == 0 and step > 0:
            val = evaluate(model, get_eval, EVAL_ITERS, device)
            history["val_step"].append(step)
            history["val_loss"].append(val)
            print(f"    [{name}] eval step {step:6d}  val_loss {val:.4f}  ppl {math.exp(val):.2f}")
            # Persist history incrementally so a crashed run is not a total loss.
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        if step in MILESTONES:
            samples = {}
            for prompt in MILESTONE_PROMPTS:
                s = generate_bytes(model, tokenizer, prompt, max_new_tokens=300,
                                   temperature=0.8, top_k=40, device=device)
                samples[prompt] = s
            history["milestone_samples"][step] = samples
            print(f"  [{name}] milestone {step} samples captured")

        if step % CHECKPOINT_INTERVAL == 0 and step > 0:
            torch.save({"model": model.state_dict(), "step": step},
                       out_dir / f"ckpt_step_{step}.pt")

    history["wall_time_s"] = time.time() - t_start
    torch.save({"model": model.state_dict(), "step": N_STEPS},
               out_dir / "ckpt_final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [{name}] done in {history['wall_time_s']:.0f}s "
          f"({history['wall_time_s']/3600:.2f}h)")
    return history


def write_report(memnls_hist, xformer_hist):
    """Write scale_up_results.md framed in continuity with long_training_results.md."""

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
    md.append("# Scale-up: post-transient regime at ~70M parameters on enwik8")
    md.append("")
    md.append("This document continues the observation from")
    md.append("`../long_training/long_training_results.md` at a substantially")
    md.append("larger scale: ~70M parameters per architecture (vs ~1.5M),")
    md.append("trained on enwik8 (~100MB Wikipedia text, vs ~1MB Shakespeare),")
    md.append("with sequence length 1024 (vs 256), for 50,000 steps each.")
    md.append("")
    md.append("The earlier run established a structural difference: Memory-NLS")
    md.append("plateaus into a generalization regime; Transformer collapses onto")
    md.append("the training distribution and val loss diverges. That observation")
    md.append("was made on a corpus small enough that overfitting is the dominant")
    md.append("dynamic at the end of training. This run is at a scale where")
    md.append("overfitting is not forced by data scarcity (~4 epochs of enwik8 in")
    md.append("50k steps at batch 8 × seq 1024), so any divergence in regime")
    md.append("between the two architectures is attributable to the architectures")
    md.append("themselves, not to data exhaustion.")
    md.append("")
    md.append("This is not a benchmark contest. See `../../CLAUDE.md` Rule 7a.")
    md.append("")
    md.append("## What was run")
    md.append("")
    md.append("Both architectures share the same shape: $d_\\text{model} = 768$,")
    md.append(f"$n_\\text{{layers}} = {N_LAYERS}$, $n_\\text{{heads}} = {N_HEADS}$,")
    md.append(f"sequence length {SEQ_LEN}, batch size {BATCH_SIZE}.")
    md.append("Memory-NLS uses FFN multiplier 5 to match the Transformer's 4× FFN")
    md.append("at the parameter-count level (Memory-NLS's mixing layer has fewer")
    md.append("parameters than attention, so the FFN compensates).")
    md.append("")
    md.append("Identical training infrastructure for both: AdamW")
    md.append(r"($\beta = (0.9, 0.95)$, weight decay 0.01), cosine schedule")
    md.append("from $3 \\times 10^{-4}$ to $3 \\times 10^{-5}$ with 500-step warmup,")
    md.append("gradient clipping at L2 norm 1.0, bfloat16 mixed precision.")
    md.append("FP8 was considered but not used: while the RTX 4060 Laptop's Ada")
    md.append("Lovelace tensor cores support FP8 numerically, torchao's FP8 path")
    md.append("on consumer Ada Lovelace measured ~2.7× *slower* than bf16 at our")
    md.append("matmul shapes (the scaling/quantization overhead dominates the")
    md.append("matmul speedup at this hardware tier).")
    md.append("")
    md.append("| Quantity | Memory-NLS | Transformer |")
    md.append("|---|---|---|")
    md.append(f"| Parameters | {memnls_hist['param_count']:,} | {xformer_hist['param_count']:,} |")
    md.append(f"| Wall time | {memnls_hist['wall_time_s']/3600:.2f} h | {xformer_hist['wall_time_s']/3600:.2f} h |")
    md.append(f"| Final train loss | {memnls_hist['train_loss'][-1]:.4f} | {xformer_hist['train_loss'][-1]:.4f} |")
    md.append(f"| Final val loss | {final_val(memnls_hist):.4f} | {final_val(xformer_hist):.4f} |")
    md.append(f"| Final val perplexity | {math.exp(final_val(memnls_hist)):.2f} | {math.exp(final_val(xformer_hist)):.2f} |")
    md.append(f"| Min val loss | {min_val(memnls_hist):.4f} (step {min_val_step(memnls_hist)}) | "
              f"{min_val(xformer_hist):.4f} (step {min_val_step(xformer_hist)}) |")
    md.append(f"| Min val perplexity | {math.exp(min_val(memnls_hist)):.2f} | "
              f"{math.exp(min_val(xformer_hist)):.2f} |")
    md.append("")
    md.append("## Trajectory shape")
    md.append("")
    md.append("Validation loss at each milestone step:")
    md.append("")
    md.append("| Step | Memory-NLS val_loss | Memory-NLS ppl | Transformer val_loss | Transformer ppl |")
    md.append("|---|---|---|---|---|")
    for milestone in MILESTONES:
        m_val = next((v for s, v in zip(memnls_hist["val_step"], memnls_hist["val_loss"])
                      if s >= milestone), None)
        x_val = next((v for s, v in zip(xformer_hist["val_step"], xformer_hist["val_loss"])
                      if s >= milestone), None)
        m_str = f"{m_val:.4f}" if m_val is not None else "—"
        m_ppl = f"{math.exp(m_val):.2f}" if m_val is not None else "—"
        x_str = f"{x_val:.4f}" if x_val is not None else "—"
        x_ppl = f"{math.exp(x_val):.2f}" if x_val is not None else "—"
        md.append(f"| {milestone:>5d} | {m_str} | {m_ppl} | {x_str} | {x_ppl} |")
    md.append("")
    md.append("## Generation samples through the trajectory")
    md.append("")
    md.append("Same prompts at each milestone:")
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
            md.append(memnls_hist["milestone_samples"].get(str(milestone), {}).get(
                prompt, memnls_hist["milestone_samples"].get(milestone, {}).get(prompt, "[missing]")
            ))
            md.append("```")
            md.append("")
            md.append("**Transformer:**")
            md.append("")
            md.append("```")
            md.append(xformer_hist["milestone_samples"].get(str(milestone), {}).get(
                prompt, xformer_hist["milestone_samples"].get(milestone, {}).get(prompt, "[missing]")
            ))
            md.append("```")
            md.append("")
    md.append("## Structural reading at this scale")
    md.append("")
    md.append("Three quantities to read from the trajectory table and samples:")
    md.append("")
    md.append("1. **Whether the post-transient regime is still distinct between")
    md.append("   the two architectures.** At 1.5M parameters on 1MB the gap was")
    md.append("   dramatic (val ppl 6.93 vs 206 at step 50k). At 70M on 100MB the")
    md.append("   gap may be smaller, comparable, or in the opposite direction —")
    md.append("   each of those is structurally informative.")
    md.append("")
    md.append("2. **Where each architecture's val loss minimum sits.** In the")
    md.append("   earlier run, Memory-NLS's minimum was near the end of training")
    md.append("   (step 44000); Transformer's was very early (step 5000) followed")
    md.append("   by monotone divergence. The location of the minimum is the")
    md.append("   structural fact: a late minimum means the architecture is still")
    md.append("   integrating; an early minimum followed by divergence means the")
    md.append("   architecture entered a memorization regime that the data scale")
    md.append("   did not punish in the earlier run.")
    md.append("")
    md.append("3. **The qualitative character of generation samples.** In the")
    md.append("   1MB run, Memory-NLS produced phonologically-coherent invented")
    md.append("   words while Transformer copy-pasted training-corpus fragments")
    md.append("   (real character names, real vocabulary, but recombined). At")
    md.append("   100MB, Transformer has enough surface coverage that copy-paste")
    md.append("   and modeling are no longer distinguishable on lexical grounds")
    md.append("   alone. The structural distinction now has to be read from the")
    md.append("   trajectory and from longer-range coherence in the samples.")
    md.append("")
    md.append("## What this experiment is not")
    md.append("")
    md.append("This is not a claim that one architecture is 'better than' the")
    md.append("other at 70M parameters. The Transformer at this size on enwik8")
    md.append("is in the parameter regime where attention is engineered to")
    md.append("perform well, and a competitive char-level benchmark project")
    md.append("would also use rotary positional embeddings, RMSNorm, SwiGLU,")
    md.append("and so on — none of which are in this baseline implementation,")
    md.append("which is kept architecturally parallel to the Memory-NLS LM by")
    md.append("design. The comparison is differentiation at this scale, not a")
    md.append("benchmark result.")
    md.append("")

    report_path = OUTPUT_DIR / "scale_up_results.md"
    report_path.write_text("\n".join(md))
    print(f"\nReport written to {report_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"Training both ~70M models for {N_STEPS} steps each on enwik8.")
    print()

    print("Loading enwik8 (byte-level)…")
    raw = load_enwik8_bytes()
    print(f"  enwik8: {len(raw):,} bytes")
    tokenizer = ByteTokenizer()
    print(f"  vocab size: {tokenizer.vocab_size} (byte-level)")

    data = torch.tensor(list(raw), dtype=torch.long)
    # Standard enwik8 split: first 90M train, next 5M val, last 5M test.
    n_train = 90_000_000
    n_val = 5_000_000
    train_data, val_data = data[:n_train], data[n_train:n_train + n_val]
    print(f"  train bytes: {len(train_data):,}, val bytes: {len(val_data):,}")

    get_train = make_batch_fn(train_data, SEQ_LEN, BATCH_SIZE)
    get_eval = make_batch_fn(val_data, SEQ_LEN, BATCH_SIZE)

    print(f"\n{'='*70}\n  Memory-NLS (70M, long horizon, enwik8)\n{'='*70}")
    memnls = build_memnls(tokenizer.vocab_size)
    print(f"  parameters: {sum(p.numel() for p in memnls.parameters()):,}")
    m_hist = train_long(memnls, tokenizer, get_train, get_eval, device, name="memnls")
    del memnls
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n  Transformer (70M, long horizon, enwik8)\n{'='*70}")
    xform = build_xformer(tokenizer.vocab_size)
    print(f"  parameters: {sum(p.numel() for p in xform.parameters()):,}")
    x_hist = train_long(xform, tokenizer, get_train, get_eval, device, name="xformer")
    del xform
    if device == "cuda":
        torch.cuda.empty_cache()

    write_report(m_hist, x_hist)


if __name__ == "__main__":
    main()
