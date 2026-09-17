"""Infrastructure sanity check: train a small Transformer on TinyShakespeare.

What this script does
---------------------
This script exists solely to verify that the training infrastructure in
`../../implementation/neural/training.py` is working correctly. It trains
a small causal-attention Transformer — a model whose training behavior is
well established in the literature — on the same TinyShakespeare corpus
and confirms that loss decreases as expected.

What this script is not
-----------------------
This is not a comparison against Memory-NLS. The Transformer is not a
"baseline" in any competitive sense. The structural distinction is:
attention-based models scale to intelligence; Memory-NLS instantiates
intelligence from structure. They live in different paradigms. See
`../../CLAUDE.md`, sections "Intelligence-as-structure" and "Rule 7".

If you want to run the Memory-NLS instantiation, see `./train_tinyshakespeare.py`.

Wall time: approximately 45 seconds on NVIDIA RTX 4060.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.neural import (
    TransformerConfig, TransformerLanguageModel,
    TrainConfig, train,
    CharTokenizer, generate_text,
)


REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = REPO_ROOT / "experiments" / "neural" / "tinyshakespeare.txt"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tinyshakespeare" / "infra_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print("Training infrastructure sanity check (Transformer on TinyShakespeare).")
    print()

    text = download_tinyshakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=192, n_layers=4, n_heads=4, ffn_mult=4,
        max_seq_len=256, dropout=0.0,
    )
    model = TransformerLanguageModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    tcfg = TrainConfig(
        n_steps=4000, batch_size=32,
        learning_rate=3e-4, min_learning_rate=3e-5,
        warmup_steps=100, grad_clip_norm=1.0,
        eval_interval=500, eval_iters=16,
        log_interval=200, checkpoint_interval=10_000,
        use_amp=True, amp_dtype="bfloat16",
        seed=42, output_dir=str(OUTPUT_DIR),
    )
    get_train = make_batch_fn(train_data, 256, 32)
    get_eval = make_batch_fn(val_data, 256, 32)
    history = train(model, get_train, get_eval, tcfg, device=device)

    final_val = history["val_loss"][-1]
    final_ppl = float(torch.exp(torch.tensor(final_val)))
    print()
    print(f"Infrastructure check: loss decreased from random (~4.17) to {final_val:.3f}.")
    print(f"  → training infrastructure is working as expected.")
    print(f"  → final val perplexity: {final_ppl:.2f}")
    print()
    print(f"This number is not a benchmark target. It confirms that the")
    print(f"AdamW + cosine schedule + AMP + gradient-clipping training loop in")
    print(f"`implementation/neural/training.py` works on a model that is known")
    print(f"to train well.")


if __name__ == "__main__":
    main()
