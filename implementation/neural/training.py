"""Training infrastructure for the memory-NLS language model.

Standard ingredients:
  - AdamW optimizer with cosine learning-rate schedule and linear warmup.
  - Gradient clipping at a configurable global L2 norm.
  - Optional mixed precision via torch.cuda.amp.
  - Checkpointing at fixed step intervals.
  - Logging of train/val loss to a JSON file.

The training loop is generic; it takes a model with `forward(input_ids, targets)`
that returns `(logits, loss)`, and dataloaders that yield `(input_ids, targets)`
pairs.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    n_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 32
    log_interval: int = 50
    checkpoint_interval: int = 1000
    use_amp: bool = True
    amp_dtype: str = "bfloat16"             # "float16", "bfloat16", or "float32"
    output_dir: str = "outputs/train_run"
    seed: int = 42


def cosine_lr_with_warmup(
    step: int, n_steps: int, warmup_steps: int,
    max_lr: float, min_lr: float,
) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _amp_dtype(name: str) -> Optional[torch.dtype]:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return None


def _evaluate(model, get_batch_eval: Callable, iters: int, device: str) -> float:
    """Average loss over `iters` evaluation batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch_eval()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(
    model: torch.nn.Module,
    get_batch_train: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    get_batch_eval: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    device: str = "cuda",
) -> dict:
    """Train `model` for `cfg.n_steps`.

    Args:
        model: a PyTorch model whose forward returns (logits, loss).
        get_batch_train: callable returning (input_ids, targets) for training.
        get_batch_eval:  callable returning (input_ids, targets) for evaluation.
        cfg: TrainConfig.
        device: "cuda" or "cpu".

    Returns:
        dict with the training history (train losses, eval losses at each
        eval interval, wall-clock time, parameter count).
    """
    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Mixed precision.
    amp_dt = _amp_dtype(cfg.amp_dtype) if cfg.use_amp and device == "cuda" else None
    scaler = GradScaler(device, enabled=(amp_dt == torch.float16))

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "step": [],
        "val_step": [],
        "param_count": sum(p.numel() for p in model.parameters()),
        "wall_time_s": 0.0,
    }

    t_start = time.time()
    running_loss = 0.0
    running_count = 0

    for step in range(cfg.n_steps):
        # Set LR.
        lr = cosine_lr_with_warmup(
            step, cfg.n_steps, cfg.warmup_steps,
            cfg.learning_rate, cfg.min_learning_rate,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Step.
        x, y = get_batch_train()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = autocast(device_type=device, dtype=amp_dt) if amp_dt else nullcontext()
        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        running_loss += loss.item()
        running_count += 1

        if step % cfg.log_interval == 0 or step == cfg.n_steps - 1:
            avg = running_loss / max(1, running_count)
            running_loss, running_count = 0.0, 0
            elapsed = time.time() - t_start
            print(f"  step {step:5d}/{cfg.n_steps}  lr {lr:.2e}  loss {avg:.4f}  ({elapsed:.1f}s)")
            history["step"].append(step)
            history["train_loss"].append(avg)
            history["lr"].append(lr)

        if (step % cfg.eval_interval == 0 and step > 0) or step == cfg.n_steps - 1:
            val_loss = _evaluate(model, get_batch_eval, cfg.eval_iters, device)
            print(f"    [eval] step {step:5d}  val_loss {val_loss:.4f}  (ppl {math.exp(val_loss):.2f})")
            history["val_step"].append(step)
            history["val_loss"].append(val_loss)

        if step % cfg.checkpoint_interval == 0 and step > 0:
            ckpt_path = out_dir / f"ckpt_step_{step}.pt"
            torch.save(
                {"model": model.state_dict(), "step": step, "config": cfg.__dict__},
                ckpt_path,
            )
            print(f"    [ckpt] saved {ckpt_path}")

    history["wall_time_s"] = time.time() - t_start

    # Final save.
    final_path = out_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": cfg.n_steps, "config": cfg.__dict__}, final_path)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Total wall time {history['wall_time_s']:.1f}s. History saved to {out_dir / 'history.json'}.")
    return history
