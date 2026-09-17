"""Memory-NLS language model.

A small autoregressive language model that uses MemoryNLSLayer as the
sequence-mixing primitive, in the place where an attention layer or a Mamba
block would otherwise appear. The architecture is intentionally a direct
translation of standard pre-norm transformer-style stacks, so that
comparisons against attention-based baselines (Transformer) and structured-
state-space baselines (Mamba) hold the rest of the architecture constant.

A block consists of:

    pre-norm -> MemoryNLSLayer (sequence mixing) -> residual
    pre-norm -> Feed-forward MLP -> residual

stacked n_layers deep, between a token embedding (input) and an output head
(linear -> vocab logits, tied weights with the embedding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import MemoryNLSLayer


@dataclass
class MemoryNLSConfig:
    vocab_size: int = 256                  # default for byte-level / char-level
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4                       # memory modes per layer
    ffn_mult: int = 4                      # hidden dim of FFN = ffn_mult * d_model
    max_seq_len: int = 1024                # used for positional encoding (optional)
    use_positional: bool = False           # SSMs typically do not need it
    dropout: float = 0.0

    # Memory-NLS specific
    nonlinearity_strength: float = -0.5
    memory_coupling_total: float = 0.3
    nu_min: float = 0.5
    nu_max: float = 10.0
    dt: float = 0.01
    fast_bias: float = 3.0
    dissipation: float = 0.0
    fdt_temperature: float = 0.0


class FeedForward(nn.Module):
    """Standard FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MemoryNLSBlock(nn.Module):
    """Pre-norm block: MemNLS mixing then FFN."""

    def __init__(self, cfg: MemoryNLSConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.mixer = MemoryNLSLayer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            nonlinearity_strength=cfg.nonlinearity_strength,
            memory_coupling_total=cfg.memory_coupling_total,
            nu_min=cfg.nu_min,
            nu_max=cfg.nu_max,
            dt=cfg.dt,
            fast_bias=cfg.fast_bias,
            dissipation=cfg.dissipation,
            fdt_temperature=cfg.fdt_temperature,
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_model * cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequence mixing (residual is inside MemoryNLSLayer)
        x = self.mixer(self.norm1(x))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class MemoryNLSLanguageModel(nn.Module):
    """Autoregressive language model with MemoryNLSLayer as sequence mixer."""

    def __init__(self, cfg: MemoryNLSConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.use_positional:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:
            self.pos_emb = None
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([MemoryNLSBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)

        # Tied weights with token embedding.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: (B, L) integer token ids.
            targets:   optional (B, L) integer targets for loss computation.

        Returns:
            logits: (B, L, vocab_size)
            loss:   scalar tensor or None
        """
        B, L = input_ids.shape
        assert L <= self.cfg.max_seq_len, f"sequence length {L} exceeds max {self.cfg.max_seq_len}"

        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)                       # (B, L, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            input_ids:     (B, L0) prompt token ids.
            max_new_tokens: number of tokens to generate.
            temperature:   sampling temperature; 0 => greedy.
            top_k:         restrict sampling to top-k tokens.
            top_p:         nucleus sampling threshold (cumulative probability).
            eos_token_id:  stop early if all sequences emit this token.

        Returns:
            (B, L0 + max_new_tokens) tensor of generated ids.
        """
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            # Crop context to model's max sequence length.
            context = out[:, -self.cfg.max_seq_len:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-9)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                threshold = v[..., -1, None]
                logits = torch.where(logits < threshold, torch.full_like(logits, -float("inf")), logits)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 0] = False  # always keep top token
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(
                    -1, sorted_indices, sorted_logits
                )

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return out
