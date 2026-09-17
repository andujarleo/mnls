"""Self-contained Memory-NLS language model for HuggingFace.

Loads the published checkpoint without requiring the full mnsm repository.

Usage:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import sys, importlib.util

    # Download files
    config_path = hf_hub_download("qvr0/mnsm-memnls-70m-enwik8", "config.json")
    weights_path = hf_hub_download("qvr0/mnsm-memnls-70m-enwik8", "model.safetensors")
    modeling_path = hf_hub_download("qvr0/mnsm-memnls-70m-enwik8", "modeling.py")

    # Import modeling module
    spec = importlib.util.spec_from_file_location("modeling", modeling_path)
    modeling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling)

    # Build model
    import json
    with open(config_path) as f:
        config = json.load(f)
    model = modeling.MemoryNLSLanguageModel(modeling.MemoryNLSConfig(**config))

    # Load weights
    state = load_file(weights_path)
    model.load_state_dict(state)
    model.eval()

    # Generate
    text = "The history of "
    input_ids = torch.tensor([list(text.encode("utf-8"))])
    out = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=40)
    print(bytes(out[0].tolist()).decode("utf-8", errors="replace"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_conv1d_fft(rho: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal 1D convolution via FFT — O(N log N) per layer in sequence length.

    rho:    (batch, length, n_modes)
    kernel: (length, n_modes)
    Returns: (batch, length, n_modes)
    """
    B, L, M = rho.shape
    n_fft = 2 * L
    rho_f = torch.fft.rfft(rho.float(), n=n_fft, dim=1)
    kernel_f = torch.fft.rfft(kernel.float(), n=n_fft, dim=0)
    out = torch.fft.irfft(rho_f * kernel_f.unsqueeze(0), n=n_fft, dim=1)
    return out[:, :L, :].to(rho.dtype)


class MemoryNLSLayer(nn.Module):
    """Memory-NLS sequence-mixing layer (FFT-convolution implementation).

    Auxiliary-field memory equation:
        y_j(t+1) = exp(-nu_j dt) y_j(t) + (1 - exp(-nu_j dt)) rho(t)
    causally convolved with kernel K_j(tau) = (1-alpha_j) alpha_j^tau.
    """

    def __init__(self, d_model, n_heads=4, nonlinearity_strength=-0.5,
                 memory_coupling_total=0.3, nu_min=0.5, nu_max=10.0,
                 dt=0.01, fast_bias=3.0, dissipation=0.0, fdt_temperature=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.Lambda = nonlinearity_strength
        self.dt = dt
        self.gamma_0 = dissipation
        self.T_bath = fdt_temperature

        if n_heads == 1:
            nus = [nu_max]
        else:
            nus = [nu_max * (nu_min / nu_max) ** (j / (n_heads - 1)) for j in range(n_heads)]
        self.register_buffer("nus", torch.tensor(nus, dtype=torch.float32))
        self.register_buffer("alphas", torch.exp(-self.nus * dt))

        if n_heads == 1:
            lambdas = [memory_coupling_total]
        else:
            raw = [fast_bias ** (1 - j / (n_heads - 1)) for j in range(n_heads)]
            total = sum(raw)
            lambdas = [r / total * memory_coupling_total for r in raw]
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))

        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _build_kernel(self, L, device, dtype):
        alphas = self.alphas.to(device=device, dtype=dtype)
        t = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)
        return (1.0 - alphas) * alphas.pow(t)

    def forward(self, x):
        B, L, D = x.shape
        x_in = self.input_proj(x)
        rho = (x_in * x_in).mean(dim=-1, keepdim=True)
        rho_heads = rho.expand(B, L, self.n_heads).contiguous()
        K = self._build_kernel(L, x.device, torch.float32)
        y = _causal_conv1d_fft(rho_heads, K)
        V_mem = (y * self.lambdas.to(y.dtype)).sum(dim=-1, keepdim=True)
        V_tot = self.Lambda * rho + V_mem
        update = V_tot * x_in
        if self.gamma_0 > 0 and self.T_bath > 0:
            noise_scale = math.sqrt(2.0 * self.gamma_0 * self.T_bath * self.dt)
            update = update + noise_scale * torch.randn_like(update)
        return x + self.output_proj(update)


@dataclass
class MemoryNLSConfig:
    vocab_size: int = 256
    d_model: int = 768
    n_layers: int = 10
    n_heads: int = 12
    ffn_mult: int = 5
    max_seq_len: int = 1024
    use_positional: bool = False
    dropout: float = 0.0
    nonlinearity_strength: float = -0.5
    memory_coupling_total: float = 0.3
    nu_min: float = 0.5
    nu_max: float = 10.0
    dt: float = 0.05
    fast_bias: float = 3.0
    dissipation: float = 0.0
    fdt_temperature: float = 0.0


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class MemoryNLSBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.mixer = MemoryNLSLayer(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            nonlinearity_strength=cfg.nonlinearity_strength,
            memory_coupling_total=cfg.memory_coupling_total,
            nu_min=cfg.nu_min, nu_max=cfg.nu_max, dt=cfg.dt,
            fast_bias=cfg.fast_bias, dissipation=cfg.dissipation,
            fdt_temperature=cfg.fdt_temperature)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_model * cfg.ffn_mult, cfg.dropout)

    def forward(self, x):
        x = self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MemoryNLSLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.use_positional else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MemoryNLSBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
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
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out
