"""Memory-NLS sequence layer in PyTorch.

This is the efficient implementation of the auxiliary-field memory structure
of the memory-augmented nonlinear Schrödinger equation. The linear recurrence
of the auxiliary fields is implemented via causal convolution in Fourier
space, giving O(N log N) cost in sequence length, matching the asymptotic
complexity of structured state space models (S4, S5, Mamba).

The equivalence with diagonal-state SSMs is detailed in
`../../interfaces/06-state-space-models.md`. The auxiliary-field equation

    y_j(t+1) = exp(-nu_j * dt) y_j(t) + (1 - exp(-nu_j * dt)) rho(t)

is a linear recurrence with closed-form kernel

    K_j(tau) = (1 - alpha_j) * alpha_j^tau ,    alpha_j = exp(-nu_j dt) ,

and y_j(t) = sum_{s <= t} K_j(t - s) rho(s) is a causal convolution that is
computed in Fourier space.

Extensions over the linear SSM baseline:
  - Cubic self-interaction Lambda * rho * x (P2 instantaneous part).
  - Multi-head memory with per-head (nu_j, lambda_j) spectra.
  - Optional FDT-locked stochastic forcing (P3).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_conv1d_fft(rho: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Causal 1D convolution via FFT.

    rho:    (batch, length, n_modes)
    kernel: (length, n_modes)

    Returns:
        (batch, length, n_modes), where out[:, t, j] = sum_{s<=t} kernel[t-s, j] * rho[:, s, j].

    The implementation zero-pads to length 2L, takes rFFT, multiplies, takes
    inverse rFFT, and crops to length L.
    """
    B, L, M = rho.shape
    n_fft = 2 * L
    # Cast to float32 for the FFT (cuFFT does not support bfloat16/float16).
    rho_f = torch.fft.rfft(rho.float(), n=n_fft, dim=1)         # (B, n_fft//2+1, M)
    kernel_f = torch.fft.rfft(kernel.float(), n=n_fft, dim=0)   # (n_fft//2+1, M)
    out_f = rho_f * kernel_f.unsqueeze(0)                       # broadcast over batch
    out = torch.fft.irfft(out_f, n=n_fft, dim=1)                # (B, n_fft, M)
    return out[:, :L, :].to(rho.dtype)                          # crop to causal output, restore dtype


class MemoryNLSLayer(nn.Module):
    """Memory-Nonlinear-Schrödinger sequence layer (efficient FFT version).

    The layer is a residual block that mixes information across the sequence
    dimension via an auxiliary-field memory potential plus a cubic self-
    interaction. The auxiliary-field update is computed in O(N log N) via
    causal convolution.

    Architecture
    ------------
    Input x of shape (B, L, D) is processed as follows:

      1. Input projection x -> x_in.
      2. Density rho = mean(x_in^2, dim=-1, keepdim=True), shape (B, L, 1).
      3. Per-head replication: rho is broadcast across n_heads memory modes
         (each head has its own (nu_j, lambda_j)).
      4. Causal convolution applied per head:
            y_j(t) = (1 - alpha_j) sum_{s<=t} alpha_j^(t-s) rho(s) .
         Implemented via FFT in O(L log L).
      5. Memory potential V_mem = sum_j lambda_j * y_j, shape (B, L, 1).
      6. Cubic interaction V_tot = Lambda * rho + V_mem.
      7. Multiplicative gate (the structural action of V_tot on x):
            update = V_tot * x_in .
      8. Optional FDT-locked noise.
      9. Output projection and residual add.

    Args
    ----
    d_model:                  Hidden dimension of input/output.
    n_heads:                  Number of memory modes (= number of nu_j).
    nonlinearity_strength:    Lambda; negative for attractive, positive for repulsive.
    memory_coupling_total:    Sigma_lambda; distributed across heads with fast-bias.
    nu_min:                   Slowest relaxation rate (smallest nu).
    nu_max:                   Fastest relaxation rate (largest nu).
    dt:                       Effective discrete time step (units of model time).
    fast_bias:                If > 1, fast modes get larger lambda. 1 = uniform.
                              Default 3.0 matches the 3:1 ratio used in the
                              physics-side crystalline regime.
    dissipation:              gamma_0 in the equation.
    fdt_temperature:          Bath temperature T. If both gamma_0 > 0 and T > 0,
                              FDT-locked noise is added at every step.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        nonlinearity_strength: float = -0.5,
        memory_coupling_total: float = 0.3,
        nu_min: float = 0.5,
        nu_max: float = 10.0,
        dt: float = 0.01,
        fast_bias: float = 3.0,
        dissipation: float = 0.0,
        fdt_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.Lambda = nonlinearity_strength
        self.dt = dt
        self.gamma_0 = dissipation
        self.T_bath = fdt_temperature

        # Geometric spectrum of relaxation rates from nu_max to nu_min.
        if n_heads == 1:
            nus = [nu_max]
        else:
            nus = [
                nu_max * (nu_min / nu_max) ** (j / (n_heads - 1))
                for j in range(n_heads)
            ]
        nus_t = torch.tensor(nus, dtype=torch.float32)
        self.register_buffer("nus", nus_t)
        alphas = torch.exp(-nus_t * dt)
        self.register_buffer("alphas", alphas)

        # Memory coupling distribution: fast-biased.
        if n_heads == 1:
            lambdas = [memory_coupling_total]
        else:
            raw = [fast_bias ** (1 - j / (n_heads - 1)) for j in range(n_heads)]
            total = sum(raw)
            lambdas = [r / total * memory_coupling_total for r in raw]
        self.register_buffer(
            "lambdas", torch.tensor(lambdas, dtype=torch.float32)
        )

        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _build_kernel(self, L: int, device, dtype) -> torch.Tensor:
        """Build the causal-convolution kernel of length L.

        K[t, j] = (1 - alpha_j) * alpha_j^t for t in [0, L-1].
        """
        alphas = self.alphas.to(device=device, dtype=dtype)
        t = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)  # (L, 1)
        return (1.0 - alphas) * alphas.pow(t)                          # (L, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, length, d_model) input sequence.

        Returns:
            (batch, length, d_model) output sequence, residual-updated.
        """
        B, L, D = x.shape
        assert D == self.d_model

        x_in = self.input_proj(x)

        # Density rho per position (scalar per position).
        rho = (x_in * x_in).mean(dim=-1, keepdim=True)  # (B, L, 1)

        # Replicate rho across the heads for the convolution.
        rho_heads = rho.expand(B, L, self.n_heads).contiguous()

        # Build the per-head kernel and apply causal FFT convolution.
        K = self._build_kernel(L, x.device, torch.float32)      # (L, n_heads), fp32
        y = _causal_conv1d_fft(rho_heads, K)                    # (B, L, n_heads)

        # Memory potential and total potential.
        V_mem = (y * self.lambdas.to(y.dtype)).sum(dim=-1, keepdim=True)  # (B, L, 1)
        V_tot = self.Lambda * rho + V_mem                                  # (B, L, 1)

        update = V_tot * x_in                                              # (B, L, D)

        if self.gamma_0 > 0 and self.T_bath > 0:
            noise_scale = math.sqrt(2.0 * self.gamma_0 * self.T_bath * self.dt)
            update = update + noise_scale * torch.randn_like(update)

        return x + self.output_proj(update)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"Lambda={self.Lambda:+.3f}, "
            f"Sigma_lambda={float(self.lambdas.sum()):.3f}, "
            f"nu_range=[{float(self.nus.min()):.2f}, {float(self.nus.max()):.2f}]"
        )
