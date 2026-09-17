"""Memory kernels in 3D.

Implements the kernel structures used in the R3 asymmetry test, extended to 3D.

Temporal structure (always multi-exponential, Markovian-embedded):
    U(tau, x, x') = sum_j lambda_j * nu_j * exp(-nu_j tau) * G(x - x')

where G is the spatial part. The Markovian embedding via auxiliary fields y_j is

    d_t y_j = nu_j * (rho_eff - y_j)
    V_mem = sum_j lambda_j y_j

In the local case (default), rho_eff = rho.
In non-local cases, rho_eff = G * rho (convolution; computed via FFT).

The R3 finding from 2D — temporal non-locality regulates collapse, spatial
non-locality destroys it — is exactly what this module makes testable in 3D.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from .precision import get_backend


@dataclass
class MemoryConfig:
    """Configuration of a multi-mode memory kernel.

    Attributes:
        nus:     list of inverse relaxation times nu_j (one per mode)
        lambdas: list of coupling strengths lambda_j
        spatial: "local" | "exp" | "gauss" | "iso"
        scale:   spatial scale (only used if non-local; lattice units)
    """
    nus: list[float] = field(default_factory=lambda: [10.0])
    lambdas: list[float] = field(default_factory=lambda: [0.3])
    spatial: str = "local"
    scale: float = 1.0

    @property
    def n_modes(self) -> int:
        return len(self.nus)

    def __post_init__(self):
        if len(self.nus) != len(self.lambdas):
            raise ValueError("nus and lambdas must have the same length")
        if self.spatial not in ("local", "exp", "gauss", "iso"):
            raise ValueError(f"unknown spatial kernel: {self.spatial}")


def build_spatial_kernel_fft_3d(N: int, L: float, kind: str, scale: float, xp=None, dtype=None):
    """Build the FFT of the spatial smoothing kernel G(r) on a 3D lattice.

    Returns G_k of shape (N, N, N), suitable for multiplication with rho_k.

    Args:
        N:      lattice size
        L:      physical box length
        kind:   "local"  — returns ones (no smoothing); never actually used
                "exp"    — G(r) ~ exp(-|r| / scale)
                "gauss"  — G(r) ~ exp(-r^2 / (2 scale^2))
                "iso"    — same as gauss; alias for clarity
        scale:  spatial scale in lattice units (lat units = L/N)

    The kernel is normalised so that the integral G d^3x = 1 in the continuum
    limit; the discrete sum is rescaled accordingly.
    """
    if xp is None:
        xp = get_backend()
    if dtype is None:
        dtype = xp.float32

    if kind == "local":
        return xp.ones((N, N, N), dtype=dtype)

    dx = L / N
    coords = xp.fft.fftfreq(N, d=1.0 / N) * dx  # physical coordinates in shifted form
    # rebuild as standard centered grid
    x = xp.arange(N, dtype=dtype) * dx - L / 2
    X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
    r2 = X * X + Y * Y + Z * Z

    if kind == "exp":
        r = xp.sqrt(r2 + (dx * 0.5) ** 2)  # regularised at origin
        G = xp.exp(-r / scale).astype(dtype)
    elif kind in ("gauss", "iso"):
        G = xp.exp(-r2 / (2.0 * scale * scale)).astype(dtype)
    else:
        raise ValueError(kind)

    # Normalise: integral G d^3x = 1
    G /= G.sum() * dx ** 3
    G *= dx ** 3  # discrete sum convention

    # FFT (no shift; cuFFT/numpy.fft uses corner-origin convention)
    G_shifted = xp.fft.ifftshift(G)
    G_k = xp.fft.fftn(G_shifted)
    return xp.real(G_k).astype(dtype)


def apply_spatial_kernel(rho, G_k, xp=None):
    """Convolve rho with the spatial kernel G via FFT.

    rho:  real array of shape (N, N, N)
    G_k:  pre-built FFT of G (real-valued, see build_spatial_kernel_fft_3d)
    """
    if xp is None:
        xp = get_backend()
    rho_k = xp.fft.fftn(rho.astype(xp.complex64 if rho.dtype.kind == "f" else rho.dtype))
    out_k = rho_k * G_k
    out = xp.real(xp.fft.ifftn(out_k))
    return out
