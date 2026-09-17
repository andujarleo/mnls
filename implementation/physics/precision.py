"""Backend and precision abstraction.

Provides a thin wrapper over numpy/cupy with explicit precision selection.
The solver and observables import from here to stay backend-agnostic.

Precision modes:
    fp64   — double precision (slowest, most accurate; norm conservation ~1e-13)
    fp32   — single precision (default for 3D; norm conservation ~1e-7)
    fp16   — half precision (FFT halved cost, halved memory; norm ~1e-3,
             use only for surveys / phase-diagram exploration)

Mixed-precision policy (default in 3D):
    Storage:         fp32 complex
    FFT:             fp32 (cuFFT native)
    Pointwise nonlin: fp32
    Auxiliary fields: fp32
    Stochastic noise: fp32
    Optional fp16 fast path: enabled with precision="fp16" — recommended for
    phase-diagram sweeps where the regime is the question, not the precision.

Tensor Core / FP8 path (Ada Lovelace 4th-gen Tensor Cores) is exposed in the
analysis layer (see sim/surrogate.py), not in the core solver. The solver
remains in standard precisions because cuFFT does not accept FP8 input.
"""

from __future__ import annotations
import os
import warnings


_BACKEND = None
_PRECISION = None


def get_backend(prefer_gpu: bool = True):
    """Return numpy or cupy as the active array module."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    if prefer_gpu and os.environ.get("MEMNLS_FORCE_CPU", "0") != "1":
        try:
            import cupy as cp
            _BACKEND = cp
            return cp
        except ImportError:
            warnings.warn("cupy not available; falling back to numpy. GPU acceleration disabled.")

    import numpy as np
    _BACKEND = np
    return np


def is_gpu() -> bool:
    """True if running on cupy."""
    xp = get_backend()
    return xp.__name__ == "cupy"


def to_cpu(arr):
    """Move array to host memory regardless of backend."""
    if hasattr(arr, "get"):  # cupy array
        return arr.get()
    return arr


class Precision:
    """Holds real and complex dtypes for the current precision mode.

    Use as:
        prec = Precision("fp32")
        psi = xp.zeros((N, N, N), dtype=prec.complex)
    """

    def __init__(self, name: str = "fp32"):
        xp = get_backend()
        self.name = name.lower()

        if self.name == "fp64":
            self.real = xp.float64
            self.complex = xp.complex128
            self.norm_tol = 1e-12
        elif self.name == "fp32":
            self.real = xp.float32
            self.complex = xp.complex64
            self.norm_tol = 1e-6
        elif self.name == "fp16":
            # cupy/numpy may not have native complex32; store as complex64
            # and cast at FFT boundary.
            self.real = xp.float16
            self.complex = xp.complex64  # transit type
            self._fft_real = xp.float16
            self.norm_tol = 1e-3
        else:
            raise ValueError(f"unknown precision: {name}")

    def __repr__(self):
        return f"Precision(name={self.name!r}, real={self.real.__name__}, complex={self.complex.__name__})"


def set_default_precision(name: str):
    """Set the global default precision."""
    global _PRECISION
    _PRECISION = Precision(name)


def get_default_precision() -> Precision:
    """Return the global default precision (fp32 if not set)."""
    global _PRECISION
    if _PRECISION is None:
        _PRECISION = Precision("fp32")
    return _PRECISION


def vram_estimate(N: int, n_aux: int = 2, d_int: int = 2, precision: str = "fp32") -> dict:
    """Estimate VRAM footprint of a 3D run.

    Args:
        N:          lattice size per dimension (volume = N^3)
        n_aux:      number of memory auxiliary fields y_j
        d_int:      internal (spinor) dimension; 1 = scalar, 2 = spinor
        precision:  "fp64", "fp32", or "fp16"

    Returns:
        dict with keys 'per_field_MB', 'total_MB', 'safe_margin_MB'
    """
    bytes_per_complex = {"fp64": 16, "fp32": 8, "fp16": 4}[precision]
    voxels = N ** 3
    per_field = voxels * bytes_per_complex / (1024 ** 2)  # MB

    # Field inventory:
    #  - Psi spinor:                 d_int
    #  - auxiliary fields y_j:       n_aux
    #  - propagator U(k) precomputed: 1  (real-valued, but store complex form)
    #  - rho (real-valued):          0.5  (only half complex)
    #  - V_mem (real):               0.5
    #  - FFT scratch (cuFFT plan):   ~1
    #  - intermediate buffers:       ~1
    fields_complex = d_int + n_aux + 1 + 1 + 1
    fields_real_half = 1.0  # rho, V_mem (real but allocated as complex for simplicity)
    total = per_field * (fields_complex + fields_real_half)

    return {
        "per_field_MB": round(per_field, 2),
        "total_estimate_MB": round(total, 2),
        "safe_for_8GB_card": total < 6000,
        "precision": precision,
        "lattice": f"{N}^3 = {voxels:,} voxels",
    }
