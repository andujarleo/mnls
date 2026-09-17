"""Physics solver for the memory-augmented nonlinear Schrödinger field equation.

This package implements the equation derived in `../../equation/01-derivation.md`
on a periodic spatial lattice using a Strang split-step integrator. The
auxiliary-field Markovian embedding of the integral memory potential
(`../../equation/02-markovian-embedding.md`) makes the solver local in time
and computationally tractable.

Modules
-------
solver_3d    : The 3D Strang split-step solver.
kernels      : Memory kernel definitions (local and spatially non-local).
observables  : Norm, peak, FWHM, IPR, radial spectrum, Bravais detection.
precision    : Backend (cupy/numpy) and precision (fp64/32/16) abstraction.
sanity       : Conservation tests (norm, dissipation, FDT thermalization).

The 2D version of the solver is derivable by setting the third spatial
dimension to size 1 in the 3D solver; a standalone 2D module is not
required for the present work.
"""

from .solver_3d import SolverConfig3D, run, step, build_initial_state
from .kernels import MemoryConfig, build_spatial_kernel_fft_3d, apply_spatial_kernel
from .observables import (
    norm_3d,
    peak_density_3d,
    fwhm_3d,
    ipr_3d,
    radial_power_spectrum_3d,
    crystallinity_3d,
    detect_bravais_3d,
    make_default_observables,
)
from .precision import get_backend, Precision, is_gpu, to_cpu, vram_estimate

__all__ = [
    "SolverConfig3D",
    "run",
    "step",
    "build_initial_state",
    "MemoryConfig",
    "build_spatial_kernel_fft_3d",
    "apply_spatial_kernel",
    "norm_3d",
    "peak_density_3d",
    "fwhm_3d",
    "ipr_3d",
    "radial_power_spectrum_3d",
    "crystallinity_3d",
    "detect_bravais_3d",
    "make_default_observables",
    "get_backend",
    "Precision",
    "is_gpu",
    "to_cpu",
    "vram_estimate",
]
