"""3D Strang split-step solver for the memory-augmented NLS equation.

Equation (scalar form; spinor extension noted below):

    i hbar d_t Psi = [ -hbar^2/(2m) D^2
                       + V_ext
                       + Lambda |Psi|^2
                       + V_mem
                       + alpha (-Delta)^(sigma/2)
                       - i Gamma ] Psi  +  eta

with V_mem = sum_j lambda_j y_j  and  d_t y_j = nu_j (rho_eff - y_j).

Strang split-step per time step dt:
    1.  V/2:    Psi <- exp(-i V_tot dt/2) Psi
    2.  K:      Psi <- IFFT[ exp(-i H_kin(k) dt) FFT[Psi] ]
    3.  V/2:    Psi <- exp(-i V_tot dt/2) Psi
    4.  OU:     y_j <- exp(-nu_j dt) y_j + (1 - exp(-nu_j dt)) rho_eff
    5.  Noise:  Psi <- Psi + sqrt(2 gamma_0 T dt) * xi  (xi: complex Gaussian)

Conservation:
    Unitary regime (Gamma=0, eta=0, gamma_s=0, memory passive):
        norm preserved to ~1e-7 (fp32) or ~1e-13 (fp64) over 1000s of steps.

3D supercriticality:
    The 3D cubic NLS is L^2-supercritical: attractive nonlinearity drives
    finite-time blow-up for any sufficiently concentrated initial state, with
    no critical-norm threshold. Memory regularisation in 3D is therefore a
    stronger statement than in 2D, where collapse is L^2-critical.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

from .precision import get_backend, get_default_precision, Precision
from .kernels import MemoryConfig, build_spatial_kernel_fft_3d, apply_spatial_kernel


@dataclass
class SolverConfig3D:
    """All parameters for a 3D simulation."""

    # Lattice
    N: int = 128                      # grid size per dimension
    L: float = 20.0                   # physical box length
    dt: float = 0.005                 # time step
    n_steps: int = 2000               # number of steps

    # Physics
    hbar: float = 1.0
    m: float = 1.0
    Lambda: float = -8.0              # local nonlinearity
    sigma: float = 2.0                # fractional exponent (2.0 = standard Laplacian)
    alpha_frac: float = 0.0           # fractional dispersion strength
    gamma_0: float = 0.0              # linear dissipation
    T: float = 0.0                    # FDT temperature

    # Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # External potential / drive
    V_ext_amp: float = 0.0
    V_drive_amp: float = 0.0
    V_drive_omega: float = 0.0

    # Initial state
    init_sigma: float = 1.2
    init_k0: tuple = (1.0, 0.5, 0.0)

    # Precision
    precision: str = "fp32"           # "fp64" | "fp32" | "fp16"

    # Adaptive dt for strong nonlinearity
    auto_halve_dt: bool = True

    # Random seed
    seed: int = 42

    # Sampling / snapshots
    sample_every: int = 50            # store observables every N steps
    snapshot_every: int = 0           # store full Psi every N steps (0 = never)

    def __post_init__(self):
        if self.auto_halve_dt and abs(self.Lambda) >= 4 and self.dt > 0.0025:
            self.dt = 0.0025


def build_initial_state(cfg: SolverConfig3D, xp, prec: Precision):
    """Single-Gaussian initial state with optional momentum boost."""
    N = cfg.N
    L = cfg.L
    dx = L / N
    x = (xp.arange(N, dtype=prec.real) - N / 2) * dx
    X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    s = cfg.init_sigma
    envelope = xp.exp(-r2 / (2.0 * s * s)).astype(prec.real)

    kx, ky, kz = cfg.init_k0
    phase = (kx * X + ky * Y + kz * Z).astype(prec.real)
    psi = (envelope * xp.exp(1j * phase)).astype(prec.complex)

    # Normalise: integral |psi|^2 d^3x = 1
    norm = xp.sqrt(xp.sum(xp.abs(psi) ** 2) * dx ** 3)
    psi /= norm
    return psi


def build_kinetic_propagator(cfg: SolverConfig3D, xp, prec: Precision):
    """Pre-compute the k-space half-step propagator exp(-i H_kin(k) dt)."""
    N = cfg.N
    L = cfg.L
    dk = 2.0 * math.pi / L
    kvec = xp.fft.fftfreq(N, d=L / N) * 2.0 * math.pi
    kx, ky, kz = xp.meshgrid(kvec, kvec, kvec, indexing="ij")
    k2 = (kx * kx + ky * ky + kz * kz).astype(prec.real)
    k_mag = xp.sqrt(k2)

    # Standard kinetic
    H_kin = (cfg.hbar ** 2 / (2.0 * cfg.m)) * k2

    # Optional fractional dispersion
    if cfg.alpha_frac != 0.0 and cfg.sigma != 2.0:
        H_kin = H_kin + cfg.alpha_frac * (k_mag ** cfg.sigma)

    # Dissipation (homogeneous)
    H_complex = H_kin.astype(prec.complex) - 1j * cfg.gamma_0

    # Full-step (NOT half-step) kinetic propagator since Strang sandwiches it
    U_k = xp.exp(-1j * H_complex * cfg.dt).astype(prec.complex)
    return U_k


def build_external_potential(cfg: SolverConfig3D, xp, prec: Precision):
    """Build the static external potential V_ext(x), if any."""
    if cfg.V_ext_amp == 0.0:
        return None
    N = cfg.N
    L = cfg.L
    dx = L / N
    x = (xp.arange(N, dtype=prec.real) - N / 2) * dx
    X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
    r2 = X * X + Y * Y + Z * Z
    return (cfg.V_ext_amp * xp.exp(-r2 / 8.0)).astype(prec.real)


def step(psi, ys, U_k, V_ext, G_k, cfg: SolverConfig3D, t: float, xp, prec: Precision, rng):
    """Single Strang split-step (in-place where possible).

    Returns the updated (psi, ys, t+dt).
    """
    dt = cfg.dt
    dx = cfg.L / cfg.N

    # --- 1. Density and memory potential
    rho = (psi.conj() * psi).real.astype(prec.real)

    if cfg.memory.spatial == "local" or G_k is None:
        rho_eff = rho
    else:
        rho_eff = apply_spatial_kernel(rho, G_k, xp=xp).astype(prec.real)

    V_mem = xp.zeros_like(rho)
    for j, lam in enumerate(cfg.memory.lambdas):
        V_mem = V_mem + lam * ys[j]

    # External drive (if any)
    V_drive = 0.0
    if cfg.V_drive_amp != 0.0 and V_ext is not None:
        V_drive = cfg.V_drive_amp * math.cos(cfg.V_drive_omega * t) * V_ext

    V_tot = cfg.Lambda * rho + V_mem
    if V_ext is not None and cfg.V_drive_amp == 0.0:
        V_tot = V_tot + V_ext
    if cfg.V_drive_amp != 0.0 and V_ext is not None:
        V_tot = V_tot + V_drive

    # --- 2. V/2 step
    phase = xp.exp(-1j * (V_tot.astype(prec.complex)) * (dt * 0.5))
    psi = psi * phase

    # --- 3. Kinetic step in k-space
    psi_k = xp.fft.fftn(psi)
    psi_k = psi_k * U_k
    psi = xp.fft.ifftn(psi_k).astype(prec.complex)

    # --- 4. V/2 step (use updated rho for second half)
    rho2 = (psi.conj() * psi).real.astype(prec.real)
    if cfg.memory.spatial == "local" or G_k is None:
        rho2_eff = rho2
    else:
        rho2_eff = apply_spatial_kernel(rho2, G_k, xp=xp).astype(prec.real)

    V_tot2 = cfg.Lambda * rho2 + V_mem
    if V_ext is not None and cfg.V_drive_amp == 0.0:
        V_tot2 = V_tot2 + V_ext
    if cfg.V_drive_amp != 0.0 and V_ext is not None:
        V_tot2 = V_tot2 + cfg.V_drive_amp * math.cos(cfg.V_drive_omega * (t + dt)) * V_ext

    phase2 = xp.exp(-1j * (V_tot2.astype(prec.complex)) * (dt * 0.5))
    psi = psi * phase2

    # --- 5. Auxiliary fields (OU exact update)
    for j, nu in enumerate(cfg.memory.nus):
        decay = math.exp(-nu * dt)
        ys[j] = decay * ys[j] + (1.0 - decay) * rho2_eff

    # --- 6. Stochastic forcing (FDT-locked)
    if cfg.gamma_0 > 0.0 and cfg.T > 0.0:
        amp = math.sqrt(2.0 * cfg.gamma_0 * cfg.T * dt)
        xi = (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape)).astype(prec.complex)
        psi = psi + amp * xi / math.sqrt(2.0)

    return psi, ys, t + dt


def run(cfg: SolverConfig3D, observables_fn: Optional[Callable] = None, verbose: bool = True) -> dict:
    """Run a 3D simulation.

    Args:
        cfg:            full configuration
        observables_fn: optional callable taking (psi, ys, t, step_idx, xp) and
                        returning a dict of scalar observables. Called every
                        cfg.sample_every steps.
        verbose:        print progress to stdout

    Returns:
        dict with keys:
            'psi_final':    final wavefunction (host array)
            'ys_final':     final auxiliary fields (host array)
            'samples':      list of dicts, one per sample
            'snapshots':    list of (t, psi_host) tuples if snapshot_every > 0
            'config':       config used (as dict)
            'wall_time':    elapsed seconds
    """
    xp = get_backend()
    prec = Precision(cfg.precision)

    # Seed
    if xp.__name__ == "cupy":
        rng = xp.random.default_rng(cfg.seed)
    else:
        rng = xp.random.default_rng(cfg.seed)

    # Initial state
    psi = build_initial_state(cfg, xp, prec)

    # Auxiliary fields
    ys = [xp.zeros_like(psi.real).astype(prec.real) for _ in range(cfg.memory.n_modes)]

    # Pre-compute propagator
    U_k = build_kinetic_propagator(cfg, xp, prec)

    # External potential (used for drive too)
    V_ext = build_external_potential(cfg, xp, prec)
    if V_ext is None and cfg.V_drive_amp != 0.0:
        # Build a Gaussian drive envelope at the centre
        N = cfg.N
        L = cfg.L
        dx = L / N
        x = (xp.arange(N, dtype=prec.real) - N / 2) * dx
        X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
        r2 = X * X + Y * Y + Z * Z
        V_ext = xp.exp(-r2 / 8.0).astype(prec.real)

    # Spatial memory kernel (if non-local)
    if cfg.memory.spatial != "local":
        G_k = build_spatial_kernel_fft_3d(
            cfg.N, cfg.L, cfg.memory.spatial, cfg.memory.scale, xp=xp, dtype=prec.real
        )
    else:
        G_k = None

    # Storage
    samples = []
    snapshots = []
    t = 0.0
    t0 = time.time()

    for step_idx in range(cfg.n_steps):
        psi, ys, t = step(psi, ys, U_k, V_ext, G_k, cfg, t, xp, prec, rng)

        # Sample observables
        if observables_fn is not None and (step_idx % cfg.sample_every == 0):
            sample = observables_fn(psi, ys, t, step_idx, xp)
            samples.append(sample)
            if verbose and step_idx % (cfg.sample_every * 10) == 0:
                msg = f"  step {step_idx:6d}/{cfg.n_steps} t={t:7.3f}"
                if "norm" in sample:
                    msg += f" norm={sample['norm']:.6f}"
                if "peak" in sample:
                    msg += f" peak={sample['peak']:.4f}"
                if "fwhm" in sample:
                    msg += f" fwhm={sample['fwhm']:.2f}"
                print(msg)

        # Snapshot full Psi
        if cfg.snapshot_every > 0 and (step_idx % cfg.snapshot_every == 0):
            snapshots.append((t, _to_host(psi, xp)))

    wall = time.time() - t0
    if verbose:
        print(f"Completed in {wall:.1f}s ({cfg.n_steps / wall:.1f} steps/s)")

    return {
        "psi_final": _to_host(psi, xp),
        "ys_final": [_to_host(y, xp) for y in ys],
        "samples": samples,
        "snapshots": snapshots,
        "config": _config_to_dict(cfg),
        "wall_time": wall,
    }


def _to_host(arr, xp):
    if xp.__name__ == "cupy":
        return arr.get()
    return arr


def _config_to_dict(cfg: SolverConfig3D) -> dict:
    d = asdict(cfg)
    d["memory"] = asdict(cfg.memory)
    return d
