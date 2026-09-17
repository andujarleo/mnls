"""3D observables and structural analysis.

Standard quantities (norm, FWHM, IPR, radial spectrum) plus the new piece
that 3D enables: identification of the Bravais symmetry that the crystalline
phase spontaneously selects from a Gaussian initial state.

The Bravais identification works by:
  1. Computing |Psi(k)|^2 as a 3D power spectrum.
  2. Finding the dominant non-zero radial wavenumber k*.
  3. Sampling the power on the spherical shell at k* and comparing the angular
     pattern to the signatures of the standard Bravais lattices:
        SC  — 6 peaks at +/- x, +/- y, +/- z (vertices of an octahedron)
        BCC — 12 peaks at (1,1,0)-type permutations
        FCC — 8 peaks at (1,1,1)-type permutations (vertices of a cube)
        HCP — hexagonal pattern at +/- z plus a 6-fold ring in-plane
  4. Reporting the best match score for each lattice and the winner.

This is the central new scientific content the 3D version exposes:
"What Bravais symmetry does the equation dynamically select?"
"""

from __future__ import annotations
import math
from typing import Optional

from .precision import get_backend, to_cpu


def norm_3d(psi, dx: float, xp=None) -> float:
    """L2 norm of psi: sqrt(integral |psi|^2 d^3x)."""
    if xp is None:
        xp = get_backend()
    return float(xp.sqrt(xp.sum(xp.abs(psi) ** 2) * dx ** 3))


def peak_density_3d(psi, xp=None) -> float:
    """Maximum value of |psi|^2 over the lattice."""
    if xp is None:
        xp = get_backend()
    return float(xp.max(xp.abs(psi) ** 2))


def fwhm_3d(psi, L: float, xp=None) -> float:
    """Estimated full-width-at-half-maximum of the density.

    Computes the radial profile around the centre of mass and finds the radius
    at which the density falls to half the peak. Approximate (assumes roughly
    spherical density); for elongated states use the per-axis variant.
    """
    if xp is None:
        xp = get_backend()
    rho = xp.abs(psi) ** 2
    N = psi.shape[0]
    dx = L / N

    # Centre of mass
    coords = xp.arange(N, dtype=xp.float32) - N / 2
    X, Y, Z = xp.meshgrid(coords, coords, coords, indexing="ij")
    total = float(xp.sum(rho))
    if total < 1e-20:
        return 0.0
    cx = float(xp.sum(rho * X) / total)
    cy = float(xp.sum(rho * Y) / total)
    cz = float(xp.sum(rho * Z) / total)

    # Radial coordinates from CoM
    R = xp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    rho_max = float(xp.max(rho))
    if rho_max < 1e-20:
        return 0.0

    # Radial binning
    r_max = N / 2
    n_bins = max(20, N // 4)
    bins = xp.linspace(0, r_max, n_bins + 1)
    R_flat = R.ravel()
    rho_flat = rho.ravel()
    # weighted histogram via digitize
    idx = xp.digitize(R_flat, bins) - 1
    radial_sum = xp.zeros(n_bins, dtype=xp.float64)
    radial_count = xp.zeros(n_bins, dtype=xp.float64)
    for b in range(n_bins):
        mask = idx == b
        radial_count[b] = float(xp.sum(mask))
        if radial_count[b] > 0:
            radial_sum[b] = float(xp.sum(rho_flat[mask]))
    radial_avg = xp.where(radial_count > 0, radial_sum / radial_count, 0.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    half = float(xp.max(radial_avg)) * 0.5
    if half <= 0:
        return 0.0
    # Find first crossing
    cpu_avg = to_cpu(radial_avg)
    cpu_centers = to_cpu(centers)
    crossing_idx = None
    for i in range(1, len(cpu_avg)):
        if cpu_avg[i - 1] >= half and cpu_avg[i] < half:
            crossing_idx = i
            break
    if crossing_idx is None:
        return float(cpu_centers[-1] * dx * 2)
    return float(cpu_centers[crossing_idx] * dx * 2)


def ipr_3d(psi, xp=None) -> float:
    """Inverse participation ratio: sum |psi|^4 / (sum |psi|^2)^2.
    High IPR = localised. Low IPR = delocalised.
    """
    if xp is None:
        xp = get_backend()
    rho = xp.abs(psi) ** 2
    s1 = float(xp.sum(rho))
    s2 = float(xp.sum(rho * rho))
    if s1 < 1e-20:
        return 0.0
    return s2 / (s1 * s1)


def radial_power_spectrum_3d(psi, L: float, xp=None) -> tuple:
    """Compute the spherically-averaged power spectrum |Psi(k)|^2.

    Returns (k_centers, P_radial) — both as numpy arrays on host.
    """
    if xp is None:
        xp = get_backend()
    N = psi.shape[0]
    psi_k = xp.fft.fftn(psi)
    P = xp.abs(psi_k) ** 2

    # k magnitudes
    kvec = xp.fft.fftfreq(N, d=L / N) * 2.0 * math.pi
    kx, ky, kz = xp.meshgrid(kvec, kvec, kvec, indexing="ij")
    k_mag = xp.sqrt(kx * kx + ky * ky + kz * kz)

    k_max = float(xp.max(k_mag))
    n_bins = N // 2
    bins = xp.linspace(0, k_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    P_radial = xp.zeros(n_bins, dtype=xp.float64)
    k_flat = k_mag.ravel()
    P_flat = P.ravel()
    idx = xp.digitize(k_flat, bins) - 1
    counts = xp.zeros(n_bins, dtype=xp.float64)
    for b in range(n_bins):
        mask = idx == b
        c = float(xp.sum(mask))
        counts[b] = c
        if c > 0:
            P_radial[b] = float(xp.sum(P_flat[mask]))
    P_radial = xp.where(counts > 0, P_radial / counts, 0.0)

    return to_cpu(centers), to_cpu(P_radial)


def crystallinity_3d(psi, L: float, k_low_cutoff: float = 0.5, xp=None) -> float:
    """Ratio of high-k power to total power.

    > 0.5 indicates clearly developed periodic structure (crystal regime).
    < 0.2 indicates dispersive / disorganised.
    """
    if xp is None:
        xp = get_backend()
    N = psi.shape[0]
    psi_k = xp.fft.fftn(psi)
    P = xp.abs(psi_k) ** 2

    kvec = xp.fft.fftfreq(N, d=L / N) * 2.0 * math.pi
    kx, ky, kz = xp.meshgrid(kvec, kvec, kvec, indexing="ij")
    k_mag = xp.sqrt(kx * kx + ky * ky + kz * kz)

    total = float(xp.sum(P))
    high = float(xp.sum(P[k_mag > k_low_cutoff]))
    if total < 1e-20:
        return 0.0
    return high / total


# -----------------------------------------------------------------------------
# Bravais symmetry detection
# -----------------------------------------------------------------------------


def bravais_signatures(k_star: float) -> dict:
    """Return the canonical k-space peak positions for each Bravais lattice.

    Each entry is a list of unit vectors (kx, ky, kz) at radius k_star.
    """
    sqrt2 = 1.0 / math.sqrt(2.0)
    sqrt3 = 1.0 / math.sqrt(3.0)
    return {
        "SC": [  # simple cubic — 6 peaks
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ],
        "BCC": [  # body-centred cubic — 12 peaks at (1,1,0)/sqrt(2)
            (sqrt2, sqrt2, 0), (sqrt2, -sqrt2, 0), (-sqrt2, sqrt2, 0), (-sqrt2, -sqrt2, 0),
            (sqrt2, 0, sqrt2), (sqrt2, 0, -sqrt2), (-sqrt2, 0, sqrt2), (-sqrt2, 0, -sqrt2),
            (0, sqrt2, sqrt2), (0, sqrt2, -sqrt2), (0, -sqrt2, sqrt2), (0, -sqrt2, -sqrt2),
        ],
        "FCC": [  # face-centred cubic — 8 peaks at (1,1,1)/sqrt(3)
            (sqrt3, sqrt3, sqrt3), (sqrt3, sqrt3, -sqrt3),
            (sqrt3, -sqrt3, sqrt3), (sqrt3, -sqrt3, -sqrt3),
            (-sqrt3, sqrt3, sqrt3), (-sqrt3, sqrt3, -sqrt3),
            (-sqrt3, -sqrt3, sqrt3), (-sqrt3, -sqrt3, -sqrt3),
        ],
        "HCP": [  # hexagonal — 6 in-plane + 2 axial
            (1.0, 0, 0), (0.5, math.sqrt(3)/2, 0), (-0.5, math.sqrt(3)/2, 0),
            (-1.0, 0, 0), (-0.5, -math.sqrt(3)/2, 0), (0.5, -math.sqrt(3)/2, 0),
            (0, 0, 1.0), (0, 0, -1.0),
        ],
    }


def detect_bravais_3d(psi, L: float, xp=None, k_min: float = 0.3) -> dict:
    """Identify the Bravais lattice that best matches the 3D power spectrum.

    Args:
        psi:    complex 3D wavefunction
        L:      physical box length
        k_min:  minimum non-zero k to consider as the dominant mode

    Returns:
        dict with:
            'k_star':         dominant non-zero radial wavenumber
            'wavelength':     2*pi/k_star
            'scores':         dict of {lattice_name: match_score in [0,1]}
            'best':           best-matching lattice name
            'crystallinity':  fraction of power above k_min
    """
    if xp is None:
        xp = get_backend()
    N = psi.shape[0]

    # Power spectrum
    psi_k = xp.fft.fftn(psi)
    P = xp.abs(psi_k) ** 2

    # k vectors
    kvec = xp.fft.fftfreq(N, d=L / N) * 2.0 * math.pi
    kx, ky, kz = xp.meshgrid(kvec, kvec, kvec, indexing="ij")
    k_mag = xp.sqrt(kx * kx + ky * ky + kz * kz)

    # Find dominant k* by radial spectrum
    k_centers, P_radial = radial_power_spectrum_3d(psi, L, xp=xp)
    # Mask out the k=0 region
    above = k_centers > k_min
    if not any(above):
        return {
            "k_star": 0.0,
            "wavelength": 0.0,
            "scores": {},
            "best": None,
            "crystallinity": 0.0,
            "note": "no non-zero mode above threshold",
        }
    P_above = P_radial.copy()
    P_above[~above] = 0.0
    idx_max = int(P_above.argmax())
    k_star = float(k_centers[idx_max])

    # Crystallinity
    cryst = crystallinity_3d(psi, L, k_low_cutoff=k_min, xp=xp)

    # Now sample P on a shell near k* and compare angular distribution to signatures
    # We use a narrow shell of width Δk = ~10% of k*
    dk_shell = max(k_star * 0.1, 2.0 * math.pi / L)
    shell_mask = (k_mag > k_star - dk_shell) & (k_mag < k_star + dk_shell)
    # Extract directions and powers on the shell
    P_shell = P[shell_mask]
    kx_shell = kx[shell_mask]
    ky_shell = ky[shell_mask]
    kz_shell = kz[shell_mask]
    k_shell_mag = k_mag[shell_mask]

    # Normalise direction vectors
    nx = kx_shell / k_shell_mag
    ny = ky_shell / k_shell_mag
    nz = kz_shell / k_shell_mag

    # For each lattice signature, compute the sum of P weighted by alignment
    # with any of its canonical directions. A perfect match would put all
    # shell power exactly on those directions; isotropic would put it everywhere.
    signatures = bravais_signatures(k_star)
    total_shell_power = float(xp.sum(P_shell))
    if total_shell_power < 1e-20:
        return {
            "k_star": k_star,
            "wavelength": 2 * math.pi / k_star if k_star > 0 else 0.0,
            "scores": {},
            "best": None,
            "crystallinity": cryst,
        }

    scores = {}
    width = 0.15  # angular width (cosine threshold)
    for name, dirs in signatures.items():
        # For each shell voxel, find the max alignment with any canonical dir
        max_align = xp.zeros_like(P_shell)
        for (dx_, dy_, dz_) in dirs:
            align = xp.abs(nx * dx_ + ny * dy_ + nz * dz_)
            max_align = xp.maximum(max_align, align)
        # Weight: 1 if aligned (cos > 1-width), 0 if not (sharp window)
        # We use a smooth window so the score is differentiable in noise:
        weight = xp.maximum(0.0, (max_align - (1.0 - width)) / width)
        score = float(xp.sum(P_shell * weight)) / total_shell_power
        scores[name] = score

    best = max(scores, key=scores.get) if scores else None
    return {
        "k_star": k_star,
        "wavelength": 2 * math.pi / k_star if k_star > 0 else 0.0,
        "scores": scores,
        "best": best,
        "best_score": scores[best] if best else 0.0,
        "crystallinity": cryst,
    }


# -----------------------------------------------------------------------------
# Convenience: build a default observables callback for run(...)
# -----------------------------------------------------------------------------


def make_default_observables(L: float):
    """Return a callable suitable for use as solver3d.run(observables_fn=...).

    Returns a dict with: t, step, norm, peak, fwhm, ipr, crystallinity.
    """
    dx = None  # filled on first call

    def fn(psi, ys, t, step_idx, xp):
        nonlocal dx
        if dx is None:
            dx = L / psi.shape[0]
        return {
            "t": t,
            "step": step_idx,
            "norm": norm_3d(psi, dx, xp=xp),
            "peak": peak_density_3d(psi, xp=xp),
            "fwhm": fwhm_3d(psi, L, xp=xp),
            "ipr": ipr_3d(psi, xp=xp),
            "crystallinity": crystallinity_3d(psi, L, xp=xp),
        }
    return fn
