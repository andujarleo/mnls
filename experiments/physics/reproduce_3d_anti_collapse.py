"""Reproduce the 3D anti-collapse result documented in
../../results/04-anti-collapse-3d.md.

Sweeps Lambda in {-2, -4, -6, -8, -10, -12} with and without memory at the
3D-rescaled total coupling Sigma_lambda = 4.0 (vs the 2D paper's 0.4).
Produces the four-to-five-orders-of-magnitude separation between the
unmemoried and memoried final peak densities in the supercritical range.

Expected wall time: approximately 2.5 minutes on RTX 4060.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the implementation package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.physics import (
    SolverConfig3D,
    MemoryConfig,
    run,
    make_default_observables,
    is_gpu,
    vram_estimate,
)


OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "anti_collapse_3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(Lambda: float, with_memory: bool, N: int = 128, precision: str = "fp32") -> dict:
    cfg = SolverConfig3D(
        N=N,
        L=20.0,
        dt=0.0025,
        n_steps=4000,
        Lambda=Lambda,
        gamma_0=0.0,
        T=0.0,
        memory=MemoryConfig(
            # Two-mode (fast + slow) at total Sigma_lambda = 4.0, the 3D-rescaled
            # coupling derived in ../../results/06-dimensional-rescaling.md.
            nus=[10.0, 0.5] if with_memory else [],
            lambdas=[3.0, 1.0] if with_memory else [],
            spatial="local",
        ),
        init_sigma=0.5,                    # 3D-supercritical concentration
        init_k0=(0.0, 0.0, 0.0),           # zero momentum, in-place dynamics
        sample_every=40,
        precision=precision,
        seed=42,
    )
    label = f"L{Lambda:+.1f}_{'mem' if with_memory else 'nomem'}"
    print(f"\n--- {label} ---")
    result = run(cfg, observables_fn=make_default_observables(cfg.L), verbose=False)

    ts = [s["t"] for s in result["samples"]]
    peaks = [s["peak"] for s in result["samples"]]
    fwhms = [s["fwhm"] for s in result["samples"]]
    norms = [s["norm"] for s in result["samples"]]

    summary = {
        "Lambda": Lambda,
        "with_memory": with_memory,
        "label": label,
        "max_peak": max(peaks),
        "final_peak": peaks[-1],
        "final_fwhm": fwhms[-1],
        "final_norm": norms[-1],
        "norm_drift": abs(norms[-1] - norms[0]),
        "wall_time": result["wall_time"],
    }
    print(f"  max_peak = {summary['max_peak']:.4f}, "
          f"final_peak = {summary['final_peak']:.4f}, "
          f"final_fwhm = {summary['final_fwhm']:.2f}")

    np.savez(
        OUTPUT_DIR / f"{label}.npz",
        ts=np.asarray(ts),
        peaks=np.asarray(peaks),
        norms=np.asarray(norms),
        fwhms=np.asarray(fwhms),
        config=json.dumps(result["config"]),
    )
    return summary


def main():
    print(f"Backend: {'cupy (GPU)' if is_gpu() else 'numpy (CPU)'}")
    print(f"Output: {OUTPUT_DIR}")
    est = vram_estimate(N=128, n_aux=2, d_int=1, precision="fp32")
    print(f"VRAM estimate (128^3, fp32, 2-mode memory): {est}")
    print("\nSweeping Lambda in {-2, -4, -6, -8, -10, -12} x {nomem, mem}.")
    print("Total: 12 trajectories at N=128, 4000 steps each.")

    lambdas = [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0]
    summaries = []
    t_total = time.time()
    for Lambda in lambdas:
        for with_mem in [False, True]:
            s = run_one(Lambda, with_mem)
            summaries.append(s)
    t_total = time.time() - t_total

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print()
    print(f"{'Lambda':>8} | {'memory':>6} | {'max_peak':>10} | {'final_peak':>11} | {'final_fwhm':>11}")
    print("-" * 70)
    for s in summaries:
        print(f"{s['Lambda']:>8.1f} | {str(s['with_memory']):>6} | "
              f"{s['max_peak']:>10.4f} | {s['final_peak']:>11.4f} | {s['final_fwhm']:>11.2f}")

    print(f"\nTotal wall time: {t_total:.1f} s.")
    print(f"All trajectories saved to {OUTPUT_DIR}.")
    print()
    print("Headline ratios (nomem final peak / mem final peak):")
    nomem = {s["Lambda"]: s for s in summaries if not s["with_memory"]}
    mem = {s["Lambda"]: s for s in summaries if s["with_memory"]}
    for Lambda in lambdas:
        if mem[Lambda]["final_peak"] > 0:
            ratio = nomem[Lambda]["final_peak"] / mem[Lambda]["final_peak"]
            print(f"  Lambda = {Lambda:>6.1f}: ratio = {ratio:.2e}")


if __name__ == "__main__":
    main()
