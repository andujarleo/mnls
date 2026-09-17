"""Reproduce the 3D Bravais lattice selection result documented in
../../results/05-bravais-selection.md.

Sweeps total memory coupling Sigma_lambda in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
at fixed Lambda = -8 and identifies the Bravais lattice symmetry of the
released crystalline state.

Expected wall time: approximately 2 minutes on RTX 4060.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from implementation.physics import (
    SolverConfig3D,
    MemoryConfig,
    run,
    make_default_observables,
    detect_bravais_3d,
    radial_power_spectrum_3d,
    get_backend,
    is_gpu,
    vram_estimate,
)


OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "bravais_sweep_3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one(Sigma_lambda: float, N: int = 128, precision: str = "fp32") -> dict:
    # Split 3:1 fast:slow, matching the m1b focal-demo recipe.
    fast = 0.75 * Sigma_lambda
    slow = 0.25 * Sigma_lambda

    cfg = SolverConfig3D(
        N=N,
        L=20.0,
        dt=0.0025,
        n_steps=6000,
        Lambda=-8.0,
        gamma_0=0.0,
        T=0.0,
        memory=MemoryConfig(nus=[10.0, 0.5], lambdas=[fast, slow], spatial="local"),
        init_sigma=0.5,
        init_k0=(0.0, 0.0, 0.0),
        sample_every=100,
        precision=precision,
        seed=42,
    )

    label = f"sl{Sigma_lambda:.2f}"
    print(f"\n--- {label} (Sigma_lambda = {Sigma_lambda}) ---")
    result = run(cfg, observables_fn=make_default_observables(cfg.L), verbose=False)

    xp = get_backend()
    psi_final = xp.asarray(result["psi_final"])
    bravais = detect_bravais_3d(psi_final, cfg.L, xp=xp)
    k_centers, P_radial = radial_power_spectrum_3d(psi_final, cfg.L, xp=xp)
    last = result["samples"][-1]

    summary = {
        "Sigma_lambda": Sigma_lambda,
        "final_peak": last["peak"],
        "final_fwhm": last["fwhm"],
        "final_ipr": last["ipr"],
        "final_crystallinity": last["crystallinity"],
        "norm_drift": abs(last["norm"] - result["samples"][0]["norm"]),
        "bravais": {
            "k_star": float(bravais["k_star"]),
            "wavelength": float(bravais["wavelength"]),
            "scores": {k: float(v) for k, v in bravais.get("scores", {}).items()},
            "best": bravais.get("best"),
            "best_score": float(bravais.get("best_score", 0.0)),
            "crystallinity": float(bravais.get("crystallinity", 0.0)),
        },
        "wall_time": result["wall_time"],
    }

    print(f"  peak = {last['peak']:.4f}, fwhm = {last['fwhm']:.2f}, "
          f"cryst = {last['crystallinity']:.3f}")
    print(f"  k* = {bravais['k_star']:.3f}, lambda_wave = {bravais['wavelength']:.3f}")
    print(f"  Bravais scores: " +
          ", ".join(f"{k}: {float(v):.3f}" for k, v in
                    sorted(bravais["scores"].items(), key=lambda kv: -kv[1])))

    np.savez(
        OUTPUT_DIR / f"{label}.npz",
        psi_final=result["psi_final"],
        k_centers=k_centers,
        P_radial=P_radial,
        bravais_scores=json.dumps(bravais["scores"]),
        config=json.dumps(result["config"]),
    )
    return summary


def main():
    print(f"Backend: {'cupy (GPU)' if is_gpu() else 'numpy (CPU)'}")
    print(f"Output: {OUTPUT_DIR}")
    est = vram_estimate(N=128, n_aux=2, d_int=1, precision="fp32")
    print(f"VRAM estimate (128^3, fp32): {est}")
    print()

    sweep = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    summaries = []
    t_total = time.time()
    for Sigma_lambda in sweep:
        summaries.append(run_one(Sigma_lambda))
    t_total = time.time() - t_total

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print()
    print(f"{'Sigma_lam':>9} | {'k_star':>7} | {'wavelen':>7} | {'best':>5} | "
          f"{'score':>6} | {'peak':>10} | {'cryst':>6}")
    print("-" * 75)
    for s in summaries:
        b = s["bravais"]
        print(f"{s['Sigma_lambda']:>9.2f} | {b['k_star']:>7.3f} | "
              f"{b['wavelength']:>7.3f} | {str(b['best']):>5} | "
              f"{b['best_score']:>6.3f} | {s['final_peak']:>10.4f} | "
              f"{s['final_crystallinity']:>6.3f}")
    print(f"\nTotal wall time: {t_total:.1f} s.")


if __name__ == "__main__":
    main()
