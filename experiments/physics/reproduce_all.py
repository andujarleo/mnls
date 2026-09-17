"""Run all physics reproduction scripts in sequence.

Total wall time: approximately 15 minutes on RTX 4060.

This script invokes each of the individual reproduction scripts as a
subprocess so that they each maintain their own random-seed and output
isolation. The output of each is saved to the same outputs/ subfolder
as if the script had been run directly.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPTS = [
    # Implemented and validated:
    "reproduce_3d_anti_collapse.py",
    "reproduce_3d_bravais_sweep.py",
    # Pending implementation (stubs):
    # "reproduce_2d_anti_collapse.py",
    # "reproduce_2d_crystallization.py",
    # "reproduce_2d_vibration_spectrum.py",
    # "reproduce_dimensional_rescaling.py",
    # "reproduce_temporal_spatial_asymmetry.py",
]


def main():
    here = Path(__file__).parent
    t_start = time.time()
    failed = []
    for script in SCRIPTS:
        script_path = here / script
        if not script_path.exists():
            print(f"  Skipping {script} (not present)")
            continue
        print(f"\n=== Running {script} ===")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(here.parent.parent),
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED: {script} (exit code {result.returncode})")
            failed.append(script)
        else:
            print(f"  OK: {script} ({elapsed:.1f}s)")
    t_total = time.time() - t_start
    print(f"\n\nTotal wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    if failed:
        print(f"Failed scripts: {failed}")
        sys.exit(1)
    else:
        print("All scripts completed successfully.")


if __name__ == "__main__":
    main()
