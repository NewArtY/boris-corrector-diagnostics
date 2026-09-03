"""
generate_results_summary.py
=============================
One-off aggregation script (not a figure): computes, for each of the five
integrators and each of the four unseen field configurations B1-B4, the
mean |dE/E0| and RMS energy error, and writes them to
output_figures/results_summary.json together with the pre-existing
output_figures/corrector_evaluation.json content (merged under the key
"b4_separation_study", without modifying that file).
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEED, set_global_seed, FIGURE_DIR
from fields import RadialField, WaveField, TiltedField, DecayingField
from diagnostics.integrator_runner import integrate, ALL_INTEGRATORS
from diagnostics.energy_drift import mean_abs_energy_drift, rms_energy_error

set_global_seed(SEED)

DT = 0.05
N_GYRO = 15
N_STEPS = int(N_GYRO * (2 * np.pi) / DT)
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
EPS = 1e-16

FIELD_SPECS = [
    ("B1", RadialField()),
    ("B2", WaveField()),
    ("B3", TiltedField()),
    ("B4", DecayingField(tau=40.0)),
]


def main():
    summary = {}
    for name in ALL_INTEGRATORS:
        summary[name] = {}
        for tag, field in FIELD_SPECS:
            rs, vs, ts = integrate(name, field, R0, V0, 0.0, DT, N_STEPS)
            mean_abs = float(np.clip(mean_abs_energy_drift(vs), EPS, 1e12))
            rms = float(np.clip(rms_energy_error(vs), EPS, 1e12))
            summary[name][tag] = {
                "mean_abs_dE_E0": mean_abs,
                "rms_energy_error": rms,
            }
        print(f"{name} done")

    # Merge in the pre-existing corrector_evaluation.json (B4 separation
    # study), produced independently by diagnostics/eval_corrector.py.
    # That file is NOT modified or deleted here -- only read and copied in.
    corrector_eval_path = os.path.join(FIGURE_DIR, "corrector_evaluation.json")
    if os.path.exists(corrector_eval_path):
        with open(corrector_eval_path) as f:
            corrector_eval = json.load(f)
        summary["b4_separation_study"] = corrector_eval
    else:
        print("WARNING: corrector_evaluation.json not found, skipping merge")

    out_path = os.path.join(FIGURE_DIR, "results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
