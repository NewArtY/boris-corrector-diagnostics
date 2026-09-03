"""
polar_decomposition.py  --  task I3.1, corrected formulation

The Cartesian split dv = dv_par + dv_perp (relative to v_ref) is exact but does
NOT isolate the energy channel: for a pure rotation the radial component s is
large and non-zero, yet the energy error vanishes identically through
cancellation against the second-order terms.

The decomposition that DOES separate the channels is polar: velocity space is
R^+ x S^2, so the velocity error splits into

    speed error      d|v| = |v_num| - |v_ref|          (energy channel)
    direction error  theta = angle(v_num, v_ref)       (phase channel)

with the exact identity

    ||dv||^2 = d|v|^2 + 2 |v_num| |v_ref| (1 - cos theta)

and the energy error depending on d|v| ALONE:

    dE = 0.5 (|v_num|^2 - |v_ref|^2) = 0.5 (|v_num| + |v_ref|) d|v| .

Writes experiments/theory/polar_decomposition.json.
"""
import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN
from experiments.theory.decompose_error import (
    load_corrector, integrate_corrected_instrumented)


def polar(vs_num, vs_ref):
    n_num = np.linalg.norm(vs_num, axis=1)
    n_ref = np.linalg.norm(vs_ref, axis=1)
    d_speed = n_num - n_ref
    cos = np.sum(vs_num * vs_ref, axis=1) / (n_num * n_ref)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    return n_num, n_ref, d_speed, theta, cos


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    n_fine = int(round(T_FINAL / DT_FINE))
    rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)
    n_work = int(round(T_FINAL / DT_WORK))
    stride = int(round(DT_WORK / DT_FINE))
    idx = np.arange(n_work + 1) * stride
    vs_ref, rs_ref = vs_r[idx], rs_r[idx]
    E0 = 0.5 * np.sum(vs_r[0] ** 2)

    model = load_corrector()
    runs = {}
    rs_b, vs_b, _ = integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field)
    runs["boris"] = (rs_b, vs_b)
    rs, vs, _, _ = integrate_corrected_instrumented(
        field, r0, v0, DT_WORK, n_work, model, False)
    runs["corrector_raw"] = (rs, vs)
    rs, vs, _, _ = integrate_corrected_instrumented(
        field, r0, v0, DT_WORK, n_work, model, True)
    runs["corrector_projected"] = (rs, vs)

    half = n_work // 2
    out = {}
    for name, (rs_n, vs_n) in runs.items():
        n_num, n_ref, d_speed, theta, cos = polar(vs_n, vs_ref)
        dv = vs_n - vs_ref
        norm_dv2 = np.sum(dv ** 2, axis=1)
        # exact identity check
        ident = d_speed ** 2 + 2 * n_num * n_ref * (1 - cos)
        resid = np.max(np.abs(norm_dv2 - ident))
        # energy from speed error alone
        dE = (0.5 * (n_num ** 2 - n_ref ** 2)) / E0
        dE_from_speed = (0.5 * (n_num + n_ref) * d_speed) / E0
        resid_E = np.max(np.abs(dE - dE_from_speed))

        pos_err = np.linalg.norm(rs_n - rs_ref, axis=1)
        out[name] = {
            "speed_err_rel_median_2nd_half": float(np.median(np.abs(d_speed / n_ref)[half:])),
            "theta_median_2nd_half_deg": float(np.degrees(np.median(theta[half:]))),
            "theta_final_deg": float(np.degrees(theta[-1])),
            "energy_err_median_2nd_half": float(np.median(np.abs(dE)[half:])),
            "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2))),
            "identity_max_residual": float(resid),
            "energy_from_speed_max_residual": float(resid_E),
            "norm_dv_median_2nd_half": float(np.median(np.sqrt(norm_dv2)[half:])),
        }

    # channel ratios that matter for the paper
    b, p = out["boris"], out["corrector_projected"]
    out["_ratios"] = {
        "speed_err_projected_over_boris": p["speed_err_rel_median_2nd_half"] /
                                          b["speed_err_rel_median_2nd_half"],
        "theta_boris_over_projected": b["theta_median_2nd_half_deg"] /
                                      p["theta_median_2nd_half_deg"],
        "pos_gain_boris_over_projected": b["pos_err_rms"] / p["pos_err_rms"],
    }

    dst = os.path.join(HERE, "polar_decomposition.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
