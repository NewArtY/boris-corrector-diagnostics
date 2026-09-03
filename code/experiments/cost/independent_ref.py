"""
independent_ref.py -- how accurate is the hybrid against a TRUSTWORTHY reference?
=================================================================================
The Article measures every error against a fine-step run of the SHIPPED Boris
integrator (dt_ref = dt_work/150). staggered.py showed that this shipped scheme
is only FIRST order in position, so its dt_ref solution carries an error of
order 1e-3 r_L -- the same order as the hybrid's reported 3.5e-3 r_L.

Worse, the corrector is TRAINED to reproduce exactly that reference. So the
reported 118x "gain" may measure how well the network reproduces a biased
reference rather than physical accuracy.

This script builds an independent, demonstrably higher-order reference with
scipy's DOP853 (8th order, adaptive, rtol=1e-12) and re-scores every scheme
against it. Writes independent_reference.json. Touches nothing else.
"""
import os
import sys
import json
import numpy as np
import torch
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN
from staggered import integrate_staggered
from bench import integrate_corrected, load_corrector

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
Q, M = -1.0, 1.0


def rhs(t, y, field):
    r, v = y[:3], y[3:]
    E = np.atleast_1d(field.E(r, t)).ravel()
    B = np.atleast_1d(field.B(r, t)).ravel()
    a = (Q / M) * (E + np.cross(v, B))
    return np.concatenate([v, a])


def dop853_reference(field, t_eval, rtol=1e-12, atol=1e-14):
    t_eval = np.clip(np.asarray(t_eval, float), 0.0, T_FINAL)
    sol = solve_ivp(rhs, (0.0, T_FINAL), np.concatenate([R0, V0]), args=(field,),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval,
                    dense_output=False)
    assert sol.success, sol.message
    return sol.y[:3].T, sol.y[3:].T


def score(rs, vs, ts, field, E0):
    """Errors of a discrete solution against DOP853 sampled at the same times."""
    r_ref, v_ref = dop853_reference(field, ts)
    E_ref = 0.5 * np.sum(v_ref ** 2, axis=1)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    e_err = np.abs(E - E_ref) / E0
    pos_err = np.linalg.norm(rs - r_ref, axis=1)
    half = len(ts) // 2
    return {"pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2))),
            "pos_err_final": float(pos_err[-1]),
            "energy_err_median_2nd_half": float(np.median(e_err[half:]))}


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    E0 = 0.5 * np.dot(V0, V0)

    # --- convergence self-check of the independent reference -------------
    t_probe = np.linspace(0, T_FINAL, 401)
    r_a, _ = dop853_reference(field, t_probe, rtol=1e-12, atol=1e-14)
    r_b, _ = dop853_reference(field, t_probe, rtol=1e-10, atol=1e-12)
    self_gap = float(np.sqrt(np.mean(np.linalg.norm(r_a - r_b, axis=1) ** 2)))
    print(f"DOP853 self-consistency (rtol 1e-12 vs 1e-10): {self_gap:.3e} r_L")

    out = {"meta": {"reference": "scipy DOP853 rtol=1e-12 atol=1e-14",
                    "self_consistency_rms_rL": self_gap,
                    "t_final": T_FINAL, "tau": TAU_MAIN},
           "vs_independent": {}, "vs_shipped_fine": {}}

    # --- the shipped fine reference, scored against DOP853 ---------------
    n_fine = int(round(T_FINAL / DT_FINE))
    rs_f, vs_f, ts_f = integrate_boris(R0, V0, 0.0, DT_FINE, n_fine, field)
    sub = np.arange(0, n_fine + 1, 50)          # subsample: DOP853 t_eval cost
    out["vs_independent"]["shipped_fine_reference_dt0.002"] = score(
        rs_f[sub], vs_f[sub], ts_f[sub], field, E0)
    print("shipped fine reference (the Article's ground truth), scored vs DOP853:",
          out["vs_independent"]["shipped_fine_reference_dt0.002"])

    # --- the three schemes of Figure 4, at the working step --------------
    n_work = int(round(T_FINAL / DT_WORK))
    model = load_corrector()

    runs = {
        "boris_shipped_dt0.3": integrate_boris(R0, V0, 0.0, DT_WORK, n_work, field),
        "hybrid_projected_dt0.3": integrate_corrected(field, R0, V0, DT_WORK, n_work, model, True),
        "hybrid_raw_dt0.3": integrate_corrected(field, R0, V0, DT_WORK, n_work, model, False),
        "boris_staggered_dt0.3": integrate_staggered(R0, V0, DT_WORK, n_work, field),
    }
    # staggered Boris at steps that cost about the same as the hybrid, and less
    for dt in (0.1, 0.05, 0.03, 0.02):
        n = int(round(T_FINAL / dt))
        runs[f"boris_staggered_dt{dt}"] = integrate_staggered(R0, V0, dt, n, field)

    for k, (rs, vs, ts) in runs.items():
        out["vs_independent"][k] = score(rs, vs, ts, field, E0)
        # also score against the Article's own reference, for direct comparison
        r_ref_i = np.vstack([np.interp(ts, ts_f, rs_f[:, j]) for j in range(3)]).T
        out["vs_shipped_fine"][k] = {
            "pos_err_rms": float(np.sqrt(np.mean(
                np.linalg.norm(rs - r_ref_i, axis=1) ** 2)))}
        print(f"{k:28s} vs DOP853: traj={out['vs_independent'][k]['pos_err_rms']:.4e}  "
              f"en={out['vs_independent'][k]['energy_err_median_2nd_half']:.4e}   |  "
              f"vs shipped-fine: traj={out['vs_shipped_fine'][k]['pos_err_rms']:.4e}")

    # --- the headline gain, recomputed --------------------------------------
    b = out["vs_independent"]["boris_shipped_dt0.3"]["pos_err_rms"]
    h = out["vs_independent"]["hybrid_projected_dt0.3"]["pos_err_rms"]
    bs = out["vs_independent"]["boris_staggered_dt0.3"]["pos_err_rms"]
    out["headline"] = {
        "gain_hybrid_over_shipped_boris_vs_independent_ref": b / h,
        "gain_hybrid_over_shipped_boris_vs_own_ref": (
            out["vs_shipped_fine"]["boris_shipped_dt0.3"]["pos_err_rms"]
            / out["vs_shipped_fine"]["hybrid_projected_dt0.3"]["pos_err_rms"]),
        "gain_hybrid_over_staggered_boris_same_dt_vs_independent_ref": bs / h,
    }
    print("\nHEADLINE:", json.dumps(out["headline"], indent=2))

    with open(os.path.join(HERE, "independent_reference.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote independent_reference.json")


if __name__ == "__main__":
    main()
