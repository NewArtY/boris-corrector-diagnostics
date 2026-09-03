"""
decompose_error.py  --  task I3.1

Orthogonal decomposition of the velocity error into a radial (speed / energy)
component and a tangential (phase / direction) component, verified against the
converged fine-step reference on field B4.

Also:
  * verifies that the hard projection is exactly the first-order retraction
    on the sphere and that it preserves |v| to machine precision;
  * measures the actual step-relative correction magnitude t = ||xi|| / ||v_b||
    and the resulting retraction-vs-exponential-map discrepancy O(t^3/3);
  * checks the exact energy identity
        dE = |v_ref| * s + 0.5 * s^2 + 0.5 * ||dv_perp||^2 .

Writes experiments/theory/decomposition.json. Reads only; modifies nothing.
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
from models.boris import boris_step, integrate_boris
from training.train_corrector_b4 import DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN


def load_corrector():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    return m


def integrate_corrected_instrumented(field, r0, v0, dt, n_steps, model, project):
    """Same as diagnostics/eval_corrector.integrate_corrected, plus diagnostics."""
    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    # per-step diagnostics of the projection
    t_rel = np.zeros(n_steps)        # ||xi|| / ||v_b||   (dimensionless)
    speed_viol = np.zeros(n_steps)   # | ||v_new|| - ||v_b|| | / ||v_b||
    retr_exp = np.zeros(n_steps)     # ||R(xi) - Exp(xi)|| / ||v_b||
    dv_radial_raw = np.zeros(n_steps)  # |dv . vhat| / ||v_b||  BEFORE projection

    r, v, t = np.array(r0, float), np.array(v0, float), 0.0
    rs[0], vs[0] = r, v
    with torch.no_grad():
        for i in range(1, n_steps + 1):
            r_b, v_b = boris_step(r, v, t, dt, field)
            B = np.atleast_1d(field.B(r, t)).ravel()
            E = np.atleast_1d(field.E(r, t)).ravel()
            x = torch.tensor(np.concatenate([r, v, B, E, [dt]]))[None, :]
            d = model(x).numpy()[0]
            dr, dv = d[:3], d[3:]

            nb = np.linalg.norm(v_b)
            vh = v_b / max(nb, 1e-300)
            dv_radial_raw[i - 1] = abs(float(np.dot(dv, vh))) / nb

            if project:
                xi = dv - np.dot(dv, vh) * vh          # tangential part only
                v_new = v_b + xi
                v_new = v_new * (nb / max(np.linalg.norm(v_new), 1e-300))

                tt = np.linalg.norm(xi) / nb
                t_rel[i - 1] = tt
                speed_viol[i - 1] = abs(np.linalg.norm(v_new) - nb) / nb
                # exponential map on the sphere of radius nb, same tangent xi
                if tt > 0:
                    xih = xi / np.linalg.norm(xi)
                    v_exp = nb * (np.cos(tt) * vh + np.sin(tt) * xih)
                    retr_exp[i - 1] = np.linalg.norm(v_new - v_exp) / nb
            else:
                v_new = v_b + dv
                t_rel[i - 1] = np.linalg.norm(dv) / nb
                speed_viol[i - 1] = abs(np.linalg.norm(v_new) - nb) / nb

            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
    diag = {"t_rel": t_rel, "speed_viol": speed_viol,
            "retr_exp": retr_exp, "dv_radial_raw": dv_radial_raw}
    return rs, vs, ts, diag


def decompose(vs_num, vs_ref):
    """Split dv = v_num - v_ref into radial (along v_ref) and tangential parts."""
    dv = vs_num - vs_ref
    nref = np.linalg.norm(vs_ref, axis=1)
    vhat = vs_ref / nref[:, None]
    s = np.sum(dv * vhat, axis=1)                 # signed radial component
    dv_par = s[:, None] * vhat
    dv_perp = dv - dv_par
    return dv, s, dv_par, dv_perp, nref


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    n_fine = int(round(T_FINAL / DT_FINE))
    rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)

    n_work = int(round(T_FINAL / DT_WORK))
    stride = int(round(DT_WORK / DT_FINE))
    assert abs(stride * DT_FINE - DT_WORK) < 1e-15
    idx = np.arange(n_work + 1) * stride
    assert idx[-1] <= n_fine
    vs_ref = vs_r[idx]          # exact sampling, no interpolation error
    rs_ref = rs_r[idx]
    E_ref = 0.5 * np.sum(vs_ref ** 2, axis=1)
    E0 = 0.5 * np.sum(vs_r[0] ** 2)

    model = load_corrector()

    runs = {}
    rs_b, vs_b, ts_b = integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field)
    runs["boris"] = (rs_b, vs_b, None)
    rs, vs, ts, dg = integrate_corrected_instrumented(
        field, r0, v0, DT_WORK, n_work, model, False)
    runs["corrector_raw"] = (rs, vs, dg)
    rs, vs, ts, dg = integrate_corrected_instrumented(
        field, r0, v0, DT_WORK, n_work, model, True)
    runs["corrector_projected"] = (rs, vs, dg)

    half = n_work // 2
    out = {"config": {"dt_work": DT_WORK, "dt_fine": DT_FINE,
                      "t_final": T_FINAL, "tau": TAU_MAIN,
                      "n_work_steps": n_work,
                      "n_params_defectnet": sum(p.numel() for p in model.parameters())}}

    for name, (rs_n, vs_n, dg) in runs.items():
        dv, s, dv_par, dv_perp, nref = decompose(vs_n, vs_ref)
        n_par = np.abs(s)
        n_perp = np.linalg.norm(dv_perp, axis=1)

        # exact energy identity
        E_num = 0.5 * np.sum(vs_n ** 2, axis=1)
        dE = (E_num - E_ref) / E0
        dE_id = (nref * s + 0.5 * s ** 2 + 0.5 * n_perp ** 2) / E0
        id_resid = np.max(np.abs(dE - dE_id))

        pos_err = np.linalg.norm(rs_n - rs_ref, axis=1)

        entry = {
            "radial_median_2nd_half": float(np.median(n_par[half:])),
            "tangential_median_2nd_half": float(np.median(n_perp[half:])),
            "radial_over_tangential": float(np.median(n_par[half:]) /
                                            max(np.median(n_perp[half:]), 1e-300)),
            "energy_err_median_2nd_half": float(np.median(np.abs(dE[half:]))),
            "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2))),
            "energy_identity_max_residual": float(id_resid),
            "energy_first_order_term_median": float(np.median(np.abs(nref * s / E0)[half:])),
            "energy_second_order_term_median": float(
                np.median(np.abs((0.5 * s ** 2 + 0.5 * n_perp ** 2) / E0)[half:])),
        }
        if dg is not None:
            entry["proj_t_rel_median"] = float(np.median(dg["t_rel"]))
            entry["proj_t_rel_max"] = float(np.max(dg["t_rel"]))
            entry["speed_violation_max"] = float(np.max(dg["speed_viol"]))
            entry["dv_radial_raw_median"] = float(np.median(dg["dv_radial_raw"]))
            if np.any(dg["retr_exp"] > 0):
                entry["retraction_vs_exp_median"] = float(np.median(dg["retr_exp"]))
                entry["retraction_vs_exp_max"] = float(np.max(dg["retr_exp"]))
                tmed = np.median(dg["t_rel"])
                entry["retraction_vs_exp_predicted_t3_over_3"] = float(tmed ** 3 / 3)
        out[name] = entry

    dst = os.path.join(HERE, "decomposition.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
