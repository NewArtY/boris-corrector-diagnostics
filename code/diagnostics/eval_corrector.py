"""
eval_corrector.py
=================
Honest evaluation of what the constrained corrector actually improves.

Three variants are compared against a converged fine-step reference:
  1. plain Boris at the working step
  2. Boris + raw learned correction (soft constraints only)
  3. Boris + HARD energy-neutral projection of the correction
     (velocity correction projected orthogonal to v and the speed rescaled
      to the Boris value, so the correction cannot change |v| at all)

Reported: relative energy error and relative trajectory (position) error.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR, FIGURE_DIR, get_logger
from fields import DecayingField
from models.boris import boris_step, integrate_boris
from training.train_corrector_b4 import DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN

logger = get_logger("eval_corrector")


def load_corrector():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    return m


def integrate_corrected(field, r0, v0, dt, n_steps, model, project=True):
    rs = np.zeros((n_steps + 1, 3)); vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
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
            if project:
                nb = np.linalg.norm(v_b)
                vh = v_b / max(nb, 1e-300)
                dv = dv - np.dot(dv, vh) * vh          # orthogonal component only
                v_new = v_b + dv
                v_new *= nb / max(np.linalg.norm(v_new), 1e-300)  # exact |v| preservation
            else:
                v_new = v_b + dv
            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
    return rs, vs, ts


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])

    n_fine = int(round(T_FINAL / DT_FINE))
    rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1); E0 = E_ref[0]

    n_work = int(round(T_FINAL / DT_WORK))
    model = load_corrector()

    runs = {}
    rs_b, vs_b, ts_b = integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field)
    runs["boris"] = (rs_b, vs_b, ts_b)
    runs["corrector_raw"] = integrate_corrected(field, r0, v0, DT_WORK, n_work, model, False)
    runs["corrector_projected"] = integrate_corrected(field, r0, v0, DT_WORK, n_work, model, True)

    out = {}
    half = n_work // 2
    for k, (rs, vs, ts) in runs.items():
        Ei = np.interp(ts, ts_r, E_ref)
        E = 0.5 * np.sum(vs ** 2, axis=1)
        e_err = np.abs(E - Ei) / E0
        r_ref_i = np.vstack([np.interp(ts, ts_r, rs_r[:, j]) for j in range(3)]).T
        pos_err = np.linalg.norm(rs - r_ref_i, axis=1)
        out[k] = {"energy_err_median_2nd_half": float(np.median(e_err[half:])),
                  "energy_err_max": float(e_err.max()),
                  "pos_err_final": float(pos_err[-1]),
                  "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2)))}
        logger.info("%-22s energy=%.3e  pos_final=%.3e  pos_rms=%.3e", k,
                    out[k]["energy_err_median_2nd_half"], out[k]["pos_err_final"],
                    out[k]["pos_err_rms"])

    phys = float(np.median(np.abs((np.interp(ts_b, ts_r, E_ref) - E0) / E0)[half:]))
    out["physical_signal_median"] = phys
    out["traj_gain_projected"] = out["boris"]["pos_err_rms"] / out["corrector_projected"]["pos_err_rms"]
    logger.info("physical signal=%.3e | trajectory gain (projected) = %.1fx",
                phys, out["traj_gain_projected"])
    with open(os.path.join(FIGURE_DIR, "corrector_evaluation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
