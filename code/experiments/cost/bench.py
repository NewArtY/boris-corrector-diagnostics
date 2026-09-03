"""
bench.py -- I1.1 "Cost at equal budget"
=======================================
Work-precision benchmark: is the hybrid's 118x trajectory gain obtained
"at fixed cost", as claimed in the Article?

Measures, for field B4 (decaying) over T_FINAL=120 against a converged
fine-step Boris reference:
  * wall-clock of the full integration (median of N_REPEAT runs)
  * RMS trajectory error in Larmor radii (r_L = 1 in normalised units)
  * median relative energy error over the second half of the run

for plain Boris over a grid of time steps, and for the Boris+Corrector
hybrid at its training step (and, to document the failure, away from it).

Outputs everything to JSON in this directory. No files outside
experiments/cost/ are written.
"""
import os
import sys
import json
import time
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

N_REPEAT = 5
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])

# Omega_c = 1 in normalised units, so Omega_c*dt == dt
DT_GRID_BORIS = [0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02]
DT_GRID_HYBRID = [0.3, 0.2, 0.1]


def load_corrector():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    return m


def integrate_corrected(field, r0, v0, dt, n_steps, model, project=True):
    """Identical to diagnostics/eval_corrector.py:integrate_corrected."""
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
                dv = dv - np.dot(dv, vh) * vh
                v_new = v_b + dv
                v_new *= nb / max(np.linalg.norm(v_new), 1e-300)
            else:
                v_new = v_b + dv
            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
    return rs, vs, ts


def errors(rs, vs, ts, rs_r, ts_r, E_ref, E0):
    """Trajectory RMS error (Larmor radii) and median |dE/E0| over 2nd half."""
    half = len(ts) // 2
    Ei = np.interp(ts, ts_r, E_ref)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    e_err = np.abs(E - Ei) / E0
    r_ref_i = np.vstack([np.interp(ts, ts_r, rs_r[:, j]) for j in range(3)]).T
    pos_err = np.linalg.norm(rs - r_ref_i, axis=1)
    return {
        "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2))),
        "pos_err_final": float(pos_err[-1]),
        "energy_err_median_2nd_half": float(np.median(e_err[half:])),
        "energy_err_max": float(e_err.max()),
    }


def timeit(fn, n_repeat=N_REPEAT):
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts)), out


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)

    # ---------------- reference ----------------
    n_fine = int(round(T_FINAL / DT_FINE))
    t0 = time.perf_counter()
    rs_r, vs_r, ts_r = integrate_boris(R0, V0, 0.0, DT_FINE, n_fine, field)
    t_ref = time.perf_counter() - t0
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1)
    E0 = E_ref[0]
    half_r = n_fine // 2
    phys_signal = float(np.median(np.abs((E_ref - E0) / E0)[half_r:]))
    print(f"reference: dt={DT_FINE:g} n={n_fine} wall={t_ref:.2f}s "
          f"physical_signal_median={phys_signal:.4e}")

    results = {
        "meta": {
            "field": "B4 decaying", "tau": TAU_MAIN, "t_final": T_FINAL,
            "dt_ref": DT_FINE, "n_ref_steps": n_fine, "ref_wall_s": t_ref,
            "r0": R0.tolist(), "v0": V0.tolist(),
            "larmor_radius": 1.0, "n_repeat": N_REPEAT,
            "physical_signal_median": phys_signal,
        },
        "boris": [], "hybrid": [],
    }

    # ---------------- Boris grid ----------------
    for dt in DT_GRID_BORIS:
        n = int(round(T_FINAL / dt))
        wall, wall_min, out = timeit(
            lambda: integrate_boris(R0, V0, 0.0, dt, n, field))
        rs, vs, ts = out
        e = errors(rs, vs, ts, rs_r, ts_r, E_ref, E0)
        rec = {"dt": dt, "n_steps": n, "wall_s": wall, "wall_min_s": wall_min,
               "us_per_step": 1e6 * wall / n, **e}
        results["boris"].append(rec)
        print(f"boris  dt={dt:6.3f} n={n:6d} wall={wall:7.4f}s "
              f"traj={e['pos_err_rms']:.4e} en={e['energy_err_median_2nd_half']:.4e}")

    # ---------------- hybrid ----------------
    model = load_corrector()
    for dt in DT_GRID_HYBRID:
        n = int(round(T_FINAL / dt))
        wall, wall_min, out = timeit(
            lambda: integrate_corrected(field, R0, V0, dt, n, model, True))
        rs, vs, ts = out
        e = errors(rs, vs, ts, rs_r, ts_r, E_ref, E0)
        rec = {"dt": dt, "n_steps": n, "wall_s": wall, "wall_min_s": wall_min,
               "us_per_step": 1e6 * wall / n, "in_training_distribution": dt == DT_WORK,
               **e}
        results["hybrid"].append(rec)
        print(f"hybrid dt={dt:6.3f} n={n:6d} wall={wall:7.4f}s "
              f"traj={e['pos_err_rms']:.4e} en={e['energy_err_median_2nd_half']:.4e}"
              f"{'' if dt == DT_WORK else '   <-- OUT OF TRAINING DISTRIBUTION'}")

    # hybrid without projection, at working step, for completeness
    n = int(round(T_FINAL / DT_WORK))
    wall, wall_min, out = timeit(
        lambda: integrate_corrected(field, R0, V0, DT_WORK, n, model, False))
    rs, vs, ts = out
    results["hybrid_raw_dt_work"] = {
        "dt": DT_WORK, "n_steps": n, "wall_s": wall, "us_per_step": 1e6 * wall / n,
        **errors(rs, vs, ts, rs_r, ts_r, E_ref, E0)}

    with open(os.path.join(HERE, "work_precision.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote work_precision.json")


if __name__ == "__main__":
    main()
