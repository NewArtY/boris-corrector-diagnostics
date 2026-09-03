"""
run.py -- do classical schemes beat the hybrid on BOTH channels?
=================================================================
The Article's central claim is that only the constrained hybrid holds both the
energy and the trajectory error below the physical signal at a usable step.
That claim was tested against the shipped (1st-order) Boris pusher and three
unconverged networks. This script tests it against the classical schemes the
Article's own introduction cites but never compares with.

Reference: DOP853, rtol 1e-12 -- independent of every scheme under test.
Physical signal: median over the second half of |E_ref(t)-E_ref(0)|/E_ref(0),
i.e. exactly the quantity diagnostics/eval_corrector.py calls
physical_signal_median.

Writes workprecision.json. Touches nothing outside this directory.
"""
import os
import sys
import json
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "cost"))
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from fields import DecayingField
from training.train_corrector_b4 import DefectNet, DT_WORK, T_FINAL, TAU_MAIN
import schemes as S

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
Q, M = -1.0, 1.0
DT_GRID = [0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02]
N_REPEAT = 3


# ------------------------------------------------------------------ hybrid
def load_corrector():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR,
                                              "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    return m


def integrate_hybrid(field, dt, n_steps, model):
    from models.boris import boris_step
    rs = np.zeros((n_steps + 1, 3)); vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    r, v, t = R0.copy(), V0.copy(), 0.0
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
            dv = dv - np.dot(dv, vh) * vh
            v_new = v_b + dv
            v_new *= nb / max(np.linalg.norm(v_new), 1e-300)
            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
    return rs, vs, ts


# --------------------------------------------------------------- reference
def dop853(field, t_eval, t_end):
    def rhs(t, y):
        r, v = y[:3], y[3:]
        E = np.atleast_1d(field.E(r, t)).ravel()
        B = np.atleast_1d(field.B(r, t)).ravel()
        return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])
    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([R0, V0]),
                    method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t_eval)
    assert sol.success, sol.message
    return sol.y[:3].T, sol.y[3:].T


def score(rs, vs, ts, r_ref, v_ref):
    E = 0.5 * np.sum(vs ** 2, axis=1)
    E_ref = 0.5 * np.sum(v_ref ** 2, axis=1)
    E0 = E_ref[0]
    half = len(ts) // 2
    e_err = np.abs(E - E_ref) / E0
    pos = np.linalg.norm(rs - r_ref, axis=1)
    return {"pos_err_rms": float(np.sqrt(np.mean(pos ** 2))),
            "pos_err_final": float(pos[-1]),
            "energy_err_median_2nd_half": float(np.median(e_err[half:])),
            "energy_err_max": float(e_err.max())}


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    model = load_corrector()
    out = {"meta": {"t_final": T_FINAL, "tau": TAU_MAIN, "dt_work": DT_WORK,
                    "reference": "DOP853 rtol=1e-12 atol=1e-14"},
           "runs": []}

    # physical signal from the reference itself, on the working grid
    n_w = int(round(T_FINAL / DT_WORK))
    ts_w = np.linspace(0.0, T_FINAL, n_w + 1)
    r_ref_w, v_ref_w = dop853(field, ts_w, T_FINAL)
    E_ref_w = 0.5 * np.sum(v_ref_w ** 2, axis=1)
    sig = float(np.median(np.abs(E_ref_w - E_ref_w[0])[len(ts_w) // 2:]
                          / E_ref_w[0]))
    out["meta"]["physical_signal_median"] = sig
    print(f"physical signal (median, 2nd half) = {sig:.6e}")

    builders = {
        "staggered": None,
        "vps2": S.make_vps2, "vps4": S.make_vps4,
        "imr": S.make_imr, "gl4": S.make_gl4,
    }

    for dt in DT_GRID:
        n = int(round(T_FINAL / dt))
        ts = np.linspace(0.0, T_FINAL, n + 1)
        r_ref, v_ref = dop853(field, ts, T_FINAL)

        # shipped (the Article's baseline)
        from models.boris import integrate_boris
        w = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            rs, vs, tt = integrate_boris(R0, V0, 0.0, dt, n, field)
            w.append(time.perf_counter() - t0)
        rec = score(rs, vs, tt, r_ref, v_ref)
        rec.update(scheme="shipped", dt=dt, n_steps=n,
                   wall_s=float(np.median(w)),
                   flops=float(n * 113))
        out["runs"].append(rec); _p(rec)

        # staggered
        w = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            rs, vs, tt = S.integrate_staggered(field, R0, V0, dt, n)
            w.append(time.perf_counter() - t0)
        rec = score(rs, vs, tt, r_ref, v_ref)
        rec.update(scheme="staggered", dt=dt, n_steps=n,
                   wall_s=float(np.median(w)), flops=float(n * 113))
        out["runs"].append(rec); _p(rec)

        for name in ("vps2", "vps4", "imr", "gl4"):
            step = builders[name](field)
            w = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                rs, vs, tt = S.integrate(step, R0, V0, dt, n)
                w.append(time.perf_counter() - t0)
            rec = score(rs, vs, tt, r_ref, v_ref)
            if hasattr(step, "stats") and step.stats["steps"]:
                ni = step.stats["iters"] / step.stats["steps"]
                fl = (S.flops_imr(ni) if name == "imr" else S.flops_gl4(ni))
                rec["mean_iters"] = float(ni)
            else:
                fl = S.FLOPS_PER_STEP[name]
            rec.update(scheme=name, dt=dt, n_steps=n,
                       wall_s=float(np.median(w)), flops=float(n * fl))
            out["runs"].append(rec); _p(rec)

    # hybrid: single operating point
    n = int(round(T_FINAL / DT_WORK))
    ts = np.linspace(0.0, T_FINAL, n + 1)
    r_ref, v_ref = dop853(field, ts, T_FINAL)
    w = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        rs, vs, tt = integrate_hybrid(field, DT_WORK, n, model)
        w.append(time.perf_counter() - t0)
    rec = score(rs, vs, tt, r_ref, v_ref)
    rec.update(scheme="hybrid", dt=DT_WORK, n_steps=n,
               wall_s=float(np.median(w)), flops=float(n * 114091))
    out["runs"].append(rec); _p(rec)

    with open(os.path.join(HERE, "workprecision.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote workprecision.json")


def _p(r):
    print(f"{r['scheme']:10s} dt={r['dt']:6.3f} "
          f"traj={r['pos_err_rms']:.4e} en={r['energy_err_median_2nd_half']:.4e} "
          f"wall={r['wall_s']:7.4f}s flops={r['flops']:.3e}")


if __name__ == "__main__":
    main()
