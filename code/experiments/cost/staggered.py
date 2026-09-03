"""
staggered.py -- is the shipped Boris implementation second-order?
=================================================================
models/boris.py updates position as

    r_{n+1} = r_n + v_{n+1} * dt                                   (as shipped)

with v treated as living at INTEGER times. That is a semi-implicit Euler
drift, which is first-order accurate in position. The textbook Boris pusher
is a staggered leapfrog: velocity lives at HALF-integer times,

    v_{n+1/2} = Boris(v_{n-1/2}, E(r_n), B(r_n)),
    r_{n+1}   = r_n + v_{n+1/2} * dt                               (staggered)

which is second-order. This script measures the convergence order of both
against the same fine reference, to establish which baseline the Article's
118x trajectory gain is actually being compared against.

Writes convergence.json in this directory. Touches nothing else.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from fields import DecayingField
from models.boris import boris_step, integrate_boris
from training.train_corrector_b4 import DT_FINE, T_FINAL, TAU_MAIN

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
N_REPEAT = 5
DT_GRID = [0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02]


def boris_kick(v, E, B, dt, q=-1.0, m=1.0):
    """Velocity-only Boris rotation + electric kicks (no position update)."""
    qmdt2 = 0.5 * q * dt / m
    v_minus = v + qmdt2 * E
    t_vec = qmdt2 * B
    t_mag2 = np.dot(t_vec, t_vec)
    s_vec = 2.0 * t_vec / (1.0 + t_mag2)
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)
    return v_plus + qmdt2 * E


def integrate_staggered(r0, v0, dt, n_steps, field, q=-1.0, m=1.0):
    """Textbook staggered-leapfrog Boris. v0 is given at t=0, so it is first
    backed up half a step to t=-dt/2; positions are returned at integer times
    and velocities re-centred to integer times for a like-for-like energy
    comparison."""
    r = np.array(r0, float)
    E = np.atleast_1d(field.E(r, 0.0)).ravel()
    B = np.atleast_1d(field.B(r, 0.0)).ravel()
    v_half = boris_kick(np.array(v0, float), E, B, -0.5 * dt, q, m)  # v_{-1/2}

    rs = np.zeros((n_steps + 1, 3)); vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    rs[0], vs[0], ts[0] = r, np.array(v0, float), 0.0
    t = 0.0
    for i in range(1, n_steps + 1):
        E = np.atleast_1d(field.E(r, t)).ravel()
        B = np.atleast_1d(field.B(r, t)).ravel()
        v_new_half = boris_kick(v_half, E, B, dt, q, m)      # v_{n+1/2}
        r = r + v_new_half * dt                              # r_{n+1}
        t = t + dt
        # velocity at integer time n+1, by averaging the two half-step values
        vs[i] = 0.5 * (v_half + v_new_half)
        v_half = v_new_half
        rs[i], ts[i] = r, t
    return rs, vs, ts


def errors(rs, vs, ts, rs_r, ts_r, E_ref, E0):
    half = len(ts) // 2
    Ei = np.interp(ts, ts_r, E_ref)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    e_err = np.abs(E - Ei) / E0
    r_ref_i = np.vstack([np.interp(ts, ts_r, rs_r[:, j]) for j in range(3)]).T
    pos_err = np.linalg.norm(rs - r_ref_i, axis=1)
    return (float(np.sqrt(np.mean(pos_err ** 2))),
            float(np.median(e_err[half:])))


def fit_order(dts, errs):
    """Least-squares slope of log(err) vs log(dt), with standard error."""
    x = np.log(np.asarray(dts)); y = np.log(np.asarray(errs))
    n = len(x)
    p, c = np.polyfit(x, y, 1)
    resid = y - (p * x + c)
    s2 = np.sum(resid ** 2) / max(n - 2, 1)
    se = np.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    return float(p), float(1.96 * se)


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    n_fine = int(round(T_FINAL / DT_FINE))

    # Reference: the SHIPPED integrator at dt_fine, exactly as the Article uses.
    rs_r, vs_r, ts_r = integrate_boris(R0, V0, 0.0, DT_FINE, n_fine, field)
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1); E0 = E_ref[0]

    # Independent reference: staggered scheme at the same fine step, to check
    # that the two references agree (i.e. the fine solution is converged).
    rs_s, vs_s, ts_s = integrate_staggered(R0, V0, DT_FINE, n_fine, field)
    ref_gap = float(np.sqrt(np.mean(np.linalg.norm(rs_r - rs_s, axis=1) ** 2)))

    out = {"meta": {"dt_ref": DT_FINE, "t_final": T_FINAL, "tau": TAU_MAIN,
                    "reference_disagreement_rms_rL": ref_gap},
           "shipped": [], "staggered": []}
    print(f"reference disagreement (shipped vs staggered @ dt_ref) = {ref_gap:.3e} r_L")

    for name, fn in (("shipped", lambda dt, n: integrate_boris(R0, V0, 0.0, dt, n, field)),
                     ("staggered", lambda dt, n: integrate_staggered(R0, V0, dt, n, field))):
        for dt in DT_GRID:
            n = int(round(T_FINAL / dt))
            ts_w = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter(); rs, vs, ts = fn(dt, n)
                ts_w.append(time.perf_counter() - t0)
            traj, en = errors(rs, vs, ts, rs_r, ts_r, E_ref, E0)
            wall = float(np.median(ts_w))
            out[name].append({"dt": dt, "n_steps": n, "wall_s": wall,
                              "us_per_step": 1e6 * wall / n,
                              "pos_err_rms": traj,
                              "energy_err_median_2nd_half": en})
            print(f"{name:10s} dt={dt:6.3f} wall={wall:7.4f}s traj={traj:.4e} en={en:.4e}")

    for name in ("shipped", "staggered"):
        d = [r["dt"] for r in out[name]]
        t = [r["pos_err_rms"] for r in out[name]]
        e = [r["energy_err_median_2nd_half"] for r in out[name]]
        p_all, ci_all = fit_order(d, t)
        # fit restricted to the coarse half, where the reference is clean
        p_c, ci_c = fit_order(d[:5], t[:5])
        p_f, ci_f = fit_order(d[5:], t[5:])
        pe, cie = fit_order(d, e)
        out[name + "_order"] = {
            "traj_all": [p_all, ci_all], "traj_coarse_dt>=0.1": [p_c, ci_c],
            "traj_fine_dt<=0.075": [p_f, ci_f], "energy_all": [pe, cie]}
        print(f"\n{name}: traj order all={p_all:.2f}+-{ci_all:.2f} | "
              f"coarse={p_c:.2f}+-{ci_c:.2f} | fine={p_f:.2f}+-{ci_f:.2f} | "
              f"energy={pe:.2f}+-{cie:.2f}")

    with open(os.path.join(HERE, "convergence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote convergence.json")


if __name__ == "__main__":
    main()
