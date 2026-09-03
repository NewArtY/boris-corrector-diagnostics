"""
harness.py -- shared machinery for the F0.1 adversarial elimination.
====================================================================
Reuses the verified classical schemes from experiments/classical/schemes.py
(orders confirmed there by measurement: vps2 -> 2, vps4 -> 4, gl4 -> 4) and
the same DOP853 reference and flop accounting, so the numbers here sit on the
same axis as I2.2's verdict table.

Rule for a verdict: a setting FAILS if a classical attack reaches comparable
or better accuracy in BOTH channels at comparable or lower flop cost. Flops,
not wall-clock, decide -- I1.1 showed wall-clock on this codebase measures
interpreter overhead on both sides, not arithmetic.
"""
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "classical"))

import schemes as S                                    # noqa: E402

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
Q, M = -1.0, 1.0

T_FINAL = 120.0          # ~19 gyrations, the Article's window
DT_WORK = 0.3            # Omega_c dt = 0.3
TAU_MAIN = 1.2e5

# One hybrid step, measured in I1.1 on the real DefectNet (52 102 params).
HYBRID_FLOPS_PER_STEP = 114091.0


def dop853(field, t_eval, t_end, r0=None, v0=None, rtol=1e-12, atol=1e-14):
    r0 = R0 if r0 is None else r0
    v0 = V0 if v0 is None else v0

    def rhs(t, y):
        r, v = y[:3], y[3:]
        E = np.atleast_1d(field.E(r, t)).ravel()
        B = np.atleast_1d(field.B(r, t)).ravel()
        return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])

    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([r0, v0]),
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
    assert sol.success, sol.message
    return sol.y[:3].T, sol.y[3:].T


_BUILDERS = {"vps2": S.make_vps2, "vps4": S.make_vps4,
             "imr": S.make_imr, "gl4": S.make_gl4}


def run_scheme(name, field, dt, n_steps, r0=None, v0=None):
    """Integrate with a classical scheme; return states and total flops."""
    r0 = R0 if r0 is None else r0
    v0 = V0 if v0 is None else v0
    if name == "staggered":
        rs, vs, ts = S.integrate_staggered(field, r0, v0, dt, n_steps)
        return rs, vs, ts, S.FLOPS_PER_STEP["staggered"] * n_steps
    step = _BUILDERS[name](field)
    rs, vs, ts = S.integrate(step, r0, v0, dt, n_steps)
    if name in ("imr", "gl4"):
        st = step.stats
        mean_it = st["iters"] / max(st["steps"], 1)
        per = S.flops_imr(mean_it) if name == "imr" else S.flops_gl4(mean_it)
    else:
        per = S.FLOPS_PER_STEP[name]
    return rs, vs, ts, per * n_steps


def score(rs, vs, ts, r_ref, v_ref):
    """Two-channel score against the reference, same definitions as I2.2."""
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


def physical_signal(v_ref):
    """Median relative energy change over the second half -- the same quantity
    diagnostics/eval_corrector.py calls physical_signal_median."""
    E = 0.5 * np.sum(v_ref ** 2, axis=1)
    half = len(E) // 2
    return float(np.median(np.abs(E - E[0])[half:] / E[0]))


def verdict(attack, hybrid_like, signal):
    """FAILS when the classical attack is no worse in both channels and no
    more expensive. `hybrid_like` is the accuracy a learned method would have
    to reach; at F0 we use the measured hybrid figures as the stand-in."""
    better_pos = attack["pos_err_rms"] <= hybrid_like["pos_err_rms"]
    better_e = (attack["energy_err_median_2nd_half"]
                <= hybrid_like["energy_err_median_2nd_half"])
    cheaper = attack["flops"] <= hybrid_like["flops"]
    return {"verdict": "FAILS" if (better_pos and better_e and cheaper)
            else "SURVIVES",
            "beats_on_position": bool(better_pos),
            "beats_on_energy": bool(better_e),
            "cheaper": bool(cheaper),
            "signal": signal}
