"""
horizon.py -- do the classical schemes keep both channels below the signal
at horizons where I1.2 showed the hybrid fails?
==========================================================================
I1.2 established that the hybrid's trajectory advantage inverts at ~101
gyrations and its energy-error envelope grows secularly (exponent 0.977 in a
quasi-static control), while the shipped Boris envelope stays bounded
(exponent 0.000 out to 1e5 gyrations).

This script asks the same of the volume-preserving splittings, which run.py
showed already beat the hybrid on both channels at the working step.

Writes horizon.json. Touches nothing outside this directory.
"""
import os
import sys
import json
import time
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DT_WORK, TAU_MAIN
import schemes as S

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
Q, M = -1.0, 1.0
GYRO = 2.0 * np.pi                      # Omega_c = 1 -> one gyroperiod
DT = DT_WORK


def dop853(field, t_eval, t_end):
    def rhs(t, y):
        r, v = y[:3], y[3:]
        E = np.atleast_1d(field.E(r, t)).ravel()
        B = np.atleast_1d(field.B(r, t)).ravel()
        return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])
    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([R0, V0]),
                    method="DOP853", rtol=1e-11, atol=1e-13, t_eval=t_eval)
    assert sol.success, sol.message
    return sol.y[:3].T, sol.y[3:].T


def envelope_exponent(ts, err, n_bins=24):
    """Power-law exponent of the running maximum of |dE|/E0, fitted on the
    last two decades so that the initial transient does not dominate."""
    m = np.maximum.accumulate(err)
    ok = (ts > ts[-1] * 1e-2) & (m > 0)
    if ok.sum() < 8:
        return float("nan")
    x, y = np.log(ts[ok]), np.log(m[ok])
    return float(np.polyfit(x, y, 1)[0])


def run(name, field, n_steps):
    if name == "shipped":
        return integrate_boris(R0, V0, 0.0, DT, n_steps, field)
    if name == "staggered":
        return S.integrate_staggered(field, R0, V0, DT, n_steps)
    step = {"vps2": S.make_vps2, "vps4": S.make_vps4}[name](field)
    return S.integrate(step, R0, V0, DT, n_steps)


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    out = {"meta": {"dt": DT, "tau": TAU_MAIN,
                    "hybrid_reference": "I1.2: inverts at ~101 gyr, "
                                        "envelope exponent 0.977 quasi-static"},
           "long": {}, "scored": {}}

    # ---- part 1: energy envelope, no reference needed --------------------
    for n_gyr in (1e3, 1e4, 1e5):
        t_end = n_gyr * GYRO
        n = int(round(t_end / DT))
        out["long"][f"{n_gyr:.0e}"] = {}
        for name in ("shipped", "vps2", "vps4"):
            t0 = time.perf_counter()
            rs, vs, ts = run(name, field, n)
            wall = time.perf_counter() - t0
            E = 0.5 * np.sum(vs ** 2, axis=1)
            # physical energy follows the adiabatic invariant: E ~ |B(t)|
            E_phys = E[0] * field.Bz_of_t(ts) / field.Bz_of_t(0.0)
            err = np.abs(E - E_phys) / E[0]
            sig = np.abs(E_phys - E[0]) / E[0]
            expo = envelope_exponent(ts[1:], err[1:])
            rec = {"n_steps": n, "wall_s": wall,
                   "envelope_exponent": expo,
                   "max_energy_err": float(err.max()),
                   "final_energy_err": float(err[-1]),
                   "physical_signal_final": float(sig[-1]),
                   "ratio_signal_over_err_final": float(sig[-1] / max(err[-1], 1e-300))}
            out["long"][f"{n_gyr:.0e}"][name] = rec
            print(f"{n_gyr:.0e} gyr {name:10s} expo={expo:+.3f} "
                  f"maxerr={err.max():.3e} sig={sig[-1]:.3e} "
                  f"ratio={rec['ratio_signal_over_err_final']:.1f} ({wall:.1f}s)")
        print()

    # ---- part 2: trajectory + energy against DOP853 at 1e3 gyrations ------
    n_gyr = 1e3
    t_end = n_gyr * GYRO
    n = int(round(t_end / DT))
    ts = np.linspace(0.0, n * DT, n + 1)
    print(f"building DOP853 reference to {n_gyr:.0e} gyrations ...")
    t0 = time.perf_counter()
    r_ref, v_ref = dop853(field, ts, n * DT)
    print(f"  reference took {time.perf_counter()-t0:.1f}s")
    E_ref = 0.5 * np.sum(v_ref ** 2, axis=1)
    E0 = E_ref[0]
    half = len(ts) // 2
    sig = float(np.median(np.abs(E_ref - E0)[half:] / E0))
    out["meta"]["physical_signal_median_1e3gyr"] = sig
    print(f"physical signal at 1e3 gyr = {sig:.4e}")

    for name in ("shipped", "vps2", "vps4"):
        rs, vs, tt = run(name, field, n)
        E = 0.5 * np.sum(vs ** 2, axis=1)
        e_err = np.abs(E - E_ref) / E0
        pos = np.linalg.norm(rs - r_ref, axis=1)
        rec = {"pos_err_rms": float(np.sqrt(np.mean(pos ** 2))),
               "energy_err_median_2nd_half": float(np.median(e_err[half:])),
               "energy_below_signal_factor":
                   float(sig / max(np.median(e_err[half:]), 1e-300))}
        out["scored"][name] = rec
        print(f"{name:10s} traj={rec['pos_err_rms']:.4e} "
              f"en={rec['energy_err_median_2nd_half']:.4e} "
              f"({rec['energy_below_signal_factor']:.1f}x below signal)")

    with open(os.path.join(HERE, "horizon.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote horizon.json")


if __name__ == "__main__":
    main()
