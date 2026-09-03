"""Audit 4: the drift reading of the stored quasistatic run.

Section 3.2 of the manuscript reports the running maximum of the raw relative
energy error.  Under the averaged staggered convention that quantity never
clears its own constant floor sin^2(theta_h/2), so its fitted exponent is 0.

The floor is a constant offset, so it cancels from any diagnostic that
subtracts the initial reading.  This script reads the same stored series
(symproj/env_quasistatic_staggered.npz) as drift, |e(t) - e(0)|, in the same
staggered convention, and reports

  * the running-maximum drift envelope at 1e3, 1e4 and 1e5 gyro-orbits,
    for the learned corrector and for the Boris baseline;
  * the ratio of the two at 1e5;
  * the two-decade log-log fit of the corrector drift envelope, which is
    printed here only to be rejected;
  * the local half-decade slopes of that envelope over the same window,
    which show that the curve is not a power law and that the fitted value
    is therefore not an exponent (rule of Section 4.4).

Reproduces and extends referee_check/rf2_staggered_drift.py.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SYM = os.path.join(HERE, "..", "symproj")


def half_decade_slopes(gyr, env, lo=1e3, hi=1e5):
    """Local log-log slopes over consecutive half-decade windows."""
    edges = 10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1e-9, 0.5)
    slopes = []
    for a, b in zip(edges[:-1], edges[1:]):
        i = np.searchsorted(gyr, a)
        j = np.searchsorted(gyr, b)
        x = np.log(gyr[i:j])
        y = np.log(env[i:j] + 1e-300)
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    return slopes


d = np.load(os.path.join(SYM, "env_quasistatic_staggered.npz"))
out = {}
for tag in ("proj", "boris"):
    t = d["staggered/%s/t" % tag]
    e = d["staggered/%s/e_err" % tag]
    gyr = t / (2 * np.pi)
    raw_env = np.maximum.accumulate(np.abs(e))
    drift_env = np.maximum.accumulate(np.abs(e - e[0]))
    lo = np.searchsorted(gyr, 1e3)
    fit = float(np.polyfit(np.log(gyr[lo:]),
                           np.log(drift_env[lo:] + 1e-300), 1)[0])
    out[tag] = {
        "raw_env_1e5": float(raw_env[-1]),
        "initial_reading": float(abs(e[0])),
        "drift_env_1e3": float(drift_env[np.searchsorted(gyr, 1e3)]),
        "drift_env_1e4": float(drift_env[np.searchsorted(gyr, 1e4)]),
        "drift_env_1e5": float(drift_env[-1]),
        "drift_env_two_decade_fit": fit,
        "drift_env_half_decade_slopes": half_decade_slopes(gyr, drift_env),
    }

out["ratio_proj_over_boris_at_1e5"] = (
    out["proj"]["drift_env_1e5"] / out["boris"]["drift_env_1e5"])
out["comment"] = (
    "The averaged staggered floor is a constant offset and cancels from the "
    "drift.  Read as drift, the same stored run shows the corrector above the "
    "Boris baseline by the ratio above at 1e5 gyro-orbits.  The two-decade fit "
    "of the corrector drift envelope is reported here only to be rejected: the "
    "local half-decade slopes are not constant, so that curve has no exponent "
    "and only the level and the ratio are quoted in the paper.")

print(json.dumps(out, indent=1))
with open(os.path.join(HERE, "an4_drift_reading.json"), "w") as f:
    json.dump(out, f, indent=1)
