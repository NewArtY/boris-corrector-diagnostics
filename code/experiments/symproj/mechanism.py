"""Decisive test: is the secular growth caused by the projection, or merely by
being on a DIFFERENT trajectory in a non-stationary field?

Run plain Boris (no network, no projection at all) from an initial position
displaced by the size of the hybrid's own trajectory error, and measure the
energy-error envelope exponent against the same E_phys = E0 exp(-t/tau).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import symproj as S

TWO_PI = 2*np.pi; DT = S.DT_WORK; TAU = S.TAU_MAIN
N = int(round(100000*TWO_PI/DT))
out = {"note": "plain Boris, no network, displaced initial position", "runs": {}}
print(f"{'смещение r0':>14s}{'показатель':>13s}{'E_err(1e5)':>14s}")
for eps in [0.0, 3.5e-3, 1.0e-2, 1.0e-1]:
    d = S.run("boris", TAU, DT, N, base="shipped", n_samples=4000,
              r0=(1.0+eps, 0.0, 0.0))
    env = np.maximum.accumulate(d["env"])
    e = S.envelope_exponent(d["t"], env)
    out["runs"][f"{eps:.1e}"] = {"envelope_exponent": float(e),
                                 "energy_err_max_1e5": float(env[-1])}
    print(f"{eps:>14.1e}{e:>13.3f}{env[-1]:>14.4e}")
json.dump(out, open(os.path.join(HERE,"mechanism.json"),"w"), indent=2)
