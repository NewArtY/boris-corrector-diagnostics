"""AN3: the two closed-form / arithmetic numbers of Secs. 4.2 and 5 that no
shipped JSON contained, written to a file so they meet the paper's own
standard ("every measured number is written by a script into a data file").

(1) Sec. 5: the erosion threshold c/Lambda at which the plateau
    kappa/[1 - rho - c h] is lost, for the generic rapidity configuration of
    `ll_probe` (alpha = 1, eps = 0.1, Lambda = 2(alpha-eps) = 1.8, h = 0.05,
    RK4).  The manuscript prints 0.956.

(2) Sec. 4.2: the horizon of the 42-cell grid, N = 2^19 steps at h = 0.3, in
    gyro-orbits.  The manuscript printed 25,046 until wave W5.3;
    `p_law/pl_core.json` ("grid_setup"/"gyros") and `p_law/pl_protocol.json`
    ("full_run_gyros") both store 25,032.908, so 25,033 is what it prints now.
    The 25,046 came from a comment in `p_law/pl_core.py`, which was wrong and
    has been corrected; the script itself always wrote the right number.
    `manuscript_prints` below is kept at the withdrawn value on purpose, as
    the record of what was corrected and against what.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "..", "ll_probe", "prereg.json")

G = json.load(open(PREREG, encoding="utf-8"))["params"]["generic"]
AL, EP, LAM = G["alpha"], G["eps"], G["Lambda"]
H_LL = 0.05


def R_rk4(z):
    return 1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24


rho = R_rk4(-LAM * H_LL)
c_over_Lambda = (1.0 - rho) / (LAM * H_LL)

# direct confirmation: bisect on the actual recurrence
def plateau(cf, n=400000, kappa=1e-6):
    c = LAM * cf
    d = 0.0
    for _ in range(n):
        d = (rho + c * H_LL) * d + kappa
        if not math.isfinite(d) or abs(d) > 1e6:
            return math.inf
    return d


lo, hi = 0.9, 1.1
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if math.isfinite(plateau(mid, n=200000)):
        lo = mid
    else:
        hi = mid

# arithmetic horizon
N = 1 << 19
H_STEP = 0.3
gyros = N * H_STEP / (2 * np.pi)

out = {
 "S5_erosion_threshold": {
  "alpha": AL, "eps": EP, "Lambda": LAM, "h": H_LL, "scheme": "RK4",
  "rho_=_R_rk4(-Lambda h)": float(rho),
  "c_over_Lambda_closed_form_(1-rho)/(Lambda h)": float(c_over_Lambda),
  "c_over_Lambda_bisection_on_the_recurrence": float(0.5 * (lo + hi)),
  "manuscript_prints": 0.956,
 },
 "S42_grid_horizon": {
  "N_steps": N, "h": H_STEP,
  "gyro_orbits_N_h_over_2pi": float(gyros),
  "shipped_pl_core_grid_setup_gyros": 25032.908041129085,
  "shipped_pl_protocol_full_run_gyros": 25032.908041129085,
  "shipped_pl_protocol_short_run_gyros": 1564.5567525705678,
  "manuscript_prints": 25046,
  "correct_rounded": int(round(gyros)),
 },
}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "an3_derived.json"), "w"), indent=1)
