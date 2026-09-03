"""Recovers `F6b_refit_clean_window` in ll_probe/results3.json from scratch.

Section 6 of the manuscript quotes two numbers from that block -- the measured
energy-channel decay rates 1.7954 (explicit Euler) and 1.4286 (implicit Euler),
compared with the predicted min(Lambda, Lambda_h) = 1.8000 and 1.4393.  The block
was computed inline during the campaign; no script in the bundle wrote it, and
the W6.2 audit listed it as one of three numbers with a file but no generator.

This script closes that gap.  It re-integrates the same Landau-Lifshitz-type
scalar model as ll_probe/fix_f3_f6.py, reproduces the neighbouring `F6b` block
(which fix_f3_f6.py does write) as a control, and then reproduces all six fields
of `F6b_refit_clean_window` to the last bit.

THE RECIPE, AND HOW IT DIFFERS FROM THE FAILED RECONSTRUCTION
------------------------------------------------------------
Everything up to the fit is identical to fix_f3_f6.py's F6b section:

    reference   : RK4 on theta' = alpha*tanh(theta) - eps*sinh(theta)*cosh(theta)
                  with hf = h/4000 = 7.5e-05, sampled back onto the coarse grid
    schemes     : explicit Euler and implicit Euler (Newton), h = 0.3, T = 30
    observable  : gamma = cosh(theta)
    deviation   : dev = |gamma_scheme - gamma_reference| / gamma_reference[0]

Only the fit window differs.  fix_f3_f6.py's `F6b` uses

    (dev > 1e-10) & (dev < 1e-2) & (t > 0)

The refit uses a *narrower* band at the top and a *deeper* one at the bottom:

    (dev > 1e-11) & (dev < 1e-3) & (t >= t0),  t0 in {4, 6, 8}

Two changes, each doing a different job:

  * upper bound 1e-2 -> 1e-3.  dev is not monotone: it rises to ~1e-1, changes
    sign near t = 3.0 (euler: dev dips to 2.6e-03 there) and only then settles
    onto the asymptotic exp(-Lambda_min t).  The 1e-2 band still admits the
    post-sign-change hump (t = 3.3 .. 5.4), which is pre-asymptotic and biases
    the slope low -- that is exactly why `F6b` reports 1.4926 / 1.2573 against a
    prediction of 1.8000 / 1.4393.  At 1e-3 the first admitted sample is
    t = 6.6 (euler) and t = 7.5 (ieuler), safely inside the clean decay.
  * lower bound 1e-10 -> 1e-11.  The reference grid's own floor for euler is
    2.36e-12 (dev goes flat from t = 17.7 on).  1e-11 buys one more decade of
    lever arm while still staying a factor ~4 above that floor; the last
    admitted sample is t = 16.5 (euler) / t = 19.8 (ieuler).

Consequence, and the fingerprint that identified the window: for both schemes
the first sample admitted by the dev band already lies beyond t = 6, so the
t0 = 4 and t0 = 6 fits use the *same* point set and are bit-identical.  The
stored JSON shows exactly that (fit_from_t4 == fit_from_t6 to all 17 digits) --
no window whose start is actually set by t0 can do that.

The earlier reconstruction attempt kept fix_f3_f6.py's own band (1e-10, 1e-2)
and only added t >= t0.  That returns 1.7875 / 1.4073 at t0 = 4, i.e. the
"1.787 / 1.407" reported in W6_2_bundle.md.  The 0.5 % / 1.5 % residual was
therefore not fit noise: it was the pre-asymptotic hump between dev = 1e-3 and
dev = 1e-2 still sitting in the window.

Run:  python rn1_f6b_refit.py            (about 40 s; the fine reference is 400k RK4 steps)
Exit code 0 iff every stored field is reproduced exactly.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.normpath(os.path.join(HERE, "..", "ll_probe"))

# ---------------------------------------------------------------- model ----
# Identical to ll_probe/fix_f3_f6.py; parameters come from the same prereg.json.
G = json.load(open(os.path.join(LL, "prereg.json"), encoding="utf-8"))["params"]["generic"]
AL, EP, THS, LAM = G["alpha"], G["eps"], G["theta_star"], G["Lambda"]


def make(al, ep):
    def f(t):
        return al * math.tanh(t) - ep * math.sinh(t) * math.cosh(t)

    def fp(t):
        ch = math.cosh(t)
        return al / (ch * ch) - ep * math.cosh(2 * t)

    return f, fp


def rk4(f, th, h):
    k1 = f(th)
    k2 = f(th + .5 * h * k1)
    k3 = f(th + .5 * h * k2)
    k4 = f(th + h * k3)
    return th + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.


# --------------------------------------------------------- the campaign ----
H, T = 0.3, 30.0
REFINE = 4000            # reference step = H / REFINE
THETA0 = 0.3

# fit bands
BAND_F6B = (1e-10, 1e-2)          # what fix_f3_f6.py uses for the `F6b` block
BAND_REFIT = (1e-11, 1e-3)        # what the clean-window refit uses
T0S = (4, 6, 8)


def trajectories():
    """Return {scheme: (t, dev)} on the coarse grid."""
    f, fp = make(AL, EP)

    hf = H / REFINE
    nf = int(T / hf) + 1
    gf = np.empty(nf)
    th = THETA0
    gf[0] = math.cosh(th)
    for i in range(1, nf):
        th = rk4(f, th, hf)
        gf[i] = math.cosh(th)

    n = int(T / H)
    tn = np.arange(n + 1) * H
    gref = gf[np.arange(n + 1) * REFINE]

    out = {}
    for nm in ("euler", "ieuler"):
        th = THETA0
        gn = np.empty(n + 1)
        gn[0] = math.cosh(th)
        for i in range(1, n + 1):
            if nm == "euler":
                th = th + H * f(th)
            else:
                x = th + H * f(th)
                for _ in range(60):
                    dd = (x - th - H * f(x)) / (1 - H * fp(x))
                    x -= dd
                    if abs(dd) < 1e-16 * max(1., abs(x)):
                        break
                th = x
            gn[i] = math.cosh(th)
        out[nm] = (tn, np.abs(gn - gref) / gref[0])
    return out


def fit(t, dev, band, t0=None):
    lo, hi = band
    m = (dev > lo) & (dev < hi)
    m &= (t > 0) if t0 is None else (t >= t0)
    rate = -np.polyfit(t[m], np.log(dev[m]), 1)[0]
    return float(rate), int(m.sum()), [float(t[m][0]), float(t[m][-1])]


def main():
    traj = trajectories()

    f6b, refit = {}, {}
    for nm, (t, dev) in traj.items():
        r, n, w = fit(t, dev, BAND_F6B)
        f6b[nm] = {"fitted_decay_rate": r, "n_points": n, "t_window": w}
        refit[nm] = {f"fit_from_t{t0}": fit(t, dev, BAND_REFIT, t0)[0] for t0 in T0S}

    # predictions, for context (these are already written by fix_f3_f6.py)
    lh = {"euler": -math.log(abs(1 - LAM * H)) / H, "ieuler": math.log(1 + LAM * H) / H}
    print("prediction min(Lambda, Lambda_h):",
          {k: min(LAM, v) for k, v in lh.items()})
    print("F6b                  :", json.dumps(f6b, indent=1))
    print("F6b_refit_clean_window:", json.dumps(refit, indent=1))

    stored = json.load(open(os.path.join(LL, "results3.json"), encoding="utf-8"))
    bad = []
    for block, got in (("F6b", f6b), ("F6b_refit_clean_window", refit)):
        want = stored[block]
        for scheme in want:
            for key in want[scheme]:
                a, b = got[scheme][key], want[scheme][key]
                ok = (a == b) if not isinstance(a, list) else (a == b)
                print(f"  {'OK ' if ok else 'BAD'} {block}.{scheme}.{key}: {a!r} vs stored {b!r}")
                if not ok:
                    bad.append(f"{block}.{scheme}.{key}")

    # the two numbers Section 6 prints
    for scheme, printed in (("euler", 1.7954), ("ieuler", 1.4286)):
        v = refit[scheme]["fit_from_t4"]
        ok = round(v, 4) == printed
        print(f"  {'OK ' if ok else 'BAD'} Section 6 prints {printed} <- {v!r}")
        if not ok:
            bad.append(f"section6.{scheme}")

    # what the failed reconstruction did, for the record
    print("\nfor reference, the band that does NOT reproduce them (F6b band + t>=4):")
    for nm, (t, dev) in traj.items():
        print(f"  {nm}: {fit(t, dev, BAND_F6B, 4)[0]:.4f}")

    if bad:
        print("\nMISMATCH:", ", ".join(bad))
        return 1
    print("\nall stored fields reproduced exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
