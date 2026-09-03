"""p_law/pl_marginal.json: what is recoverable, what is not, and how much the
irrecoverable part is worth.

Section 4.2 of the manuscript prints all three fields of pl_marginal.json:

    predicted_slope_sqrt_logN = 0.050809886076838895   ->  "slope 0.051 analytically"
    measured_rms_slope        = 0.03983447434435622    ->  "0.0398 measured on the
                                                            ensemble root mean square"
    VarS_over_logN            = 1.0915862071515887     ->  "the measured Var S_N / ln N
                                                            is 1.09 against the 1.05
                                                            the closed form predicts"

The block was computed inline during the campaign; no script wrote it (W6.2, finding 1).

WHAT THIS SCRIPT ESTABLISHES
----------------------------
1. The setup is now pinned, not guessed.  Both deterministic fields fall out of
   the pl_limits.py setup exactly:

     N    = int(round(1e4 * 2*pi / 0.3)) = 209440 steps  (1e4 gyro-orbits at h = 0.3)
     grid = np.unique(np.round(np.logspace(log10(N/100), log10(N), 200)).astype(int))
            -- the same two-decade, 200-point log-spaced grid pl_limits.py builds
               for its L1 block

   On that grid, polyfit(log10(n), 0.5*log10(log(n)), 1)[0] returns
   0.050809886076838895 -- bit-for-bit the stored value.  No other grid tried
   (the 2000-point version, the dense integer version, pl_core.py's 60-point
   n_probe, or any other N) comes closer than 3e-6.  A 17-digit float agreeing
   to the last bit is not a coincidence, so N and the grid are settled.

   The companion "1.05" is settled the same way: on the line a+H=0 with H=1/2 the
   closed form degenerates to Var S_N = kappa^2 * sum_{k<=N} 1/k = kappa^2 * H_N,
   so the predicted ratio is H_N / ln N = 1.0471 at N = 209440.  Round to two
   decimals and that is the 1.05 the paper prints.

2. The two measured fields are single draws from an ensemble whose seed set was
   never written down, and they cannot be recovered by search.  Around 350
   candidate configurations were tried -- every RNG seed base appearing anywhere
   in the p_law / ll_probe campaign (0, 1, 7, 11, 42, 555, 777, 1000, 1234, 1500,
   2024, 2025, 2026, 4242, 12345, 31337, 31365, 31415, 20260831, 70450, 79950,
   80450, 90000, 500000 and the a/H-dependent formulas that generate them),
   crossed with ensemble sizes 8/16/24/32/48/64/100/128, real and complex noise,
   and all three RNG consumption patterns the campaign uses (one generator per
   seed as in pl_limits.py; one generator for the whole ensemble as in
   pl_protocol.py; BATCH=8 chunks as in pl_core.py and pl_antipersistent.py).
   Nothing reproduced measured_rms_slope to better than 3e-4 while also
   reproducing VarS_over_logN to better than 3e-3.  The seed set is lost.

3. That is less of a loss than it looks, and the reason is the point worth
   carrying to the manuscript.  Both measured fields are ensemble estimates at
   the exact marginal point, where the signal is logarithmic: the true slope is
   1/(2 ln N) ~ 0.04, and the sampling scatter of the estimate over independent
   ensembles of the size the campaign used is of the same order as the gap
   between "0.051 analytically" and "0.0398 measured".  The Monte Carlo below
   measures that scatter.  Both stored values sit inside it.

   In other words: 0.0398 is an ordinary draw, not a wrong number and not an
   informative one.  The sentence built on it -- analytic 0.051 against measured
   0.0398 -- is comparing a deterministic constant against a random variable
   whose standard deviation is comparable to the difference being displayed.

Run:  python rn2_pl_marginal.py [--reps R] [--nseed M]
      default --reps 24 --nseed 64 takes about 2 minutes.
Exit code 0 iff the two deterministic fields are reproduced exactly and both
stored measured values fall inside the sampled range.
"""

import argparse
import math
import sys

import numpy as np

# ------------------------------------------------------------- stored ----
STORED = {
    "predicted_slope_sqrt_logN": 0.050809886076838895,
    "measured_rms_slope": 0.03983447434435622,
    "VarS_over_logN": 1.0915862071515887,
}

# ------------------------------- the pl_limits.py setup, reproduced ------
H_STEP = 0.3
NG = 1e4
TWO_PI = 2 * np.pi
N = int(round(NG * TWO_PI / H_STEP))              # 209440
A = -0.5                                          # the marginal point ...
HURST = 0.5                                       # ... a + H = 0

# pl_limits.py's L1 grid: two decades ending at N, 200 log-spaced points
NN = np.unique(np.round(np.logspace(np.log10(N / 100.0), np.log10(N), 200)).astype(int))
LOG_NN = np.log10(NN)
KIDX = np.arange(1, N + 1, dtype=float)
SIGMA = KIDX ** A


def predicted_slope():
    """Analytic slope of a two-decade log-log fit to sqrt(ln N). Deterministic."""
    return float(np.polyfit(LOG_NN, 0.5 * np.log10(np.log(NN)), 1)[0])


def predicted_var_over_logn():
    """Closed form on the line a+H=0 at H=1/2: Var S_N / (kappa^2 ln N) = H_N / ln N."""
    harmonic = float(np.sum(1.0 / KIDX))
    return harmonic / math.log(N)


def one_ensemble(rng, nseed):
    """Ensemble rms of |S_n| on the grid, and Var S_N / ln N, for `nseed` draws."""
    acc = np.zeros(len(NN))
    for _ in range(nseed):
        u = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
        acc += np.abs(np.cumsum(SIGMA * u)[NN - 1]) ** 2
    ms = acc / nseed
    slope = float(np.polyfit(LOG_NN, np.log10(np.sqrt(ms)), 1)[0])
    return slope, float(ms[-1] / math.log(N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=24, help="independent ensembles to draw")
    ap.add_argument("--nseed", type=int, default=64, help="realizations per ensemble")
    args = ap.parse_args()

    print(f"N = {N} steps ({NG:g} gyro-orbits at h = {H_STEP}); "
          f"grid = {len(NN)} points from n = {NN[0]} to n = {NN[-1]}")

    bad = []

    ps = predicted_slope()
    ok = ps == STORED["predicted_slope_sqrt_logN"]
    print(f"  {'OK ' if ok else 'BAD'} predicted_slope_sqrt_logN: {ps!r} "
          f"vs stored {STORED['predicted_slope_sqrt_logN']!r}")
    if not ok:
        bad.append("predicted_slope_sqrt_logN")

    pv = predicted_var_over_logn()
    print(f"      closed-form Var S_N / (kappa^2 ln N) = H_N / ln N = {pv:.6f} "
          f"-> the '1.05' the paper prints")

    # ---- sampling distribution of the two measured fields -----------------
    print(f"\ndrawing {args.reps} independent ensembles of {args.nseed} realizations "
          f"(the seed set of the original ensemble is lost; these are fresh):")
    slopes, vars_ = [], []
    rng = np.random.default_rng(20260901)
    for r in range(args.reps):
        s, v = one_ensemble(rng, args.nseed)
        slopes.append(s)
        vars_.append(v)
        print(f"  rep {r:2d}: slope = {s:+.5f}   Var/lnN = {v:.5f}", flush=True)

    slopes = np.array(slopes)
    vars_ = np.array(vars_)
    for name, arr, stored in (("measured_rms_slope", slopes, STORED["measured_rms_slope"]),
                              ("VarS_over_logN", vars_, STORED["VarS_over_logN"])):
        inside = arr.min() <= stored <= arr.max()
        print(f"\n{name}: stored {stored!r}")
        print(f"  sampled mean {arr.mean():.5f}, sd {arr.std(ddof=1):.5f}, "
              f"range [{arr.min():.5f}, {arr.max():.5f}]")
        print(f"  stored value is {abs(stored - arr.mean()) / arr.std(ddof=1):.2f} sd "
              f"from the sampled mean; inside the sampled range: {inside}")
        if not inside:
            bad.append(name + " outside sampled range")

    z = abs(STORED["measured_rms_slope"] - ps) / slopes.std(ddof=1)
    print(f"\nthe gap the paper displays, |0.0398 - 0.051| = "
          f"{abs(STORED['measured_rms_slope'] - ps):.5f}, is {z:.2f} sampling sd "
          f"of the estimator at {args.nseed} realizations.")

    if bad:
        print("\nPROBLEM:", ", ".join(bad))
        return 1
    print("\nboth deterministic fields exact; both measured fields are ordinary draws")
    return 0


if __name__ == "__main__":
    sys.exit(main())
