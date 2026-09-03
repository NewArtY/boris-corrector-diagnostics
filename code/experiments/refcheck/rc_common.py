"""rc_common.py -- the closed-form ruler for the Section 7 horizon numbers.

WHY THIS DIRECTORY EXISTS
-------------------------
Every trajectory number of Section 7 is measured against a *Boris* run at
h/150.  The corrector was *trained* against a Boris run at h/150
(`training/train_corrector_b4.py`: target = (r_ref, v_ref)_{n+1} -
BorisStep(r_n, v_n; dt_work), DT_FINE = DT_WORK / 150).  Ruler and pupil are
the same object, so the reported advantage is an advantage over a ruler the
corrector was fitted to, and the ruler's own error cannot be subtracted in
quadrature -- quadrature subtraction assumes independence, and independence is
exactly what is missing.

W16 measured the ruler: over the window of Table~\\ref{tab:family} the h/150
Boris propagation is 1.3959e-3 Larmor radii from the truth
(`experiments/seeds/sd5_summary.json:reference_floor`).  That is not far below
the corrector's reported 3.47e-3.  This directory replaces the ruler and
re-reads every number.

THE REPLACEMENT RULER
---------------------
Not Boris at h/1500 -- a finer ruler of the same family inherits the same
first-order position drift and is still not independent.  The closed form.

In the Larmor frame the transverse motion of B4 obeys
zeta'' + (B_z^2/4) zeta = 0 with B_z = B_0 e^{-t/tau}; the substitution
s = (B_0 tau / 2) e^{-t/tau} turns it into Bessel's equation of order zero.
The solution is implemented in `../spectral/sw_common.py` and is **reused
here, not rewritten**: `exact_from_basis` is imported and called verbatim.
`sw1_reference.py` has already priced it -- initial-condition residual
1.9e-16, float64 reconstruction 4.4e-16 against the same closed form carried
end to end in mpmath, and identical at 40 and 60 digits.

It gives r(t) and v(t), so it covers all three quantities the horizon
experiments read: position, energy (E = |v|^2/2) and the adiabatic invariant
(mu = (E_perp/B) normalised).  The check is in `rc1_calibration.py`.

THE ONE THING THAT IS NEW HERE
------------------------------
`sw_common.bessel_basis` evaluates J_0, Y_0, J_1, Y_1 in mpmath at 40 digits.
That costs 2.7 ms per sample.  The horizon grids are 20944, 209440 and 2094395
samples long, so the committed basis would cost 1.6 hours for the longest grid
alone, and W18 needs several of them.

`fast_basis` below evaluates the same four functions with `scipy.special` in
float64 and forms the phase with `expm1` so that theta = s_0 (1 - e^{-t/tau})
does not lose its leading digits at small t.  It is **validated against the
committed mpmath basis on every grid it is used on** (`rc1_calibration.py`,
`basis_agreement`), and the agreement is reported beside every number that
depends on it.  The residual is the float64 representation of s itself:
s(0) = 6e4 carries an absolute error of 6e-12, and |J_1| <= sqrt(2/(pi s))
turns that into 2e-14 in the basis and ~1e-11 in the reconstructed position --
eight orders below the 3.5e-3 the measurement is about, and eleven orders
below the 1.4e-3 error of the ruler being replaced.

Nothing else in the bundle is modified.  This directory only reads
`../horizon/`, `../seeds/`, `../stats/`, `../spectral/` and `checkpoints/`.

COST
----
In flops, on the model of Section 9 of the manuscript and of
`../classical/schemes.py`, reused unchanged: one flop per arithmetic
operation, `FLOP_TRANSCENDENTAL` = 20 per transcendental.  `flops_reference`
and `flops_closed_form` below.

SEEDS
-----
**This directory draws nothing.**  One initial condition, fixed schemes,
committed checkpoints, deterministic throughout.  The number of random draws
declared before the first run is **zero**, and it is asserted at the end of
every script by `assert_no_draws()`.  Should a later extension need seeds,
the block reserved for W18 -- declared here before any code was run and
verified free by the audit in `rc0_seed_audit.py` -- is

    18_000_000 .. 18_999_999

Occupied blocks at the time of writing (see `rc0_seed_audit.py`):
    <= 5e5        the bundle, including the committed corrector
    9.0e6-9.8e6   W9.1 external architectures
    11.0e6-11.8e6 W10 SympMat
    11.0e6-14.0e6 W12 HPO  (overlaps W10 and W13/W11, see the audit)
    13.0e6        W13 spectral, W11 spectrum
    14.0e6        W14 map (and W15 gtable through it)
    16.0e6+16     W16 seeds
    20260830/31   I1.3 and the p-law probes
"""
import math
import os
import sys

import numpy as np
from scipy.special import j0 as _sj0, j1 as _sj1, y0 as _sy0, y1 as _sy1

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, EXP,
           os.path.join(EXP, "external_arch"),
           os.path.join(EXP, "classical"),
           os.path.join(EXP, "spectral"),
           os.path.join(EXP, "horizon"),
           os.path.join(EXP, "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sw_common as SW                                          # noqa: E402
from ea_common import check_or_write, FLOP_TRANSCENDENTAL       # noqa: E402
import schemes as SCH                                           # noqa: E402

# ------------------------------------------------------------------ setup --
# Every constant is imported, none is chosen here.
from training.train_corrector_b4 import (                       # noqa: E402
    DT_WORK, DT_FINE, T_FINAL, TAU_MAIN)

TWO_PI = 2.0 * np.pi
B0 = SW.B0                 # 1.0
Q, M = SW.Q, SW.M          # -1.0, 1.0
TAU = TAU_MAIN             # 1.2e5 -- identical to sw_common.TAU (asserted below)
DT = DT_WORK               # 0.3
REFINE = int(round(DT_WORK / DT_FINE))          # 150
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])

assert abs(SW.TAU - TAU) < 1e-9, "spectral and horizon disagree about tau"
assert abs(SW.DT - DT) < 1e-15, "spectral and horizon disagree about the step"
assert REFINE == 150, "DT_FINE is no longer DT_WORK/150"

#: the Larmor radius of the initial condition, rho = m|v_perp| / |q| B_0 = 1.
#: Every trajectory error in this directory is already in Larmor radii.
R_LARMOR = 1.0

OUT_DIR = os.environ.get("RC_OUT", HERE)


def outpath(name):
    return os.path.join(OUT_DIR, name)


# --------------------------------------------------------- the closed form --
def fast_basis(ts, tau=TAU, b0=B0):
    """The Bessel basis of `sw_common.bessel_basis`, in float64.

    Same keys, same meaning, same convention; only the arithmetic is
    different.  See the module docstring for why, and `rc1_calibration.py`
    for the agreement against the committed mpmath basis on the grids this
    is actually used on.
    """
    ts = np.asarray(ts, dtype=float)
    s0 = b0 * tau / 2.0
    x = ts / tau
    s = s0 * np.exp(-x)
    # theta = s0 (1 - e^{-t/tau}) reduced mod 2 pi.  expm1 keeps the leading
    # digits at small t, where 1 - e^{-x} cancels.
    theta = np.mod(-s0 * np.expm1(-x), 2.0 * np.pi)
    return {"t": ts, "s": s, "s0": float(s0), "theta_mod2pi": theta,
            "j0": _sj0(s), "y0": _sy0(s), "j1": _sj1(s), "y1": _sy1(s),
            "tau": float(tau), "dps": "float64"}


def closed_form(ts, r0=R0, v0=V0, tau=TAU):
    """Exact (r, v) on the grid `ts`.  `exact_from_basis` is reused verbatim.

    `exact_from_basis` fixes its two integration constants at the first sample
    and therefore asserts that the grid starts at t = 0.  A grid that does not
    (the horizon scripts index their output from the first *step*, not from the
    initial condition) gets t = 0 prepended here and dropped again, so that the
    constants are still fixed at the initial condition and nothing about the
    committed routine changes.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.size and ts[0] != 0.0:
        R, V = SW.exact_from_basis(
            fast_basis(np.concatenate([[0.0], ts]), tau=tau), r0, v0)
        return R[1:], V[1:]
    return SW.exact_from_basis(fast_basis(ts, tau=tau), r0, v0)


def basis_agreement(ts, r0=R0, v0=V0, tau=TAU, n_spot=61):
    """Price `fast_basis` against the committed mpmath basis on this grid.

    Returns the worst absolute difference in the reconstructed position and
    velocity over `n_spot` samples spread across the whole grid.  This is the
    number that has to be small next to whatever the grid is used to measure.
    """
    ts = np.asarray(ts, dtype=float)
    idx = np.unique(np.linspace(0, len(ts) - 1, n_spot).astype(int))
    sub = ts[idx]
    r_fast, v_fast = SW.exact_from_basis(fast_basis(sub, tau=tau), r0, v0)
    r_mp, v_mp = SW.exact_from_basis(SW.bessel_basis(sub, tau=tau), r0, v0)
    return {"n_spot": int(len(idx)), "t_max": float(sub[-1]),
            "position_max_abs": float(np.abs(r_fast - r_mp).max()),
            "velocity_max_abs": float(np.abs(v_fast - v_mp).max())}


# ------------------------------------------------------- derived readouts --
def energy(V):
    return 0.5 * np.sum(np.asarray(V) ** 2, axis=-1)


def bz(ts, tau=TAU, b0=B0):
    return b0 * np.exp(-np.asarray(ts, dtype=float) / tau)


def mu_error(V, ts, V_ref, ts_ref=None, tau=TAU):
    """|mu/mu_0 - 1| with mu = E/B, the convention of ../horizon/fast.py.

    `fast.py` forms mu from the *total* energy and the instantaneous B, and
    normalises by its own initial value; the only thing that changes here is
    which E_0 and which B the comparison is against -- nothing, since both are
    exact.  Kept as a function so that the reference version and the run
    version are formed by the same code.
    """
    ts = np.asarray(ts, dtype=float)
    E = energy(V)
    E0 = energy(np.asarray(V_ref)[0]) if ts_ref is None else energy(
        np.asarray(V_ref)[0])
    return np.abs((E / bz(ts, tau)) / (E0 / bz(0.0, tau)) - 1.0)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def running_rms(err):
    """Cumulative rms over the first k samples, k = 1..n -- crossover.py's."""
    err = np.asarray(err, dtype=float)
    return np.sqrt(np.cumsum(err ** 2) / np.arange(1, len(err) + 1))


def first_crossing(x, level, grid):
    """First point of `grid` at which `x` crosses `level` from below."""
    i = np.where(np.asarray(x) > level)[0]
    return float(grid[i[0]]) if len(i) else None


def first_below(x, level, grid):
    i = np.where(np.asarray(x) < level)[0]
    return float(grid[i[0]]) if len(i) else None


# ------------------------------------------------------------ flop model ---
#: the committed per-step counts, imported, not chosen.  `schemes.py` counts
#: the raw arithmetic of the shipped Boris step at 93 and notes in its own
#: docstring that `experiments/cost` -- the accounting \ref{sec:app_setups}
#: prints and the rest of the paper uses -- carries it at 113 with the margin.
#: 113 is used here, as `../spectral/sw_common.py:FLOPS_BORIS` does, so that
#: this directory's costs sit on the manuscript's scale; the raw count is kept
#: beside it so the choice is visible rather than buried.
FLOPS_BORIS_STEP = SW.FLOPS_BORIS                       # 113
FLOPS_BORIS_STEP_RAW = SCH.FLOPS_PER_STEP["shipped"]    # 93
FLOPS_CORRECTOR_STEP = SW.FLOPS_CORRECTOR               # 114091


def flops_boris_reference(n_out, refine=REFINE):
    """A Boris ruler at h/refine, per `n_out` output samples."""
    return float(n_out) * refine * FLOPS_BORIS_STEP


def flops_closed_form(n_out):
    """The closed form, per `n_out` output samples.

    Per sample: exp, expm1, fmod, J0, Y0, J1, Y1, sin, cos -- nine
    transcendentals at FLOP_TRANSCENDENTAL each -- plus the reconstruction,
    which is two real linear combinations of the basis for zeta, two for
    dzeta, one complex rotation each for z and w, and the parallel component:
    46 arithmetic operations, counted from `sw_common.exact_from_basis` line
    by line.  The two-by-two solve is done once for the whole grid and is not
    counted per sample.
    """
    per_sample = 9 * FLOP_TRANSCENDENTAL + 46
    return float(n_out) * per_sample


def flops_corrector_run(n_steps):
    return float(n_steps) * FLOPS_CORRECTOR_STEP


def flops_boris_run(n_steps):
    return float(n_steps) * FLOPS_BORIS_STEP


# ------------------------------------------------------------- discipline --
_DRAWS_DECLARED = 0


def assert_no_draws(n=0):
    """Every script in this directory ends with this call."""
    assert n == _DRAWS_DECLARED, (
        "this directory declared %d random draws before the first run and "
        "has just made %d" % (_DRAWS_DECLARED, n))


def rel(new, old):
    """The shift a report has to print: new/old, and the signed per-cent."""
    if old is None or new is None:
        return {"old": old, "new": new, "ratio": None, "percent": None}
    if old == 0:
        return {"old": old, "new": new, "ratio": None, "percent": None}
    return {"old": float(old), "new": float(new),
            "ratio": float(new) / float(old),
            "percent": 100.0 * (float(new) - float(old)) / abs(float(old))}


__all__ = ["SW", "check_or_write", "closed_form", "fast_basis",
           "basis_agreement", "energy", "bz", "mu_error", "rms", "running_rms",
           "first_crossing", "first_below", "rel", "outpath",
           "flops_boris_reference", "flops_closed_form", "flops_corrector_run",
           "flops_boris_run", "assert_no_draws",
           "DT", "TAU", "REFINE", "R0", "V0", "TWO_PI", "T_FINAL", "R_LARMOR",
           "HERE", "EXP", "ROOT"]
