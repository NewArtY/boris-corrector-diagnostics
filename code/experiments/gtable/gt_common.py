"""Shared machinery for wave W15 -- the table of orders, channel by channel.

WHAT THIS DIRECTORY IS FOR
--------------------------
The first author asks for one number per (configuration, scheme, channel):

    G = log10( E_Boris / E_scheme )

taken verbatim, so that G > 0 reads "better than the Boris scheme" and the
Boris scheme itself is the row with G = 0.  Four channels are named in the
pre-registration: the trajectory, the phase, the total energy, and the
spectral power in the band f/Omega_c < 0.2.  Five configurations, five
schemes, one step, Omega h = 0.3.

This is a measurement and not an assembly.  Of the four channels, two were
already on the map of W14 (trajectory and energy) and two were not: the
spectral wave W13 covered the working configuration alone, and the phase
channel exists nowhere in the bundle outside the single Boris run of
`../theory_check/t1_boris_channels.py`.  Both are measured here on all five
configurations.

WHAT IS IMPORTED AND WHAT IS NEW
--------------------------------
Nothing is retrained and nothing outside this directory is written.  The
three-dimensional batched rollout, the field bridge, the closed forms, the
Larmor radii, the flop model and the rank correlation come from
`../map/map_common.py`; the periodogram, the band accounting and the windows
from `../spectral/sw_common.py`; the JSON gatekeeper from
`../external_arch/ea_common.py`.  A change to any of them changes these
numbers too.  What is new here is the phase channel, the per-configuration
band, and the assembly of the four into one table.

THE RECORD
----------
Every channel is computed on the same record: the state at every step of the
scheme, from n = 0 to n = N-1, at h = 0.3.  N = 400 is the window of
Table~\\ref{tab:family}, 19.1 gyro-orbits; N = 2120 is the crossover horizon of
W14, 101.2 gyro-orbits.  The half-open convention (N samples, the last at
t = (N-1)h) is W13's, taken so that the spectral figures of this directory
compare directly with the committed `../spectral/sw2_spectra.json`.  W14's
trajectory and energy figures are computed on N+1 samples instead; the
difference is measured in `gt1_calibration.py` rather than assumed away.

THE BAND, WHERE Omega_c DEPENDS ON POSITION  --  DECLARED BEFORE THE RUNS
-------------------------------------------------------------------------
"f/Omega_c < 0.2" is unambiguous only where Omega_c is constant.  It is
constant in three of the five configurations (uniform and B3 are spatially
uniform and static; B4 decays by one part in a thousand over the paper
window) and is not in the other two: B1's gyrofrequency rises with rho and
B2's oscillates with the wave.

The reference gyrofrequency of a configuration is declared here, before any
run, as

    Omega_c^ref = |q/m| * |B(r_0, 0)|   ,   evaluated at that configuration's
                                            own initial condition,

that is, the initial value of the gyrofrequency.  Three reasons, all of them
fixed in advance:

  1.  It does not depend on which scheme is being scored.  A gyrofrequency
      averaged along "the" trajectory would be a different number for every
      scheme, so the band would be a different window for every scheme and the
      ratio of two in-band powers would no longer be a comparison of schemes.
  2.  It is the same |B(r_0, 0)| that sets the Larmor radius r_L in which the
      trajectory channel of W14 is measured, so the two channels are scaled by
      one field magnitude and not by two.
  3.  It requires no integration, so it cannot inherit the error of a
      reference.

`BAND_EDGE_RATIO` below is the first author's 0.2, verbatim; the band of a
configuration is nu < 0.2 * Omega_c^ref.  The alternative -- the time mean of
|B| along the *reference* orbit, which is also scheme-independent -- is
carried through the whole measurement as a declared sensitivity check, not as
a second choice: `gt3_gtable.py` reports whether any G or any rank moves.

THE PHASE CHANNEL, AND WHY IT IS NOT arccos
--------------------------------------------
The angle of Eq.~(\\ref{eq:polar}) is computed as

    theta = atan2( |v x v_ref| , v . v_ref ) ,

and never as arccos(v.v_ref / (|v| |v_ref|)).  The two are the same angle and
they are not the same measurement.  Near theta = 0 the cosine is 1 - theta^2/2,
so a double-precision cosine resolves no angle below about sqrt(eps) = 1.5e-8;
gl4 and vps4 hold theta four orders below that, and an arccos would return
their phase error as zero or as noise at 1e-8 and the corresponding G as a
number about the floating-point format.  The two-argument arctangent has full
relative accuracy at every angle.  `gt1_calibration.py` shows both: it
reproduces the 38.11 degrees of Section~\\ref{sec:channels} with either
formula, because at that angle they agree, and then exhibits the angle at
which they part.

METRICS, ONE PER CHANNEL, EACH THE COMMITTED ONE
-------------------------------------------------
Each channel is summarised by the statistic the manuscript already uses for
that channel, so that no new convention enters through W15:

    trajectory   root mean square over the record, in Larmor radii
                 (Table~\\ref{tab:family}, W14)
    phase        median over the second half of the record, in radians
                 (Section~\\ref{sec:channels}: "the median angle over the
                  second half of the run is 38.11 degrees")
    energy       median relative error over the second half
                 (Table~\\ref{tab:family}, W14)
    spectral     the integral of the PSD of the position error over
                 nu < 0.2 Omega_c^ref, Hann window
                 (W13, `../spectral/sw2_spectra.py`)

The four statistics are not the same functional, and that is deliberate: each
is the one the manuscript prints.  Every channel is *also* summarised by the
root mean square over the record, and `gt3_gtable.py` reports whether any rank
changes when the whole table is recomputed that way.

G IS NOT COMMENSURABLE ACROSS CHANNELS, AND THE SPECTRAL COLUMN IS WHY
-----------------------------------------------------------------------
Three of the four channels are linear in the error; the spectral channel is a
power and therefore quadratic in it.  log10 of a ratio of powers is twice
log10 of the corresponding ratio of amplitudes.  The pre-registration names
the channel quantity as "the integral of the PSD", so the power ratio is what
G means in that column and it is reported as such; the amplitude-equivalent
G/2 is reported beside it so that the size of a spectral G can be read against
a trajectory G.  Ranks are identical under either, log10 being monotone, so
nothing in the rank-agreement result turns on this.

SEEDS
-----
This directory draws nothing.  The eight initial conditions are
`../map/map_common.py:initial_conditions`, built there from MAP_SEED =
14_000_000 with the twenty-one draws declared in W14; no generator is
constructed here.  The number of random draws in this directory is **zero**.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, os.path.join(EXP, "external_arch"),
           os.path.join(EXP, "classical"), os.path.join(EXP, "spectral"),
           os.path.join(EXP, "map")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_common as MC                                       # noqa: E402
import sw_common as SW                                        # noqa: E402
from ea_common import check_or_write                          # noqa: E402

check_or_write = check_or_write        # re-exported for the scripts below
clean = MC.clean
spearman = MC.spearman

TWO_PI = 2.0 * math.pi
Q, M = MC.Q, MC.M

# ------------------------------------------------------------------- axes ---
#: the step of the pre-registration.  W15 varies the channel, not the step;
#: the step axis is the map of W14.
DT = 0.3

#: the record lengths, in samples at h = 0.3.  H_paper is the window every
#: number of Table~\ref{tab:family} lives in.  H_crossover is the horizon of
#: W14 and is carried because the band forces it: a record of N samples at
#: spacing h resolves 0.2 * (gyro-orbits) independent bins strictly inside
#: f/Omega_c < 0.2, so H_paper carries 3.8 of them and H_crossover 20.2.  A
#: spectral G reported at H_paper alone would rest on four bins.
HORIZONS = {"H_paper": 400, "H_crossover": 2120}
HORIZON_T = {k: v * DT for k, v in HORIZONS.items()}

SCHEMES = MC.SCHEMES                       # boris, corrector, vps2, vps4, gl4
BASE = "boris"
FIELD_NAMES = MC.FIELD_NAMES
CHANNELS = ["trajectory", "phase", "energy", "spectral"]

#: the first author's band edge, verbatim, as a ratio to Omega_c^ref
BAND_EDGE_RATIO = SW.BAND                  # 0.2

#: a cell is reported as possibly reference-limited when its channel value
#: falls within this factor of the measured floor of the reference.  Ten, the
#: factor W14 already uses (`../map/mp3_maps.json`:meta.ref_limit_factor).
REF_LIMIT_FACTOR = 10.0

#: the trajectory floors W14 measured, imported rather than re-derived, so
#: that the two waves adjudicate the reference by one number.  The other three
#: channels have no committed floor and `gt1_calibration.py` measures theirs
#: by the same method: a second, independent reference put through the same
#: four channels as if it were a scheme.
W14_POSITION_FLOOR_JSON = os.path.join(EXP, "map", "mp3_maps.json")

N_RANDOM_DRAWS = 0


# ============================================================ the band ======
def reference_gyrofrequency(field, R0):
    """Omega_c^ref = |q/m| |B(r_0, 0)| per initial condition.

    Declared in the module docstring before any run.  Scheme-independent by
    construction, which is the property the band needs.
    """
    B = np.atleast_2d(field.B(R0, 0.0))
    return abs(Q / M) * np.linalg.norm(B, axis=1)


def gyrofrequency_along(field, Rr, ts):
    """|q/m| |B(r(t), t)| along a given orbit, for the sensitivity check and
    for reporting how far from constant the gyrofrequency actually is.

    `Rr` is (n, nb, 3); the orbit passed in is always the *reference* one, so
    this too is the same number for every scheme.
    """
    n, nb, _ = Rr.shape
    out = np.empty((n, nb))
    for j in range(n):
        B = np.atleast_2d(field.B(Rr[j], float(ts[j])))
        out[j] = np.linalg.norm(B, axis=1)
    return abs(Q / M) * out


# ========================================================= the four channels =
def channel_series(Rs, Vs, Rr, Vr, r_L):
    """The four residual series of the pre-registration, per sample per member.

    Rs, Vs, Rr, Vr are (n, nb, 3); r_L is (nb,).  Returns the three pointwise
    series -- the spectral channel is not pointwise and is formed from
    `position_vector` by `band_power`.

      trajectory  |r - r_ref| / r_L                          (Larmor radii)
      phase       atan2(|v x v_ref|, v . v_ref)              (radians)
      energy      (E - E_ref) / E_ref(0),  E = |v|^2 / 2     (relative)
    """
    r_L = np.asarray(r_L, dtype=float)[None, :]
    dR = (Rs - Rr) / r_L[..., None]
    traj = np.linalg.norm(dR, axis=-1)

    cross = np.cross(Vs, Vr)
    theta = np.arctan2(np.linalg.norm(cross, axis=-1),
                       np.sum(Vs * Vr, axis=-1))

    e = 0.5 * np.sum(Vs * Vs, axis=-1)
    er = 0.5 * np.sum(Vr * Vr, axis=-1)
    energy = (e - er) / er[0][None, :]
    return {"trajectory": traj, "phase": theta, "energy": energy,
            "position_vector": dR}


def polar_identity_residual(Vs, Vr, theta):
    """Eq.~(\\ref{eq:polar}) checked pointwise, as Section~\\ref{sec:channels}
    checks it:  |v - v_ref|^2 = (v - v_ref)^2 + 2 v v_ref (1 - cos theta).

    Returned as the largest relative residual over the record, per member.
    The identity holds for any two vectors, so a residual above rounding would
    mean the angle is not the angle between them.
    """
    lhs = np.sum((Vs - Vr) ** 2, axis=-1)
    v = np.linalg.norm(Vs, axis=-1)
    vr = np.linalg.norm(Vr, axis=-1)
    # 1 - cos theta = 2 sin^2(theta/2), which is accurate at small theta where
    # 1 - cos theta is not.
    rhs = (v - vr) ** 2 + 4.0 * v * vr * np.sin(0.5 * theta) ** 2
    scale = np.maximum(np.abs(lhs), 1e-300)
    return np.max(np.abs(lhs - rhs) / scale, axis=0)


def band_power(dR, dt, omega_c, edge_ratio=BAND_EDGE_RATIO, window="hann"):
    """The integral of the PSD of the position-error vector over
    nu < edge_ratio * omega_c, for one batch member.

    `dR` is (n, 3) in Larmor radii.  The periodogram is
    `../spectral/sw_common.py:psd` unchanged: one-sided, one segment,
    normalised so that the summed spectrum is the window-weighted mean square,
    which is the normalisation under which the total power of this channel is
    the square of the windowed root-mean-square position error and the band
    ratio factors exactly.  Parseval is checked on every series by the caller.
    """
    return SW.band_powers(dR, dt, edge=edge_ratio * float(omega_c),
                          window=window)


#: the statistic each channel is summarised by.  Each is the one the
#: manuscript already prints for that channel; see the module docstring.
def summarise(series_by_channel, dt, omega_c, r_L=None):
    """Every declared figure for one run, per batch member.

    Returns {channel: {"primary": (nb,), "rms": (nb,), ...}}.  "primary" is
    the committed statistic of that channel and is what G is formed from;
    "rms" is the same series under one common statistic and is the robustness
    column of `gt3_gtable.py`.
    """
    traj = series_by_channel["trajectory"]
    phase = series_by_channel["phase"]
    energy = series_by_channel["energy"]
    dR = series_by_channel["position_vector"]
    n, nb = traj.shape
    half = n // 2

    out = {
        "trajectory": {
            "primary": _rms(traj), "rms": _rms(traj),
            "max": np.max(traj, axis=0), "final": traj[-1],
            "median_2nd_half": np.median(np.abs(traj[half:]), axis=0),
            "primary_is": "rms over the record, Larmor radii",
        },
        "phase": {
            "primary": np.median(np.abs(phase[half:]), axis=0),
            "rms": _rms(phase),
            "max": np.max(np.abs(phase), axis=0), "final": np.abs(phase[-1]),
            "median_2nd_half": np.median(np.abs(phase[half:]), axis=0),
            "median_2nd_half_deg": np.degrees(
                np.median(np.abs(phase[half:]), axis=0)),
            "primary_is": "median over the second half, radians",
        },
        "energy": {
            "primary": np.median(np.abs(energy[half:]), axis=0),
            "rms": _rms(energy),
            "max": np.max(np.abs(energy), axis=0), "final": np.abs(energy[-1]),
            "median_2nd_half": np.median(np.abs(energy[half:]), axis=0),
            "primary_is": "median relative error over the second half",
        },
    }

    p_band = np.empty(nb); p_tot = np.empty(nb); p_nodc = np.empty(nb)
    p_band_bh = np.empty(nb); n_bins = np.empty(nb, dtype=np.int64)
    parseval = np.empty(nb)
    for i in range(nb):
        bp = band_power(dR[:, i, :], dt, omega_c[i])
        bh = band_power(dR[:, i, :], dt, omega_c[i], window="blackmanharris")
        p_band[i] = bp["p_band"]; p_tot[i] = bp["p_total"]
        p_nodc[i] = bp["p_band_nodc"]; p_band_bh[i] = bh["p_band"]
        n_bins[i] = bp["n_bins_in_band"]
        ok, lhs, rhs = SW.spectral_selfcheck(dR[:, i, :], dt)
        parseval[i] = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)
    out["spectral"] = {
        "primary": p_band, "rms": np.sqrt(p_band),
        "p_band": p_band, "p_band_nodc": p_nodc, "p_total": p_tot,
        "p_band_blackmanharris": p_band_bh,
        "frac_in_band": p_band / np.maximum(p_tot, 1e-300),
        "n_bins_in_band": n_bins,
        "parseval_rel_residual": parseval,
        "primary_is": "PSD integral over nu < 0.2 Omega_c^ref, Hann; a POWER, "
                      "so its G is twice the amplitude-equivalent G",
    }
    return out


def _rms(a):
    return np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2, axis=0))


# ==================================================================== G ======
def G(e_base, e_scheme):
    """G = log10(E_Boris / E_scheme), the first author's definition verbatim.

    Positive means better than the Boris scheme.  A scheme whose error is
    exactly zero would give +inf and is reported as such rather than clipped;
    a Boris error of exactly zero would make G undefined and is reported as
    such.  Neither occurs at Omega h = 0.3.
    """
    b = float(e_base); s = float(e_scheme)
    if not (math.isfinite(b) and math.isfinite(s)):
        return float("nan")
    if b <= 0.0:
        return float("nan")
    if s <= 0.0:
        return float("inf")
    return math.log10(b / s)


# ======================================================= rank agreement ======
#: Declared before the runs: the rank statistic of this wave is **Spearman's
#: rho**, average ranks on ties, computed by `../map/map_common.py:spearman`
#: -- the same function whose median over the map of W14 is the +0.00 this
#: wave is asked to check on four channels.  Kendall's tau-b is reported
#: beside it because at these tiny n the two can differ visibly, and reporting
#: only the more favourable one would be a choice made after the result.
RANK_STAT = "spearman rho (primary), kendall tau-b (reported alongside)"


def kendall_tau_b(a, b):
    """Kendall's tau-b, with the tie correction, on two short vectors."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return float("nan")
    conc = disc = 0
    ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1; tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da * db > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2.0
    den = math.sqrt((n0 - ta) * (n0 - tb))
    return float((conc - disc) / den) if den > 0 else float("nan")


def rank_agreement(err_by_channel, schemes):
    """Do two channels rank the same schemes in the same order?

    `err_by_channel` maps a channel to {scheme: error}.  Ranking is by the
    error ascending (best first); ranking by G descending is the same order,
    G being log10(E_base) - log10(E_scheme) with a per-channel constant, so
    nothing here depends on which of the two is ranked.

    Returned per unordered channel pair.
    """
    out = {}
    chans = [c for c in CHANNELS if c in err_by_channel]
    for i, ca in enumerate(chans):
        for cb in chans[i + 1:]:
            va = [err_by_channel[ca][s] for s in schemes]
            vb = [err_by_channel[cb][s] for s in schemes]
            out["%s|%s" % (ca, cb)] = {
                "spearman": spearman(va, vb),
                "kendall_tau_b": kendall_tau_b(va, vb),
            }
    return out


# ------------------------------------------------------------------ output --
def stat(a):
    a = np.asarray(a, dtype=float)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return {"ic0": float(a[0]) if a.size else float("nan"),
                "median": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_finite": 0}
    return {"ic0": float(a[0]), "median": float(np.median(f)),
            "min": float(f.min()), "max": float(f.max()),
            "n_finite": int(f.size)}
