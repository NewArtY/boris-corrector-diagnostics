"""Figure 2 -- the readout floor hides the growth.  Cited in Section 3.

THE POINT
    One integrator, one trained corrector, one physical problem.  Read the
    velocity at integer times, as the code ships it, and the energy error of
    the corrector grows secularly with exponent p = 0.977.  Read the same
    integrator at half-integer times with average re-centring -- the ordinary
    leapfrog convention -- and the fitted exponent is 0.000, because the whole
    of the growth stays under a constant pedestal, sin^2(theta_h/2) = 0.02200,
    that the re-centring itself introduces.  The convention does not scale the
    conclusion.  It reverses it.

WHAT EACH PANEL SHOWS
    (a) Measured envelopes of the relative energy error against gyroperiods,
        log-log, over the five decades that were run.  Four measured series;
        the readout floor is one of them, not an assumption.  Fitted exponents
        are printed next to the two corrector curves.
    (b) The same integer-time envelope divided by the readout floor.  Solid
        where measured, dotted where the fitted power law is extrapolated.
        The growth crosses the floor -- becomes visible under the staggered
        convention at all -- only after about 2e7 gyroperiods.

INPUT  (all committed, nothing is re-run and nothing is re-trained)
    ../../code/experiments/symproj/env_quasistatic_shipped.npz
        shipped/proj/{t,env}   corrector, integer-time readout
        shipped/boris/{t,env}  plain Boris, integer-time readout
    ../../code/experiments/symproj/env_quasistatic_staggered.npz
        staggered/proj/{t,env}   corrector, half-integer readout
        staggered/boris/{t,env}  plain Boris, half-integer readout = the floor
    ../../code/experiments/symproj/summary.json
        exponents, used only to check that this script's refit reproduces the
        committed values.

    Configuration `quasistatic` of experiments/symproj/main.py: h = 0.3,
    Omega_0 = 1, tau = 1.2e8, 1e5 gyroperiods, envelope = running maximum of
    |E_n - E_phys|/E_0.  `proj` is the shipped one-sided projection of the
    trained defect network (checkpoints/boris_corrector_b4.pt).

    The floor drawn in both panels is measured, not asserted: it is the
    half-integer-readout envelope of *plain Boris*, whose recursion is
    bitwise identical to the integer-time one (experiments/verify_theory/
    vt2_map_identity.json: max_abs_dr = max_abs_dv = 0).  For that pair the
    two curves differ by the readout and by nothing else.

OUTPUT
    fig2_readout.pdf          -- the figure
    fig2_readout_values.json  -- every number drawn or annotated
"""
#
# ---------------------------------------------------------------------
# BUNDLE COPY.  This file is byte-for-byte the manuscript's figure script
# except for the one line that locates the data: in the manuscript tree
# the scripts sit in article/figures/ and reach the experiments through
# ../../code/bundle/code/experiments/, here they sit next to them and
# reach them through ../experiments/.  repro.py --check-sync verifies
# that this is still the only difference.
# ---------------------------------------------------------------------


import json
import os

import numpy as np

import paper_style as ps
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SYMPROJ = os.path.join(HERE, os.pardir, "experiments", "symproj")
TWO_PI = 2.0 * np.pi
H = 0.3                      # DT_WORK of training/train_corrector_b4.py
CONFIG = "quasistatic"


def series(npz, key):
    """Gyroperiods and monotone envelope for one run stored in an npz."""
    t = npz[f"{key}/t"]
    env = np.maximum.accumulate(npz[f"{key}/env"])
    return t / TWO_PI, env


def exponent(t, env, decades=2.0):
    """Refit of experiments/symproj/symproj.py::envelope_exponent."""
    env = np.maximum.accumulate(env)
    sel = (t > t[-1] / 10.0 ** decades) & (env > 0)
    slope, intercept = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
    return float(slope), float(intercept)


zs = np.load(os.path.join(SYMPROJ, f"env_{CONFIG}_shipped.npz"))
zt = np.load(os.path.join(SYMPROJ, f"env_{CONFIG}_staggered.npz"))
ref = json.load(open(os.path.join(SYMPROJ, "summary.json"), encoding="utf-8"))
ref = ref["results"][CONFIG]

g_int, e_int = series(zs, "shipped/proj")        # corrector, integer time
g_sta, e_sta = series(zt, "staggered/proj")      # corrector, half-integer
g_bi, e_bi = series(zs, "shipped/boris")         # Boris, integer time
g_bs, e_bs = series(zt, "staggered/boris")       # Boris, half-integer = floor

p_int, c_int = exponent(g_int * TWO_PI, e_int)
p_sta, _ = exponent(g_sta * TWO_PI, e_sta)

floor = float(e_bs[-1])
floor_closed = (H / 2) ** 2 / (1.0 + (H / 2) ** 2)   # sin^2(theta_h/2), theta_h = 2 atan(hOm/2)
level_sta = float(e_sta[-1])
end_gyro = float(g_int[-1])
end_env = float(e_int[-1])
frac_of_floor = end_env / floor
cross_gyro = end_gyro * (floor / end_env) ** (1.0 / p_int)

# ---------------------------------------------------------------------------
# self-check -- nothing is drawn before this
# ---------------------------------------------------------------------------
assert np.isclose(p_int, ref["shipped/proj"]["envelope_exponent"], rtol=1e-9), p_int
assert np.isclose(p_sta, ref["staggered/proj"]["envelope_exponent"],
                  rtol=0.0, atol=1e-12), p_sta
assert np.isclose(end_env, ref["shipped/proj"]["E_err_1e+05"], rtol=1e-12)
assert np.isclose(floor, ref["staggered/boris"]["E_err_1e+05"], rtol=1e-12)
assert np.isclose(level_sta, ref["staggered/proj"]["E_err_1e+05"], rtol=1e-12)
# the floor is the closed form sin^2(theta_h/2), to seven digits
assert abs(floor / floor_closed - 1.0) < 1e-6, (floor, floor_closed)
# flat means flat: the half-integer envelope never moves
assert e_sta.max() == e_sta.min() and e_bs.max() == e_bs.min()

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
ps.use_style()
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(ps.TEXTWIDTH_IN, 2.55),
    gridspec_kw=dict(wspace=0.34, left=0.095, right=0.995,
                     bottom=0.165, top=0.935))

# ---- (a) the measurement --------------------------------------------------
# The half-integer corrector level (0.02233) and the half-integer Boris level
# (0.02200 = the floor) differ by 1.5 %, so on nine decades they coincide.
# That coincidence is the result, so the floor is drawn as a broad pale band
# and the corrector curve rides on it.
axA.axhline(floor, color=ps.VERMILLION, lw=3.2, alpha=0.45,
            solid_capstyle="butt", zorder=2)
axA.plot(g_sta, e_sta, color=ps.BLUE, ls=(0, (5.0, 2.2)), lw=1.3, zorder=4)
axA.plot(g_bi, e_bi, color=ps.GREY, ls=(0, (5.5, 1.4, 1.0, 1.4)), lw=1.1,
         zorder=3)
axA.plot(g_int, e_int, color=ps.BLACK, ls="-", lw=1.4, zorder=5)

g_fit = np.array([2e3, 3e5])
axA.plot(g_fit, 10.0 ** (c_int + p_int * np.log10(g_fit * TWO_PI)),
         color=ps.BLACK, ls=(0, (2.0, 1.6)), lw=0.85, zorder=6)

axA.text(4.6, floor * 2.6,
         f"readout floor  $\\sin^2(\\theta_h/2) = {floor:.5f}$",
         fontsize=6.8, color=ps.VERMILLION, ha="left", va="bottom")
axA.text(4.6, floor * 0.42,
         f"corrector, half-integer readout:  $p = {abs(p_sta):.3f}$",
         fontsize=6.8, color=ps.BLUE, ha="left", va="top")
axA.text(4.6, e_bi[-1] * 2.0, "Boris, integer-time readout",
         fontsize=6.8, color=ps.GREY, ha="left", va="bottom")
axA.annotate("corrector,\ninteger-time readout\n"
             f"$p = {p_int:.3f}$",
             xy=(1.2e4, 1.3e-5), xytext=(1.3e2, 3.5e-4),
             fontsize=7.0, color=ps.BLACK, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.BLACK,
                             shrinkA=3, shrinkB=3))

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(3.5, 2.2e5)
axA.set_ylim(8e-10, 0.5)
axA.set_xlabel("gyroperiods")
axA.set_ylabel(r"envelope of $|\Delta E|/E_0$")
ps.panel_label(axA, "(a)")

# ---- (b) the consequence --------------------------------------------------
g_ext = np.logspace(np.log10(end_gyro), np.log10(1.2e8), 200)
r_ext = (end_env / floor) * (g_ext / end_gyro) ** p_int

axB.axhspan(1e-7, 1.0, color=ps.BLACK, alpha=0.05, lw=0, zorder=0)
axB.axhline(1.0, color=ps.VERMILLION, lw=3.2, alpha=0.45,
            solid_capstyle="butt", zorder=3)
axB.plot(g_int, e_int / floor, color=ps.BLACK, ls="-", lw=1.4, zorder=5,
         label="measured")
axB.plot(g_ext, r_ext, color=ps.BLACK, ls=(0, (1.0, 1.7)), lw=1.2, zorder=4,
         label=f"$p = {p_int:.4f}$, extrapolated")
axB.plot([cross_gyro], [1.0], marker="o", ms=4.4, mfc="white", mew=1.1,
         color=ps.BLACK, zorder=6)
axB.plot([end_gyro], [frac_of_floor], marker="s", ms=3.8, mfc="white",
         mew=1.1, color=ps.BLACK, zorder=6)

axB.text(4.5, 1.7, "readout floor", fontsize=7.0, color=ps.VERMILLION,
         ha="left", va="bottom")
axB.annotate(f"${frac_of_floor * 100:.2f}$ % of the floor\n"
             f"at $10^5$ gyroperiods",
             xy=(end_gyro, frac_of_floor), xytext=(12.0, 6.0e-2),
             fontsize=7.0, ha="left", va="center", color=ps.BLACK,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.BLACK,
                             shrinkA=2, shrinkB=4))
axB.annotate(f"${cross_gyro / 1e7:.1f}\\times10^{{7}}$",
             xy=(cross_gyro, 1.0), xytext=(-5.0, 7.0),
             textcoords="offset points", fontsize=7.4, ha="right", va="bottom",
             color=ps.BLACK)

axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlim(3.5, 1.5e8)
axB.set_ylim(1e-7, 40.0)
axB.set_xlabel("gyroperiods")
axB.set_ylabel(r"envelope $/$ readout floor")
axB.legend(loc="lower right", bbox_to_anchor=(1.02, -0.02))
ps.panel_label(axB, "(b)")

ps.save_pdf(fig, "fig2_readout")

# ---------------------------------------------------------------------------
values = {
    "source": f"experiments/symproj/env_{CONFIG}_*.npz; exponents refitted here "
              "and asserted equal to experiments/symproj/summary.json",
    "setup": {"h": H, "tau": 1.2e8, "gyroperiods_run": end_gyro,
              "config": CONFIG},
    "panel_a": {
        "exponent_integer_time_readout": p_int,
        "exponent_half_integer_readout": p_sta,
        "readout_floor_measured": floor,
        "readout_floor_closed_form_sin2_theta_h_half": floor_closed,
        "corrector_level_half_integer_readout": level_sta,
        "boris_level_integer_time_readout": float(e_bi[-1]),
        "envelope_at_1e5_gyroperiods": end_env,
    },
    "panel_b": {
        "fraction_of_floor_at_1e5_gyroperiods": frac_of_floor,
        "crossing_gyroperiods_anchored_at_last_point": cross_gyro,
        "crossing_gyroperiods_from_lsq_intercept":
            float(10.0 ** ((np.log10(floor) - c_int) / p_int) / TWO_PI),
    },
}
with open(os.path.join(HERE, "fig2_readout_values.json"), "w",
          encoding="utf-8") as fh:
    json.dump(values, fh, indent=1)
print(json.dumps(values, indent=1))
