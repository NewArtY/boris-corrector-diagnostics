"""Figure 3 -- work against precision for the classical family and the hybrid.
Cited in Section 7 (the scheme-family comparison).

THE POINT
    Cost is measured in flops, not seconds.  Seconds are a property of this
    implementation: the Boris step is pure-Python NumPy on 3-vectors, some 43
    us of interpreter overhead around 113 flops of arithmetic, while the
    corrector network runs in compiled BLAS.  Wall-clock therefore flatters
    the hybrid by two to three orders of magnitude relative to a compiled PIC
    code (experiments/cost/breakeven.json, "why_they_disagree").  Flops are
    the same number on every machine, so that is the axis.

    On that axis the learned hybrid sits to the RIGHT of and ABOVE the entire
    classical family, in both channels at once.  Its own operating point,
    Omega_c dt = 0.3, is the only step at which it works at all: at 0.2 and
    0.1 -- steps outside the distribution it was trained on, and steps at
    which every classical scheme is more accurate, not less -- the trajectory
    error is 13.8 and 27.6 Larmor radii.

WHAT EACH PANEL SHOWS
    (a) Root-mean-square trajectory error over the run, in Larmor radii,
        against total flops.  Five classical schemes, each a curve over the
        step grid dt = 0.3 ... 0.02 (cost rises as the step falls).  The
        hybrid is a single filled star at its operating point; the two open
        stars are the same network at dt = 0.2 and 0.1.  The right-angle
        annotation compares vps4 with the hybrid at the SAME step dt = 0.3:
        418x fewer flops for 64.8x less error.
    (b) The same abscissa, the energy channel: median relative energy error
        over the second half of the run.  The horizontal line is the physical
        signal -- the median energy change the decaying field actually
        produces -- so a scheme is only certifying anything below it.  The
        same right-angle annotation gives 62.2x at 418x fewer flops.

    No fitted exponent appears anywhere on this figure.  Convergence orders
    for these curves exist in experiments/cost/breakeven.json and are NOT
    drawn, because that file itself shows the shipped scheme fitting 1.82 on
    the coarse half of the grid and 0.96 on the fine half: a single number
    would misdescribe the curve.  The curves are drawn instead, and the
    reader reads the local slope directly.  (plan/reports/W0_2_exponent_1540.md)

INPUT  (all committed; nothing is re-run and nothing is re-trained)
    ../../code/bundle/code/experiments/classical/workprecision.json
        meta.physical_signal_median and the 8 x 6 grid of scheme runs plus the
        hybrid at its operating point.  Source of every classical point and of
        the hybrid's filled star.
    ../../code/bundle/code/experiments/classical/verdict.json
        The ratios quoted in the text.  This script recomputes all three of
        them from workprecision.json and asserts equality; it does not read
        them for plotting.
    ../../code/bundle/code/experiments/cost/work_precision.json
        The hybrid at dt = 0.2 and 0.1 (the two open stars), with the
        in_training_distribution flag.  This suite scores against its own
        reference (Boris at dt = 0.002) rather than DOP853, so the script
        checks that the two suites agree where they overlap -- the hybrid at
        dt = 0.3 -- before overlaying them.
    ../../code/bundle/code/experiments/cost/breakeven.json
        The analytic flop model: 113 flop per Boris step, 114091 per hybrid
        step.  Used to give the two out-of-distribution hybrid runs an
        abscissa, and asserted consistent with the flop counts committed in
        workprecision.json.

    Setup, identical for every point: B4 decaying field, B = B0 exp(-t/tau)
    zhat with tau = 1.2e5, induced E = (Bz/2 tau)(-y, x, 0), q = -1, m = 1,
    r0 = (1,0,0), v0 = (0,1,0), t_final = 120, Larmor radius 1.  Reference
    DOP853 at rtol 1e-12, atol 1e-14.

NOT DRAWN
    The implicit midpoint rule (imr) was run on the same grid and is in the
    same JSON.  It is left off the artwork to keep the series count at the
    five schemes of the manuscript's table, and every one of its numbers is
    written to fig3_work_precision_values.json under "schemes_not_drawn" so
    that the omission changes nothing a reader can check.  It does not touch
    the claim: at its cheapest useful setting it reaches 2.3e-3 Larmor radii
    for 3.4e6 flops, still better and cheaper than the hybrid.

OUTPUT
    fig3_work_precision.pdf          -- the figure
    fig3_work_precision_values.json  -- every number drawn or annotated
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
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, os.pardir,
                   "experiments")
WP = os.path.join(EXP, "classical", "workprecision.json")
VERDICT = os.path.join(EXP, "classical", "verdict.json")
COST_WP = os.path.join(EXP, "cost", "work_precision.json")
BREAKEVEN = os.path.join(EXP, "cost", "breakeven.json")

DT_OP = 0.3                       # the hybrid's operating step, = DT_WORK

wp = json.load(open(WP, encoding="utf-8"))
verdict = json.load(open(VERDICT, encoding="utf-8"))
cost = json.load(open(COST_WP, encoding="utf-8"))
brk = json.load(open(BREAKEVEN, encoding="utf-8"))

runs = wp["runs"]
signal = wp["meta"]["physical_signal_median"]


def curve(scheme):
    """(flops, trajectory error, energy error, dt) sorted by cost."""
    rs = sorted((r for r in runs if r["scheme"] == scheme),
                key=lambda r: r["flops"])
    return (np.array([r["flops"] for r in rs]),
            np.array([r["pos_err_rms"] for r in rs]),
            np.array([r["energy_err_median_2nd_half"] for r in rs]),
            np.array([r["dt"] for r in rs]))


def at_dt(scheme, dt):
    for r in runs:
        if r["scheme"] == scheme and r["dt"] == dt:
            return r
    raise KeyError((scheme, dt))


# ---------------------------------------------------------------------------
# self-check -- nothing is drawn before this
# ---------------------------------------------------------------------------
hyb = at_dt("hybrid", DT_OP)

# 1. every ratio the manuscript quotes, recomputed from workprecision.json
#    and asserted equal to the committed verdict.
for name, blk in verdict["schemes"].items():
    r = at_dt(name, DT_OP)
    for key, got in (("traj", r["pos_err_rms"]),
                     ("energy", r["energy_err_median_2nd_half"]),
                     ("flops", r["flops"]),
                     ("traj_vs_hybrid", hyb["pos_err_rms"] / r["pos_err_rms"]),
                     ("energy_vs_hybrid",
                      hyb["energy_err_median_2nd_half"]
                      / r["energy_err_median_2nd_half"]),
                     ("flops_cheaper_than_hybrid", hyb["flops"] / r["flops"]),
                     ("below_signal", signal / r["energy_err_median_2nd_half"])):
        assert np.isclose(got, blk[key], rtol=1e-12, atol=0.0), \
            f"{name}.{key}: recomputed {got!r} != committed {blk[key]!r}"

# 2. the flop model of experiments/cost reproduces the counts of
#    experiments/classical, so it may be used to price the two extra hybrid runs
boris_per_step = at_dt("shipped", DT_OP)["flops"] / at_dt("shipped", DT_OP)["n_steps"]
hybrid_per_step = hyb["flops"] / hyb["n_steps"]
assert boris_per_step == brk["flops"]["boris_step_total"], boris_per_step
assert hybrid_per_step == brk["flops"]["hybrid_step_total"], hybrid_per_step
assert hyb["flops"] == brk["hybrid_operating_point"]["total_flops"]

# 3. the two suites agree where they overlap, so their points may share an axis
cost_hyb = {r["dt"]: r for r in cost["hybrid"]}
assert cost_hyb[DT_OP]["in_training_distribution"] is True
assert not cost_hyb[0.2]["in_training_distribution"]
assert not cost_hyb[0.1]["in_training_distribution"]
traj_suite_ratio = cost_hyb[DT_OP]["pos_err_rms"] / hyb["pos_err_rms"]
ener_suite_ratio = (cost_hyb[DT_OP]["energy_err_median_2nd_half"]
                    / hyb["energy_err_median_2nd_half"])
assert abs(traj_suite_ratio - 1.0) < 0.03, traj_suite_ratio
assert abs(ener_suite_ratio - 1.0) < 0.01, ener_suite_ratio
assert abs(cost["meta"]["physical_signal_median"] / signal - 1.0) < 1e-4
assert cost["meta"]["larmor_radius"] == 1.0        # errors are already in r_L

# 4. the headline comparison, recomputed here
v4 = at_dt("vps4", DT_OP)
gain_traj = hyb["pos_err_rms"] / v4["pos_err_rms"]
gain_energy = (hyb["energy_err_median_2nd_half"]
               / v4["energy_err_median_2nd_half"])
gain_flops = hyb["flops"] / v4["flops"]
assert np.isclose(gain_traj, 64.79103049098778, rtol=1e-12)
assert np.isclose(gain_energy, 62.184756308974805, rtol=1e-12)
assert np.isclose(gain_flops, 417.9157509157509, rtol=1e-12)

# 5. the out-of-distribution hybrid points, priced by the asserted flop model
ood = []
for dt in (0.2, 0.1):
    rec = cost_hyb[dt]
    ood.append(dict(dt=dt, n_steps=rec["n_steps"],
                    flops=rec["n_steps"] * hybrid_per_step,
                    traj=rec["pos_err_rms"],
                    energy=rec["energy_err_median_2nd_half"]))
assert abs(ood[0]["traj"] - 13.839226741460223) < 1e-12
assert abs(ood[1]["traj"] - 27.56079826394451) < 1e-12

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
SERIES = [
    # key          label                 colour          dash                       marker
    ("shipped",   "Boris, as shipped",   ps.BLACK,       "-",                       "o"),
    ("staggered", "Boris, staggered",    ps.GREY,        (0, (5.5, 1.4, 1.0, 1.4)), "s"),
    ("vps2",      "vps2 (splitting)",    ps.BLUE,        (0, (4.5, 1.8)),           "^"),
    ("vps4",      "vps4 (Yoshida)",      ps.VERMILLION,  (0, (1.3, 1.5)),           "D"),
    ("gl4",       "gl4 (Gauss-Legendre)", ps.GREEN,      (0, (6.0, 1.5, 1.0, 1.5)), "v"),
]

ps.use_style()
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(ps.TEXTWIDTH_IN, 3.25),
    gridspec_kw=dict(wspace=0.30, left=0.093, right=0.995,
                     bottom=0.240, top=0.950))

hyb_x = [hyb["flops"]] + [o["flops"] for o in ood]
hyb_y = {"traj": [hyb["pos_err_rms"]] + [o["traj"] for o in ood],
         "energy": [hyb["energy_err_median_2nd_half"]] + [o["energy"] for o in ood]}

for ax, chan, ylab in ((axA, "traj", r"trajectory error  (Larmor radii)"),
                       (axB, "energy", r"energy error  $|\Delta E|/E_0$")):
    for key, label, col, dash, mk in SERIES:
        f, tr, en, _ = curve(key)
        y = tr if chan == "traj" else en
        ax.plot(f, y, color=col, ls=dash, lw=1.2, marker=mk, ms=2.9,
                mfc="none", mew=0.85, zorder=3, label=label)

    # the hybrid: one filled star at the operating point, two open stars where
    # it is outside the distribution it was trained on
    ax.plot(hyb_x, hyb_y[chan], color=ps.PURPLE, ls=(0, (1.0, 1.5)), lw=1.0,
            zorder=5)
    ax.plot(hyb_x[1:], hyb_y[chan][1:], ls="none", marker="*", ms=7.5,
            mfc="white", mec=ps.PURPLE, mew=1.0, zorder=6)
    ax.plot(hyb_x[:1], hyb_y[chan][:1], ls="none", marker="*", ms=8.5,
            color=ps.PURPLE, mec=ps.PURPLE, mew=1.0, zorder=7,
            label="learned corrector")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2.4e4, 4.5e8)
    ax.set_xlabel("cost  (flop per run)")
    ax.set_ylabel(ylab)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=14))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0,
                                          subs=tuple(np.arange(2, 10) * 0.1),
                                          numticks=80))
    ax.yaxis.set_minor_formatter(NullFormatter())

# ---- the right-angle comparison, vps4 against the hybrid at the same step --
def right_angle(ax, x0, y0, x1, y1, lab_h, lab_v):
    ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color=ps.GREY,
                                shrinkA=3, shrinkB=1,
                                mutation_scale=7))
    ax.annotate("", xy=(x1, y1), xytext=(x1, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color=ps.GREY,
                                shrinkA=1, shrinkB=4,
                                mutation_scale=7))
    ax.text(x1 * 0.26, y0 * 0.30, lab_h, fontsize=6.9, color=ps.GREY,
            ha="center", va="top")
    ax.text(x1 * 1.35, np.sqrt(y0 * y1), lab_v, fontsize=6.9, color=ps.GREY,
            ha="left", va="center")


right_angle(axA, v4["flops"], v4["pos_err_rms"], hyb["flops"],
            hyb["pos_err_rms"],
            f"${gain_flops:.0f}\\times$ the flops",
            f"${gain_traj:.1f}\\times$\nthe error")
right_angle(axB, v4["flops"], v4["energy_err_median_2nd_half"], hyb["flops"],
            hyb["energy_err_median_2nd_half"],
            f"${gain_flops:.0f}\\times$ the flops",
            f"${gain_energy:.1f}\\times$\nthe error")

# ---- panel (a): axes, step annotation, out-of-distribution label ----------
axA.set_ylim(3.0e-10, 300.0)
axA.text(3.1e4, 3.6e-9, r"$\Omega_c\Delta t$:  $0.3 \rightarrow 0.02$",
         fontsize=6.8, color=ps.BLACK, ha="left", va="bottom")
axA.annotate("", xy=(3.0e5, 2.8e-9), xytext=(3.6e4, 2.8e-9),
             arrowprops=dict(arrowstyle="-|>", lw=0.7, color=ps.BLACK,
                             mutation_scale=7))
axA.annotate(f"$\\Omega_c\\Delta t = {ood[0]['dt']}$ and ${ood[1]['dt']}$,\n"
             "outside the training distribution",
             xy=(ood[0]["flops"], ood[0]["traj"]), xytext=(4.5e4, 35.0),
             fontsize=6.9, color=ps.PURPLE, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.PURPLE,
                             shrinkA=2, shrinkB=5))
axA.annotate(f"$\\Omega_c\\Delta t = {DT_OP}$",
             xy=(hyb["flops"], hyb["pos_err_rms"]), xytext=(-10.0, 4.0),
             textcoords="offset points", fontsize=6.9, color=ps.PURPLE,
             ha="right", va="bottom")
ps.panel_label(axA, "(a)")

# ---- panel (b): axes and the physical signal ------------------------------
axB.set_ylim(6.0e-13, 3.0e-1)
axB.axhline(signal, color=ps.ORANGE, lw=2.6, alpha=0.55, zorder=1,
            solid_capstyle="butt")
axB.text(2.9e4, signal * 1.7,
         f"physical signal  ${signal * 1e4:.3f}\\times10^{{-4}}$",
         fontsize=6.8, color="#B07000", ha="left", va="bottom")
ps.panel_label(axB, "(b)")

# one legend for both panels, below the artwork: the lower-right corner of
# panel (a) is occupied by the tails of vps4 and gl4, so nothing may sit there
h, l = axA.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.52, -0.012), ncol=3,
           columnspacing=1.4)

ps.save_pdf(fig, "fig3_work_precision")

# ---------------------------------------------------------------------------
values = {
    "source": "experiments/classical/workprecision.json (all curves and the "
              "hybrid operating point), experiments/cost/work_precision.json "
              "(hybrid at dt = 0.2 and 0.1), experiments/cost/breakeven.json "
              "(flop model). Every ratio below is recomputed here from "
              "workprecision.json and asserted equal to "
              "experiments/classical/verdict.json at rtol 1e-12.",
    "setup": {
        "field": "B4 decaying, B0 = 1, tau = 1.2e5",
        "t_final": wp["meta"]["t_final"],
        "reference": wp["meta"]["reference"],
        "larmor_radius": cost["meta"]["larmor_radius"],
        "dt_grid": sorted({r["dt"] for r in runs if r["scheme"] != "hybrid"},
                          reverse=True),
        "physical_signal_median": signal,
        "cost_metric": "total flops for the run; flop per step from "
                       "experiments/cost/breakeven.json, "
                       f"Boris {boris_per_step:.0f}, hybrid {hybrid_per_step:.0f}",
        "why_flops_not_seconds": brk["verdict"]["why_they_disagree"],
    },
    "headline_at_equal_step": {
        "dt": DT_OP,
        "vps4_traj_err_larmor": v4["pos_err_rms"],
        "hybrid_traj_err_larmor": hyb["pos_err_rms"],
        "hybrid_over_vps4_traj": gain_traj,
        "vps4_energy_err": v4["energy_err_median_2nd_half"],
        "hybrid_energy_err": hyb["energy_err_median_2nd_half"],
        "hybrid_over_vps4_energy": gain_energy,
        "vps4_flops": v4["flops"],
        "hybrid_flops": hyb["flops"],
        "hybrid_over_vps4_flops": gain_flops,
    },
    "hybrid_points_drawn": [
        {"dt": DT_OP, "flops": hyb["flops"], "traj_err_larmor": hyb["pos_err_rms"],
         "energy_err": hyb["energy_err_median_2nd_half"],
         "in_training_distribution": True,
         "source": "experiments/classical/workprecision.json"},
    ] + [
        {"dt": o["dt"], "flops": o["flops"], "traj_err_larmor": o["traj"],
         "energy_err": o["energy"], "in_training_distribution": False,
         "source": "experiments/cost/work_precision.json; flops = n_steps x "
                   "hybrid_step_total of breakeven.json"}
        for o in ood
    ],
    "cross_suite_check_at_dt_0.3": {
        "traj_cost_over_classical": traj_suite_ratio,
        "energy_cost_over_classical": ener_suite_ratio,
        "note": "the two suites score against different references (DOP853 "
                "rtol 1e-12 vs Boris at dt = 0.002); they agree to 2.0 % and "
                "0.07 % at the overlapping point, which is why the open stars "
                "may share an axis with the curves.",
    },
    "curves_drawn": {
        key: [{"dt": float(d), "flops": float(f), "traj_err_larmor": float(t),
               "energy_err": float(e)}
              for f, t, e, d in zip(*curve(key))]
        for key, *_ in SERIES
    },
    "schemes_not_drawn": {
        "imr": {
            "reason": "implicit midpoint; run on the same grid and committed "
                      "in the same JSON, left off the artwork to match the "
                      "five-scheme table of the manuscript. Its numbers are "
                      "here so the omission hides nothing.",
            "points": [{"dt": float(d), "flops": float(f),
                        "traj_err_larmor": float(t), "energy_err": float(e)}
                       for f, t, e, d in zip(*curve("imr"))],
            "best_traj_still_beats_hybrid": {
                "dt": 0.02,
                "traj_err_larmor": at_dt("imr", 0.02)["pos_err_rms"],
                "flops": at_dt("imr", 0.02)["flops"],
                "hybrid_traj_err_larmor": hyb["pos_err_rms"],
                "hybrid_flops": hyb["flops"],
            },
        }
    },
    "no_fitted_exponent_on_this_figure": {
        "rule": "plan/reports/W0_2_exponent_1540.md -- a fitted exponent must "
                "be printed with its local half-decade slopes.",
        "discharged_by": "no exponent is fitted or printed here; the curves "
                         "themselves are drawn, so the local slope is visible "
                         "at every point.",
        "committed_orders_deliberately_not_drawn":
            brk["convergence_orders"],
    },
}
with open(os.path.join(HERE, "fig3_work_precision_values.json"), "w",
          encoding="utf-8") as fh:
    json.dump(values, fh, indent=1)
print(json.dumps(values["headline_at_equal_step"], indent=1))
