"""Figure 4 -- the dissipation sweep.  Cited in Section 5 (protection by
contraction).

THE POINT
    Along the whole sweep the relativistic kinematics, the code, the drive
    alpha, the step h, the initial rapidity and the horizon T are held fixed.
    The one thing that changes is the radiation-reaction coefficient eps, and
    with it the contraction rate Lambda = 2(alpha - eps) at the attractor.
    At eps = 0 there is no attractor and no contraction, and the injected
    defect integrates freely: the envelope of the rapidity error grows with a
    two-decade fitted exponent of 0.944, rising through 0.990 on its last
    half-decade towards the exact linear law q0 t.  Switch the friction on and
    the same defect, the same code and the same kinematics give a bounded
    plateau q0/Lambda and an exponent of zero.  The factor responsible for the
    growth is the dissipation, not the relativity.
    (experiments/ll_probe/prereg2.json, F7_eps_sweep.confound_statement.)

WHAT EACH PANEL SHOWS
    (a) Envelope of the rapidity error, in units of the injection rate
        q0 = kappa/h, against time.  In these units the free-growth law is the
        diagonal Delta theta / q0 = t, drawn once; every run follows it until
        t ~ 1/Lambda and then leaves it for the plateau 1/Lambda.  The
        Lambda = 0 run never leaves.  Open circles mark t = 1/Lambda on each
        curve -- the boundary Lambda T ~ 1.
    (b) Local half-decade slopes of those envelopes against Lambda t.  The
        five dissipative runs collapse: slope 1 while Lambda t < 1, zero once
        Lambda t > 5.  The band is the same statistic for the Lambda = 0 run,
        which has no Lambda to rescale by and never leaves 1.
    (c) The growth exponent against Lambda, with the point Lambda = 0
        included.  Filled markers are the committed two-decade log-log fits;
        the open marks stacked at the same abscissa are the local half-decade
        slopes inside that same fit window, and the vertical bar is their
        range.  This is what keeps the panel honest: at Lambda = 0.002 the fit
        returns 0.374, but the four half-decade slopes are 0.90 / 0.62 / 0.07
        / 0.00 -- the run is crossing Lambda t = 1 inside the fit window, so
        0.374 is a crossover, not an exponent.  At Lambda = 0 the local slopes
        are 0.78 / 0.91 / 0.97 / 0.99, still climbing to the linear law: 0.944
        understates it.  Nowhere else in the sweep is any slope nonzero.
        (Rule adopted in plan/reports/W0_2_exponent_1540.md, where fitting the
        last two decades of a curve that is not a power law there produced a
        false exponent twice.)
    (d) The plateau law across three decades in Lambda.  Ordinate is
        plateau x Lambda / q0, which the continuum argument puts at 1.  The
        curve is the discrete correction z / (1 - R(-z)) with z = Lambda h and
        R the RK4 stability function -- no fitted parameter -- and it accounts
        for the entire departure, 4.6 % at the stiffest point.

INPUT  (committed)
    ../../code/bundle/code/experiments/ll_probe/results2.json
        block F7: the measured exponents, plateaux and the two windowed
        exponents at eps/alpha = 0.999.  Every reproduced value below is
        asserted equal to this file before anything is drawn.
    ../../code/bundle/code/experiments/ll_probe/prereg2.json
        block F7_eps_sweep: the grid, frozen before the measurement ran, and
        the source of the Lambda assigned to each run.

    The per-step series is not committed -- only the summary is -- so the
    script reproduces the sweep.  The model, the integrator and the envelope
    fit below are a verbatim copy of ``make``, ``rk4`` and ``env_exp`` from
    experiments/ll_probe/followup.py, driven exactly as its ``f7`` drives
    them; the run is deterministic, takes about five seconds, and the
    assertions are what prove the copy faithful (they hold to the last bit).

    Model: dtheta/dt = alpha tanh(theta) - eps sinh(theta) cosh(theta), the
    rapidity form of the Landau-Lifshitz equation used throughout Section 5;
    alpha = 1, eps/alpha in {0, 0.1, 0.5, 0.9, 0.99, 0.999}, h = 0.05,
    theta_0 = 0.3, defect kappa injected on every step, horizon T = 300 for
    eps = 0 and T = 2e4 otherwise.

ONE LABEL THAT MUST NOT BE TAKEN AT FACE VALUE
    results2.json stores, for the eps = 0 run, "Lambda": 2.0.  That field is
    the formula 2(alpha - eps) evaluated where the attractor it describes does
    not exist: with eps = 0 the flow has no fixed point, f'(theta) = alpha
    sech^2 theta -> 0, and the contraction rate is 0, which is what prereg2
    fixed for that row before the run.  The distinction is not cosmetic.  Plot
    the stored field and the only growing point in the sweep lands at the
    LARGEST Lambda, reversing the conclusion of the figure.  This script takes
    Lambda from prereg2.json, asserts that results2.json carries 2.0 there,
    and records both in the values file.

OUTPUT
    fig4_dissipation.pdf          -- the figure
    fig4_dissipation_values.json  -- every number drawn or annotated
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
import math
import os

import numpy as np

import paper_style as ps
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.join(HERE, os.pardir,
                  "experiments", "ll_probe")

H = 0.05
THETA0 = 0.3
# (eps/alpha, kappa, T) -- verbatim SWEEP of experiments/ll_probe/followup.py
SWEEP = [(0.0, 1e-6, 300.0), (0.1, 1e-6, 2e4), (0.5, 1e-6, 2e4),
         (0.9, 1e-6, 2e4), (0.99, 1e-7, 2e4), (0.999, 1e-8, 2e4)]


# ---------------------------------------------------------------------------
# verbatim from experiments/ll_probe/followup.py
# ---------------------------------------------------------------------------
def make(al, ep):
    def f(t):
        return al * math.tanh(t) - ep * math.sinh(t) * math.cosh(t)
    return f


def rk4(f, th, h):
    k1 = f(th); k2 = f(th + .5 * h * k1); k3 = f(th + .5 * h * k2); k4 = f(th + h * k3)
    return th + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def R_rk4(z):
    return 1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24


def env_exp(t, d):
    e = np.maximum.accumulate(np.abs(d)); s = (t > t[-1] / 100.) & (e > 0)
    return (float(np.polyfit(np.log10(t[s]), np.log10(e[s]), 1)[0]) if s.sum() > 10
            else float("nan")), e


# ---------------------------------------------------------------------------
def half_decade_slopes(t, env, t_lo, t_hi):
    """Slope of log10(envelope) against log10(t) inside each half decade.

    The rule of plan/reports/W0_2_exponent_1540.md: a fitted exponent is
    reported together with these, so that a fit taken across a crossover
    cannot be mistaken for a power law.
    """
    lo, hi = math.log10(t_lo), math.log10(t_hi)
    edges = np.arange(lo, hi + 1e-9, 0.5)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= 10.0 ** a) & (t <= 10.0 ** b) & (env > 0)
        if m.sum() > 4:
            out.append((float(10.0 ** (0.5 * (a + b))),
                        float(np.polyfit(np.log10(t[m]), np.log10(env[m]), 1)[0])))
    return out


# ---------------------------------------------------------------------------
# reproduce the F7 sweep of followup.py
# ---------------------------------------------------------------------------
res = json.load(open(os.path.join(LL, "results2.json"), encoding="utf-8"))["F7"]
pre = json.load(open(os.path.join(LL, "prereg2.json"),
                     encoding="utf-8"))["F7_eps_sweep"]

runs = []
for epr, kap, T in SWEEP:
    al = 1.0; ep = epr * al
    f = make(al, ep)
    n = int(T / H)
    idx = np.unique(np.round(np.logspace(0, math.log10(n), 3000)).astype(int))
    thu = thp = THETA0
    j = 0
    ts = np.empty(len(idx)); dth = np.empty(len(idx))
    for k in range(n):
        thu = rk4(f, thu, H); thp = rk4(f, thp, H) + kap
        if j < len(idx) and k + 1 == idx[j]:
            ts[j] = (k + 1) * H; dth[j] = thp - thu; j += 1
    p, env = env_exp(ts, dth)
    late = ts > ts[-1] / 10
    runs.append(dict(
        epr=epr, kap=kap, T=T, t=ts, env=env, p=p,
        q0=kap / H,
        dth_final=float(dth[-1]),
        slope_per_time=float(np.polyfit(ts[late], dth[late], 1)[0]),
        key=f"eps_over_alpha={epr}",
        pre_key="eps_over_alpha=0" if epr == 0.0 else f"eps_over_alpha={epr}",
    ))

# Lambda comes from the preregistered grid, never from the results file: see
# the docstring, "ONE LABEL THAT MUST NOT BE TAKEN AT FACE VALUE".
for r in runs:
    r["Lambda"] = float(pre["grid"][r["pre_key"]]["Lambda"])
    r["law"] = (None if r["Lambda"] == 0.0
                else abs(r["dth_final"]) * r["Lambda"] / r["q0"])

# ---------------------------------------------------------------------------
# self-check -- nothing is drawn before this
# ---------------------------------------------------------------------------
for r in runs:
    c = res[r["key"]]
    for name, got, want in (("exponent_theta", r["p"], c["exponent_theta"]),
                            ("dtheta_final", r["dth_final"], c["dtheta_final"]),
                            ("dtheta_slope_per_unit_time", r["slope_per_time"],
                             c["dtheta_slope_per_unit_time"])):
        assert np.isclose(got, want, rtol=1e-9, atol=1e-18), \
            f"{r['key']}.{name}: reproduced {got!r} != committed {want!r}"
    if r["law"] is not None:
        assert np.isclose(r["law"], c["law_plateau_times_Lambda_over_q0"],
                          rtol=1e-9), (r["key"], r["law"])
    else:
        assert c["law_plateau_times_Lambda_over_q0"] is None

# the Lambda = 0 row: prereg says 0, results carries the formula 2(alpha-eps)
zero = [r for r in runs if r["epr"] == 0.0][0]
assert zero["Lambda"] == 0.0
assert res[zero["key"]]["Lambda"] == 2.0
assert pre["grid"]["eps_over_alpha=0"]["attractor"] is None

# the two windowed exponents that locate the crossover, at eps/alpha = 0.999
soft = [r for r in runs if r["epr"] == 0.999][0]
windows = {}
for lo, hi, lbl in ((1., 1e2, "early_LambdaT_lt_1"), (1e3, 2e4, "late_LambdaT_gt_1")):
    m = (soft["t"] >= lo) & (soft["t"] <= hi)
    windows[lbl] = float(np.polyfit(np.log10(soft["t"][m]),
                                    np.log10(soft["env"][m]), 1)[0])
    assert np.isclose(windows[lbl], res[soft["key"]][f"window_{lbl}_exponent"],
                      rtol=1e-9), lbl

# the discrete plateau of the preregistered grid is kappa / (1 - R_rk4(-Lambda h))
for r in runs:
    if r["Lambda"] == 0.0:
        continue
    want = pre["grid"][r["pre_key"]]["predicted_plateau_dtheta"]
    assert np.isclose(r["kap"] / (1.0 - R_rk4(-r["Lambda"] * H)), want, rtol=1e-12)
    assert np.isclose(pre["grid"][r["pre_key"]]
                      ["predicted_plateau_continuum_q0_over_Lambda"],
                      r["q0"] / r["Lambda"], rtol=1e-12)

# the free-growth run really does integrate the defect at the injection rate
assert abs(zero["slope_per_time"] / zero["q0"] - 1.0) < 1e-8, zero["slope_per_time"]

# local half-decade slopes, over the fit window env_exp actually uses
for r in runs:
    r["local_fit_window"] = half_decade_slopes(r["t"], r["env"],
                                               r["t"][-1] / 100.0, r["t"][-1])
    r["local_full"] = half_decade_slopes(r["t"], r["env"], r["t"][0], r["t"][-1])

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
STYLE = {
    0.0:   (ps.BLACK,      "-",                       "o", r"$\Lambda = 0$"),
    0.999: (ps.PURPLE,     (0, (1.3, 1.5)),           "D", r"$\Lambda = 0.002$"),
    0.99:  (ps.VERMILLION, (0, (3.0, 1.4, 1.0, 1.4)), "v", r"$\Lambda = 0.02$"),
    0.9:   (ps.ORANGE,     (0, (4.5, 1.8)),           "^", r"$\Lambda = 0.2$"),
    0.5:   (ps.GREEN,      (0, (6.0, 1.5, 1.0, 1.5)), "s", r"$\Lambda = 1$"),
    0.1:   (ps.BLUE,       (0, (2.4, 1.3)),           "o", r"$\Lambda = 1.8$"),
}
ORDER = [0.0, 0.999, 0.99, 0.9, 0.5, 0.1]
by_epr = {r["epr"]: r for r in runs}

ps.use_style()
fig, axes = plt.subplots(
    2, 2, figsize=(ps.TEXTWIDTH_IN, 4.46),
    gridspec_kw=dict(wspace=0.315, hspace=0.42, left=0.098, right=0.995,
                     bottom=0.152, top=0.960))
(axA, axB), (axC, axD) = axes

# ---- (a) envelopes in units of the injection rate -------------------------
tt = np.logspace(-1.6, 3.47, 50)
axA.plot(tt, tt, color=ps.GREY, lw=2.6, alpha=0.40, solid_capstyle="butt",
         zorder=1)
axA.text(0.14, 0.60, r"free growth  $q_0 t$", fontsize=6.8, color="#5A5A5A",
         rotation=39, ha="left", va="bottom", rotation_mode="anchor")

for epr in ORDER:
    r = by_epr[epr]
    col, dash, mk, lab = STYLE[epr]
    axA.plot(r["t"], r["env"] / r["q0"], color=col, ls=dash,
             lw=1.5 if epr == 0.0 else 1.2, zorder=4 if epr == 0.0 else 3,
             label=lab)
    if r["Lambda"] > 0:
        tk = 1.0 / r["Lambda"]
        axA.plot([tk], [np.interp(tk, r["t"], r["env"]) / r["q0"]], marker="o",
                 ms=3.6, mfc="white", mec=col, mew=1.0, ls="none", zorder=6)

axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlim(0.035, 4.0e4)
axA.set_ylim(0.035, 3.0e3)
axA.set_xlabel(r"time  $t$")
axA.set_ylabel(r"$\Delta\theta \,/\, q_0$")
axA.text(0.028, 0.965, "open circles:  $t = 1/\\Lambda$",
         transform=axA.transAxes, fontsize=6.8, ha="left", va="top",
         color=ps.BLACK)
ps.panel_label(axA, "(a)")

# ---- (b) local half-decade slopes, collapsed on Lambda t ------------------
z0 = [s for _, s in by_epr[0.0]["local_fit_window"]]
axB.axhspan(min(z0), max(z0), color=ps.BLACK, alpha=0.10, lw=0, zorder=0)
axB.axhline(0.0, color=ps.BLACK, lw=0.7, ls=(0, (4, 2)), zorder=1)
axB.axvline(1.0, color=ps.GREY, lw=2.4, alpha=0.45, zorder=1)
axB.text(1.25, 1.19, r"$\Lambda t = 1$", fontsize=6.9, color="#5A5A5A",
         ha="left", va="center")
axB.text(2.0e-4, float(np.mean(z0)), "$\\Lambda = 0$\n(fit window)",
         fontsize=6.8, ha="left", va="center", color=ps.BLACK)

for epr in ORDER:
    r = by_epr[epr]
    if r["Lambda"] == 0.0:
        continue
    col, dash, mk, lab = STYLE[epr]
    c = np.array([x for x, _ in r["local_full"]]) * r["Lambda"]
    s = np.array([y for _, y in r["local_full"]])
    axB.plot(c, s, color=col, ls=dash, lw=1.1, marker=mk, ms=3.0, mfc="none",
             mew=0.85, zorder=3)

axB.set_xscale("log")
axB.set_xlim(1.5e-4, 4.0e4)
axB.set_ylim(-0.16, 1.34)
axB.set_xlabel(r"$\Lambda t$")
axB.set_ylabel("local half-decade slope")
axB.set_yticks([0.0, 0.5, 1.0])
ps.panel_label(axB, "(b)")

# ---- (c) exponent against Lambda, with the local slopes -------------------
LIN = 1.0e-3
for epr in ORDER:
    r = by_epr[epr]
    col, dash, mk, lab = STYLE[epr]
    x = r["Lambda"]
    loc = [s for _, s in r["local_fit_window"]]
    axC.plot([x, x], [min(loc), max(loc)], color=col, lw=0.9, zorder=2,
             solid_capstyle="butt")
    axC.plot([x] * len(loc), loc, ls="none", marker="_", ms=5.2, mew=1.0,
             color=col, zorder=3)
    axC.plot([x], [r["p"]], ls="none", marker=mk, ms=4.2, color=col,
             mec=col, mew=0.9, zorder=4)

axC.axhline(0.0, color=ps.BLACK, lw=0.7, ls=(0, (4, 2)), zorder=1)
axC.axhline(1.0, color=ps.GREY, lw=0.7, ls=(0, (1.4, 1.6)), zorder=1)
axC.set_xscale("symlog", linthresh=LIN, linscale=0.42)
axC.set_xlim(-2.2e-4, 3.4)
axC.set_ylim(-0.16, 1.20)
axC.set_xticks([0.0, 1e-2, 1e-1, 1e0])
axC.set_xticklabels(["0", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
axC.set_xlabel(r"contraction rate  $\Lambda = 2(\alpha - \epsilon)$")
axC.set_ylabel("growth exponent")
axC.annotate("two-decade fit $0.374$;\nhalf-decade slopes\n$0.90/0.62/0.07/0.00$",
             xy=(2.0e-3, 0.50), xytext=(6.5e-3, 0.815),
             fontsize=6.6, color=ps.PURPLE, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.PURPLE,
                             shrinkA=2, shrinkB=3))
axC.annotate(f"$p = {by_epr[0.0]['p']:.3f}$",
             xy=(0.0, by_epr[0.0]["p"]), xytext=(6.0, -1.0),
             textcoords="offset points", fontsize=6.9, color=ps.BLACK,
             ha="left", va="center")
ps.panel_label(axC, "(c)")

# ---- (d) the plateau law over three decades in Lambda ---------------------
zz = np.logspace(-4.2, -0.85, 200)                 # z = Lambda h
axD.plot(zz / H, zz / (1.0 - R_rk4(-zz)), color=ps.GREY, lw=2.4, alpha=0.45,
         zorder=1, solid_capstyle="butt")
axD.axhline(1.0, color=ps.BLACK, lw=0.7, ls=(0, (4, 2)), zorder=2)
for epr in ORDER:
    r = by_epr[epr]
    if r["law"] is None:
        continue
    col, dash, mk, lab = STYLE[epr]
    axD.plot([r["Lambda"]], [r["law"]], ls="none", marker=mk, ms=4.4,
             color=col, mec=col, mew=0.9, zorder=4)

law_dev = max(abs(r["law"] - 1.0) for r in runs if r["law"] is not None)
axD.set_xscale("log")
axD.set_xlim(1.1e-3, 3.4)
axD.set_ylim(0.982, 1.062)
axD.set_xlabel(r"contraction rate  $\Lambda$")
axD.set_ylabel(r"plateau $\times\,\Lambda / q_0$")
axD.text(1.5e-3, 1.0535,
         "continuum law  $q_0/\\Lambda$,\n"
         f"held to ${law_dev * 100:.1f}\\,\\%$ over three decades",
         fontsize=6.8, color=ps.BLACK, ha="left", va="top")
axD.annotate(r"$z/[1-R(-z)]$, $z = \Lambda h$" + "\n(no fitted parameter)",
             xy=(0.62, float(0.62 * H / (1.0 - R_rk4(-0.62 * H)))),
             xytext=(1.35e-3, 1.0265), fontsize=6.6, color="#5A5A5A",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="#5A5A5A",
                             shrinkA=2, shrinkB=3))
axD.text(1.35e-3, 0.9845, r"($\Lambda = 0$ has no plateau)", fontsize=6.6,
         color=ps.BLACK, ha="left", va="bottom")
ps.panel_label(axD, "(d)")

for ax in (axA,):
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0,
                                          subs=tuple(np.arange(2, 10) * 0.1),
                                          numticks=60))
    ax.yaxis.set_minor_formatter(NullFormatter())

# one legend for all four panels, below the artwork: in panel (a) the free
# corner is where the two stiffest plateaux run, and in panel (b) the zero
# plateau crosses the whole width, so no panel has room for it
_h, _l = axA.get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", bbox_to_anchor=(0.52, -0.008), ncol=6,
           columnspacing=0.9, handlelength=1.9)

ps.save_pdf(fig, "fig4_dissipation")

# ---------------------------------------------------------------------------
values = {
    "source": "experiments/ll_probe/followup.py, block F7, reproduced here "
              "verbatim and asserted equal to experiments/ll_probe/"
              "results2.json at rtol 1e-9 (the reproduction is bit-identical); "
              "Lambda and the predicted plateaux come from the preregistered "
              "experiments/ll_probe/prereg2.json.",
    "setup": {
        "model": "dtheta/dt = alpha tanh(theta) - eps sinh(theta) cosh(theta)",
        "alpha": 1.0, "h": H, "theta_0": THETA0,
        "integrator": "RK4",
        "defect": "kappa added to theta on every step",
        "held_fixed_across_the_sweep": pre["confound_statement"],
        "caveat_from_prereg": pre["caveat"],
    },
    "lambda_label_warning": {
        "issue": "results2.json stores Lambda = 2.0 for the eps = 0 run, the "
                 "formula 2(alpha - eps) evaluated where its attractor does "
                 "not exist. With eps = 0 the flow has no fixed point and the "
                 "contraction rate is 0.",
        "value_used_here": 0.0,
        "value_stored_in_results2_json": res["eps_over_alpha=0.0"]["Lambda"],
        "authority": "prereg2.json F7_eps_sweep.grid['eps_over_alpha=0']"
                     ".Lambda = 0.0, frozen before the run; attractor: null",
        "why_it_matters": "plotted at the stored value the only growing point "
                          "in the sweep would land at the largest Lambda and "
                          "reverse the conclusion of the figure.",
    },
    "runs": [
        {
            "eps_over_alpha": r["epr"],
            "Lambda": r["Lambda"],
            "kappa": r["kap"], "q0_kappa_over_h": r["q0"], "T": r["T"],
            "Lambda_times_T": r["Lambda"] * r["T"],
            "envelope_exponent_two_decade_fit": r["p"],
            "local_half_decade_slopes_in_fit_window":
                [round(s, 6) for _, s in r["local_fit_window"]],
            "local_half_decade_slopes_whole_run":
                [round(s, 6) for _, s in r["local_full"]],
            "plateau_or_final_dtheta": r["dth_final"],
            "plateau_over_q0": abs(r["dth_final"]) / r["q0"],
            "plateau_times_Lambda_over_q0": r["law"],
            "discrete_prediction_kappa_over_1_minus_R":
                (None if r["Lambda"] == 0.0
                 else r["kap"] / (1.0 - R_rk4(-r["Lambda"] * H))),
        }
        for r in runs
    ],
    "panel_c": {
        "rule": "plan/reports/W0_2_exponent_1540.md -- every fitted exponent "
                "is printed together with its local half-decade slopes.",
        "Lambda_0_fit": by_epr[0.0]["p"],
        "Lambda_0_local_slopes": [round(s, 6) for _, s in
                                  by_epr[0.0]["local_fit_window"]],
        "Lambda_0_reading": "still climbing towards the exact linear law; the "
                            "last half decade gives "
                            f"{by_epr[0.0]['local_fit_window'][-1][1]:.3f}, and "
                            "the measured slope of Delta theta against t is "
                            f"{by_epr[0.0]['slope_per_time']:.6e} against "
                            f"q0 = {by_epr[0.0]['q0']:.6e}.",
        "Lambda_0.002_fit": soft["p"],
        "Lambda_0.002_local_slopes": [round(s, 6) for _, s in
                                      soft["local_fit_window"]],
        "Lambda_0.002_reading": "not an exponent: the fit window straddles "
                                "Lambda t = 1.",
        "Lambda_0.002_windowed_exponents": windows,
    },
    "panel_d": {
        "law": pre["law"],
        "max_deviation_from_continuum_law": law_dev,
        "explained_by": "the discrete plateau kappa / (1 - R_rk4(-Lambda h)), "
                        "i.e. z / (1 - R(-z)) with z = Lambda h; the residual "
                        "after that correction is at most "
                        + f"{max(abs(r['dth_final'] * (1.0 - R_rk4(-r['Lambda'] * H)) / r['kap'] - 1.0) for r in runs if r['law'] is not None) * 100:.2f} %",
        "prereg_tolerance_rel": pre["law_tolerance_rel"],
    },
}
with open(os.path.join(HERE, "fig4_dissipation_values.json"), "w",
          encoding="utf-8") as fh:
    json.dump(values, fh, indent=1)
print(json.dumps(values["panel_c"], indent=1))
