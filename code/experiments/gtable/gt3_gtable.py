"""gt3_gtable.py -- the table of orders, and the question it was built to answer.

    python gt3_gtable.py [--force]

Reads the five `gt2_channels__<field>.json` and `gt1_calibration.json`, writes
`gt3_gtable.json` and `gt3_report.txt` beside this file, and exits non-zero if
a rerun no longer reproduces the committed JSON.

WHAT IS ASSEMBLED
-----------------
    G = log10(E_Boris / E_scheme)
for every (configuration, scheme, channel), with the Boris scheme standing at
G = 0 by construction.  Then the thing the table was actually built for: do
the four channels rank the five schemes in the same order?

TWO WAYS A CELL CAN FAIL TO MEAN WHAT IT LOOKS LIKE, BOTH MARKED
-----------------------------------------------------------------
1.  THE CHANNEL IS ANALYTICALLY DEGENERATE.  Four of the five configurations
    carry no electric field at all (`code/fields/`: E0 = 0 and no induced
    term).  The magnetic force does no work, so |v| is an exact invariant of
    the continuous motion and, separately, all five schemes hold |v| fixed by
    construction or to rounding.  The energy "error" of every scheme in those
    four configurations is therefore accumulated floating-point rounding and
    nothing else, and a ratio of two rounding levels is not a measurement.
    Those twenty cells are computed, printed and excluded from the primary
    rank analysis, and the reason they are excluded is a fact about the field
    and not about the numbers that came out.

    This cuts against the manuscript's own thesis rather than for it: a
    channel filled with rounding ranks the schemes at random and would
    manufacture exactly the disagreement prediction P3 hopes to find.  It is
    excluded for that reason.

2.  THE RESIDUAL HAS FALLEN TO THE REFERENCE.  `gt1_calibration.py` measures
    what each configuration's reference is worth in each of the four channels,
    by scoring a second, independent reference through the same four channels
    as if it were a scheme.  A cell within `REF_LIMIT_FACTOR` of that floor is
    marked, and what it measures is the reference and not the scheme.

WHAT IS DECLARED HERE AND WHAT WAS DECLARED EARLIER
----------------------------------------------------
Earlier, before any run: the definition of G, the four channels and their
statistics, the record, Omega_c^ref, and Spearman's rho as the rank statistic.

Here, because the pre-registration fixed no threshold for "the channels
agree": a pair of channels is called AGREED in a cell when rho = +1, that is
when the two orderings of the four measured schemes are identical, and
DISAGREED when rho <= 0.  The full distribution of rho is written out beside
the verdict so that any other threshold can be applied to the same numbers.
"""
import json
import math
import os
import sys
import time

import numpy as np

import gt_common as G
import map_common as MC
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gt3_gtable.json")
REPORT = os.path.join(HERE, "gt3_report.txt")

#: the four configurations in which the energy channel carries no signal, and
#: why.  A statement about `code/fields/`, checked below against the field
#: objects themselves rather than trusted.
NO_ELECTRIC_FIELD = ["uniform", "B1_radial", "B2_wave", "B3_tilted"]

#: cells whose channel is not small but ZERO, exactly, for an analytic reason,
#: so that what the table reports there is double-precision rounding and the
#: G beside it is a statement about the floating-point format.
#:
#: The uniform configuration and B3 are static and spatially uniform, and both
#: carry no electric field.  vps2 advances the velocity by an exact Rodrigues
#: rotation through -(q/m)|B|h about b_hat; with |B| and b_hat constant that
#: is exactly the rotation the continuous motion performs over one step, so
#: the numerical velocity is the exact velocity and the angle between them is
#: zero.  vps4 is the Yoshida triple jump of vps2 and its three sub-rotations
#: sum to gamma_1 + gamma_0 + gamma_1 = 1 times the same angle, so the same
#: holds.  The Boris scheme rotates by 2 arctan(|B|h/2) instead and has a real
#: phase error; gl4 has a truncation error; the corrector adds a learned
#: increment.  Nothing in this list is inferred from the numbers -- the
#: numbers are checked against it below.
EXACT_PHASE_BY_CONSTRUCTION = [("uniform", "phase", "vps2"),
                               ("uniform", "phase", "vps4"),
                               ("B3_tilted", "phase", "vps2"),
                               ("B3_tilted", "phase", "vps4")]
#: the value below which the check accepts "this is rounding".  Every scheme's
#: genuine phase error in this bundle at Omega h = 0.3 is above 1e-10.
ROUNDING_CEILING = 1e-12

NICE = {"uniform": "uniform", "B1_radial": "B1 radial",
        "B2_wave": "B2 wave", "B3_tilted": "B3 tilted",
        "B4_decaying": "B4 decaying"}
NICE_S = {"boris": "Boris", "corrector": "corrector", "vps2": "vps2",
          "vps4": "vps4", "gl4": "gl4"}
NICE_C = {"trajectory": "trajectory", "phase": "phase",
          "energy": "energy", "spectral": "spectral"}


# ------------------------------------------------------------------ loading -
def load():
    shards = {}
    for f in G.FIELD_NAMES:
        p = os.path.join(HERE, "gt2_channels__%s.json" % f)
        if not os.path.exists(p):
            raise SystemExit("missing %s; run gt2_channels.py first" % p)
        shards[f] = json.load(open(p, encoding="utf-8"))
    cal = json.load(open(os.path.join(HERE, "gt1_calibration.json"),
                         encoding="utf-8"))
    return shards, cal


def verify_no_electric_field():
    """The claim that four configurations carry no electric field, checked
    against the shipped field objects on a scatter of states rather than read
    off the source."""
    fields = MC.make_fields()
    rng_states = [(np.array([0.7, -0.3, 0.2]), 0.0),
                  (np.array([1.3, 0.9, -0.4]), 61.7),
                  (np.array([-0.2, 1.1, 0.05]), 635.7)]
    out = {}
    for name, f in fields.items():
        m = 0.0
        for r, t in rng_states:
            m = max(m, float(np.abs(np.asarray(f.E(r, t))).max()))
        out[name] = {"max_abs_E_on_probe_states": m,
                     "electric_field_identically_zero": bool(m == 0.0)}
    return out


# ------------------------------------------------------------------- G table
def applicable_floor(cal, f, hname, c, stat_key):
    """The floor that applies to a configuration, in the units of a channel.

    Where `gt2_channels.py` scores against a closed form, the floor is what
    evaluating that closed form in float64 costs, measured against the same
    form carried in mpmath at forty digits.  Where it scores against DOP853 at
    rtol 3e-14, the floor is the distance to an independent DOP853 at rtol
    1e-13.  Using the second where the first applies would report DOP853's
    error as the closed form's, which is three orders too pessimistic.
    """
    tight = cal["reference_floors"]["closed_form_float64_vs_mpmath40_ic0"]
    if f in tight and stat_key == "primary":
        return {"ic0": tight[f][hname][c], "median": tight[f][hname][c],
                "kind": "closed form in float64 vs mpmath at 40 digits"}
    fl = cal["reference_floors"]["per_configuration"][f][hname][c][
        "floor_primary_statistic" if stat_key == "primary" else "floor_rms"]
    return {"ic0": fl["ic0"], "median": fl["median"],
            "kind": "primary reference vs an independent second reference"}


def build_table(shards, cal, stat_key="primary"):
    """G for every (horizon, field, channel, scheme), plus the flags."""
    tbl = {}
    for hname in G.HORIZONS:
        for f in G.FIELD_NAMES:
            runs = shards[f]["runs"]
            for c in G.CHANNELS:
                base = np.asarray(runs["%s|%s" % (hname, G.BASE)][c][stat_key],
                                  dtype=float)
                fl = applicable_floor(cal, f, hname, c, stat_key)
                rec = {"floor_ic0": fl["ic0"],
                       "floor_median": fl["median"],
                       "floor_kind": fl["kind"],
                       "degenerate": bool(c == "energy"
                                          and f in NO_ELECTRIC_FIELD),
                       "degenerate_reason":
                           ("E = 0 identically in this configuration, so |v| "
                            "is an exact invariant of the continuous motion "
                            "and every scheme's energy error is accumulated "
                            "rounding")
                           if (c == "energy" and f in NO_ELECTRIC_FIELD)
                           else None}
                for s in G.SCHEMES:
                    e = np.asarray(runs["%s|%s" % (hname, s)][c][stat_key],
                                   dtype=float)
                    g = np.array([G.G(base[i], e[i]) for i in range(len(e))])
                    exact = (f, c, s) in EXACT_PHASE_BY_CONSTRUCTION
                    rec[s] = {
                        "error": G.stat(e),
                        "G": G.stat(g),
                        "G_ic0": float(g[0]),
                        "reference_limited_ic0": bool(
                            math.isfinite(fl["ic0"]) and fl["ic0"] > 0
                            and e[0] <= G.REF_LIMIT_FACTOR * fl["ic0"]),
                        "error_over_floor_ic0": (float(e[0] / fl["ic0"])
                                                 if fl["ic0"] > 0
                                                 else float("inf")),
                        "exact_by_construction": bool(exact),
                        "rounding_check_passes": (bool(e[0] < ROUNDING_CEILING)
                                                  if exact else None),
                    }
                    if c == "spectral":
                        # a power, so its G is twice the amplitude-equivalent
                        rec[s]["G_amplitude_equivalent_ic0"] = float(
                            g[0] / 2.0) if math.isfinite(g[0]) else g[0]
                tbl["%s|%s|%s" % (hname, f, c)] = rec
    return tbl


# ------------------------------------------------- the question of the table
def rank_analysis(shards, tbl, hname, stat_key="primary",
                  schemes=None, include_degenerate=False,
                  spectral_key=None, tie_exact=False, drop_pairs=(),
                  fields=None):
    """Do the channels rank the schemes alike?

    One rho per (configuration, initial condition, channel pair).  Channels
    that are analytically degenerate in a configuration are dropped unless
    `include_degenerate`, in which case the run is the control that shows what
    keeping them would have produced.

    `tie_exact` replaces the value of a cell that is exactly zero by
    construction with a single common value, so that the order in which
    rounding happens to place two schemes whose true error is zero does not
    enter the rank statistic.  `drop_pairs` removes a named channel pair, and
    is used for the one pair with a built-in reason to agree.
    """
    schemes = schemes or [s for s in G.SCHEMES if s != G.BASE]
    fields = fields or G.FIELD_NAMES
    per_cell = {}
    for f in fields:
        runs = shards[f]["runs"]
        chans = [c for c in G.CHANNELS
                 if include_degenerate
                 or not tbl["%s|%s|%s" % (hname, f, c)]["degenerate"]]
        for i in range(MC.N_IC):
            err = {}
            for c in chans:
                key = (spectral_key if (c == "spectral" and spectral_key)
                       else stat_key)
                err[c] = {s: float(runs["%s|%s" % (hname, s)][c][key][i])
                          for s in schemes}
                if tie_exact:
                    tied = [s for s in schemes
                            if (f, c, s) in EXACT_PHASE_BY_CONSTRUCTION]
                    for s in tied:
                        err[c][s] = 0.0
            for pair, v in G.rank_agreement(err, schemes).items():
                if pair in drop_pairs:
                    continue
                per_cell.setdefault(pair, {})[
                    "%s|ic%d" % (f, i)] = v
    summ = {}
    allr = []
    for pair, cells in per_cell.items():
        rho = np.array([v["spearman"] for v in cells.values()])
        tau = np.array([v["kendall_tau_b"] for v in cells.values()])
        allr.append(rho)
        summ[pair] = {
            "n_cells": int(len(rho)),
            "spearman_median": float(np.nanmedian(rho)),
            "spearman_mean": float(np.nanmean(rho)),
            "spearman_min": float(np.nanmin(rho)),
            "spearman_max": float(np.nanmax(rho)),
            "kendall_median": float(np.nanmedian(tau)),
            "n_agreed_rho_eq_1": int(np.sum(rho >= 1.0 - 1e-12)),
            "n_disagreed_rho_le_0": int(np.sum(rho <= 0.0)),
            "per_cell": cells,
        }
    flat = np.concatenate(allr) if allr else np.array([])
    return {"pairs": summ,
            "overall": {
                "n_pair_cells": int(flat.size),
                "spearman_median": float(np.nanmedian(flat)) if flat.size
                else float("nan"),
                "spearman_mean": float(np.nanmean(flat)) if flat.size
                else float("nan"),
                "n_agreed_rho_eq_1": int(np.sum(flat >= 1.0 - 1e-12)),
                "fraction_agreed": float(np.mean(flat >= 1.0 - 1e-12))
                if flat.size else float("nan"),
                "n_disagreed_rho_le_0": int(np.sum(flat <= 0.0)),
                "fraction_disagreed": float(np.mean(flat <= 0.0))
                if flat.size else float("nan"),
                "schemes_ranked": schemes,
                "degenerate_channels_included": bool(include_degenerate),
            }}


def coarse_agreement(shards, tbl, hname, stat_key="primary"):
    """Where the agreement of the channels comes from, and where it fails.

    Spearman's rho over four schemes is a summary and hides the shape of the
    thing.  This asks the coarser question directly: do the channels agree on
    *which two schemes are the best two*?  If they do, the agreement is at the
    level of "the fourth-order schemes beat the second-order one and the
    corrector" and the disagreement lives inside each pair.
    """
    schemes = [s for s in G.SCHEMES if s != G.BASE]
    out = {"per_cell": {}, "top2_counts": {}}
    n_tot = 0
    n_vps4_gl4 = 0
    fails = []
    for f in G.FIELD_NAMES:
        runs = shards[f]["runs"]
        for c in G.CHANNELS:
            if tbl["%s|%s|%s" % (hname, f, c)]["degenerate"]:
                continue
            v = sorted((float(runs["%s|%s" % (hname, s)][c][stat_key][0]), s)
                       for s in schemes)
            top2 = tuple(sorted(s for _, s in v[:2]))
            out["per_cell"]["%s|%s" % (f, c)] = list(top2)
            key = "+".join(top2)
            out["top2_counts"][key] = out["top2_counts"].get(key, 0) + 1
            n_tot += 1
            if top2 == ("gl4", "vps4"):
                n_vps4_gl4 += 1
            else:
                fails.append("%s|%s -> %s" % (f, c, key))
    out["n_cells"] = n_tot
    out["n_top2_is_vps4_and_gl4"] = n_vps4_gl4
    out["cells_where_it_is_not"] = fails
    out["reading"] = (
        "the two fourth-order schemes are the best two in %d of the %d "
        "measured (configuration, channel) cells; every exception is in the "
        "phase channel, where vps2 rises because its rotation angle is the "
        "exact one whenever |B| is constant along the step" % (n_vps4_gl4,
                                                               n_tot))
    return out


def ranking_strings(shards, hname, stat_key="primary"):
    """The literal ordering each channel puts the schemes in, per
    configuration, at the canonical initial condition.  This is the table's
    answer in the form a reader can check by eye."""
    out = {}
    for f in G.FIELD_NAMES:
        runs = shards[f]["runs"]
        d = {}
        for c in G.CHANNELS:
            v = [(float(runs["%s|%s" % (hname, s)][c][stat_key][0]), s)
                 for s in G.SCHEMES]
            v.sort()
            d[c] = " < ".join(s for _, s in v)
        out[f] = d
    return out


# --------------------------------------------------------------- the LaTeX --
def fmt_g(x, limited=False, degenerate=False, exact=False):
    if degenerate:
        return "$\\dagger$"
    if not math.isfinite(x):
        return "$\\infty$" if x > 0 else "---"
    body = "%+.2f" % x if abs(x) >= 0.005 else "0.00"
    marks = ("\\ddagger" if exact else "") + ("\\ast" if limited else "")
    if marks:
        return "$%s^{%s}$" % (body, marks)
    return "$%s$" % body


def latex_table(tbl, hname="H_paper"):
    L = []
    L.append("%% Built by experiments/gtable/gt3_gtable.py from")
    L.append("%% gt3_gtable.json.  Do not edit by hand.")
    L.append("\\begin{table}[tbp]")
    L.append("  \\centering")
    L.append("  \\caption{Orders gained on the Boris scheme, channel by "
             "channel.  Each entry is")
    L.append("    $G = \\log_{10}(E_{\\mathrm{Boris}}/E_{\\mathrm{scheme}})$, "
             "so a positive $G$ is an")
    L.append("    advantage over the Boris scheme and the Boris column is "
             "zero by construction.")
    L.append("    All runs are at $\\Omega h = 0.3$ over $19.1$ gyro-orbits, "
             "each configuration")
    L.append("    scored against its own reference.  The trajectory channel "
             "is the root mean")
    L.append("    square of the position error in Larmor radii, the phase "
             "channel the median")
    L.append("    angle between the numerical and the reference velocity over "
             "the second half")
    L.append("    of the run, the energy channel the median relative energy "
             "error over the same")
    L.append("    window, and the spectral channel the integral of the power "
             "spectrum of the")
    L.append("    position error below $f/\\Omega_c = 0.2$, taken with "
             "$\\Omega_c$ at its initial")
    L.append("    value in each configuration.  The spectral entry is a ratio "
             "of powers and is")
    L.append("    therefore twice the corresponding ratio of amplitudes.  "
             "$\\dagger$: the")
    L.append("    configuration carries no electric field, so the kinetic "
             "energy is an exact")
    L.append("    invariant and every entry in that row would be a ratio of "
             "rounding errors.")
    L.append("    $\\ddagger$: the scheme's error in that channel is not "
             "small but exactly zero,")
    L.append("    the field being constant and the scheme rotating the "
             "velocity through the exact")
    L.append("    angle, so the entry measures double precision and not the "
             "scheme.")
    L.append("    $\\ast$: the residual has fallen within a factor of ten of "
             "the reference.")
    L.append("    The corrector is a single committed checkpoint, not an "
             "ensemble.}")
    L.append("  \\label{tab:gtable}")
    L.append("  \\footnotesize")
    L.append("  \\setlength{\\tabcolsep}{4pt}")
    L.append("  \\begin{tabular}{@{}llrrrrr@{}}")
    L.append("    \\toprule")
    L.append("    Configuration & Channel & Boris & corrector & vps2 & vps4 "
             "& gl4 \\\\")
    L.append("    \\midrule")
    for f in G.FIELD_NAMES:
        for j, c in enumerate(G.CHANNELS):
            rec = tbl["%s|%s|%s" % (hname, f, c)]
            head = NICE[f] if j == 0 else ""
            cells = []
            for s in G.SCHEMES:
                cells.append(fmt_g(rec[s]["G_ic0"],
                                   rec[s]["reference_limited_ic0"],
                                   rec["degenerate"],
                                   rec[s]["exact_by_construction"]))
            L.append("    %-14s & %-10s & %s \\\\"
                     % (head, NICE_C[c], " & ".join(cells)))
        if f != G.FIELD_NAMES[-1]:
            L.append("    \\addlinespace")
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


def latex_rank_table(rk, hname="H_paper"):
    L = []
    L.append("%% Built by experiments/gtable/gt3_gtable.py.  Do not edit.")
    L.append("\\begin{table}[tbp]")
    L.append("  \\centering")
    L.append("  \\caption{Rank agreement between the channels of "
             "Table~\\ref{tab:gtable}.  For each")
    L.append("    configuration and each of the eight initial conditions the "
             "four measured")
    L.append("    schemes are ordered by each channel and the two orderings "
             "compared by")
    L.append("    Spearman's $\\rho$; the table reports the median over those "
             "cells, the number")
    L.append("    in which the two orderings are identical, and the number in "
             "which they are")
    L.append("    uncorrelated or opposed.  The energy channel is excluded "
             "where the")
    L.append("    configuration carries no electric field.}")
    L.append("  \\label{tab:gtable_ranks}")
    L.append("  \\footnotesize")
    L.append("  \\begin{tabular}{@{}lrrrr@{}}")
    L.append("    \\toprule")
    L.append("    Channel pair & cells & median $\\rho$ & identical & "
             "$\\rho \\le 0$ \\\\")
    L.append("    \\midrule")
    for pair, v in sorted(rk["pairs"].items()):
        a, b = pair.split("|")
        L.append("    %s vs.\\ %s & %d & $%+.2f$ & %d & %d \\\\"
                 % (NICE_C[a], NICE_C[b], v["n_cells"],
                    v["spearman_median"], v["n_agreed_rho_eq_1"],
                    v["n_disagreed_rho_le_0"]))
    o = rk["overall"]
    L.append("    \\midrule")
    L.append("    all pairs & %d & $%+.2f$ & %d & %d \\\\"
             % (o["n_pair_cells"], o["spearman_median"],
                o["n_agreed_rho_eq_1"], o["n_disagreed_rho_le_0"]))
    L.append("    \\bottomrule")
    L.append("  \\end{tabular}")
    L.append("\\end{table}")
    return "\n".join(L)


# --------------------------------------------------------------------- main -
def main():
    force = "--force" in sys.argv
    t0 = time.time()
    shards, cal = load()

    out = {"meta": {
        "what": "G = log10(E_Boris/E_scheme), four channels, five "
                "configurations, five schemes, Omega h = 0.3",
        "G_definition": "log10(E_Boris / E_scheme); positive is better than "
                        "the Boris scheme; the Boris row is 0 by construction",
        "channels": G.CHANNELS,
        "channel_statistics": {
            "trajectory": "rms of the position error over the record, r_L",
            "phase": "median of atan2(|v x v_ref|, v.v_ref) over the second "
                     "half, radians",
            "energy": "median relative energy error over the second half",
            "spectral": "PSD integral of the position error over "
                        "nu < 0.2 Omega_c^ref, Hann window -- a POWER",
        },
        "spectral_is_a_power": "G in the spectral column is a ratio of powers "
                               "and is therefore twice the corresponding "
                               "ratio of amplitudes; G/2 is reported beside "
                               "it.  Ranks are the same under either",
        "channels_are_not_independent": "the spectral channel is a band-"
            "limited functional of the same position-error series the "
            "trajectory channel takes the root mean square of, so the pair "
            "(trajectory, spectral) is the one pair with a built-in reason to "
            "agree; it is reported with the rest and flagged here",
        "rank_statistic": G.RANK_STAT,
        "agreement_rule_declared_here": "a pair of channels is AGREED in a "
            "cell when rho = +1 exactly (identical orderings of the four "
            "measured schemes) and DISAGREED when rho <= 0.  The "
            "pre-registration fixed no threshold, so this one is fixed here "
            "and the whole distribution of rho is written out beside it",
        "base_scheme": G.BASE,
        "schemes_ranked_primary": [s for s in G.SCHEMES if s != G.BASE],
        "why_boris_is_excluded_from_the_ranking": "the Boris scheme is the "
            "base, so its G is 0 in every channel by construction; including "
            "it in the rank comparison would let a constant contributed by "
            "the definition count as agreement between two channels",
        "ref_limit_factor": G.REF_LIMIT_FACTOR,
        "dt": G.DT, "horizons_samples": G.HORIZONS,
        "n_initial_conditions": MC.N_IC,
        "n_random_draws": G.N_RANDOM_DRAWS,
        "corrector_checkpoint": "boris_corrector_b4.pt -- ONE checkpoint; the "
                                "corrector column is a single run and not an "
                                "ensemble over seeds (that is W16)",
    }}

    out["electric_field_check"] = verify_no_electric_field()
    bad = [f for f in NO_ELECTRIC_FIELD
           if not out["electric_field_check"][f]["electric_field_identically_zero"]]
    if bad:
        print("the no-electric-field claim fails for %s" % bad)
        return 1
    out["meta"]["degenerate_energy_configurations"] = NO_ELECTRIC_FIELD

    out["table"] = build_table(shards, cal, "primary")
    out["table_rms"] = build_table(shards, cal, "rms")

    out["rankings"] = {h: ranking_strings(shards, h) for h in G.HORIZONS}
    out["coarse_agreement"] = {h: coarse_agreement(shards, out["table"], h)
                               for h in G.HORIZONS}

    # the analytic exactness claim, checked against the numbers rather than
    # asserted over them
    ex = {}
    for f, c, s in EXACT_PHASE_BY_CONSTRUCTION:
        for h in G.HORIZONS:
            rec = out["table"]["%s|%s|%s" % (h, f, c)][s]
            ex["%s|%s|%s|%s" % (h, f, c, s)] = {
                "measured": rec["error"]["ic0"],
                "rounding_ceiling": ROUNDING_CEILING,
                "passes": bool(rec["error"]["ic0"] < ROUNDING_CEILING)}
    out["exact_by_construction_check"] = ex
    if not all(v["passes"] for v in ex.values()):
        print("an exact-by-construction cell is not at the rounding level")
        return 1

    out["rank_agreement"] = {}
    for h in G.HORIZONS:
        out["rank_agreement"][h] = {
            "primary_4_schemes": rank_analysis(shards, out["table"], h),
            "all_5_schemes_incl_boris": rank_analysis(
                shards, out["table"], h,
                schemes=list(G.SCHEMES)),
            "control_degenerate_energy_kept": rank_analysis(
                shards, out["table"], h, include_degenerate=True),
            "control_exact_cells_tied": rank_analysis(
                shards, out["table"], h, tie_exact=True),
            "control_built_in_pair_dropped": rank_analysis(
                shards, out["table"], h,
                drop_pairs=("trajectory|spectral",)),
            "robustness_all_channels_as_rms": rank_analysis(
                shards, out["table_rms"], h, stat_key="rms"),
            "sensitivity_omega_c_alternative": rank_analysis(
                shards, out["table"], h,
                spectral_key="p_band_omega_c_alt"),
            "B4_only_energy_carries_signal": rank_analysis(
                shards, out["table"], h, fields=["B4_decaying"]),
        }

    # ------------------------------------- the W14 number, taken apart
    # W14 reports a median rank agreement of +0.00 between the energy and the
    # trajectory channels over its 120 cells.  Ninety-six of those cells are
    # in the four configurations that carry no electric field, where the
    # energy channel is rounding.  The split is computed here from W14's own
    # committed file, not from anything measured in this directory.
    mp3 = json.load(open(os.path.join(G.EXP, "map", "mp3_maps.json"),
                         encoding="utf-8"))
    byf = {}
    for k, v in mp3["map_C"].items():
        h, f, dts = k.split("|")
        byf.setdefault(f, []).append(
            v["_rank_agreement_energy_vs_position"]["median"])
    allv = [x for v in byf.values() for x in v]
    signal = byf["B4_decaying"]
    rounding = [x for f, v in byf.items() if f != "B4_decaying" for x in v]
    out["W14_number_decomposed"] = {
        "W14_reported_median_over_all_cells":
            mp3["summary"]["rank_agreement_median_over_all_cells"],
        "recomputed_median_over_all_cells": float(np.median(allv)),
        "n_cells": len(allv),
        "median_where_the_energy_channel_carries_signal_B4":
            float(np.median(signal)),
        "n_cells_B4": len(signal),
        "median_where_the_energy_channel_is_rounding":
            float(np.median(rounding)),
        "n_cells_rounding": len(rounding),
        "per_field_median": {f: float(np.median(v)) for f, v in byf.items()},
        "what_this_means":
            "the +0.00 the manuscript's map reports is the median of a "
            "genuine +0.80, measured in the one configuration whose energy "
            "channel has anything in it, and a -0.20 produced by ranking five "
            "schemes on their accumulated rounding in the four that do not.  "
            "Four fifths of the cells behind the number are of the second "
            "kind",
        "cross_check_against_this_directory":
            "the W14 cell H_paper|B4_decaying|0.3 gives "
            + "%.3f" % mp3["map_C"]["H_paper|B4_decaying|0.3"][
                "_rank_agreement_energy_vs_position"]["median"]
            + "; the same quantity measured here over the same five schemes "
              "is reported under rank_agreement.H_paper."
              "all_5_schemes_incl_boris.pairs['trajectory|energy']",
        "measured_here_H_paper_B4_all5_trajectory_energy":
            rank_analysis(shards, out["table"], "H_paper",
                          schemes=list(G.SCHEMES),
                          fields=["B4_decaying"])["pairs"]
            ["trajectory|energy"]["spearman_median"],
        "W14_H_paper_B4_0.3": mp3["map_C"]["H_paper|B4_decaying|0.3"][
            "_rank_agreement_energy_vs_position"]["median"],
    }

    # ---------------------------------------------- the Omega_c sensitivity
    sens = {}
    for h in G.HORIZONS:
        for f in G.FIELD_NAMES:
            runs = shards[f]["runs"]
            base_p = runs["%s|%s" % (h, G.BASE)]["spectral"]
            worst = 0.0
            for s in G.SCHEMES:
                if s == G.BASE:
                    continue
                r = runs["%s|%s" % (h, s)]["spectral"]
                g1 = G.G(base_p["p_band"][0], r["p_band"][0])
                g2 = G.G(base_p["p_band_omega_c_alt"][0],
                         r["p_band_omega_c_alt"][0])
                worst = max(worst, abs(g1 - g2))
            sens["%s|%s" % (h, f)] = {
                "omega_c_ref": shards[f]["meta"]["omega_c_ref_per_ic"][0],
                "omega_c_alt": shards[f]["meta"][
                    "omega_c_alt_per_ic_%s" % h][0],
                "worst_abs_change_in_spectral_G_ic0": worst,
            }
    out["omega_c_sensitivity"] = sens
    out["omega_c_sensitivity_worst_over_all"] = max(
        v["worst_abs_change_in_spectral_G_ic0"] for v in sens.values())

    # -------------------------------------------------- reference limitation
    lim = []
    for k, rec in out["table"].items():
        for s in G.SCHEMES:
            if rec[s]["reference_limited_ic0"] and not rec["degenerate"]:
                lim.append("%s|%s (error %.3e, floor %.3e)"
                           % (k, s, rec[s]["error"]["ic0"], rec["floor_ic0"]))
    out["reference_limited_cells"] = lim
    out["n_reference_limited_cells"] = len(lim)

    # --------------------------------------------------------- the verdicts
    rkP = out["rank_agreement"]["H_paper"]["primary_4_schemes"]["overall"]
    rkC = out["rank_agreement"]["H_crossover"]["primary_4_schemes"]["overall"]

    corr_pos = {}
    for h in G.HORIZONS:
        n_pos = n_tot = 0
        where = []
        for f in G.FIELD_NAMES:
            for c in G.CHANNELS:
                rec = out["table"]["%s|%s|%s" % (h, f, c)]
                if rec["degenerate"]:
                    continue
                n_tot += 1
                if rec["corrector"]["G_ic0"] > 0:
                    n_pos += 1
                    where.append("%s|%s" % (f, c))
        corr_pos[h] = {"n_positive": n_pos, "n_cells": n_tot,
                       "where": where}
    out["corrector_positive_G"] = corr_pos

    classical = {}
    for h in G.HORIZONS:
        neg = []
        n_tot = 0
        for f in G.FIELD_NAMES:
            for c in G.CHANNELS:
                rec = out["table"]["%s|%s|%s" % (h, f, c)]
                if rec["degenerate"]:
                    continue
                for s in ("vps2", "vps4", "gl4"):
                    n_tot += 1
                    if not rec[s]["G_ic0"] > 0:
                        neg.append("%s|%s|%s (G = %+.3f)"
                                   % (f, c, s, rec[s]["G_ic0"]))
        classical[h] = {"n_cells": n_tot, "n_not_positive": len(neg),
                        "where": neg}
    out["classical_not_positive"] = classical

    # P1 makes a claim about *which* configurations, not only about a count.
    by_field = {}
    for f in G.FIELD_NAMES:
        cells = [out["table"]["H_paper|%s|%s" % (f, c)] for c in G.CHANNELS]
        cells = [r for r in cells if not r["degenerate"]]
        pos = sum(1 for r in cells if r["corrector"]["G_ic0"] > 0)
        by_field[f] = {"n_positive": pos, "n_channels": len(cells),
                       "majority_positive": bool(pos * 2 > len(cells))}
    p1_predicted_positive = {"uniform", "B4_decaying"}
    p1_field_pattern_holds = all(
        by_field[f]["majority_positive"] == (f in p1_predicted_positive)
        for f in G.FIELD_NAMES)

    out["prereg_status"] = {
        "P1": {
            "claim": "G of the corrector is negative almost everywhere; "
                     "positive in the decaying and uniform configurations and "
                     "negative in the other three",
            "measured_positive_cells_H_paper":
                corr_pos["H_paper"]["n_positive"],
            "of_cells": corr_pos["H_paper"]["n_cells"],
            "where": corr_pos["H_paper"]["where"],
            "per_field": by_field,
            "field_pattern_predicted_holds": bool(p1_field_pattern_holds),
            "status": ("partially confirmed -- the field pattern it names is "
                       "right and 'negative almost everywhere' is not"
                       if p1_field_pattern_holds
                       and corr_pos["H_paper"]["n_positive"] * 2
                       >= corr_pos["H_paper"]["n_cells"]
                       else ("confirmed"
                             if corr_pos["H_paper"]["n_positive"] * 2
                             < corr_pos["H_paper"]["n_cells"]
                             and p1_field_pattern_holds else "refuted")),
        },
        "P2": {
            "claim": "G of the classical schemes is positive in every channel "
                     "and every configuration",
            "n_cells_H_paper": classical["H_paper"]["n_cells"],
            "n_not_positive_H_paper": classical["H_paper"]["n_not_positive"],
            "where": classical["H_paper"]["where"],
            "status": "confirmed" if classical["H_paper"]["n_not_positive"] == 0
                      else "refuted",
        },
        "P3": {
            "claim": "the channels are not agreed by rank, and this is the "
                     "main result of the table",
            "H_paper_median_rho": rkP["spearman_median"],
            "H_paper_fraction_identical_orderings": rkP["fraction_agreed"],
            "H_paper_fraction_rho_le_0": rkP["fraction_disagreed"],
            "H_crossover_median_rho": rkC["spearman_median"],
            "verdict_by_the_identical_ordering_rule":
                "confirmed" if rkP["fraction_agreed"] < 0.5 else "refuted",
            "verdict_by_the_median_rho":
                "refuted" if rkP["spearman_median"] >= 0.5 else "confirmed",
            "status": ("refuted in the sense that matters: the median rho is "
                       "%+.2f, substantial positive agreement.  The "
                       "identical-ordering rule declared above reads the "
                       "other way only because an identical ordering of four "
                       "schemes is a demanding thing to ask"
                       % rkP["spearman_median"])
                      if rkP["spearman_median"] >= 0.5 else "confirmed",
        },
        "P4": {
            "claim": "negative control: if the channels are agreed by rank, "
                     "the manuscript's thesis is weaker than written and that "
                     "is the first paragraph of the report",
            "triggered": bool(rkP["spearman_median"] >= 0.5),
            "status": ("TRIGGERED" if rkP["spearman_median"] >= 0.5
                       else "not triggered"),
            "on_what": "median Spearman rho over the 144 (configuration, "
                       "initial condition, channel pair) cells at H_paper",
        },
    }

    out["latex"] = {
        "tab_gtable": latex_table(out["table"], "H_paper"),
        "tab_gtable_ranks": latex_rank_table(
            out["rank_agreement"]["H_paper"]["primary_4_schemes"], "H_paper"),
    }

    out["meta"]["wall_s"] = time.time() - t0
    write_report(out)
    print(open(REPORT, encoding="utf-8").read())
    return check_or_write(OUT, json.loads(json.dumps(G.clean(out))),
                          force=force)


def write_report(out):
    L = []
    W = L.append
    W("W15 -- the table of orders, channel by channel")
    W("=" * 78)
    W("")
    w14 = out["W14_number_decomposed"]
    W("HEADLINE")
    W("The four channels are substantially AGREED by rank: median Spearman")
    W("rho = %+.2f over the %d (configuration, initial condition, channel"
      % (out["rank_agreement"]["H_paper"]["primary_4_schemes"]["overall"]
         ["spearman_median"],
         out["rank_agreement"]["H_paper"]["primary_4_schemes"]["overall"]
         ["n_pair_cells"]))
    W("pair) cells at the paper horizon.  Prediction P4, the negative")
    W("control, is TRIGGERED.  The +0.00 the manuscript's map reports for the")
    W("energy-against-trajectory agreement decomposes as %+.2f over the %d"
      % (w14["median_where_the_energy_channel_carries_signal_B4"],
         w14["n_cells_B4"]))
    W("cells whose energy channel carries a signal and %+.2f over the %d"
      % (w14["median_where_the_energy_channel_is_rounding"],
         w14["n_cells_rounding"]))
    W("cells in which every scheme's energy error is accumulated rounding,")
    W("the field carrying no electric part.  Four fifths of the cells behind")
    W("the number are of the second kind.")
    W("")
    W("=" * 78)
    W("")
    W("G = log10(E_Boris / E_scheme) at Omega h = 0.3, canonical initial")
    W("condition, over 19.1 gyro-orbits (H_paper).  '.' marks a cell whose")
    W("channel is analytically degenerate in that configuration; '*' a cell")
    W("within a factor of ten of the reference; '+' a cell whose error is")
    W("exactly zero by construction, so that the entry measures rounding.")
    W("")
    hdr = "%-13s %-11s" % ("configuration", "channel")
    for s in G.SCHEMES:
        hdr += "%11s" % s
    W(hdr)
    W("-" * len(hdr))
    for f in G.FIELD_NAMES:
        for j, c in enumerate(G.CHANNELS):
            rec = out["table"]["H_paper|%s|%s" % (f, c)]
            row = "%-13s %-11s" % (NICE[f] if j == 0 else "", c)
            for s in G.SCHEMES:
                if rec["degenerate"]:
                    row += "%11s" % "."
                else:
                    g = rec[s]["G_ic0"]
                    mark = ""
                    if rec[s]["exact_by_construction"]:
                        mark = "+"
                    elif rec[s]["reference_limited_ic0"]:
                        mark = "*"
                    row += "%11s" % ((("%+.2f" % g) if math.isfinite(g)
                                      else "inf") + mark)
            W(row)
        W("")
    W("orderings by channel, best first, canonical initial condition")
    W("-" * 78)
    for f in G.FIELD_NAMES:
        W("  %s" % NICE[f])
        for c in G.CHANNELS:
            deg = out["table"]["H_paper|%s|%s" % (f, c)]["degenerate"]
            W("    %-11s %s%s" % (c, out["rankings"]["H_paper"][f][c],
                                  "   (degenerate: rounding only)"
                                  if deg else ""))
    W("")
    W("rank agreement, four measured schemes, Spearman rho")
    W("-" * 78)
    rk = out["rank_agreement"]["H_paper"]["primary_4_schemes"]
    for pair, v in sorted(rk["pairs"].items()):
        W("  %-26s cells %3d  median rho %+.2f  identical %2d  rho<=0 %2d"
          % (pair, v["n_cells"], v["spearman_median"],
             v["n_agreed_rho_eq_1"], v["n_disagreed_rho_le_0"]))
    o = rk["overall"]
    W("  %-26s cells %3d  median rho %+.2f  identical %2d  rho<=0 %2d"
      % ("ALL PAIRS", o["n_pair_cells"], o["spearman_median"],
         o["n_agreed_rho_eq_1"], o["n_disagreed_rho_le_0"]))
    W("")
    W("controls, robustness and sensitivity")
    W("-" * 78)
    for k in ("all_5_schemes_incl_boris", "control_degenerate_energy_kept",
              "control_exact_cells_tied", "control_built_in_pair_dropped",
              "robustness_all_channels_as_rms",
              "sensitivity_omega_c_alternative",
              "B4_only_energy_carries_signal"):
        v = out["rank_agreement"]["H_paper"][k]["overall"]
        W("  %-34s median rho %+.2f  identical %d/%d"
          % (k, v["spearman_median"], v["n_agreed_rho_eq_1"],
             v["n_pair_cells"]))
    v = out["rank_agreement"]["H_crossover"]["primary_4_schemes"]["overall"]
    W("  %-34s median rho %+.2f  identical %d/%d"
      % ("H_crossover, primary", v["spearman_median"],
         v["n_agreed_rho_eq_1"], v["n_pair_cells"]))
    W("  %-34s %.3e"
      % ("worst |dG| from the Omega_c choice",
         out["omega_c_sensitivity_worst_over_all"]))
    W("  %-34s %d" % ("reference-limited cells",
                      out["n_reference_limited_cells"]))
    W("")
    W("")
    W("where the agreement comes from: the best two schemes, per cell")
    W("-" * 78)
    ca = out["coarse_agreement"]["H_paper"]
    W("  the two fourth-order schemes are the best two in %d of %d cells"
      % (ca["n_top2_is_vps4_and_gl4"], ca["n_cells"]))
    for s in ca["cells_where_it_is_not"]:
        W("    not there: %s" % s)
    W("")
    W("the W14 number taken apart (from ../map/mp3_maps.json, not remeasured)")
    W("-" * 78)
    for f, v in w14["per_field_median"].items():
        W("  %-13s median rho (energy vs trajectory, 24 cells) %+.3f" % (f, v))
    W("  %-13s %+.3f over %d cells"
      % ("all", w14["recomputed_median_over_all_cells"], w14["n_cells"]))
    W("  W14 H_paper|B4|0.3 = %+.3f ; the same measured here = %+.3f"
      % (w14["W14_H_paper_B4_0.3"],
         w14["measured_here_H_paper_B4_all5_trajectory_energy"]))
    W("")
    W("pre-registration")
    W("-" * 78)
    for k, v in out["prereg_status"].items():
        W("  %s  %s" % (k, v["status"]))
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
