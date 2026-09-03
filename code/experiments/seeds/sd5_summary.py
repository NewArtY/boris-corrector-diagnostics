"""SD5: the ensemble, assembled.

    python sd5_summary.py            assemble, then check against the file
    python sd5_summary.py --force    overwrite sd5_summary.json

Reads sd1_calibration.json, sd2_training.json, sd3_ensemble.json and
sd4_external.json.  Writes sd5_summary.json and sd5_report.txt, and prints
everything the report quotes, so that no number in the report is typed by hand.

WHAT IS DECIDED HERE, AND IT WAS DECIDED BEFORE THE NUMBERS
-----------------------------------------------------------
Spread is reported as the median and the interquartile range, which is what
the pre-registration asks for, with the minimum, the maximum and the ratio of
the two beside them because W9.1 and W12 quote a spread as a ratio.  The
committed checkpoint is placed by the count of ensemble members strictly below
it, and it is never a member of the sample it is placed in.

Four questions are asked of the spread, and a yes to any of them is a result
that changes a conclusion rather than a decoration on one:

  Q1  does any cell of the corrector column of Table~\\ref{tab:gtable} change
      SIGN across the ensemble?  A sign change is the difference between
      "better than the scheme it corrects" and "worse".
  Q2  does the corrector ever reach vps4 in any channel at the working step?
  Q3  does the separation between the corrector's energy error and the
      physical signal ever fall below one, that is, does the error ever exceed
      the signal it is supposed to leave alone?
  Q4  does the trajectory advantage over the Boris scheme ever fall below one
      at the window of the manuscript?

A fifth question is not about the spread at all and is asked because SD1
measured the answer: how much of the reported trajectory advantage is the
error of the ruler.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sd_common as SD                                        # noqa: E402

OUT = os.path.join(HERE, "sd5_summary.json")
REPORT = os.path.join(HERE, "sd5_report.txt")

#: what Section 7 of the manuscript prints for the five-seed ensemble
MANUSCRIPT = {
    "traj_gain_committed": 117.8,
    "traj_gain_median_of_five": 135.0,
    "traj_gain_range_of_five": [50.0, 447.0],
    "energy_separation_committed": 45.8,
    "energy_separation_sd_of_five": 0.064,
    "tab_family_trajectory": 3.47e-3,
    "tab_family_energy": 1.64e-5,
    "vps4_trajectory": 5.35e-5,
    "vps4_energy": 2.64e-7,
    "physical_signal": 7.497e-4,
}

#: the energy channel carries no signal where the configuration has no
#: electric field: |v| is then an exact invariant of the continuous motion and
#: every entry of that row is a ratio of rounding errors.  W15 established the
#: list, marked those cells with a dagger in Table~\ref{tab:gtable} and
#: excluded them from its rank analysis; they are excluded here for the same
#: reason and reported separately rather than dropped.
DEGENERATE_ENERGY = ("uniform", "B1_radial", "B2_wave", "B3_tilted")

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def load(name):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


def members(sd3):
    """The independent ensemble, and the committed run kept aside."""
    ens = {k: v for k, v in sd3["runs"].items()
           if v["source"] in ("W16", "I1.3")}
    com = sd3["runs"]["committed"]
    rep = sd3["runs"].get("i13_reproduce42")
    return ens, com, rep


def col(ens, path):
    """One column of the ensemble, addressed by a dotted path."""
    out = []
    for k in sorted(ens):
        v = ens[k]
        for p in path.split("."):
            v = v[p]
        out.append(float(v))
    return out


def get(rec, path):
    v = rec
    for p in path.split("."):
        v = v[p]
    return float(v)


def block(ens, com, path, label, smaller_is_better=True):
    s = col(ens, path)
    q = SD.quartiles(s)
    p = SD.percentile_of(get(com, path), s)
    return {"label": label, "path": path, "ensemble": q, "committed": p,
            "smaller_is_better": smaller_is_better,
            "committed_is_favourable": (
                (p["percentile_below"] < 50.0) if smaller_is_better
                else (p["percentile_below"] > 50.0)),
            "values": s}


def fmt_block(b):
    q, p = b["ensemble"], b["committed"]
    return ("%-46s committed %11.4g | median %11.4g  IQR [%10.4g, %10.4g]  "
            "range [%10.4g, %10.4g]  x%.2f | %d of %d below, %.0f%%"
            % (b["label"], p["value"], q["median"], q["q1"], q["q3"],
               q["min"], q["max"], q["ratio_max_min"],
               p["n_below"], p["n"], p["percentile_below"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    sd1 = load("sd1_calibration.json")
    sd2 = load("sd2_training.json")
    sd3 = load("sd3_ensemble.json")
    sd4 = load("sd4_external.json")
    ens, com, rep = members(sd3)
    n = len(ens)

    say("=" * 100)
    say("W16 -- the corrector over %d independent retrainings, and the "
        "committed checkpoint placed inside them" % n)
    say("=" * 100)

    # ---------------------------------------------------------------- 0 ----
    floor = sd1["reference_floor"]
    say()
    say("0.  THE RULER, MEASURED BEFORE THE ENSEMBLE IS READ")
    say("    the reference of ../horizon/ and ../stats/ is the Boris scheme at "
        "h/150.")
    say("    its own trajectory error over the window of tab:family is "
        "%.4e Larmor radii," % floor["its_own_rms_error"])
    say("    against a closed form whose own cost is %.2e (DOP853 at rtol "
        "1e-12 measures %.2e)."
        % (0.0, floor["dop853_rtol_1e-12"]["rms_vs_closed_form"]))
    say("    fitted order of the shipped Boris position readout:")
    for k, r in floor["boris_ladder"].items():
        say("       %-12s h = %-10.6f rms %.6e  %s"
            % (k, r["h"], r["rms_vs_closed_form"],
               ("order %.3f" % r["fitted_order"]) if "fitted_order" in r
               else ""))
    thr = floor["gain_at_which_a_report_falls_below_this_floor"]
    say("    a trajectory advantage above %.1f is a corrector error BELOW the "
        "error of the ruler." % thr)
    say("    a trajectory advantage above %.1f is within the factor of ten "
        "W14 and W15 use to flag" % floor["gain_within_a_factor_of_ten_of_the_floor"])
    say("    a cell as reference-limited.")

    # ---------------------------------------------------------------- 1 ----
    say()
    say("1.  THE READOUT SECTION 7 PRINTS (Boris reference at h/150)")
    b_gain = block(ens, com, "paper_recipe.traj_gain_projected",
                   "trajectory advantage over Boris", smaller_is_better=False)
    b_sep = block(ens, com, "paper_recipe.energy_separation_hybrid",
                  "energy separation, projected corrector",
                  smaller_is_better=False)
    b_sepr = block(ens, com, "paper_recipe.energy_separation_raw",
                   "energy separation, unprojected correction",
                   smaller_is_better=False)
    b_pos = block(ens, com, "paper_recipe.corrector_projected.pos_err_rms",
                  "corrector trajectory error (h/150 ruler)")
    b_en = block(ens, com,
                 "paper_recipe.corrector_projected.energy_err_median_2nd_half",
                 "corrector energy error (h/150 ruler)")
    for b in (b_gain, b_sep, b_sepr, b_pos, b_en):
        say("    " + fmt_block(b))
    say()
    say("    against what Section 7 prints for its five: committed %.1f, "
        "median %.0f, range %.0f to %.0f"
        % (MANUSCRIPT["traj_gain_committed"],
           MANUSCRIPT["traj_gain_median_of_five"],
           MANUSCRIPT["traj_gain_range_of_five"][0],
           MANUSCRIPT["traj_gain_range_of_five"][1]))
    say("    the energy separation is quoted as %.1f with a standard deviation "
        "of %.3f over five;" % (MANUSCRIPT["energy_separation_committed"],
                                MANUSCRIPT["energy_separation_sd_of_five"]))
    say("    over %d it is %.4f with a standard deviation of %.4f."
        % (n, float(np.mean(b_sep["values"])), float(np.std(b_sep["values"],
                                                            ddof=1))))
    below_floor = [v for v in b_gain["values"] if v > thr]
    say("    members whose REPORTED advantage puts the corrector below the "
        "ruler's own error: %d of %d" % (len(below_floor), n))

    # ---------------------------------------------------------------- 2 ----
    say()
    say("2.  THE SAME QUANTITIES ON A REFERENCE THAT IS NOT THE TRAINING "
        "TARGET (tab:family, DOP853 rtol 1e-12)")
    boris_tf = sd3["deterministic"]["boris"]["tab_family"]["trajectory"]
    gain_dop = {k: boris_tf / v["tab_family"]["trajectory"]
                for k, v in ens.items()}
    q_gain = SD.quartiles(list(gain_dop.values()))
    p_gain = SD.percentile_of(boris_tf / com["tab_family"]["trajectory"],
                              list(gain_dop.values()))
    b_traj = block(ens, com, "tab_family.trajectory",
                   "corrector trajectory error (DOP853 ruler)")
    b_ener = block(ens, com, "tab_family.energy",
                   "corrector energy error (DOP853 ruler)")
    say("    " + fmt_block(b_traj))
    say("    " + fmt_block(b_ener))
    say("    %-46s committed %11.4g | median %11.4g  IQR [%10.4g, %10.4g]  "
        "range [%10.4g, %10.4g]  x%.2f | %d of %d below, %.0f%%"
        % ("trajectory advantage over Boris", p_gain["value"],
           q_gain["median"], q_gain["q1"], q_gain["q3"], q_gain["min"],
           q_gain["max"], q_gain["ratio_max_min"], p_gain["n_below"],
           p_gain["n"], p_gain["percentile_below"]))
    say()
    say("    the spread of the advantage narrows from x%.2f on the training "
        "target to x%.2f on the closed-form-grade reference."
        % (b_gain["ensemble"]["ratio_max_min"], q_gain["ratio_max_min"]))
    sig = MANUSCRIPT["physical_signal"]
    sep_dop = {k: sig / v["tab_family"]["energy"] for k, v in ens.items()}
    q_sep = SD.quartiles(list(sep_dop.values()))
    p_sep = SD.percentile_of(sig / com["tab_family"]["energy"],
                             list(sep_dop.values()))
    say("    energy separation on the same reference: committed %.4f, median "
        "%.4f, range [%.4f, %.4f], x%.4f"
        % (p_sep["value"], q_sep["median"], q_sep["min"], q_sep["max"],
           q_sep["ratio_max_min"]))

    # ---------------------------------------------------------------- 3 ----
    say()
    say("3.  THE CORRECTOR COLUMN OF tab:gtable, CELL BY CELL "
        "(H_paper, canonical initial condition)")
    say("    %-14s %-11s %9s %9s %9s %9s %9s %7s"
        % ("configuration", "channel", "committed", "median", "q1", "q3",
           "min..max", "below"))
    gcells = {}
    sign_flips = []
    for fname in SD.FIELD_NAMES:
        for ch in SD.CHANNELS:
            key = "H_paper|%s" % ch
            path = "G.%s.%s" % (fname, key)
            vals = col(ens, path)
            if not all(math.isfinite(v) for v in vals):
                continue
            cv = get(com, path)
            q = SD.quartiles(vals)
            p = SD.percentile_of(cv, vals)
            allv = vals + [cv]
            deg = (ch == "energy" and fname in DEGENERATE_ENERGY)
            flip = (min(allv) < 0.0 < max(allv))
            gcells["%s|%s" % (fname, ch)] = {
                "committed": cv, "ensemble": q, "committed_in_ensemble": p,
                "degenerate": bool(deg),
                "degenerate_why": ("the configuration carries no electric "
                                   "field, so every entry is a ratio of "
                                   "rounding errors (W15 Section 5.1)")
                                  if deg else "",
                "sign_changes_across_the_ensemble": bool(flip and not deg),
                "n_positive": int(sum(1 for v in vals if v > 0)),
                "n_negative": int(sum(1 for v in vals if v < 0)),
                "values": vals}
            if flip and not deg:
                sign_flips.append("%s|%s" % (fname, ch))
            say("    %-14s %-11s %+9.3f %+9.3f %+9.3f %+9.3f "
                "%+7.2f..%+6.2f %3d/%d %s"
                % (fname, ch, cv, q["median"], q["q1"], q["q3"], q["min"],
                   q["max"], p["n_below"], p["n"],
                   "dagger: degenerate" if deg
                   else ("SIGN FLIP" if flip else "")))
    say()
    say("    Q1  cells whose sign changes across the ensemble: %d of %d  %s"
        % (len(sign_flips), len(gcells), sign_flips if sign_flips else ""))

    # how many cells each member wins, the analogue of tab:map's "4 of 120"
    keys = [k for k in gcells if not gcells[k]["degenerate"]]
    wins = {}
    for i, k in enumerate(sorted(ens)):
        wins[k] = int(sum(1 for c in keys if gcells[c]["values"][i] > 0))
    wins_com = int(sum(1 for c in keys if gcells[c]["committed"] > 0))
    qw = SD.quartiles(list(wins.values()))
    pw = SD.percentile_of(wins_com, list(wins.values()))
    say("    cells of %d in which the corrector beats the scheme it corrects: "
        "committed %d, ensemble median %.1f, range %d to %d (%d of %d members "
        "below the committed)"
        % (len(keys), wins_com, qw["median"], int(qw["min"]), int(qw["max"]),
           pw["n_below"], pw["n"]))
    cells_won = {"n_cells": len(keys), "committed": wins_com,
                 "ensemble": qw, "committed_in_ensemble": pw,
                 "per_member": wins}

    # ---------------------------------------------------------------- 4 ----
    say()
    say("4.  THE FOUR QUESTIONS")
    q2_traj = min(col(ens, "tab_family.trajectory")
                  + [com["tab_family"]["trajectory"]])
    q2_en = min(col(ens, "tab_family.energy")
                + [com["tab_family"]["energy"]])
    q2 = {"best_corrector_trajectory": q2_traj,
          "vps4_trajectory": sd3["deterministic"]["vps4"]["tab_family"]["trajectory"],
          "margin_trajectory": q2_traj / sd3["deterministic"]["vps4"]["tab_family"]["trajectory"],
          "best_corrector_energy": q2_en,
          "vps4_energy": sd3["deterministic"]["vps4"]["tab_family"]["energy"],
          "margin_energy": q2_en / sd3["deterministic"]["vps4"]["tab_family"]["energy"],
          "answer": None}
    q2["answer"] = bool(q2["margin_trajectory"] <= 1.0
                        or q2["margin_energy"] <= 1.0)
    q3_min = min(b_sep["values"] + [b_sep["committed"]["value"]])
    q3_min_dop = min(list(sep_dop.values()) + [p_sep["value"]])
    q4_min = min(b_gain["values"] + [b_gain["committed"]["value"]])
    q4_min_dop = min(list(gain_dop.values()) + [p_gain["value"]])
    say("    Q1 sign change in the tab:gtable corrector column: %s"
        % ("YES -- %s" % sign_flips if sign_flips else "no"))
    say("    Q2 corrector reaches vps4 at the working step: %s "
        "(best of the ensemble is %.3g against vps4's %.3g on the trajectory, "
        "a factor of %.1f)"
        % ("YES" if q2["answer"] else "no", q2_traj, q2["vps4_trajectory"],
           q2["margin_trajectory"]))
    say("    Q3 energy error ever above the physical signal: %s "
        "(worst separation %.4f on the h/150 ruler, %.4f on DOP853)"
        % ("YES" if min(q3_min, q3_min_dop) < 1.0 else "no", q3_min,
           q3_min_dop))
    say("    Q4 trajectory advantage ever below one: %s "
        "(worst %.2f on the h/150 ruler, %.2f on DOP853)"
        % ("YES" if min(q4_min, q4_min_dop) < 1.0 else "no", q4_min,
           q4_min_dop))

    # ---------------------------------------------------------------- 5 ----
    say()
    say("5.  THE EXTERNAL ARCHITECTURES, FOUR REPETITIONS AGAINST TEN")
    say("    %-9s %-26s %10s %10s %8s %10s %10s %8s %7s"
        % ("arch", "channel", "med(4)", "range(4)", "x(4)", "med(10)",
           "range(10)", "x(10)", "wider"))
    ext = {}
    for arch, chans in sd4["summary"].items():
        for c, s in chans.items():
            f, t = s["four_committed"], s["ten"]
            ext["%s|%s" % (arch, c)] = s
            say("    %-9s %-26s %10.4g %5.2g..%-5.2g %8.2f %10.4g "
                "%5.2g..%-5.2g %8.2f %7.2f"
                % (arch, c, f["median"], f["min"], f["max"],
                   f["ratio_max_min"], t["median"], t["min"], t["max"],
                   t["ratio_max_min"], s["spread_widened_by"]))
    widened = [k for k, s in ext.items() if s["spread_widened_by"] > 1.0]
    say("    channels whose spread widened at ten repetitions: %d of %d"
        % (len(widened), len(ext)))

    # ---------------------------------------------------------------- 6 ----
    say()
    say("6.  WHAT HAS NO SPREAD, AND WHY")
    for s, d in sd3["deterministic"].items():
        say("    %-10s trajectory %.6e  energy %.6e   %s"
            % (s, d["tab_family"]["trajectory"], d["tab_family"]["energy"],
               d["spread_over_seeds"]))
    tc = sd1["training_cost"]
    say("    the flop model, the closed forms and the classical schemes are "
        "arithmetic, not draws.")
    say("    one retraining of the corrector costs %.3e flops (%.3e data, "
        "%.3e optimisation);" % (tc["total_flops"], tc["data_flops"],
                                 tc["optimisation_flops"]))
    say("    the ensemble of %d cost %.3e; the cost column of tab:family "
        "carries none of it." % (n, n * tc["total_flops"]))

    # ---------------------------------------------------------------- 7 ----
    say()
    say("7.  THE REPRODUCTION CHECK")
    if rep is not None:
        say("    ../stats/checkpoints/corrector_b4_seed42.pt is a retraining "
            "of the committed run at the committed seed.")
        say("    trajectory advantage %.5f against the committed %.5f; "
            "energy separation %.6f against %.6f"
            % (rep["paper_recipe"]["traj_gain_projected"],
               com["paper_recipe"]["traj_gain_projected"],
               rep["paper_recipe"]["energy_separation_hybrid"],
               com["paper_recipe"]["energy_separation_hybrid"]))
    r42 = os.path.join(HERE, "sd2_reproduce42.json")
    if os.path.exists(r42):
        d = json.load(open(r42, encoding="utf-8"))
        say("    this directory's own retraining at seed 42: bitwise "
            "identical to the committed checkpoint = %s, "
            "largest relative difference %.3e"
            % (d["identical_bitwise"], d["max_rel_diff"]))
    say("    validation defect error, ensemble: median %.6f  IQR [%.6f, "
        "%.6f]; committed %.6f (%d of %d below)"
        % (sd2["val_rel_defect_error"]["median"],
           sd2["val_rel_defect_error"]["q1"],
           sd2["val_rel_defect_error"]["q3"],
           sd2["committed_in_ensemble_val_rel_defect_error"]["value"],
           sd2["committed_in_ensemble_val_rel_defect_error"]["n_below"],
           sd2["committed_in_ensemble_val_rel_defect_error"]["n"]))

    payload = {
        "meta": {"wave": "W16", "n_independent_members": n,
                 "members": sorted(ens),
                 "statistics": "median, q1, q3 by numpy.percentile linear "
                               "interpolation; the committed run placed by the "
                               "count of members strictly below it and never "
                               "counted as a member",
                 "manuscript_values_quoted_for_comparison": MANUSCRIPT},
        "reference_floor": floor,
        "paper_recipe": {"traj_gain": b_gain, "energy_separation": b_sep,
                         "energy_separation_unprojected": b_sepr,
                         "trajectory_error": b_pos, "energy_error": b_en,
                         "members_reported_below_the_ruler": len(below_floor)},
        "dop853_recipe": {
            "trajectory_error": b_traj, "energy_error": b_ener,
            "traj_gain": {"committed": p_gain, "ensemble": q_gain,
                          "values": gain_dop},
            "energy_separation": {"committed": p_sep, "ensemble": q_sep,
                                  "values": sep_dop}},
        "gtable_cells": gcells,
        "cells_won": cells_won,
        "questions": {"Q1_sign_flips": sign_flips,
                      "Q2_reaches_vps4": q2,
                      "Q3_worst_energy_separation": {
                          "h150_ruler": q3_min, "dop853": q3_min_dop,
                          "below_one": bool(min(q3_min, q3_min_dop) < 1.0)},
                      "Q4_worst_trajectory_advantage": {
                          "h150_ruler": q4_min, "dop853": q4_min_dop,
                          "below_one": bool(min(q4_min, q4_min_dop) < 1.0)}},
        "external": sd4["summary"],
        "deterministic": {k: {"tab_family": v["tab_family"],
                              "spread_over_seeds": v["spread_over_seeds"]}
                          for k, v in sd3["deterministic"].items()},
        "training_cost": {"per_retraining": tc,
                          "ensemble_total_flops": n * tc["total_flops"]},
    }
    payload["latex"] = latex(payload, sd3, sd4)
    say()
    say(payload["latex"]["tab_seeds"])
    say()
    say(payload["latex"]["tab_seed_channels"])

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return SD.write(OUT, payload, force=a.force)


# ---------------------------------------------------------------- latex ------
def latex(p, sd3, sd4):
    n = p["meta"]["n_independent_members"]

    def num(x, fmt):
        """A number in LaTeX, with a real \times10^{} where the format is
        exponential, so that the block below pastes into the manuscript
        without a pass of hand editing."""
        t = fmt % x
        m = re.match(r"^(-?[0-9.]+)e([-+])0*([0-9]+)$", t)
        if m:
            sign = "-" if m.group(2) == "-" else ""
            return "%s\\times10^{%s%s}" % (m.group(1), sign, m.group(3))
        return t

    def row(label, b, fmt="%.4g"):
        q, c = b["ensemble"], b["committed"]
        return ("    %-42s & $%s$ & $%s$ & $%s$--$%s$ & $%d/%d$ \\\\"
                % (label, num(c["value"], fmt), num(q["median"], fmt),
                   num(q["q1"], fmt), num(q["q3"], fmt),
                   c["n_below"], c["n"]))

    t1 = ["\\begin{table}[tbp]", "  \\centering",
          "  \\caption{The learned corrector over $%d$ independent "
          "retrainings at the architecture and hyper-parameters of the "
          "committed run.  The committed checkpoint is the one every "
          "corrector number in this paper stands on; it is placed inside the "
          "ensemble and is not one of its members, and the last column counts "
          "the members that fall below it.  The upper block is scored against "
          "the reference the corrector was trained on, a Boris run at $h/150$, "
          "which is the reference Section~\\ref{sec:family} quotes; that "
          "reference carries a trajectory error of $%s$ Larmor radii of its "
          "own, so an advantage above $%.0f$ reports a corrector error below "
          "the error of the ruler.  The lower block is scored against DOP853 "
          "at $\\mathrm{rtol}=10^{-12}$, which is the reference of "
          "Table~\\ref{tab:family} and is not the training target.  The "
          "classical schemes contain no random draw and have no spread.}"
          % (n, num(p["reference_floor"]["its_own_rms_error"], "%.3e"),
             p["reference_floor"]["gain_at_which_a_report_falls_below_this_floor"]),
          "  \\label{tab:seeds}", "  \\footnotesize",
          "  \\begin{tabular}{@{}lrrrr@{}}", "    \\toprule",
          "    Quantity & committed & median & IQR & below \\\\",
          "    \\midrule",
          "    \\multicolumn{5}{@{}l}{\\emph{against the Boris reference at "
          "$h/150$, the training target}} \\\\",
          row("trajectory advantage over Boris",
              p["paper_recipe"]["traj_gain"], "%.1f"),
          row("trajectory error (Larmor radii)",
              p["paper_recipe"]["trajectory_error"], "%.2e"),
          row("energy error", p["paper_recipe"]["energy_error"], "%.3e"),
          row("signal per energy error",
              p["paper_recipe"]["energy_separation"], "%.3f"),
          row("signal per energy error, no projection",
              p["paper_recipe"]["energy_separation_unprojected"], "%.2f"),
          "    \\addlinespace",
          "    \\multicolumn{5}{@{}l}{\\emph{against DOP853 at "
          "$\\mathrm{rtol}=10^{-12}$}} \\\\",
          row("trajectory error (Larmor radii)",
              p["dop853_recipe"]["trajectory_error"], "%.2e"),
          row("energy error", p["dop853_recipe"]["energy_error"], "%.3e"),
          ]
    d = p["dop853_recipe"]["traj_gain"]
    t1.append("    %-42s & $%.1f$ & $%.1f$ & $%.1f$--$%.1f$ & $%d/%d$ \\\\"
              % ("trajectory advantage over Boris", d["committed"]["value"],
                 d["ensemble"]["median"], d["ensemble"]["q1"],
                 d["ensemble"]["q3"], d["committed"]["n_below"],
                 d["committed"]["n"]))
    s = p["dop853_recipe"]["energy_separation"]
    t1.append("    %-42s & $%.2f$ & $%.2f$ & $%.2f$--$%.2f$ & $%d/%d$ \\\\"
              % ("signal per energy error", s["committed"]["value"],
                 s["ensemble"]["median"], s["ensemble"]["q1"],
                 s["ensemble"]["q3"], s["committed"]["n_below"],
                 s["committed"]["n"]))
    t1 += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    # the gtable corrector column with its spread
    t2 = ["\\begin{table}[tbp]", "  \\centering",
          "  \\caption{The corrector column of Table~\\ref{tab:gtable} with "
          "its spread over $%d$ retrainings.  Each entry is "
          "$G=\\log_{10}(E_{\\mathrm{Boris}}/E_{\\mathrm{corrector}})$ at "
          "$\\Omega h = 0.3$ over $19.1$ gyro-orbits, so a positive $G$ is an "
          "advantage over the scheme being corrected and a change of sign "
          "across the ensemble would be a change of verdict.  The energy "
          "channel is omitted where the configuration carries no electric "
          "field and every entry would be a ratio of rounding errors.  "
          "$\\ast$: the sign of the entry is not stable across the ensemble, "
          "so the committed checkpoint and a typical retraining disagree on "
          "whether the corrector helps in that cell.}" % n,
          "  \\label{tab:seed_channels}", "  \\footnotesize",
          "  \\begin{tabular}{@{}llrrrr@{}}", "    \\toprule",
          "    Configuration & Channel & committed & median & IQR & below \\\\",
          "    \\midrule"]
    prev = None
    for key, c in p["gtable_cells"].items():
        f, ch = key.split("|")
        if c["degenerate"]:
            continue
        name = f.replace("_", " ") if f != prev else ""
        prev = f
        mark = "^{\\ast}" if c["sign_changes_across_the_ensemble"] else ""
        t2.append("    %-14s & %-11s & $%+.2f%s$ & $%+.2f$ & $%+.2f$--$%+.2f$ "
                  "& $%d/%d$ \\\\"
                  % (name, ch, c["committed"], mark, c["ensemble"]["median"],
                     c["ensemble"]["q1"], c["ensemble"]["q3"],
                     c["committed_in_ensemble"]["n_below"],
                     c["committed_in_ensemble"]["n"]))
    t2 += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    return {"tab_seeds": "\n".join(t1),
            "tab_seed_channels": "\n".join(t2)}


if __name__ == "__main__":
    raise SystemExit(main())
