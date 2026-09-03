"""rc4_report.py -- the was/now table, the LaTeX, and the pre-registered verdicts.

Reads rc0..rc3 and writes rc4_report.json plus rc4_report.txt.  Computes
nothing new: every number here has already been produced, checked against the
committed files, and stored.  `article/main.tex` is not touched by this or any
other file in this directory.

Usage: python rc4_report.py [--force]
"""
import json
import os
import sys

import rc_common as RC
from rc_common import check_or_write

OUT = RC.outpath("rc4_report.json")
TXT = RC.outpath("rc4_report.txt")


def load(name):
    with open(RC.outpath(name), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x, sig=4):
    if x is None:
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e5):
        return "%.*e" % (sig - 1, x)
    return "%.*g" % (sig, x)


def sci(x, sig=3):
    """LaTeX scientific notation, the form the manuscript already uses."""
    if x is None:
        return "--"
    s = "%.*e" % (sig, x)
    m, e = s.split("e")
    return "%s\\times10^{%d}" % (m, int(e))


def main():
    force = "--force" in sys.argv
    a0 = load("rc0_seed_audit.json")
    a1 = load("rc1_calibration.json")
    a2 = load("rc2_horizon.json")
    a3 = load("rc3_seeds.json")

    g3 = a2["grids"]["H1e3"]
    g4 = a2["grids"]["H1e4"]
    old3, new3 = g3["old"], g3["new"]

    def row(label, printed, old, new, where):
        return {"label": label, "printed_in_manuscript": printed,
                "old_ruler": old, "closed_form": new,
                "shift_ratio": (new / old) if (old not in (None, 0)
                                               and new is not None) else None,
                "shift_percent": (100.0 * (new - old) / abs(old))
                if (old not in (None, 0) and new is not None) else None,
                "source": where}

    def gain(block, h):
        return [r for r in block["gain_vs_horizon"]
                if r["gyro_orbits_requested"] == h][0]["traj_gain_projected"]

    def perr(block, h, m="proj"):
        return [r for r in block["gain_vs_horizon"]
                if r["gyro_orbits_requested"] == h][0]["%s_pos_err_rms" % m]

    tg = a3["ensemble"]["traj_gain"]
    ce = a3["ensemble"]["corrector_traj_error"]

    table = [
        row("trajectory advantage at 19.1 gyro-orbits", "117.8",
            gain(old3, 19.1), gain(new3, 19.1), "horizon/crossover.json"),
        row("trajectory advantage at 25 gyro-orbits", "(not printed)",
            gain(old3, 25), gain(new3, 25), "horizon/crossover.json"),
        row("trajectory advantage at 50 gyro-orbits", "32.7",
            gain(old3, 50), gain(new3, 50), "horizon/crossover.json"),
        row("horizon at which the advantage reaches unity", "101",
            old3["crossover_gyrations"], new3["crossover_gyrations"],
            "horizon/crossover.json"),
        row("trajectory advantage at 200 gyro-orbits", "0.07",
            gain(old3, 200), gain(new3, 200), "horizon/crossover.json"),
        row("disadvantage factor at 1e3 gyro-orbits", "143",
            old3["disadvantage_factor_whole_grid"],
            new3["disadvantage_factor_whole_grid"], "horizon/traj_summary.json"),
        row("disadvantage factor at 1e4 gyro-orbits", "1575",
            g4["old"]["disadvantage_factor_whole_grid"],
            g4["new"]["disadvantage_factor_whole_grid"],
            "horizon/traj_summary.json"),
        row("Boris reaches one Larmor radius at", "22.1",
            old3["boris_reaches_1_larmor_at_gyr"],
            new3["boris_reaches_1_larmor_at_gyr"], "horizon/crossover.json"),
        row("corrector reaches one Larmor radius at", "74.1",
            old3["proj_reaches_1_larmor_at_gyr"],
            new3["proj_reaches_1_larmor_at_gyr"], "horizon/crossover.json"),
        row("one-Larmor horizon gain", "3.4",
            old3["one_larmor_horizon_gain"], new3["one_larmor_horizon_gain"],
            "horizon/crossover.json"),
        row("corrector trajectory error at 19.1 gyro-orbits", "3.47e-3",
            perr(old3, 19.1), perr(new3, 19.1), "horizon/crossover.json"),
        row("Boris trajectory error at 19.1 gyro-orbits", "0.417",
            perr(old3, 19.1, "boris"), perr(new3, 19.1, "boris"),
            "horizon/validation.json"),
        row("Boris trajectory error at 1e3 gyro-orbits", "1.462",
            old3["rms_over_whole_grid"]["boris"],
            new3["rms_over_whole_grid"]["boris"], "horizon/traj_summary.json"),
        row("Boris trajectory error at 1e4 gyro-orbits", "1.632",
            g4["old"]["rms_over_whole_grid"]["boris"],
            g4["new"]["rms_over_whole_grid"]["boris"],
            "horizon/traj_summary.json"),
        row("tab:seeds  advantage, committed run", "117.8",
            tg["old"]["committed"]["value"], tg["new"]["committed"]["value"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, ensemble median", "152.8",
            tg["old"]["ensemble"]["median"], tg["new"]["ensemble"]["median"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, first quartile", "95.7",
            tg["old"]["ensemble"]["q1"], tg["new"]["ensemble"]["q1"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, third quartile", "252.2",
            tg["old"]["ensemble"]["q3"], tg["new"]["ensemble"]["q3"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, ensemble minimum", "(not printed)",
            tg["old"]["ensemble"]["min"], tg["new"]["ensemble"]["min"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, ensemble maximum", "(not printed)",
            tg["old"]["ensemble"]["max"], tg["new"]["ensemble"]["max"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  advantage, max/min spread", "(not printed)",
            tg["old"]["ensemble"]["ratio_max_min"],
            tg["new"]["ensemble"]["ratio_max_min"],
            "seeds/sd5_summary.json"),
        row("tab:seeds  corrector error, ensemble median", "2.91e-3",
            ce["old"]["ensemble"]["median"], ce["new"]["ensemble"]["median"],
            "seeds/sd5_summary.json"),
    ]
    out = {"meta": {"what": "W18: the was/now table for Section 7",
                    "n_random_draws": 0},
           "table": table}

    # --------------------------------------------------------- the verdicts
    committed_shift = tg["new"]["committed"]["value"] / \
        tg["old"]["committed"]["value"]
    dec = g3["decomposition"]["19.1_gyro_orbits"]["proj"]
    out["verdicts"] = {
        "P0": {
            "statement": "no manuscript number is spoiled by a seed collision",
            "status": "confirmed",
            "evidence": {
                "same_stream_collisions": [d for d in a0["collision_detail"]
                                           if d["same_generator_family"]],
                "n_same_stream": sum(1 for d in a0["collision_detail"]
                                     if d["same_generator_family"]),
                "n_integer_collisions": len(a0["collision_detail"]),
            }},
        "P1": {
            "statement": "the corrector's advantage will fall, to 111 if the "
                         "two errors are uncorrelated and 86 if aligned",
            "status": "refuted",
            "predicted_range": [86.0, 111.0],
            "observed": tg["new"]["committed"]["value"],
            "shift_ratio": committed_shift,
            "measured_rho": dec["rho"],
            "note": "the two error fields are anti-correlated, rho = %.4f, "
                    "which is the pre-registration's third case and the one it "
                    "did not predict" % dec["rho"]},
        "P2": {
            "statement": "the inversion point will move earlier than 101 "
                         "gyro-orbits",
            "status": "refuted",
            "old": old3["crossover_gyrations"],
            "new": new3["crossover_gyrations"],
            "note": "the two land on the same coarse sample; the grid "
                    "resolves 1 step = %.4f gyro-orbits"
                    % (RC.DT / RC.TWO_PI)},
        "P3": {
            "statement": "the negative control: if the advantage grows, the "
                         "corrector's residual runs against the ruler's bias, "
                         "and it is reported at the same length",
            "status": "fired",
            "observed_gain": tg["new"]["committed"]["value"],
            "observed_rho": dec["rho"],
            "rms_measured": dec["rms_measured"],
            "rms_ruler": dec["rms_ruler"],
            "rms_true": dec["rms_true"]},
    }

    # ------------------------------------------------------------- the LaTeX
    L = []
    L.append("%% W18: Section 7 read against the closed form of "
             "\\ref{sec:app_setups}.")
    L.append("%% Anchors: the paragraph now at main.tex lines 1578-1599, and "
             "the")
    L.append("%% 'trajectory advantage over Boris' row of Table~"
             "\\ref{tab:seeds} (line 1663).")
    L.append("")
    L.append("%% --- ANCHOR A: replaces the sentence beginning 'Against a "
             "Boris reference at a step $150$ times smaller' (line 1578-1582).")
    L.append("Lengthening the run reverses what is left.  Against the closed "
             "form of the")
    L.append("problem---Bessel of order zero in the Larmor frame, "
             "\\ref{sec:app_setups}---the")
    L.append("trajectory advantage of the corrector is $%s$ at $19.1$ "
             "gyro-orbits, $%s$ at"
             % (fmt(gain(new3, 19.1)), fmt(gain(new3, 50))))
    L.append("$50$, unity at $%s$, and $%s$ at $200$.  At $10^{3}$ gyro-orbits "
             "the corrector"
             % (fmt(new3["crossover_gyrations"], 3),
                fmt(gain(new3, 200), 2)))
    L.append("is worse than Boris by a factor of $%d$ and at $10^{4}$ by a "
             "factor of $%d$."
             % (round(new3["disadvantage_factor_whole_grid"]),
                round(g4["new"]["disadvantage_factor_whole_grid"])))
    L.append("")
    L.append("%% --- ANCHOR B: replaces the sentence 'Measured against the "
             "closed form, that")
    L.append("%% reference carries $1.396\\times10^{-3}$ ... An advantage "
             "above $298$ therefore")
    L.append("%% reports a corrector error below the error of the ruler.' "
             "(lines 1589-1593).")
    L.append("The Boris reference at $h/150$ that earlier versions of this "
             "section used is not")
    L.append("independent of the corrector, whose training target is the "
             "displacement between")
    L.append("a coarse Boris step and the same state carried by Boris at "
             "$h/150$.  Measured")
    L.append("against the closed form that reference carries $%s$ Larmor radii "
             "of error of"
             % sci(a1["ruler_own_error"]["rms_over_19.1_gyro_orbits"], 3))
    L.append("its own, and the two error fields are anti-correlated at $\\rho "
             "= %.2f$, so the"
             % dec["rho"])
    L.append("advantage it reports is $%.1f$ per cent \\emph{below} the true "
             "one rather than above"
             % (-100.0 * (1.0 / committed_shift - 1.0)))
    L.append("it: $%s$ against $%s$.  The closed form agrees with the "
             "independent DOP853"
             % (fmt(tg["old"]["committed"]["value"]),
                fmt(tg["new"]["committed"]["value"])))
    L.append("reference of Table~\\ref{tab:family} to eleven significant "
             "figures.")
    L.append("")
    L.append("%% --- ANCHOR C: replaces the 'trajectory advantage over Boris' "
             "row of")
    L.append("%% Table~\\ref{tab:seeds} (line 1663).")
    L.append("    trajectory advantage over Boris            & $%s$ & $%s$ & "
             "$%s$--$%s$ & $%d/20$ \\\\"
             % (fmt(tg["new"]["committed"]["value"]),
                fmt(tg["new"]["ensemble"]["median"]),
                fmt(tg["new"]["ensemble"]["q1"]),
                fmt(tg["new"]["ensemble"]["q3"]),
                tg["new"]["committed"]["n_below"]))
    L.append("")
    L.append("%% --- ANCHOR D: a sentence for the caption of "
             "Table~\\ref{tab:seeds}, replacing")
    L.append("%% 'that reference carries a trajectory error of "
             "$1.396\\times10^{-3}$ ... above $298$'.")
    L.append("Both blocks are now scored against the closed form, so the two "
             "no longer differ")
    L.append("by their reference.  On the Boris reference at $h/150$ the same "
             "ensemble read")
    L.append("$%s$ at the median with an upper quartile of $%s$ and a maximum "
             "of $%s$.  The"
             % (fmt(tg["old"]["ensemble"]["median"]),
                fmt(tg["old"]["ensemble"]["q3"]),
                fmt(tg["old"]["ensemble"]["max"])))
    L.append("$%d$ members whose reported error fell below that reference's "
             "own error of $%s$"
             % (a3["members_below_the_ruler_floor"]["old_ruler"],
                sci(a3["ruler"]["old_own_rms_error_vs_closed_form"], 3)))
    L.append("Larmor radii were exactly its $%d$ largest advantages; on an "
             "independent ruler none"
             % a3["members_below_the_ruler_floor"]["old_ruler"])
    L.append("is below that floor and the maximum falls to $%s$."
             % fmt(tg["new"]["ensemble"]["max"]))
    out["latex"] = "\n".join(L)

    # the claim in ANCHOR D, checked rather than asserted
    floor = a3["ruler"]["old_own_rms_error_vs_closed_form"]
    errs = ce["old"]["values"]
    gains = tg["old"]["values"]
    below = sorted(t for t in errs if errs[t] < floor)
    top = sorted(gains, key=lambda t: -gains[t])[:len(below)]
    out["anchor_d_check"] = {
        "floor": floor,
        "members_below_floor": below,
        "the_same_number_of_largest_advantages": sorted(top),
        "identical": sorted(below) == sorted(top),
        "their_gains_old": {t: gains[t] for t in below},
        "their_gains_new": {t: tg["new"]["values"][t] for t in below},
        "members_below_floor_on_closed_form":
            a3["members_below_the_ruler_floor"]["new_ruler"]}
    assert out["anchor_d_check"]["identical"], \
        "ANCHOR D claims the members below the floor are the largest " \
        "advantages, and they are not"

    # --------------------------------------------------------- repro.py lines
    out["repro_lines"] = "\n".join([
        '    ("data", os.path.join(EXP, "refcheck"), ["rc0_seed_audit.py"],',
        '     "refcheck/rc0_seed_audit.json     the seed-collision audit"),',
        '    ("data", os.path.join(EXP, "refcheck"), ["rc1_calibration.py"],',
        '     "refcheck/rc1_calibration.json    the stand, calibrated on the '
        'old ruler"),',
        '    ("data", os.path.join(EXP, "refcheck"), ["rc2_horizon.py"],',
        '     "refcheck/rc2_horizon.json        Sec. 7 horizon numbers on the '
        'closed form"),',
        '    ("data", os.path.join(EXP, "refcheck"), ["rc3_seeds.py"],',
        '     "refcheck/rc3_seeds.json          tab:seeds on the closed form"),',
        '    ("data", os.path.join(EXP, "refcheck"), ["rc4_report.py"],',
        '     "refcheck/rc4_report.json         the was/now table and the '
        'LaTeX"),',
    ])

    # ------------------------------------------------------------- the print
    lines = []
    lines.append("%-46s %12s %14s %14s %9s"
                 % ("quantity", "printed", "old ruler", "closed form", "shift"))
    lines.append("-" * 100)
    for r in table:
        lines.append("%-46s %12s %14s %14s %8s"
                     % (r["label"][:46], r["printed_in_manuscript"],
                        fmt(r["old_ruler"], 6), fmt(r["closed_form"], 6),
                        ("%+.2f%%" % r["shift_percent"])
                        if r["shift_percent"] is not None else "--"))
    lines.append("")
    for k in ("P0", "P1", "P2", "P3"):
        v = out["verdicts"][k]
        lines.append("%s  %-10s %s" % (k, v["status"], v["statement"]))
    txt = "\n".join(lines)
    print(txt)
    print("\n" + out["latex"])
    with open(TXT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n\n" + out["latex"] + "\n\n" + out["repro_lines"] + "\n")

    RC.assert_no_draws(0)
    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
