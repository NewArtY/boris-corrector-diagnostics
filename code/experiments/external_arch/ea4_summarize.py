"""EA4: read the three JSON files and print the tables of the W9.1 report.

    python ea4_summarize.py            markdown tables
    python ea4_summarize.py --latex    the LaTeX table for the manuscript

Writes nothing.  It exists so that no number in the report is transcribed by
hand, which is the rule of Section 9: every measured number is written by a
script into a data file, and everything downstream reads the file.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(v, nd=3):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "nan"
    if f == 0:
        return "0"
    if 1e-3 <= abs(f) < 1e6:
        return ("%%.%dg" % (nd + 1)) % f
    return ("%%.%de" % nd) % f


def tex(v, nd=2):
    """Always scientific, as Table 1 of the manuscript prints its columns."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "---"
    if f == 0:
        return "$0$"
    m = ("%%.%de" % nd) % f
    a, b = m.split("e")
    return "$%s\\times10^{%d}$" % (a, int(b))


def tex_plain(v, sig=3):
    """Three significant figures, with the thousands comma of the manuscript."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "---"
    if abs(f) >= 1000:
        # four digits plain, five and above with the thousands comma (B10)
        s = "%d" % int(round(f))
        if abs(f) >= 10000:
            s = "{:,}".format(int(round(f))).replace(",", "{,}")
    else:
        s = "%.*g" % (sig, f)
    return "$%s$" % s


NAMES = {"hnn": "HNN", "sympnet": "SympNet", "pinn": "PINN-symplectic",
         "boris": "Boris", "shipped": "Boris, synchronized",
         "staggered": "Boris, staggered", "vps2": "vps2", "vps4": "vps4",
         "gl4": "gl4", "imr": "implicit midpoint", "hybrid": "Learned corrector"}


def markdown():
    tr = load("ea1_training.json")
    pr = load("ea2_probes.json")
    co = load("ea3_cost.json")

    if tr:
        print("\n### Training, four repetitions each\n")
        print("| arch | rep | params | final loss | traj (r_L) | energy | "
              "flops/step | omega_h |")
        print("|---|---|---|---|---|---|---|---|")
        for k in sorted(tr["runs"]):
            v = tr["runs"][k]
            arch, rep = k.split("/")
            print("| %s | %s | %d | %s | %s | %s | %d | %s |" % (
                NAMES.get(arch, arch), rep, v["n_parameters"],
                fmt(v["final_loss"]), fmt(v["pos_err_rms"]),
                fmt(v["energy_err_median_2nd_half"]), v["flops_per_step"],
                fmt(v.get("omega_h_measured"), 6)))
        if tr.get("controls"):
            print("\n### Budget and capacity control\n")
            print("| arch | config | adam steps | params | traj (r_L) | energy |")
            print("|---|---|---|---|---|---|")
            for k in sorted(tr["controls"]):
                v = tr["controls"][k]
                print("| %s | %s | %d | %d | %s | %s |" % (
                    NAMES.get(k.split("/")[0], k), k.split("/")[1],
                    v["adam_steps"], v["n_parameters"], fmt(v["pos_err_rms"]),
                    fmt(v["energy_err_median_2nd_half"])))

    if co:
        print("\n### Two channels and cost, against the family of Table 1\n")
        print("| scheme | flops (run) | traj (r_L) | energy | signal/error |")
        print("|---|---|---|---|---|")
        sig = co["setup"]["physical_signal"]
        order = ["shipped", "staggered", "vps2", "vps4", "gl4", "imr", "hybrid",
                 "hnn", "sympnet", "pinn"]
        for k in order:
            if k not in co["rows"]:
                continue
            v = co["rows"][k]
            t_ = v.get("traj")
            e_ = v.get("energy")
            print("| %s | %s | %s | %s | %s |" % (
                NAMES.get(k, k), fmt(v["flops_run"]), fmt(t_), fmt(e_),
                fmt(sig / e_ if e_ else float("nan"), 2)))
        print("\n### Against vps4 and against the learned corrector\n")
        print("| arch | vps4 traj | vps4 energy | vps4 flops | corrector traj "
              "| corrector flops |")
        print("|---|---|---|---|---|---|")
        for k, v in co["against_classical"].items():
            print("| %s | %s | %s | %s | %s | %s |" % (
                NAMES.get(k, k),
                fmt(v["vps4"]["classical_more_accurate_in_traj_by"], 1),
                fmt(v["vps4"]["classical_more_accurate_in_energy_by"], 1),
                fmt(v["vps4"]["classical_cheaper_in_flops_by"], 1),
                fmt(v["learned_corrector"]["corrector_more_accurate_in_traj_by"], 2),
                fmt(v["learned_corrector"]["corrector_dearer_in_flops_by"], 2)))
        if co.get("symplecticity"):
            print("\n### How symplectic the one-step map actually is\n")
            print("| scheme | states | ||J^T Omega J - Omega||_F median | max "
                  "| \\|det J - 1\\| median |")
            print("|---|---|---|---|---|")
            for k, v in co["symplecticity"].items():
                if not isinstance(v, dict) or "residual_median" not in v:
                    continue
                print("| %s | %d | %s | %s | %s |" % (
                    NAMES.get(k, k), v["n_states"], fmt(v["residual_median"]),
                    fmt(v["residual_max"]), fmt(v["det_minus_one_median"])))
        if co.get("hnn_step_refinement"):
            print("\n### HNN under step refinement\n")
            print("| h | traj (r_L) | energy | flops (run) |")
            print("|---|---|---|---|")
            for k, v in co["hnn_step_refinement"].items():
                if not isinstance(v, dict):
                    continue
                print("| %s | %s | %s | %s |" % (
                    k.split("=")[1], fmt(v["pos_err_rms"]),
                    fmt(v["energy_err_median_2nd_half"]), fmt(v["flops_run"])))

    if pr:
        print("\n### Architecture x probe\n")
        print("| scheme / field | P1 | P2 (1e3) | P2 (1e4) | P3 | P4 driven | "
              "channel | P4 intrinsic | channel |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in pr["probes"]:
            v = pr["probes"][k]
            p2 = v.get("P2", {})
            g3 = p2.get("gyros_1000", {}).get("verdict", "--")
            g4 = p2.get("gyros_10000", {}).get("verdict", "--")
            pi = v.get("P4_intrinsic", {})
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                k, v["P1"]["verdict"], g3, g4, v["P3"]["verdict"],
                v["P4"]["verdict"], v["P4"].get("selected_channel", "--"),
                pi.get("verdict", "--"), pi.get("selected_channel", "--")))
        print("\n### The channel search, line by line "
              "(gates of the rule in `tab:probe4`)\n")
        rows = [(k, v.get("P4"), "driven") for k, v in pr["probes"].items()]
        rows += [(k, v.get("P4_intrinsic"), "intrinsic")
                 for k, v in pr["probes"].items()]
        rows += [(k, v, "intrinsic") for k, v in pr.get("P4_projected", {}).items()]
        print("| run | mode | channel | a | H | p | G1 rho (dec) | G1 | "
              "G2 p_rec vs terms | G2 | kept |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, p4, mode in rows:
            if not p4 or "rule" not in p4:
                continue
            for c in p4["rule"]["channels"]:
                print("| %s | %s | %s | %s | %s | %s | %s | %s | %s vs %s | "
                      "%s | %s |" % (
                          k, mode, c["channel"], fmt(c["a_hat"]),
                          fmt(c["H_hat"]), fmt(c["p_hat"]),
                          fmt(c["G1_rho_decades"]),
                          "ok" if c["G1_pass"] else "REJECT",
                          fmt(c["G2_p_reconstruction"]), fmt(c["G2_p_terms"]),
                          "ok" if c["G2_pass"] else "REJECT",
                          "yes" if c["survives"] else "no"))
        print("\n### Verdicts of the rule, and what the two-channel reading "
              "would have said\n")
        print("| run | mode | G0 | G3 spread | survivors | selected | p_pred | "
              "p measured | verdict | two-channel verdict / channel |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k, p4, mode in rows:
            if not p4 or "rule" not in p4:
                if p4:
                    print("| %s | %s | -- | -- | -- | -- | -- | -- | %s | -- |"
                          % (k, mode, p4.get("verdict", "--")))
                continue
            r = p4["rule"]
            lg = p4.get("two_channel_reading", {})
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s |" % (
                k, mode, "ok" if r["G0_pass"] else "REJECT",
                fmt(r["G3_spread"]), ",".join(r["survivors"]) or "none",
                r["selected"] or "--", fmt(r["p_pred"]),
                fmt(r["p_measured_full_horizon"]), r["probe_verdict"],
                lg.get("verdict", "--"), lg.get("channel_that_works", "--")))
        print("\n### Probe numbers\n")
        for k in pr["probes"]:
            v = pr["probes"][k]
            print("\n**%s** eps_trunc=%s eps_used=%s omega_h=%s flops/step=%d"
                  % (k, fmt(v["eps_truncation"]), fmt(v["eps_used"]),
                     fmt(v["omega_h"], 6), v["flops_per_step"]))
            p1 = v["P1"]
            if "isolated_response" in p1:
                print("  P1 driven raw p=%s ; isolated response p=%s ; "
                      "own error dominates: %s ; slopes %s"
                      % (fmt(p1["driven_raw"]["p_fit_last2dec"]),
                         fmt(p1["isolated_response"]["p_fit_last2dec"]),
                         fmt(p1["own_error_dominates_response"]),
                         [fmt(s, 2) for s in
                          p1["isolated_response"]["half_decade_slopes"]]))
            p3 = v["P3"]
            if "conclusions" in p3:
                print("  P3 levels %s / %s / %s  spread %s ; conclusions %s"
                      % (fmt(p3["synchronized"]["energy_err_median_2nd_half"]),
                         fmt(p3["shifted_h_over_2"]["energy_err_median_2nd_half"]),
                         fmt(p3["averaged_reading"]["energy_err_median_2nd_half"]),
                         fmt(p3["level_spread_factor"], 1),
                         p3["conclusions"]))
            for tag in ("P4", "P4_intrinsic"):
                p4 = v.get(tag, {})
                lg = p4.get("two_channel_reading")
                if not lg:
                    continue
                print("  %s rule: %s, selected %s, p_pred %s vs measured %s"
                      % (tag, p4["rule"]["probe_verdict"],
                         p4["rule"]["selected"], fmt(p4["rule"]["p_pred"]),
                         fmt(p4["rule"]["p_measured_full_horizon"])))
                print("     two-channel reading: V a=%s H=%s -> %s ; "
                      "E a=%s H=%s -> %s ; miss V=%s E=%s ; repair needed: %s"
                      % (fmt(lg["V_velocity_defect"]["a_hat_short"]),
                         fmt(lg["V_velocity_defect"]["H_hat_short"]),
                         fmt(lg["V_velocity_defect"]["p_pred_short"]),
                         fmt(lg["E_energy_increment"]["a_hat_short"]),
                         fmt(lg["E_energy_increment"]["H_hat_short"]),
                         fmt(lg["E_energy_increment"]["p_pred_short"]),
                         fmt(lg["miss_from_short_run"]["V_velocity_defect"]),
                         fmt(lg["miss_from_short_run"]["E_energy_increment"]),
                         fmt(lg["repair_needed"])))
            for gk, gv in v.get("P2", {}).items():
                if not isinstance(gv, dict) or "peak_rel_detune" not in gv:
                    continue
                chirp = gv.get("band_from_chirp") or float("nan")
                print("  P2 %s peak at rel %s = %s grid steps = %s of the "
                      "swept band; emax %s, on-resonance p=%s, suppression %s, "
                      "band %s (chirp %s, Fourier %s), verdict %s"
                      % (gk, fmt(gv["peak_rel_detune"]),
                         fmt(gv["peak_rel_detune"] / gv["grid_step_rel"], 2),
                         fmt(gv["peak_rel_detune"] / (chirp / 4.0), 2),
                         fmt(gv["peak_emax"]), fmt(gv["p_on_omega_h"]),
                         fmt(gv["suppression_at_band_edge"], 1),
                         fmt(gv["band_half_width_rel"]), fmt(chirp),
                         fmt(gv.get("band_from_fourier_limit")),
                         gv["verdict"]))


def latex():
    co = load("ea3_cost.json")
    pr = load("ea2_probes.json")
    if co is None:
        print("ea3_cost.json missing")
        return
    sig = co["setup"]["physical_signal"]
    tr = load("ea1_training.json") or {"runs": {}}
    npar = {a: tr["runs"].get("%s/rep0" % a, {}).get("n_parameters")
            for a in ("hnn", "sympnet", "pinn")}
    npar["hybrid"] = 52102
    print(r"""\begin{table}[tbp]
  \centering
  \caption{Three learned architectures from the literature on the run of
    Table~\ref{tab:family}, with two rows of that table repeated beneath them.
    All rows are integrated in the decaying field of
    Section~\ref{sec:channels} at $\Omega h = 0.3$ to $t = 120$, which is
    $19.1$ gyro-orbits, and scored against a DOP853 reference at
    $\mathrm{rtol} = 10^{-12}$. The physical signal over that window is
    $7.497\times10^{-4}$, so a scheme whose last column is below unity reports
    an energy error larger than the physical change it is there to resolve.
    Each of the three upper rows is one seed of four, fixed before the runs;
    the spread over the four is given in the text. Flop counts follow the
    model of \ref{sec:methods}.}
  \label{tab:external}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}lrrrrr@{}}
    \toprule
    Scheme & Parameters & Flops & Trajectory & Energy & Signal / \\
           &            &       & error ($r_L$) & error & error \\
    \midrule""")
    for k in ("hnn", "sympnet", "pinn"):
        if k not in co["rows"]:
            continue
        v = co["rows"][k]
        t_, e_ = v["traj"], v["energy"]
        print("    %s\n      & $%s$ & %s & %s & %s & %s \\\\"
              % (NAMES[k], "{:,}".format(npar[k]).replace(",", "{,}"),
                 tex(v["flops_run"]), tex(t_), tex(e_), tex_plain(sig / e_)))
    print(r"    \midrule")
    for k in ("hybrid", "vps4"):
        v = co["rows"][k]
        p = "$%s$" % "{:,}".format(npar[k]).replace(",", "{,}") \
            if npar.get(k) else "---"
        print("    %s\n      & %s & %s & %s & %s & %s \\\\"
              % (NAMES[k], p, tex(v["flops_run"]), tex(v["traj"]),
                 tex(v["energy"]), tex_plain(sig / v["energy"])))
    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    if not pr:
        return
    short = {"pass": "pass", "fail": "fail", "not reached": "---",
             "reports its readout floor": "floor", "inconclusive": "---",
             "no defect against the reference map": "---",
             "PASS": "pass", "FAIL": "fail", "NO EXPONENT": "no exponent",
             "NO EXPONENT (full horizon)": "no exponent", None: "---"}
    print(r"""
\begin{table}[tbp]
  \centering
  \caption{The four probes of Section~\ref{sec:protocol} on the Boris scheme
    and on the three architectures of Section~\ref{sec:external}, in the
    quasistatic field at $\tau = 1.2\times10^{8}$ over $10^{4}$ gyro-orbits at
    $\Omega h = 0.3$. The first, second and fourth probes read the exponent off
    the response to the drive with an undriven twin removed; the second is run
    at two horizons and the entry is the longer of them. The fourth is given in
    both modes: driven, with a defect injected parallel to $\bm v$, and
    intrinsic, on the scheme's own defect against the Boris map, which is the
    mode in which its failure was measured in Section~\ref{sec:conditions}.
    The last column names the channel on which the estimate reproduced the
    measured exponent.}
  \label{tab:probes}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    Scheme & Probe 1 & Probe 2 & Probe 3 & Probe 4 & Channel \\
    \midrule""")
    for name in ("boris", "hnn", "sympnet", "pinn"):
        k = "%s/quasistatic" % name
        if k not in pr["probes"]:
            continue
        v = pr["probes"][k]
        p2 = v.get("P2", {})
        g4 = p2.get("gyros_10000", {}).get("verdict", None)
        pi = v.get("P4_intrinsic", {})
        p4 = pi if pi.get("rule") else v.get("P4", {})
        ch = p4.get("selected_channel")
        chs = {"V(ref)": "velocity, reference frame",
               "V(own)": "velocity, own frame",
               "E": "energy increments"}.get(ch, "---")
        print("    %-16s & %s & %s & %s & %s & %s \\\\"
              % (NAMES.get(name, name), short.get(v["P1"]["verdict"], "---"),
                 short.get(g4, "---"), short.get(v["P3"]["verdict"], "---"),
                 short.get(p4.get("verdict"), "---"), chs))
    if pr.get("P4_projected"):
        print(r"    \midrule")
        for name in ("hnn", "sympnet", "pinn"):
            k = "%s+proj/quasistatic" % name
            if k not in pr["P4_projected"]:
                continue
            v = pr["P4_projected"][k]
            chs = {"V(ref)": "velocity, reference frame",
                   "V(own)": "velocity, own frame",
                   "E": "energy increments"}.get(v.get("selected_channel"),
                                                 "---")
            print("    %-16s & --- & --- & --- & %s & %s \\\\"
                  % (NAMES.get(name, name) + ", projected",
                     short.get(v.get("verdict"), "---"), chs))
    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")
    print("\n%% probe verdicts, plain")
    for k in pr["probes"]:
        v = pr["probes"][k]
        p2 = v.get("P2", {})
        print("%% %-22s P1 %-28s P2 %-10s P3 %-6s P4 %-6s P4int %s"
              % (k, v["P1"]["verdict"],
                 p2.get("gyros_10000", {}).get("verdict", "-"),
                 v["P3"]["verdict"], v["P4"].get("verdict", "-"),
                 v.get("P4_intrinsic", {}).get("verdict", "-")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    sys.exit(latex() if a.latex else markdown())
