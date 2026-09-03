"""SP4: print the tables of the Wave 11 report from the committed JSON.

    python sp4_summarize.py

Reads sp1_calibration.json, sp2_spectra.json and sp3_horizon.json and prints
Markdown.  No number in the report is typed by hand: the tables are this
command's output pasted in.  Nothing is computed here that is not already in
the JSON, except the arithmetic of formatting.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NAMES = {"hnn": "HNN", "sympnet": "G-SympNet", "pinn": "PINN-symplectic",
         "boris": "Boris"}


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        sys.exit("missing %s -- run the scripts in order" % name)
    return json.load(open(p, encoding="utf-8"))


def fnum(x, digits=6):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 1e-3 and abs(x) < 1e4:
        return ("%%.%df" % digits) % x
    return "%.3e" % x


def table1_calibration(d1):
    print("\n### Calibration: maps whose spectrum is known in closed form\n")
    print("| map | symplectic defect | \\|det J - 1\\| | \\|lambda\\| | "
          "arg of the gyration pair | max\\|lambda\\| - 1 |")
    print("| :-- | --: | --: | :-- | --: | --: |")
    for key, label in (("at_working_step_exact_flow", "exact flow map"),
                       ("at_working_step_boris", "Boris"),
                       ("at_working_step_rk4_on_exact_field",
                        "RK4 on the exact field")):
        v = d1["closed_forms"][key]
        arg = max(abs(a) for a in v["arg"])
        print("| %s | %s | %s | %s | %s | %s |" % (
            label, fnum(v["symplectic_defect"]), fnum(v["det_minus_one"]),
            ", ".join("%.9f" % a for a in v["abs"]),
            "%.9f" % arg, fnum(v["max_abs_minus_1"])))
    s = d1["setup"]
    print("\nWorst disagreement between the finite-difference Jacobian and the "
          "closed-form Boris matrix over the step ladder %s: `%s` on the matrix "
          "entries, `%s` on |lambda|." % (
              d1["setup"]["dt_ladder"],
              fnum(s["worst_matrix_error_over_ladder"]),
              fnum(s["worst_abs_error_over_ladder"])))
    b = d1["boris_angle"]
    print("Boris angle from the eigenvalues `%.16f` against the closed form "
          "`2 arctan(Omega h / 2)` = `%.16f`, difference `%s`."
          % (b["theta_from_eigenvalues"],
             b["theta_closed_form_2atan_Omega_h_over_2"],
             fnum(b["difference"])))
    r = d1["reciprocity_lemma"]
    print("Reciprocal pairing on %d symplectic matrices drawn without any "
          "training (seed `%d`): worst residual `%s`; smallest spectral radius "
          "among them `%.6f`, which is the inequality rho >= 1."
          % (r["n_draws"], r["seed"], fnum(r["worst_reciprocity"]),
             r["min_of_max_abs"]))
    sm = d1["sympmat_wave10"]
    print("SympMat of Wave 10, read from `sm4_gyrocentre.json`: %d records, "
          "worst reciprocity residual `%s`, largest `max|lambda|` = `%.6f` "
          "(%s)." % (sm["n_records"], fnum(sm["worst_reciprocity"]),
                     sm["largest_max_abs"], sm["largest_case"]))

    print("\n### SympMat of Wave 10, every stored record\n")
    print("| training level | omega_0 dt | B/B_0 | seed | max\\|lambda\\| - 1 "
          "| growth over 1e5 steps |")
    print("| :-- | --: | --: | --: | --: | --: |")
    for r in sm["rows"]:
        case = r["case"]                      # e.g. dt2_b2.5
        dt = case.split("_")[0][2:]
        b = case.split("_")[1][1:]
        print("| %s | %s | %s | %d | %s | 10^%.1f |" % (
            r["budget"].replace("at_", "").replace("_", " "), dt, b, r["seed"],
            fnum(r["max_abs"] - 1.0), r["log10_growth_over_1e5_steps"]))


def table2_spectra(d2):
    print("\n### Spectrum of the one-step Jacobian, %d points per cell, "
          "frozen field\n" % 64)
    print("| architecture | seed | rho min | rho median | rho max | "
          "points with rho > 1 + 1e-6 | points with rho < 1 | "
          "symplectic defect (median) | reciprocity (median) |")
    print("| :-- | --: | --: | --: | --: | --: | --: | --: | --: |")
    for arch in ("hnn", "sympnet", "pinn"):
        for rep in range(4):
            c = d2["cells"]["%s_r%d_frozen" % (arch, rep)]
            pp = [r for r in c["per_point"] if r.get("finite")]
            off = sum(1 for r in pp if r["max_abs"] > 1 + 1e-6)
            print("| %s | %d | %.9f | %.9f | %.6f | %d / %d | %d / %d | %s | %s |"
                  % (NAMES[arch], rep, c["max_abs"]["min"],
                     c["max_abs"]["median"], c["max_abs"]["max"],
                     off, len(pp), c["n_points_with_max_abs_below_1"], len(pp),
                     fnum(c["symplectic_defect"]["median"]),
                     fnum(c["reciprocity"]["median"])))
    print("\n### Finite-time Lyapunov spectrum along the orbit\n")
    print("| architecture | seed | lambda_max at n=1000 | at n=4000 | "
          "ratio (4.0 = pure transient, 1.0 = real exponent) | "
          "extrapolated | sum of the four | pairing residual |")
    print("| :-- | --: | --: | --: | --: | --: | --: | --: |")
    for arch in ("hnn", "sympnet", "pinn"):
        v = d2["by_architecture"][arch]
        for rep in range(4):
            c = d2["cells"]["%s_r%d_frozen" % (arch, rep)]
            cv = c["lyapunov_convergence"]
            print("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                NAMES[arch], rep, fnum(cv["lambda_short"]),
                fnum(cv["lambda_long"]), fnum(cv["ratio_short_over_long"], 2),
                fnum(cv["lambda_extrapolated"]),
                fnum(v["lyapunov_sum_per_step"][rep]),
                fnum(v["lyapunov_pairing_residual"][rep])))
    cc = d2["classical_control"]["boris_frozen"]
    cv = cc["lyapunov_convergence"]
    print("| %s (control, true value 0) | -- | %s | %s | %s | %s | %s | %s |" % (
        NAMES["boris"], fnum(cv["lambda_short"]), fnum(cv["lambda_long"]),
        fnum(cv["ratio_short_over_long"], 2), fnum(cv["lambda_extrapolated"]),
        fnum(cc["lyapunov"]["sum_per_step"]),
        fnum(cc["lyapunov"]["pairing_residual"])))
    print("\n`||M^n||` of the Boris one-step matrix, which is where its "
          "spurious exponent comes from: %s." % ", ".join(
              "n=%s: %.4f" % (k, v)
              for k, v in cc["norm_of_matrix_power"].items()))


def table3_horizon(d2, d3):
    print("\n### Architecture x seed x max|lambda| x error at %d steps "
          "(%.0f gyro-orbits)\n" % (d3["setup"]["n_steps"],
                                    d3["setup"]["gyro_orbits"]))
    print("| architecture | seed | rho on the orbit | rho max over all points "
          "| lambda_max extrapolated | measured growth per step | "
          "amplitude max | blow-up step | position error | energy error |")
    print("| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for c in d3["cells"]:
        print("| %s | %d | %.6f | %.6f | %s | %s | %s | %s | %s | %s |" % (
            NAMES[c["arch"]], c["rep"],
            float(np.exp(c["P1_log_median_rho_on_orbit"])),
            float(np.exp(c["P2_log_max_rho_all_points"])),
            fnum(c["P4_lyapunov_extrapolated"]),
            fnum(c["g_measured_growth_per_step"]),
            fnum(c["amplitude_running_max"], 4),
            "--" if c["blowup_step"] is None else str(c["blowup_step"]),
            fnum(c["final_position_error"], 4),
            fnum(c["final_energy_rel_err"], 4)))
    b = d3["runs"]["boris"]
    last = str(max(int(k) for k in b["checkpoints"]))
    cc = d2["classical_control"]["boris_frozen"]["lyapunov_convergence"]
    print("| %s | -- | 1.000000 | 1.000000 | %s | %s | %s | -- | %s | %s |" % (
        NAMES["boris"], fnum(cc["lambda_extrapolated"]),
        fnum(b["growth_per_step_measured"]),
        fnum(b["amplitude_running_max"], 4),
        fnum(b["checkpoints"][last]["position_err"], 4),
        fnum(b["checkpoints"][last]["energy_rel_err"], 4)))

    print("\n### Do the spectral quantities predict the growth?\n")
    keys = (("P1_log_median_rho_on_orbit", "log rho on the orbit"),
            ("P2_log_max_rho_all_points", "log rho, largest over all points"),
            ("P3_lyapunov_max_per_step", "leading Lyapunov exponent, n = 4000"),
            ("P4_lyapunov_extrapolated", "the same, extrapolated"))
    print("| predictor | Spearman | ratio predicted / measured, growing cells "
          "| ranks of the two cells that grew, out of 12 |")
    print("| :-- | --: | --: | --: |")
    for key, label in keys:
        p = d3["predictors"][key]
        rng = p["ratio_range_over_growing_cells"]
        rk = d3["ranking"][key]["ranks_of_the_cells_that_grew"]
        print("| %s | %s | %s (range %s) | %s |" % (
            label, fnum(p["spearman_with_measured_growth"], 3),
            fnum(p["ratio_median_over_growing_cells"], 1),
            "--" if rng is None else "%.1f to %.1f" % (rng[0], rng[1]),
            ", ".join(str(x) for x in rk)))
    if d3["predictors"]["blowup_calls"]:
        print("\n| cell | actual blow-up step | predicted by rho on orbit | "
              "by rho max | by lambda_max | by lambda extrapolated |")
        print("| :-- | --: | --: | --: | --: | --: |")
        for r in d3["predictors"]["blowup_calls"]:
            print("| %s | %d | %s | %s | %s | %s |" % (
                r["cell"], r["actual_blowup_step"],
                fnum(r.get("predicted_step_P1"), 1),
                fnum(r.get("predicted_step_P2"), 1),
                fnum(r.get("predicted_step_P3"), 1),
                fnum(r.get("predicted_step_P4"), 1)))

    print("\n### The other channel: the argument\n")
    print("| cell | complex pair in the spectrum? | arg of that pair | "
          "phase advance measured on the run | arg - measured | "
          "relative frequency error against Omega h | steps to pi of drift |")
    print("| :-- | :-- | --: | --: | --: | --: | --: |")
    for r in d3["frequency_channel"]["rows"]:
        print("| %s | %s | %.9f | %.9f | %s | %s | %s |" % (
            r["cell"], "yes" if r["spectrum_has_a_complex_pair"] else
            "**no, all four real**",
            r["arg_from_spectrum"], r["frequency_measured_on_run"],
            fnum(r["arg_minus_measured"]),
            fnum(r["relative_frequency_error_vs_exact"], 5),
            fnum(r["steps_to_pi_of_phase_drift"], 0)))
    fc = d3["frequency_channel"]
    print("\nOn the Boris scheme the argument reproduces the phase advance of "
          "its own run to `%s`. On the learned maps the two disagree by up to "
          "`%s`, and %d of the twelve cells have no complex eigenvalue on the "
          "orbit at all."
          % (fnum(fc["boris_arg_minus_measured"]),
             fnum(fc["worst_arg_minus_measured_learned"], 4),
             fc["n_learned_cells_with_no_complex_pair"]))


def main():
    d1 = load("sp1_calibration.json")
    d2 = load("sp2_spectra.json")
    d3 = load("sp3_horizon.json")
    table1_calibration(d1)
    table2_spectra(d2)
    table3_horizon(d2, d3)
    print("\n### The structural test\n")
    st = d2["structure"]
    print("| architecture | smallest rho over 256 points | "
          "points strictly inside the unit circle | "
          "median \\|det J - 1\\| | "
          "sum of the Lyapunov exponents (median over seeds) |")
    print("| :-- | --: | --: | --: | --: |")
    for arch in ("hnn", "sympnet", "pinn"):
        s = float(np.median(d2["by_architecture"][arch]["lyapunov_sum_per_step"]))
        print("| %s | %.10f | %d / 256 | %s | %s |" % (
            NAMES[arch], st["min_rho_over_every_point_and_seed"][arch],
            st["points_strictly_inside_the_unit_circle"][arch],
            fnum(st["det_minus_one_median_of_medians"][arch]), fnum(s)))
    print("\n%s" % d3["verdict"]["note"])


if __name__ == "__main__":
    main()
