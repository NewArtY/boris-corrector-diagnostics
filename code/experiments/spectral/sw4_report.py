"""sw4_report.py -- the tables of the W13 report.  Writes nothing.

Every number in `plan/reports/W13_1_spectral.md` is printed by this script from
the three committed JSON files, and pasted from its output.  Nothing in that
report is typed by hand.

Usage: python sw4_report.py
"""
import json
import os

import numpy as np

import sw_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = json.load(open(os.path.join(HERE, "sw1_reference.json"), encoding="utf-8"))
S2 = json.load(open(os.path.join(HERE, "sw2_spectra.json"), encoding="utf-8"))
S3 = json.load(open(os.path.join(HERE, "sw3_ensemble.json"), encoding="utf-8"))

SCHEMES = ["boris", "corrector", "vps2", "vps4", "gl4"]
NAMES = {"boris": "Boris", "corrector": "corrector", "vps2": "vps2",
         "vps4": "vps4", "gl4": "gl4"}
HZ = ["H1_paper", "H2_100orb", "H3_long"]


def rule(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ------------------------------------------------------------------ T1 -----
rule("T1  cost ledger.  The budget is one corrector step, 114,091 flops.")
c = S2["cost"]
print("%-10s %10s %8s %12s %10s" % ("scheme", "flop/step", "substeps",
                                    "step h", "budget"))
for k in SCHEMES:
    print("%-10s %10.1f %8d %12.4e %10.4f"
          % (NAMES[k], c["flops_per_step"][k], c["equal_cost_substeps"][k],
             c["equal_cost_step"][k], c["budget_ratio"][k]))
print("\ngl4 fixed point on its own iteration count:")
for r in c["gl4_fixed_point_trace"]:
    print("   m=%-5d mean iters %.3f -> %.1f flop/step" %
          (r["m"], r["mean_iters"], r["flops_per_step"]))
print("\n" + c["note"])


# ------------------------------------------------------------------ T2 -----
rule("T2  is the reference the floor?  Position error of DOP853 itself, "
     "against the closed form, in Larmor radii.")
print("%-10s %-18s %12s %12s" % ("horizon", "reference", "rms", "max"))
for hz, d in S1["reference_own_error"].items():
    for tag, v in d.items():
        print("%-10s %-18s %12.3e %12.3e"
              % (hz, tag, v["pos_err_rms"], v["pos_err_max"]))
cf = S1["closed_form_check"]
print("\nclosed form: initial-condition residual %.1e, float64 reconstruction "
      "against end-to-end mpmath %.1e over %d samples, 40 vs 60 digits %.1e"
      % (cf["initial_condition_residual"], cf["float64_reconstruction_max_abs"],
         cf["n_spot_samples"], cf["dps40_vs_dps60_max_abs"]))

rule("T2a  the equal-cost figure the appendix prints, re-measured. "
     "vps4 position error, root mean square, Larmor radii, over t = 120.")
w = S1["w12_equal_cost_remeasured"]
print("%-28s %10s %14s %14s %10s"
      % ("step", "h", "vs closed form", "vs DOP853 1e-12", "inflated"))
for k in ("h1.0e-3", "h7.19e-4_corrector_budget"):
    v = w[k]
    print("%-28s %10.3e %14.4e %14.4e %10.1f"
          % (k, v["h"], v["pos_err_rms_vs_closed_form"],
             v["pos_err_rms_vs_dop853_rtol1e-12"],
             v["inflation_of_the_paper_reference"]))
print("\nequal-cost factor against the best of the ninety-six learned runs "
      "(%.2e Larmor radii): the paper prints %.2e, the closed form gives %.3e"
      % (w["sympnet_best_traj"], w["equal_cost_factor_paper"],
         w["equal_cost_factor_closed_form"]))

rule("T2b  how far the in-band power moves when the reference is refined "
     "(position channel).  A run whose number moves is measuring the "
     "reference.")
print("%-28s %12s %12s %12s %s"
      % ("run", "exact", "DOP853 1e-12", "DOP853 3e-14", "verdict"))
for k, v in sorted(S1["verdict"].items()):
    p = v["position"]
    print("%-28s %12.4e %12.4e %12.4e %s"
          % (k, p["p_band_exact"], p["p_band_dop853_paper"],
             p["p_band_dop853_tight"],
             "REFERENCE-LIMITED" if p["reference_limited"] else "sound"))


# ------------------------------------------------------------------ T3 -----
def table_runs(hz, mode, ch):
    rule("T3  %s / %s / %s channel.   %d samples, t = %.1f, %.1f gyro-orbits, "
         "%d bins strictly inside f/Omega_c < 0.2"
         % (hz, mode, ch, S2["horizons"][hz]["n_samples"],
            S2["horizons"][hz]["t_final"], S2["horizons"][hz]["gyro_orbits"],
            S2["horizons"][hz]["bins_in_band"]))
    runs = S2["horizons"][hz]["runs"][mode]
    print("%-10s %12s %12s %12s %8s %10s %10s"
          % ("scheme", "in band", "out of band", "total", "in-band",
             "at Omega_c", "Boris/x"))
    print("%-10s %12s %12s %12s %8s %10s %10s"
          % ("", "", "", "", "fraction", "fraction", "orders"))
    for k in SCHEMES:
        r = runs[k][ch]
        b = (S2["boris_over_scheme"][hz][mode][ch][k]["band_ratio_orders"]
             if k != "boris" else 0.0)
        print("%-10s %12.4e %12.4e %12.4e %8.4f %10.4f %10.3f"
              % (NAMES[k], r["p_band"], r["p_out"], r["p_total"],
                 r["frac_in_band"], r["harmonics"]["per_harmonic"][0], b))


for hz in HZ:
    for mode in ("fixed_step", "equal_total_flops"):
        for ch in ("position", "energy"):
            table_runs(hz, mode, ch)


# ------------------------------------------------------------------ T4 -----
rule("T4  the decomposition.  band ratio = (amplitude ratio)^2 x concentration, "
     "exactly.  Position channel.")
for hz in HZ:
    for mode in ("fixed_step", "equal_total_flops"):
        print("\n-- %s / %s" % (hz, mode))
        print("%-10s %10s %10s %10s %12s %12s %10s"
              % ("scheme", "band ord", "total ord", "conc ord", "amp ratio",
                 "plain rms x", "identity"))
        for k in SCHEMES:
            if k == "boris":
                continue
            v = S2["boris_over_scheme"][hz][mode]["position"][k]
            print("%-10s %10.3f %10.3f %10.3f %12.4g %12.4g %10.1e"
                  % (NAMES[k], v["band_ratio_orders"], v["total_ratio_orders"],
                     v["concentration_orders"], v["amplitude_ratio"],
                     v["plain_rms_ratio"], v["identity_residual"]))


# ------------------------------------------------------------------ T5 -----
rule("T5  leakage control.  In-band power under Hann and under "
     "Blackman-Harris (sidelobes 61 dB lower).  A ratio far from one means "
     "the in-band figure is the skirt of the line at Omega_c, not the scheme.")
print("%-10s %-18s %-10s %12s %12s %10s"
      % ("horizon", "mode", "scheme", "Hann", "Bl.-Harris", "ratio"))
for hz in HZ:
    for mode in ("fixed_step", "equal_total_flops"):
        for k in SCHEMES:
            r = S2["horizons"][hz]["runs"][mode][k]["position"]
            print("%-10s %-18s %-10s %12.4e %12.4e %10.4g"
                  % (hz, mode, NAMES[k], r["p_band"], r["bh"]["p_band"],
                     r["bh"]["ratio_to_hann_band"]))


# ------------------------------------------------------------------ T6 -----
rule("T6  the band edge is not a choice.  In-band power against the "
     "threshold, H1_paper, fixed step, position channel.")
runs = S2["horizons"]["H1_paper"]["runs"]["fixed_step"]
print("%8s " % "edge" + " ".join("%12s" % NAMES[k] for k in SCHEMES)
      + " %10s %10s" % ("B/corr", "B/vps4"))
for i, e in enumerate(C.SWEEP_EDGES):
    row = [runs[k]["position"]["sweep_p_band"][i] for k in SCHEMES]
    bc = np.log10(row[0] / row[1]) if row[1] > 0 else float("inf")
    bv = np.log10(row[0] / row[3]) if row[3] > 0 else float("inf")
    print("%8.3f " % e + " ".join("%12.4e" % v for v in row)
          + " %10.3f %10.3f" % (bc, bv))

rule("T6b  the same sweep at equal total flops, H1_paper, position channel.")
runs = S2["horizons"]["H1_paper"]["runs"]["equal_total_flops"]
print("%8s " % "edge" + " ".join("%12s" % NAMES[k] for k in SCHEMES)
      + " %10s" % "corr/vps4")
for i, e in enumerate(C.SWEEP_EDGES):
    row = [runs[k]["position"]["sweep_p_band"][i] for k in SCHEMES]
    cv = np.log10(row[1] / row[3]) if row[3] > 0 else float("inf")
    print("%8.3f " % e + " ".join("%12.4e" % v for v in row)
          + " %10.3f" % cv)


# ------------------------------------------------------------------ T7 -----
rule("T7  the time-domain metrics the pre-registration fixes, H1_paper, "
     "fixed step.")
for ch in ("position", "energy"):
    print("\n-- %s channel" % ch)
    print("%-10s %12s %12s %12s %12s"
          % ("scheme", "max", "rms", "final", "envelope end"))
    for k in SCHEMES:
        t = S2["horizons"]["H1_paper"]["runs"]["fixed_step"][k][ch]["time"]
        print("%-10s %12.4e %12.4e %12.4e %12.4e"
              % (NAMES[k], t["max"], t["rms"], t["final"],
                 t["envelope_deciles"][-1]))


# ------------------------------------------------------------------ T8 -----
rule("T8  eight initial conditions, fixed step, window of Table 4.  "
     "Boris over scheme, in-band power, orders of magnitude.")
for ch in ("position", "energy", "position_scalar_norm"):
    print("\n-- %s" % ch)
    print("%-10s %10s %10s %10s %14s"
          % ("scheme", "median", "min", "max", "draws >= 6 ord"))
    for k, v in S3["boris_over_scheme_orders"][ch].items():
        print("%-10s %10.3f %10.3f %10.3f %10d / 8"
              % (NAMES[k], v["median"], v["min"], v["max"],
                 v["n_draws_above_six_orders"]))


# ------------------------------------------------------------------ T9 -----
rule("T9  the gate.  G0: at equal total flops the vps4 in-band power is "
     "below the corrector's.")
for k, v in S2["G0"].items():
    print("%-24s %-5s corrector/vps4 = %.4e  (%.2f orders)"
          % (k, "yes" if v["vps4_below_corrector"] else "NO",
             v["ratio_corrector_over_vps4"],
             np.log10(v["ratio_corrector_over_vps4"])))

rule("T10  resolution.  Bins strictly inside the band, by horizon.")
print("%-12s %10s %12s %14s %12s"
      % ("horizon", "samples", "gyro-orbits", "bin spacing", "bins in band"))
for hz in HZ:
    h = S2["horizons"][hz]
    print("%-12s %10d %12.1f %14.4e %12d"
          % (hz, h["n_samples"], h["gyro_orbits"], h["df_over_omega_c"],
             h["bins_in_band"]))
