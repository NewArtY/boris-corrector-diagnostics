"""rc2_horizon.py -- Section 7's horizon numbers, read against the closed form.

`rc1_calibration.py` has shown that this stand reproduces every one of them on
the ruler that printed them.  This script keeps the runs and changes the ruler.

WHAT IS RECOMPUTED
------------------
Everything in Section 7 that is measured against Boris at h/150:

    117.8 at 19.1 gyro-orbits, 32.7 at 50, unity at 101, 0.07 at 200
        (../horizon/crossover.py -> crossover.json)
    the factors 143 at 1e3 and 1575 at 1e4
        (../horizon/traj.py -> traj_summary.json, reciprocals of
         traj_gain_projected)
    22.1 -> 74.1 gyro-orbits at a threshold of one Larmor radius
        (crossover.json, {boris,proj}_reaches_1_larmor_at_gyr)
    0.417, 1.462 and 1.632 Larmor radii, the Boris saturation

Both rulers are applied to the *same* coarse runs, so every shift reported
here is the ruler and nothing else.  The 1e4 grid is run on both rulers too,
so the 1575 has its own calibration and is not taken on trust from the 1e3 one.

THE DECOMPOSITION THE PRE-REGISTRATION ASKED FOR
------------------------------------------------
Write e_meas = r_run - r_ruler (what Section 7 prints), e_ruler = r_ruler -
r_exact (the ruler's own error, 1.40e-3 Larmor radii) and e_true = r_run -
r_exact.  Then e_true = e_meas + e_ruler exactly, and

    rms(e_true)^2 = rms(e_meas)^2 + rms(e_ruler)^2
                    + 2 rho rms(e_meas) rms(e_ruler)

with rho the correlation of the two error fields over the record.  The
pre-registration's three cases -- uncorrelated (rho = 0, gain 111), aligned
(rho = +1, gain 86), opposite (rho = -1, gain 201) -- are the three values of
rho, and rho is measured here rather than assumed.  It is the number that says
*why* the answer came out where it did.

ENERGY AND MU
-------------
`../horizon/fast.py` does not use the Boris ruler for these: it scores the
energy against E_0 e^{-t/tau} and mu against (E/B)/(E_0/B_0), both adiabatic
statements.  The closed form scores those rulers too, and the residual it
leaves is the floor under every energy and mu number of Section 7.  It is
measured here out to 1e5 gyro-orbits, the longest horizon the section quotes.

Writes rc2_horizon.json.  Draws nothing.
Usage: python rc2_horizon.py [--force]
"""
import json
import os
import sys
import time

import numpy as np

import rc_common as RC
from rc_common import check_or_write

import fast as F
import traj as T

OUT = RC.outpath("rc2_horizon.json")
HORIZON = os.path.join(RC.EXP, "horizon")

GAIN_HORIZONS = [19.1, 25, 50, 100, 200, 300, 500, 1000]
GRIDS = [("H1e3", 1e3, True), ("H1e4", 1e4, True)]
MODES = ("boris", "raw", "proj")


def cache_path(tag):
    return RC.outpath("rc_coarse_%s.npz" % tag)


def coarse_runs(tag, n_w):
    p = cache_path(tag)
    if tag == "H1e3":
        p1 = RC.outpath("rc1_coarse_1e3.npz")
        if os.path.exists(p1):
            z = np.load(p1)
            if int(z["n_w"]) == n_w:
                return {m: z[m] for m in MODES}, float(z["seconds"])
    if os.path.exists(p):
        z = np.load(p)
        if int(z["n_w"]) == n_w:
            return {m: z[m] for m in MODES}, float(z["seconds"])
    t0 = time.time()
    R = {m: T.coarse(m, n_w) for m in MODES}
    el = time.time() - t0
    np.savez_compressed(p, n_w=n_w, seconds=el, **R)
    return R, el


def decompose(R_run, R_ruler, R_exact):
    """rms of the measured, the true and the ruler error, and their rho."""
    e_meas = R_run - R_ruler
    e_ruler = R_ruler - R_exact
    e_true = R_run - R_exact
    a = RC.rms(np.linalg.norm(e_meas, axis=1))
    b = RC.rms(np.linalg.norm(e_ruler, axis=1))
    c = RC.rms(np.linalg.norm(e_true, axis=1))
    cross = float(np.mean(np.sum(e_meas * e_ruler, axis=1)))
    rho = cross / (a * b) if a > 0 and b > 0 else None
    return {"rms_measured": a, "rms_ruler": b, "rms_true": c,
            "rho": rho,
            "identity_check": float(abs(c ** 2 - (a ** 2 + b ** 2
                                                  + 2 * cross))
                                    / max(c ** 2, 1e-300))}


def gain_table(tg, crms, key_num="boris", key_den="proj"):
    rows = []
    for h in GAIN_HORIZONS:
        j = np.searchsorted(tg, h) - 1
        if j < 1:
            continue
        row = {"gyro_orbits_requested": float(h),
               "gyro_orbits_sampled": float(tg[j])}
        for m in MODES:
            row["%s_pos_err_rms" % m] = float(crms[m][j])
        row["traj_gain_projected"] = float(crms[key_num][j] / crms[key_den][j])
        rows.append(row)
    return rows


def main():
    force = "--force" in sys.argv
    with open(os.path.join(HORIZON, "crossover.json"), encoding="utf-8") as fh:
        cr = json.load(fh)
    with open(os.path.join(HORIZON, "traj_summary.json"), encoding="utf-8") as fh:
        tr = json.load(fh)

    out = {"meta": {
        "what": "Section 7's horizon numbers on the closed form, with the "
                "shift of each against the printed value",
        "n_random_draws": 0, "tau": RC.TAU, "dt": RC.DT, "refine": RC.REFINE,
        "rulers": {"old": "Boris at h/150, the reference of Section 7 and the "
                          "training target of the corrector",
                   "new": "the closed form of ../spectral/sw_common.py, "
                          "zeta'' + (Bz^2/4) zeta = 0 -> Bessel order 0"},
    }, "grids": {}}

    for tag, H, do_old in GRIDS:
        n_w = int(round(H * RC.TWO_PI / RC.DT))
        ts = np.arange(1, n_w + 1) * RC.DT
        tg = ts / RC.TWO_PI
        g = {"gyro_orbits": H, "n_coarse": n_w}

        R, t_coarse = coarse_runs(tag, n_w)
        g["coarse_seconds"] = t_coarse

        t0 = time.time()
        Rex, Vex = RC.closed_form(ts)
        g["closed_form_seconds"] = time.time() - t0

        t0 = time.time()
        _, Rold, _, _ = F.fine_reference(RC.TAU, RC.DT / RC.REFINE,
                                         n_w * RC.REFINE, RC.REFINE)
        Rold = Rold[:n_w]
        g["old_ruler_seconds"] = time.time() - t0
        print("%s: %d coarse steps, ruler %d fine steps in %.1f s, closed "
              "form in %.2f s" % (tag, n_w, n_w * RC.REFINE,
                                  g["old_ruler_seconds"],
                                  g["closed_form_seconds"]))

        crms = {}
        for ruler, Rr in (("old", Rold), ("new", Rex)):
            e = {m: np.linalg.norm(R[m] - Rr, axis=1) for m in MODES}
            crms[ruler] = {m: RC.running_rms(e[m]) for m in MODES}
            block = {
                "gain_vs_horizon": gain_table(tg, crms[ruler]),
                "crossover_gyrations": RC.first_below(
                    crms[ruler]["boris"] / crms[ruler]["proj"], 1.0, tg),
                "rms_over_whole_grid": {m: RC.rms(e[m]) for m in MODES},
                "final_over_whole_grid": {m: float(e[m][-1]) for m in MODES},
            }
            block["traj_gain_projected_whole_grid"] = (
                block["rms_over_whole_grid"]["boris"]
                / block["rms_over_whole_grid"]["proj"])
            block["disadvantage_factor_whole_grid"] = (
                1.0 / block["traj_gain_projected_whole_grid"])
            for m in MODES:
                block["%s_reaches_1_larmor_at_gyr" % m] = RC.first_crossing(
                    e[m], 1.0, tg)
            if block["proj_reaches_1_larmor_at_gyr"] and \
                    block["boris_reaches_1_larmor_at_gyr"]:
                block["one_larmor_horizon_gain"] = (
                    block["proj_reaches_1_larmor_at_gyr"]
                    / block["boris_reaches_1_larmor_at_gyr"])
            g[ruler] = block

        # the ruler's own error, and the decomposition, on this grid
        d_ruler = np.linalg.norm(Rold - Rex, axis=1)
        g["ruler_own_error"] = {
            "rms_over_whole_grid": RC.rms(d_ruler),
            "max": float(d_ruler.max()), "final": float(d_ruler[-1])}
        j19 = np.searchsorted(tg, 19.1) - 1
        if j19 > 1:
            g["ruler_own_error"]["rms_over_19.1_gyro_orbits"] = \
                RC.rms(d_ruler[:j19 + 1])

        g["decomposition"] = {}
        windows = [("whole_grid", slice(0, n_w))]
        if j19 > 1:
            windows.append(("19.1_gyro_orbits", slice(0, j19 + 1)))
        j50 = np.searchsorted(tg, 50.0) - 1
        if 1 < j50 < n_w:
            windows.append(("50_gyro_orbits", slice(0, j50 + 1)))
        for wname, sl in windows:
            g["decomposition"][wname] = {
                m: decompose(R[m][sl], Rold[sl], Rex[sl]) for m in MODES}

        # cost of the two rulers on this grid, in flops
        g["cost_flops"] = {
            "old_ruler": RC.flops_boris_reference(n_w),
            "new_ruler": RC.flops_closed_form(n_w),
            "ratio": RC.flops_boris_reference(n_w) / RC.flops_closed_form(n_w),
            "the_runs_being_measured": {
                "boris": RC.flops_boris_run(n_w),
                "corrector": RC.flops_corrector_run(n_w)}}
        out["grids"][tag] = g

    # ------------------------------------------------- the was/now comparison
    g3, g4 = out["grids"]["H1e3"], out["grids"]["H1e4"]
    old_rows = {r["gyro_orbits_requested"]: r for r in g3["old"]["gain_vs_horizon"]}
    new_rows = {r["gyro_orbits_requested"]: r for r in g3["new"]["gain_vs_horizon"]}
    shifts = {}
    for h in (19.1, 25, 50, 100, 200, 300, 500, 1000):
        shifts["gain_at_%g_gyro_orbits" % h] = RC.rel(
            new_rows[h]["traj_gain_projected"],
            old_rows[h]["traj_gain_projected"])
        shifts["corrector_error_at_%g_gyro_orbits" % h] = RC.rel(
            new_rows[h]["proj_pos_err_rms"], old_rows[h]["proj_pos_err_rms"])
        shifts["boris_error_at_%g_gyro_orbits" % h] = RC.rel(
            new_rows[h]["boris_pos_err_rms"], old_rows[h]["boris_pos_err_rms"])
    shifts["crossover_gyrations"] = RC.rel(g3["new"]["crossover_gyrations"],
                                           g3["old"]["crossover_gyrations"])
    for m in MODES:
        shifts["%s_reaches_1_larmor" % m] = RC.rel(
            g3["new"]["%s_reaches_1_larmor_at_gyr" % m],
            g3["old"]["%s_reaches_1_larmor_at_gyr" % m])
    shifts["one_larmor_horizon_gain"] = RC.rel(
        g3["new"]["one_larmor_horizon_gain"],
        g3["old"]["one_larmor_horizon_gain"])
    shifts["disadvantage_factor_at_1e3"] = RC.rel(
        g3["new"]["disadvantage_factor_whole_grid"],
        g3["old"]["disadvantage_factor_whole_grid"])
    shifts["disadvantage_factor_at_1e4"] = RC.rel(
        g4["new"]["disadvantage_factor_whole_grid"],
        g4["old"]["disadvantage_factor_whole_grid"])
    shifts["boris_saturation_at_1e3"] = RC.rel(
        g3["new"]["rms_over_whole_grid"]["boris"],
        g3["old"]["rms_over_whole_grid"]["boris"])
    shifts["boris_saturation_at_1e4"] = RC.rel(
        g4["new"]["rms_over_whole_grid"]["boris"],
        g4["old"]["rms_over_whole_grid"]["boris"])
    out["shifts"] = shifts

    # calibration of the 1e4 grid against the committed file
    out["calibration_1e4"] = {
        "committed_traj_gain": tr["H1e+04_ref150x"]["traj_gain_projected"],
        "stand_traj_gain": g4["old"]["traj_gain_projected_whole_grid"],
        "committed_boris_rms": tr["H1e+04_ref150x"]["boris"]["pos_err_rms"],
        "stand_boris_rms": g4["old"]["rms_over_whole_grid"]["boris"]}
    c = out["calibration_1e4"]
    bad4 = (abs(c["stand_traj_gain"] - c["committed_traj_gain"])
            > 1e-9 * abs(c["committed_traj_gain"])
            or abs(c["stand_boris_rms"] - c["committed_boris_rms"])
            > 1e-9 * abs(c["committed_boris_rms"]))
    out["calibration_1e4"]["ok"] = not bool(bad4)
    if bad4:
        print("CALIBRATION FAILED on the 1e4 grid")
        return 1

    # ---------------------------------------- energy and mu: the other rulers
    n5 = int(round(1e5 * RC.TWO_PI / RC.DT))
    ts5 = np.arange(n5 + 1) * RC.DT
    t0 = time.time()
    _, Vex5 = RC.closed_form(ts5)
    E5 = RC.energy(Vex5)
    E0 = float(E5[0])
    env5 = E0 * np.exp(-ts5 / RC.TAU)
    mu5 = np.abs((E5 / RC.bz(ts5)) / (E0 / RC.bz(0.0)) - 1.0)
    adia = {"seconds": time.time() - t0}
    for H in (1e3, 1e4, 1e5):
        m = ts5 <= H * RC.TWO_PI
        adia["%.0e" % H] = {
            "energy_ruler_floor_max": float(np.max(np.abs(E5[m] - env5[m])) / E0),
            "energy_ruler_floor_at_end": float(abs(E5[m][-1] - env5[m][-1]) / E0),
            "mu_ruler_floor_max": float(mu5[m].max()),
            "mu_ruler_floor_at_end": float(mu5[m][-1])}
    with open(os.path.join(HORIZON, "long_runs_summary.json"),
              encoding="utf-8") as fh:
        lr = json.load(fh)["results"]["paper_tau1.2e5"]
    for H in ("1e+03", "1e+04", "1e+05"):
        k = "%.0e" % float(H)
        adia[k]["boris_energy_err_max_reported"] = \
            lr["boris"]["horizons"][H]["energy_err_max"]
        adia[k]["boris_mu_err_max_reported"] = \
            lr["boris"]["horizons"][H]["mu_err_max"]
        adia[k]["proj_energy_err_max_reported"] = \
            lr["proj"]["horizons"][H]["energy_err_max"]
        adia[k]["floor_as_fraction_of_boris_energy_err"] = (
            adia[k]["energy_ruler_floor_max"]
            / max(lr["boris"]["horizons"][H]["energy_err_max"], 1e-300))
        adia[k]["floor_as_fraction_of_boris_mu_err"] = (
            adia[k]["mu_ruler_floor_max"]
            / max(lr["boris"]["horizons"][H]["mu_err_max"], 1e-300))
    adia["note"] = ("../horizon/long_runs.py scores the energy against "
                    "E_0 e^{-t/tau} and mu against (E/B)/(E_0/B_0).  Neither "
                    "is the Boris ruler, so the energy and mu numbers of "
                    "Section 7 are NOT affected by the ruler swap this wave "
                    "is about.  They carry a floor of their own, measured "
                    "here: it is the score the exact solution gets on those "
                    "same rulers.")
    out["adiabatic_ruler_floor"] = adia

    # ----------------------------------------------------------------- print
    print("\n=== the trajectory advantage, was and now ===")
    print("%-14s %14s %14s %10s" % ("horizon", "old ruler", "closed form",
                                    "x shift"))
    for h in (19.1, 25, 50, 100, 200, 300, 500, 1000):
        s = shifts["gain_at_%g_gyro_orbits" % h]
        print("%-14s %14.4f %14.4f %10.3f"
              % ("%g gyro-orb" % h, s["old"], s["new"], s["ratio"]))
    print("\n%-34s %14s %14s %10s" % ("", "old", "new", "x"))
    for k in ("crossover_gyrations", "boris_reaches_1_larmor",
              "proj_reaches_1_larmor", "one_larmor_horizon_gain",
              "disadvantage_factor_at_1e3", "disadvantage_factor_at_1e4",
              "boris_saturation_at_1e3", "boris_saturation_at_1e4"):
        s = shifts[k]
        print("%-34s %14.5g %14.5g %10.3f" % (k, s["old"], s["new"],
                                              s["ratio"]))

    print("\n=== the decomposition: why ===")
    for wname in ("19.1_gyro_orbits", "50_gyro_orbits"):
        d = out["grids"]["H1e3"]["decomposition"].get(wname)
        if not d:
            continue
        print("  window %s" % wname)
        for m in MODES:
            v = d[m]
            print("    %-6s measured %.4e   ruler %.4e   true %.4e   "
                  "rho %+.4f" % (m, v["rms_measured"], v["rms_ruler"],
                                 v["rms_true"], v["rho"]))

    print("\n=== the adiabatic ruler of the energy and mu readouts ===")
    for H in ("1e+03", "1e+04", "1e+05"):
        k = "%.0e" % float(H)
        a = adia[k]
        print("  %s gyro-orbits: exact solution scores %.3e on the energy "
              "ruler (Boris reports %.3e, ratio %.2e) and %.3e on the mu "
              "ruler (Boris reports %.3e, ratio %.2e)"
              % (H, a["energy_ruler_floor_max"],
                 a["boris_energy_err_max_reported"],
                 a["floor_as_fraction_of_boris_energy_err"],
                 a["mu_ruler_floor_max"], a["boris_mu_err_max_reported"],
                 a["floor_as_fraction_of_boris_mu_err"]))

    RC.assert_no_draws(0)
    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
