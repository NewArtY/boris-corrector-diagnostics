"""rc1_calibration.py -- the stand, before it is allowed to say anything.

Three things, in this order, and the script exits non-zero if any of them
fails.  Nothing downstream is worth reading if this one does not pass.

1.  THE CLOSED FORM IS PRICED.  `fast_basis` is checked against the committed
    mpmath basis of `../spectral/sw_common.py` on every grid W18 uses.  The
    number that has to be small is the reconstructed position difference,
    against the 3.5e-3 Larmor radii the measurement is about.

2.  THE CLOSED FORM COVERS WHAT THE HORIZON EXPERIMENTS READ.  They read three
    things: position, energy, and the adiabatic invariant mu = (E/B)/(E_0/B_0).
    The closed form returns r AND v, so all three follow; this section shows
    that they do, by checking the closed-form energy against the analytic
    envelope E_0 e^{-t/tau} that `../horizon/fast.py` uses as its energy ruler,
    and by exhibiting mu on the exact solution.  If the closed form did not
    cover one of the three, the recomputation of that one would have to stop
    here.

3.  THE OLD NUMBERS COME BACK ON THE OLD RULER.  The stand rebuilds the Boris
    h/150 reference and the three coarse runs and must reproduce, to 1e-9,
    every Section 7 number that stands on them: 117.8, 32.7, the crossover at
    101, 0.07 at 200, 22.1 -> 74.1, and the 1e3-gyro-orbit row of
    `../horizon/traj_summary.json` behind the factor 143.  A stand that cannot
    reproduce them is measuring something else and its new numbers mean
    nothing.

The reference-side integrator and the three coarse runs are `../horizon/fast.py`
and `../horizon/traj.py`, imported and called unchanged: reusing the code that
produced the printed numbers is what makes step 3 a test rather than a
coincidence.

The closed-form ruler for the same grid is built here too, so that this file
also carries the ruler's own error -- the quantity the whole wave turns on.

Writes rc1_calibration.json.  Draws nothing.
Usage: python rc1_calibration.py [--force]
"""
import json
import os
import sys
import time

import numpy as np

import rc_common as RC
from rc_common import check_or_write

import fast as F                                     # ../horizon/fast.py
import traj as T                                     # ../horizon/traj.py

OUT = RC.outpath("rc1_calibration.json")
CACHE = RC.outpath("rc1_coarse_1e3.npz")

H_MAIN = 1e3                       # the grid crossover.py runs on
GAIN_HORIZONS = [19.1, 25, 50, 100, 200, 300, 500, 1000]

#: what the committed files say, read at run time, not copied.
HORIZON = os.path.join(RC.EXP, "horizon")


def committed():
    with open(os.path.join(HORIZON, "crossover.json"), encoding="utf-8") as fh:
        cr = json.load(fh)
    with open(os.path.join(HORIZON, "traj_summary.json"), encoding="utf-8") as fh:
        tr = json.load(fh)
    with open(os.path.join(HORIZON, "validation.json"), encoding="utf-8") as fh:
        va = json.load(fh)
    return cr, tr, va


def coarse_runs(n_w):
    """boris / raw / proj at the working step.  `traj.coarse` unchanged."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if int(z["n_w"]) == n_w:
            return {m: z[m] for m in ("boris", "raw", "proj")}, float(z["seconds"])
    t0 = time.time()
    R = {m: T.coarse(m, n_w) for m in ("boris", "raw", "proj")}
    el = time.time() - t0
    np.savez_compressed(CACHE, n_w=n_w, seconds=el, **R)
    return R, el


def gain_table(tg, crms_b, crms_p, crms_r):
    rows = []
    for h in GAIN_HORIZONS:
        j = np.searchsorted(tg, h) - 1
        if j < 1:
            continue
        rows.append({"gyro_orbits_requested": float(h),
                     "gyro_orbits_sampled": float(tg[j]),
                     "boris_pos_err_rms": float(crms_b[j]),
                     "proj_pos_err_rms": float(crms_p[j]),
                     "raw_pos_err_rms": float(crms_r[j]),
                     "traj_gain_projected": float(crms_b[j] / crms_p[j])})
    return rows


def main():
    force = "--force" in sys.argv
    cr, tr, va = committed()
    out = {"meta": {
        "what": "W18 stand calibration: the closed form priced, and the "
                "Section 7 numbers reproduced on the ruler that printed them",
        "n_random_draws": 0,
        "tau": RC.TAU, "dt": RC.DT, "refine": RC.REFINE,
        "r0": list(RC.R0), "v0": list(RC.V0),
        "reused_verbatim": ["../horizon/fast.py:fine_reference",
                            "../horizon/traj.py:coarse",
                            "../spectral/sw_common.py:exact_from_basis",
                            "../spectral/sw_common.py:bessel_basis"],
    }}

    n_w = int(round(H_MAIN * RC.TWO_PI / RC.DT))
    ts = np.arange(1, n_w + 1) * RC.DT               # crossover.py's grid
    ts0 = np.arange(0, n_w + 1) * RC.DT              # with t = 0, for energy
    tg = ts / RC.TWO_PI

    # ------------------------------------------------- 1. price the closed form
    t0 = time.time()
    agree = {"H1e3_coarse_grid": RC.basis_agreement(ts0)}
    n_19 = int(round(RC.T_FINAL / RC.DT))
    agree["H19.1_paper_window"] = RC.basis_agreement(
        np.arange(n_19 + 1) * RC.DT)
    agree["H1e4_coarse_grid"] = RC.basis_agreement(
        np.arange(int(round(1e4 * RC.TWO_PI / RC.DT)) + 1) * RC.DT)
    agree["H1e5_coarse_grid"] = RC.basis_agreement(
        np.arange(int(round(1e5 * RC.TWO_PI / RC.DT)) + 1) * RC.DT)
    agree["seconds"] = time.time() - t0
    out["closed_form_price"] = agree
    worst = max(v["position_max_abs"] for k, v in agree.items()
                if isinstance(v, dict))
    print("closed form: float64 basis agrees with the committed mpmath basis "
          "to %.2e Larmor radii on every grid W18 uses" % worst)
    if worst > 1e-8:
        print("FAILED: the float64 basis is not good enough for this wave")
        return 1

    # -------------------------------- 2. does it cover position, energy and mu?
    Rex0, Vex0 = RC.closed_form(ts0)
    E_ex = RC.energy(Vex0)
    E0 = float(E_ex[0])
    env = E0 * np.exp(-ts0 / RC.TAU)                 # fast.py's energy ruler
    mu_ex = np.abs((E_ex / RC.bz(ts0)) / (E0 / RC.bz(0.0)) - 1.0)
    cover = {
        "position": "r(t) returned by exact_from_basis",
        "energy_max_abs_dev_from_adiabatic_envelope_over_E0":
            float(np.max(np.abs(E_ex - env)) / E0),
        "energy_dev_from_adiabatic_envelope_at_end_over_E0":
            float(abs(E_ex[-1] - env[-1]) / E0),
        "mu_error_of_the_exact_solution_max": float(mu_ex.max()),
        "mu_error_of_the_exact_solution_at_end": float(mu_ex[-1]),
        "t_end": float(ts0[-1]), "gyro_orbits": float(H_MAIN),
        "note": "../horizon/fast.py measures the energy error against "
                "E_0 e^{-t/tau} and mu against (E/B)/(E_0/B_0).  Both are "
                "adiabatic statements, not exact ones.  The two numbers above "
                "are what the exact solution itself scores on those rulers, "
                "i.e. the floor those readouts carry before any integrator "
                "is run.",
    }
    out["coverage"] = cover
    print("coverage: closed form gives r and v, so position, energy and mu all "
          "follow; the exact solution scores %.3e on fast.py's energy ruler "
          "and %.3e on its mu ruler at 1e3 gyro-orbits"
          % (cover["energy_max_abs_dev_from_adiabatic_envelope_over_E0"],
             cover["mu_error_of_the_exact_solution_max"]))

    # ---------------------------------- 3. the old numbers on the old ruler
    t0 = time.time()
    _, Rr, _, _ = F.fine_reference(RC.TAU, RC.DT / RC.REFINE,
                                   n_w * RC.REFINE, RC.REFINE)
    t_ref = time.time() - t0
    R, t_coarse = coarse_runs(n_w)
    print("Boris h/150 ruler: %d fine steps in %.1f s;  coarse runs %.1f s"
          % (n_w * RC.REFINE, t_ref, t_coarse))

    err_old = {m: np.linalg.norm(R[m] - Rr[:n_w], axis=1) for m in R}
    crms_old = {m: RC.running_rms(err_old[m]) for m in err_old}
    ratio_old = crms_old["boris"] / crms_old["proj"]

    rep = {"reference_refinement": RC.REFINE,
           "reference_seconds": t_ref, "coarse_seconds": t_coarse,
           "crossover_gyrations": RC.first_below(ratio_old, 1.0, tg),
           "gain_vs_horizon": gain_table(tg, crms_old["boris"],
                                         crms_old["proj"], crms_old["raw"])}
    for m in ("boris", "proj", "raw"):
        rep["%s_reaches_1_larmor_at_gyr" % m] = RC.first_crossing(
            err_old[m], 1.0, tg)
    rep["one_larmor_horizon_gain"] = (rep["proj_reaches_1_larmor_at_gyr"]
                                      / rep["boris_reaches_1_larmor_at_gyr"])
    # the traj.py row behind the factor 143: rms over the whole 1e3 window
    rep["traj_H1e3_ref150x"] = {
        m: {"pos_err_rms": RC.rms(err_old[m]),
            "pos_err_final": float(err_old[m][-1])} for m in err_old}
    rep["traj_H1e3_ref150x"]["traj_gain_projected"] = (
        rep["traj_H1e3_ref150x"]["boris"]["pos_err_rms"]
        / rep["traj_H1e3_ref150x"]["proj"]["pos_err_rms"])
    out["reproduction_on_the_old_ruler"] = rep

    # --------------------------------------------------------- the comparison
    checks = []

    def chk(name, got, want, tol=1e-9):
        ok = want is not None and got is not None and \
            abs(got - want) <= tol * max(abs(want), 1e-300)
        checks.append({"name": name, "committed": want, "stand": got,
                       "rel_diff": (abs(got - want) / abs(want))
                       if (want not in (None, 0) and got is not None) else None,
                       "ok": bool(ok)})

    chk("crossover_gyrations (manuscript: 101)",
        rep["crossover_gyrations"], cr["crossover_gyrations"])
    for row_s, row_c in zip(rep["gain_vs_horizon"], cr["gain_vs_horizon"]):
        h = row_c["gyro_orbits_requested"]
        chk("gain at %g gyro-orbits" % h,
            row_s["traj_gain_projected"], row_c["traj_gain_projected"])
        chk("boris rms at %g" % h,
            row_s["boris_pos_err_rms"], row_c["boris_pos_err_rms"])
        chk("proj rms at %g" % h,
            row_s["proj_pos_err_rms"], row_c["proj_pos_err_rms"])
    for m in ("boris", "proj", "raw"):
        chk("%s reaches 1 Larmor radius" % m,
            rep["%s_reaches_1_larmor_at_gyr" % m],
            cr["%s_reaches_1_larmor_at_gyr" % m])
    tj = tr["H1e+03_ref150x"]
    for m in ("boris", "proj", "raw"):
        chk("traj 1e3 %s pos_err_rms" % m,
            rep["traj_H1e3_ref150x"][m]["pos_err_rms"], tj[m]["pos_err_rms"])
    chk("traj 1e3 gain (manuscript: 1/143)",
        rep["traj_H1e3_ref150x"]["traj_gain_projected"],
        tj["traj_gain_projected"])
    out["calibration_checks"] = checks

    print("\n%-46s %16s %16s %10s" % ("check", "committed", "stand", "rel"))
    for c in checks:
        print("%-46s %16.9g %16.9g %10.1e %s"
              % (c["name"][:46], c["committed"], c["stand"],
                 c["rel_diff"] if c["rel_diff"] is not None else float("nan"),
                 "" if c["ok"] else "   <-- FAILED"))
    n_bad = sum(1 for c in checks if not c["ok"])
    out["calibration_failures"] = n_bad
    if n_bad:
        print("\nCALIBRATION FAILED on %d of %d checks -- the stand is not "
              "measuring what Section 7 measured, and nothing downstream is "
              "meaningful." % (n_bad, len(checks)))
        return 1
    print("\ncalibration: %d of %d numbers reproduced to 1e-9, 117.8 among "
          "them" % (len(checks), len(checks)))

    # ------------------------------------ the ruler's own error, on this grid
    Rex = Rex0[1:]                                   # closed form at ts
    d_ruler = np.linalg.norm(Rr[:n_w] - Rex, axis=1)
    j19 = np.searchsorted(tg, 19.1) - 1
    out["ruler_own_error"] = {
        "what": "the Boris h/150 reference of Section 7, against the closed "
                "form, on the same grid",
        "rms_over_19.1_gyro_orbits": RC.rms(d_ruler[:j19 + 1]),
        "rms_over_1e3_gyro_orbits": RC.rms(d_ruler),
        "max_over_1e3_gyro_orbits": float(d_ruler.max()),
        "final_at_1e3_gyro_orbits": float(d_ruler[-1]),
        "w16_value_over_the_table4_window": 0.0013959214902257689,
        "corrector_reported_error_19.1": cr["gain_vs_horizon"][0]
                                           ["proj_pos_err_rms"],
    }
    r = out["ruler_own_error"]
    print("\nthe ruler's own error: %.4e Larmor radii rms over 19.1 gyro-"
          "orbits (W16 got %.4e over the 401-sample window), against the "
          "corrector's reported %.4e"
          % (r["rms_over_19.1_gyro_orbits"],
             r["w16_value_over_the_table4_window"],
             r["corrector_reported_error_19.1"]))
    print("the ruler is %.0f%% of what it is used to measure"
          % (100 * r["rms_over_19.1_gyro_orbits"]
             / r["corrector_reported_error_19.1"]))

    # ------------------------------------------------------------------- cost
    out["cost_flops"] = {
        "boris_ruler_h_over_150_per_sample": RC.flops_boris_reference(1),
        "closed_form_per_sample": RC.flops_closed_form(1),
        "ratio": RC.flops_boris_reference(1) / RC.flops_closed_form(1),
        "boris_ruler_H1e3_total": RC.flops_boris_reference(n_w),
        "closed_form_H1e3_total": RC.flops_closed_form(n_w),
        "note": "the model of Section 9 and of ../classical/schemes.py, "
                "reused unchanged: 1 flop per arithmetic operation, 20 per "
                "transcendental",
    }
    print("cost: the closed form is %.0f times cheaper per sample than the "
          "Boris ruler it replaces (%.0f vs %.0f flops)"
          % (out["cost_flops"]["ratio"],
             out["cost_flops"]["closed_form_per_sample"],
             out["cost_flops"]["boris_ruler_h_over_150_per_sample"]))

    RC.assert_no_draws(0)
    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
