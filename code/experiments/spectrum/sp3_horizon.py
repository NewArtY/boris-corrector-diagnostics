"""SP3: does the spectrum predict the error at a long horizon?

Writes sp3_horizon.json.  Rerunning recomputes and exits non-zero on any
disagreement with the committed file.  Requires sp2_spectra.json.

    python sp3_horizon.py [--force]

THE QUESTION
------------
Wave 10 found eigenvalues off the unit circle in a trained SympMat and read an
exponential blow-up off them.  That reading is sound there because SympMat is a
*linear* map: rho > 1 and ||z_n|| ~ rho^n are the same statement.  Whether it
survives on a nonlinear architecture is an empirical question, and this file
asks it in the only way that settles it -- run the maps out to 10^5 steps and
see whether the number measured beforehand predicted what happened.

Four candidate predictors are tested against one measurement.

  P1  log of the median spectral radius over the points the run visits
  P2  log of the largest spectral radius over all 64 linearisation points
  P3  the leading finite-time Lyapunov exponent along the orbit at n = 4000
  P4  the same, extrapolated to n -> infinity from the pair (1000, 4000), which
      is the only one of the four that returns zero on a scheme whose true
      exponents are zero

against

  g   the fitted growth per step of the running maximum of |w| over the last
      decade of a 10^5-step run in the frozen field, where the exact solution
      is the closed form r = (cos t, sin t) and no reference integrator enters.

A predictor is scored three ways, and they are not the same question:

  does it get the rate right          ratio P / g
  does it get the ranking right       Spearman rank correlation over the
                                      twelve cells
  does it call the blow-ups           the predicted step at which |w| passes
                                      10^3 against the step at which it did

THE SECOND CHANNEL
------------------
The modulus of an eigenvalue is not the only thing a spectrum carries.  For a
linear scheme the argument of the gyration pair *is* the per-step phase advance
-- for the Boris scheme it is parker1991's 2 arctan(Omega h / 2) -- and the
trajectory error of a bounded run is made of the phase drift that argument
predicts rather than of any growth of amplitude.  Whether that identity
survives on a learned map is the second question asked here, it costs nothing
once the eigenvalues are in hand, and the answer is no.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sp_common as S            # noqa: E402
import ea_arch as A              # noqa: E402

OUT = os.path.join(HERE, "sp3_horizon.json")
SP2 = os.path.join(HERE, "sp2_spectra.json")


def _rank(v):
    """Average ranks, so that ties do not bias the correlation."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    # average the ranks of tied values
    for val in np.unique(v):
        m = v == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / d) if d > 0 else float("nan")


def _orbit_rho(cell):
    """The median spectral radius over the points the run actually visits."""
    v = [r["max_abs"] for r in cell["per_point"]
         if r.get("finite") and r["point"].startswith("orbit")]
    return float(np.median(v)) if v else float("nan")


def _all_rho_max(cell):
    v = [r["max_abs"] for r in cell["per_point"] if r.get("finite")]
    return float(np.max(v)) if v else float("nan")


def _orbit_arg(cell):
    v = [r["rotation_arg"] for r in cell["per_point"]
         if r.get("finite") and r["point"].startswith("orbit")]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    assert os.path.exists(SP2), "run sp2_spectra.py first"
    sp2 = json.load(open(SP2, encoding="utf-8"))

    out = {"setup": {
        "field": "frozen: tau = infinity, B_z = 1, E = 0; exact solution "
                 "r(t) = (cos t, sin t), v(t) = (-sin t, cos t)",
        "n_steps": S.N_LONG,
        "gyro_orbits": float(S.N_LONG * S.DT / (2.0 * np.pi)),
        "dt": S.DT,
        "blowup_threshold": S.BLOWUP,
        "amplitude": "|w| in canonical coordinates; exactly sqrt(5)/2 on the "
                     "exact solution at every time",
        "growth_fit": "log of the running maximum against step number, over "
                      "the last decade of the run",
    }, "runs": {}, "predictors": {}, "frequency_channel": {}, "verdict": {}}

    # ------------------------------------------------------------ the runs
    cells = []
    bs = A.BorisStepper(S.TAU_FROZEN)
    out["runs"]["boris"] = S.long_run(bs, S.TAU_FROZEN)
    out["runs"]["boris"]["measured_frequency_per_step"] = \
        S.measured_frequency(bs, S.TAU_FROZEN)
    out["runs"]["boris"]["closed_form_frequency_per_step"] = S.boris_angle(S.DT)

    for arch in S.ARCHS:
        for rep in range(S.N_REP):
            st = S.load(arch, rep, S.TAU_FROZEN)
            key = "%s_r%d" % (arch, rep)
            r = S.long_run(st, S.TAU_FROZEN)
            r["measured_frequency_per_step"] = \
                S.measured_frequency(st, S.TAU_FROZEN)
            out["runs"][key] = r
            c2 = sp2["cells"]["%s_frozen" % key]
            cells.append({
                "cell": key, "arch": arch, "rep": rep,
                "P1_log_median_rho_on_orbit": float(np.log(_orbit_rho(c2))),
                "P2_log_max_rho_all_points": float(np.log(_all_rho_max(c2))),
                "P3_lyapunov_max_per_step": c2["lyapunov"]["lambda_max_per_step"],
                "P4_lyapunov_extrapolated": c2["lyapunov_convergence"][
                    "lambda_extrapolated"],
                "g_measured_growth_per_step": r["growth_per_step_measured"],
                "blowup_step": r["blowup_step"],
                "amplitude_running_max": r["amplitude_running_max"],
                "final_position_error": r["checkpoints"][
                    str(max(int(k) for k in r["checkpoints"]))]["position_err"],
                "final_energy_rel_err": r["checkpoints"][
                    str(max(int(k) for k in r["checkpoints"]))]["energy_rel_err"],
                "median_rotation_arg_on_orbit": _orbit_arg(c2),
                "measured_frequency_per_step": r["measured_frequency_per_step"],
            })

    out["cells"] = cells

    # ------------------------------------------------------ the four scores
    g = np.array([c["g_measured_growth_per_step"] for c in cells])
    for name in ("P1_log_median_rho_on_orbit", "P2_log_max_rho_all_points",
                 "P3_lyapunov_max_per_step", "P4_lyapunov_extrapolated"):
        p = np.array([c[name] for c in cells])
        m = np.isfinite(p) & np.isfinite(g)
        ratios = []
        for c in cells:
            gi = c["g_measured_growth_per_step"]
            pi = c[name]
            ratios.append(float(pi / gi) if np.isfinite(gi) and abs(gi) > 1e-9
                          else float("nan"))
        # the ratio is only a fair question where something actually grew: on a
        # bounded run the denominator is the residue of a flat fit and the
        # ratio says nothing about the predictor.
        fr = [r for r, c in zip(ratios, cells)
              if np.isfinite(r) and r > 0 and (
                  c["blowup_step"] is not None
                  or c["amplitude_running_max"] >= 10.0)]
        out["predictors"][name] = {
            "spearman_with_measured_growth": spearman(p[m], g[m]),
            "ratio_predicted_over_measured": ratios,
            "ratio_median_over_growing_cells": float(np.median(fr)) if fr
            else float("nan"),
            "ratio_range_over_growing_cells":
                [float(min(fr)), float(max(fr))] if fr else None,
        }

    # blow-up calls: only cells that actually blew up can be scored on timing
    blow = [c for c in cells if c["blowup_step"] is not None]
    calls = []
    for c in blow:
        a0 = out["runs"][c["cell"]]["amplitude_0"]
        target = float(np.log(S.BLOWUP / a0))
        row = {"cell": c["cell"], "actual_blowup_step": c["blowup_step"]}
        for name in ("P1_log_median_rho_on_orbit", "P2_log_max_rho_all_points",
                     "P3_lyapunov_max_per_step", "P4_lyapunov_extrapolated"):
            p = c[name]
            row["predicted_step_" + name[:2]] = float(target / p) if p > 0 \
                else None
        calls.append(row)
    out["predictors"]["blowup_calls"] = calls
    out["predictors"]["blowup_note"] = (
        "The predicted step is log(10^3 / |w_0|) divided by the predictor.  A "
        "predictor that had the rate right would land on the actual step; the "
        "ones here are short by three to four orders of magnitude, which is "
        "another way of saying that none of them is the growth rate of the "
        "state.")

    # ------------------------------------------------------ the phase channel
    #
    # The question asked here was whether the argument of the eigenvalue can
    # stand in for the phase advance of the scheme, which for the Boris scheme
    # it does exactly (parker1991 gives that angle in closed form).  It cannot,
    # and the reason is the same one that defeats the modulus: the Jacobian
    # rotates the *tangent space*, and only for a linear map is that the same
    # as the rotation of the state.  The frequency error below is therefore
    # taken from the run, as Section~7 of the manuscript already takes it, and
    # the argument is printed beside it to record the size of the disagreement
    # rather than to be used.
    exact = S.DT
    rows = []
    for c in cells:
        f = c["measured_frequency_per_step"]
        arg = c["median_rotation_arg_on_orbit"]
        rows.append({
            "cell": c["cell"],
            "arg_from_spectrum": arg,
            "spectrum_has_a_complex_pair": bool(abs(arg) > 1e-12),
            "frequency_measured_on_run": f,
            "arg_minus_measured": float(arg - f),
            "relative_frequency_error_vs_exact": float((f - exact) / exact),
            "steps_to_pi_of_phase_drift":
                float(np.pi / abs(f - exact)) if abs(f - exact) > 0
                else float("inf"),
        })
    b = out["runs"]["boris"]
    fb = b["measured_frequency_per_step"]
    rows.append({
        "cell": "boris",
        "arg_from_spectrum": S.boris_angle(S.DT),
        "spectrum_has_a_complex_pair": True,
        "frequency_measured_on_run": fb,
        "arg_minus_measured": float(S.boris_angle(S.DT) - fb),
        "relative_frequency_error_vs_exact": float((fb - exact) / exact),
        "steps_to_pi_of_phase_drift": float(np.pi / abs(fb - exact)),
    })
    learned = [r for r in rows if r["cell"] != "boris"]
    out["frequency_channel"] = {
        "exact_per_step": exact,
        "rows": rows,
        "boris_arg_minus_measured": rows[-1]["arg_minus_measured"],
        "worst_arg_minus_measured_learned": float(max(
            abs(r["arg_minus_measured"]) for r in learned)),
        "n_learned_cells_with_no_complex_pair": int(sum(
            1 for r in learned if not r["spectrum_has_a_complex_pair"])),
        "note": "On the Boris scheme the argument of the gyration eigenvalue "
                "and the phase advance measured on the run agree to 4e-16, "
                "which is parker1991's 2 arctan(Omega h / 2) recovered from a "
                "numerical Jacobian.  On the learned maps they do not agree at "
                "all: the four HNN cells have no complex eigenvalue on the "
                "orbit whatsoever -- all four eigenvalues are real, in "
                "reciprocal pairs -- and the SympNet's complex pair sits at "
                "0.263 to 0.269 where the run advances by 0.29997 per step.  "
                "The Jacobian rotates the tangent space; only for a linear map "
                "is that the rotation of the state.  So the argument channel "
                "fails to transfer from the linear case for exactly the reason "
                "the modulus channel does, and the frequency error quoted here "
                "and in Section 7 of the manuscript has to come from the run.",
    }
    assert abs(out["frequency_channel"]["boris_arg_minus_measured"]) < 1e-12, \
        "the argument of the Boris eigenvalue no longer reproduces the phase " \
        "advance of its own run; without that the disagreement on the learned " \
        "maps could be an artefact of the frequency measurement"

    # --------------------------------------------------------------- verdict
    n_bounded = sum(1 for c in cells if c["blowup_step"] is None
                    and c["amplitude_running_max"] < 10.0)
    grew = [c["cell"] for c in cells if c["blowup_step"] is not None
            or c["amplitude_running_max"] >= 10.0]

    # Ranking, as opposed to rate.  A predictor that cannot give the growth
    # rate may still say which cells will grow, and that is a separate and
    # weaker claim worth scoring separately: where in the ordering of all
    # twelve cells, by each predictor, do the cells that actually grew fall?
    ranking = {}
    for name in ("P1_log_median_rho_on_orbit", "P2_log_max_rho_all_points",
                 "P3_lyapunov_max_per_step", "P4_lyapunov_extrapolated"):
        order = sorted(cells, key=lambda c: -c[name])
        ranks = [i + 1 for i, c in enumerate(order) if c["cell"] in grew]
        ranking[name] = {
            "ranks_of_the_cells_that_grew": ranks,
            "best_possible": list(range(1, len(grew) + 1)),
            "ordering": [c["cell"] for c in order],
        }
    out["ranking"] = ranking
    out["ranking"]["note"] = (
        "Two of the twelve cells grew.  A predictor that put them first and "
        "second would have ranked perfectly; the probability of that by chance "
        "is 1/66. The leading Lyapunov exponent does it; the pointwise "
        "spectral radius does not.")

    out["verdict"] = {
        "n_cells": len(cells),
        "n_bounded_over_1e5_steps": n_bounded,
        "n_blown_up": len(blow),
        "cells_that_grew": grew,
        "sympnet_max_rho_over_all_points": float(np.exp(max(
            c["P2_log_max_rho_all_points"] for c in cells
            if c["arch"] == "sympnet"))),
        "sympnet_min_lambda_max_per_step": float(min(
            c["P3_lyapunov_max_per_step"] for c in cells
            if c["arch"] == "sympnet")),
        "sympnet_min_lambda_extrapolated": float(min(
            c["P4_lyapunov_extrapolated"] for c in cells
            if c["arch"] == "sympnet")),
        "sympnet_bounded": bool(all(
            c["blowup_step"] is None and c["amplitude_running_max"] < 10.0
            for c in cells if c["arch"] == "sympnet")),
        "note": "Every SympNet cell has a spectral radius above one at almost "
                "every point and a positive leading Lyapunov exponent, and "
                "every SympNet run stays bounded for 10^5 steps.  The "
                "inference that carried in Wave 10 -- eigenvalue off the "
                "circle, therefore exponential blow-up -- carried there "
                "because SympMat is linear, and does not carry to a nonlinear "
                "symplectic architecture.",
    }

    return S.check_or_write(OUT, out, rtol=1e-6, atol=1e-12, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
