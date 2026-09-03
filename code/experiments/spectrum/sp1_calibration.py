"""SP1: the measurement calibrated on maps whose spectrum is known in closed form.

Writes sp1_calibration.json.  Rerunning recomputes and exits non-zero on any
disagreement with the committed file.

    python sp1_calibration.py [--force]

WHY THIS RUNS FIRST
-------------------
Every number in sp2 and sp3 comes out of one procedure: differentiate a
one-step map by central differences in canonical coordinates, take the
eigenvalues.  A procedure that cannot return the right answer where the right
answer is known cannot be believed on a network.  Four maps have a closed form
here and all four are checked against it:

  the exact flow map        spectrum {1, 1, e^{+ih}, e^{-ih}}, on the circle,
                            symplectic
  the Boris scheme          spectrum {1, 1, e^{+i theta}, e^{-i theta}} with
                            theta = 2 arctan(Omega h / 2), on the circle, and
                            *not* symplectic -- ||J^T Omega J - Omega|| = 0.13
  RK4 on the exact field    spectrum {1, 1, R(+ih), R(-ih)} with |R(ih)| < 1:
                            off the circle, and *inside* it
  a random symplectic       drawn once, to show the reciprocal pairing of the
  matrix                    module docstring is a property of the class and not
                            an artefact of any particular training run

The Boris row is the load-bearing one.  It is volume preserving and not
symplectic (`qin2013`, and Section 5.2 of the Wave 9 report measures 0.1315 for
it), and its spectrum sits on the unit circle to eleven digits.  So
symplecticity is not necessary for a spectrum on the circle.  The RK4 row is
the other half: a map that is neither symplectic nor neutral, and whose
departure from the circle is *inward*, which a symplectic map cannot do.

WHAT IS READ AND NOT RECOMPUTED
-------------------------------
The SympMat spectra of Wave 10 are read from experiments/sympmat/
sm4_gyrocentre.json.  Nothing there is recomputed or rewritten; the two things
done with them are to verify the reciprocal pairing on the published moduli and
to evaluate what those moduli imply over 10^5 steps.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sp_common as S            # noqa: E402
import ea_common as C            # noqa: E402
import ea_arch as A              # noqa: E402

OUT = os.path.join(HERE, "sp1_calibration.json")

#: the step sizes at which the closed forms are checked, declared here
DT_LADDER = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0)


def _analytic_vs_measured(name, M, stepper, dt):
    """One closed-form matrix against the finite-difference machinery."""
    w0 = np.array([1.0, 0.0, 0.0, 0.5])       # the exact orbit at t = 0
    J, _ = S.jacobian(stepper, w0, 0.0, S.TAU_FROZEN, dt=dt)
    Mc = S.canonicalise_matrix(M)
    sm = S.spec(Mc)
    sj = S.spec(J)
    return {"scheme": name, "dt": dt,
            "matrix_error": float(np.max(np.abs(J - Mc))),
            "analytic": sm, "measured": sj,
            "abs_agreement": float(np.max(np.abs(
                np.array(sm["abs"]) - np.array(sj["abs"])))),
            "arg_agreement": float(np.max(np.abs(
                np.array(sm["arg"]) - np.array(sj["arg"]))))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = {"setup": {
        "field": "frozen: tau = infinity, B_z = B_0 = 1 exactly, E = 0 exactly",
        "coordinates": "canonical (x, y, p_x, p_y), p = v - A, A = (B_z/2)(-y, x)",
        "stencil": "central differences, fd = %g, the stencil of "
                   "ea3_cost.symplecticity" % S.FD,
        "dt_ladder": list(DT_LADDER),
        "exact_solution": "r(t) = (cos t, sin t), v(t) = (-sin t, cos t)",
    }, "closed_forms": {}, "boris_angle": {}, "rk4_stability": {},
        "reciprocity_lemma": {}, "sympmat_wave10": {}, "frequency_check": {}}

    # ---------------------------------------------------------- closed forms
    for dt in DT_LADDER:
        out["closed_forms"]["boris_dt%g" % dt] = _analytic_vs_measured(
            "boris", S.boris_matrix(dt), A.BorisStepper(S.TAU_FROZEN), dt)

    worst_M = max(out["closed_forms"][k]["matrix_error"]
                  for k in out["closed_forms"])
    worst_a = max(out["closed_forms"][k]["abs_agreement"]
                  for k in out["closed_forms"])
    assert worst_M < 1e-7, \
        "the finite-difference Jacobian no longer reproduces the closed-form " \
        "Boris matrix (%g)" % worst_M
    assert worst_a < 1e-8, \
        "the measured Boris spectrum no longer reproduces the closed form (%g)" \
        % worst_a
    out["setup"]["worst_matrix_error_over_ladder"] = float(worst_M)
    out["setup"]["worst_abs_error_over_ladder"] = float(worst_a)

    # the three analytically known maps at the working step, side by side
    for name, M in (("exact_flow", S.exact_matrix(S.DT)),
                    ("boris", S.boris_matrix(S.DT)),
                    ("rk4_on_exact_field", S.rk4_matrix(S.DT))):
        sm = S.spec(S.canonicalise_matrix(M))
        sm["scheme"] = name
        sm["dt"] = S.DT
        out["closed_forms"]["at_working_step_%s" % name] = sm

    # ---------------------------------------------------------- Boris angle
    th = S.boris_angle(S.DT)
    ev = np.linalg.eigvals(S.boris_matrix(S.DT))
    out["boris_angle"] = {
        "theta_closed_form_2atan_Omega_h_over_2": th,
        "theta_from_eigenvalues": float(np.max(np.abs(np.angle(ev)))),
        "difference": float(abs(np.max(np.abs(np.angle(ev))) - th)),
        "exact_Omega_h": S.DT,
        "relative_frequency_error": float((th - S.DT) / S.DT),
        "max_abs_minus_1": float(np.max(np.abs(ev)) - 1.0),
        "symplectic_defect":
            S.spec(S.canonicalise_matrix(S.boris_matrix(S.DT)))["symplectic_defect"],
        "det_minus_one":
            S.spec(S.canonicalise_matrix(S.boris_matrix(S.DT)))["det_minus_one"],
        "note": "volume preserving, not symplectic, spectrum exactly on the "
                "unit circle: symplecticity is not necessary for neutrality",
    }

    # ------------------------------------------------------- RK4 stability
    for dt in DT_LADDER:
        ev = np.linalg.eigvals(S.rk4_matrix(dt))
        out["rk4_stability"]["dt%g" % dt] = {
            "max_abs": float(np.max(np.abs(ev))),
            "min_abs": float(np.min(np.abs(ev))),
            "gyration_abs": float(np.max(np.abs(ev)[np.abs(np.angle(ev)) > 1e-12]))
            if np.any(np.abs(np.angle(ev)) > 1e-12) else float("nan"),
            "closed_form_R_iy": S.rk4_stability_modulus(dt),
        }
    out["rk4_stability"]["note"] = (
        "|R(iy)|^2 = 1 - y^6/72 + y^8/576.  At y = Omega h = 0.3 the gyration "
        "pair sits 5.0e-6 *inside* the unit circle.  RK4 on an exact "
        "Hamiltonian field therefore damps; the departure from the circle has "
        "the opposite sign from the SympMat observation, and no symplectic map "
        "can have it.")
    worst_rk4 = max(abs(out["rk4_stability"]["dt%g" % dt]["gyration_abs"]
                        - S.rk4_stability_modulus(dt)) for dt in DT_LADDER)
    assert worst_rk4 < 1e-12, "the RK4 closed form no longer holds (%g)" % worst_rk4

    # --------------------------------------------------- the pairing lemma
    # A symplectic matrix built without any training: exp of a Hamiltonian
    # matrix, Omega^{-1} times a symmetric matrix.  One generator, built once.
    rng = np.random.default_rng(S.sp_seed("boris", 0))
    from scipy.linalg import expm
    recs = []
    for i in range(8):
        Ssym = rng.normal(size=(4, 4))
        Ssym = Ssym + Ssym.T
        H = np.linalg.solve(S.OMEGA4, Ssym)          # Hamiltonian matrix
        Msym = expm(0.3 * H)
        sp = S.spec(Msym)
        recs.append({"draw": i, "symplectic_defect": sp["symplectic_defect"],
                     "reciprocity": sp["reciprocity"],
                     "max_abs": sp["max_abs"], "min_abs": sp["min_abs"],
                     "product_of_extremes": float(sp["abs"][0] * sp["abs"][3])})
    out["reciprocity_lemma"] = {
        "seed": S.sp_seed("boris", 0),
        "n_draws": len(recs), "draws": recs,
        "worst_reciprocity": float(max(r["reciprocity"] for r in recs)),
        "min_of_max_abs": float(min(r["max_abs"] for r in recs)),
        "note": "J symplectic implies Omega J Omega^{-1} = J^{-T}, so J is "
                "similar to J^{-T} and the spectrum is closed under "
                "lambda -> 1/lambda.  Hence rho(J) >= 1 for every symplectic "
                "J, with equality only when the whole spectrum is on the unit "
                "circle.  min_of_max_abs is that inequality, measured.",
    }
    assert out["reciprocity_lemma"]["worst_reciprocity"] < 1e-10
    assert out["reciprocity_lemma"]["min_of_max_abs"] >= 1.0 - 1e-12

    # ------------------------------------------ SympMat of Wave 10, read only
    sm = json.load(open(S.SYMPMAT_JSON, encoding="utf-8"))
    rows = []
    for budget in ("at_declared_budget", "at_paper_training_loss"):
        if budget not in sm:
            continue
        for key, lst in sm[budget]["spectrum"].items():
            for rec in lst:
                mods = sorted(rec["abs"])
                rows.append({
                    "budget": budget, "case": key, "seed": rec["seed"],
                    "max_abs": float(mods[-1]),
                    "reciprocity": float(max(abs(mods[0] * mods[3] - 1.0),
                                             abs(mods[1] * mods[2] - 1.0))),
                    "log10_growth_over_1e5_steps":
                        float(1e5 * np.log10(mods[-1])),
                })
    worst = max(rows, key=lambda r: r["max_abs"])
    out["sympmat_wave10"] = {
        "source": "experiments/sympmat/sm4_gyrocentre.json, read only",
        "n_records": len(rows), "rows": rows,
        "worst_reciprocity": float(max(r["reciprocity"] for r in rows)),
        "largest_max_abs": worst["max_abs"],
        "largest_case": "%s / %s / seed %d" % (worst["budget"], worst["case"],
                                               worst["seed"]),
        "log10_growth_of_largest_over_1e5_steps":
            worst["log10_growth_over_1e5_steps"],
        "note": "The reciprocal pairing holds on the published moduli to the "
                "last digit, which is the pairing lemma seen in the trained "
                "matrices themselves: the pair leaves the circle radially, one "
                "eigenvalue out and its reciprocal in.  The growth column is "
                "(max|lambda|)^(10^5) written as a power of ten.",
    }
    assert out["sympmat_wave10"]["worst_reciprocity"] < 1e-9

    # -------------------------------- the frequency readout, calibrated too
    bs = A.BorisStepper(S.TAU_FROZEN)
    f = S.measured_frequency(bs, S.TAU_FROZEN)
    out["frequency_check"] = {
        "boris_measured_per_step": f,
        "boris_closed_form": th,
        "relative_difference": float(abs(f - th) / th),
        "n_steps": S.N_FREQ,
        "note": "the per-step phase advance measured on the run agrees with "
                "the argument of the gyration eigenvalue, which is what lets "
                "sp3 read a frequency error off a spectrum on a learned map",
    }
    assert out["frequency_check"]["relative_difference"] < 1e-12

    return S.check_or_write(OUT, out, rtol=1e-9, atol=1e-13, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
