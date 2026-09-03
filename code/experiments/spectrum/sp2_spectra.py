"""SP2: the spectrum of the three Wave 8 architectures, seed by seed.

Writes sp2_spectra.json.  Rerunning recomputes and exits non-zero on any
disagreement with the committed file.

    python sp2_spectra.py [--force]

WHAT IS MEASURED, AND ON WHAT
-----------------------------
The twelve checkpoints of Wave 8 -- HNN, G-SympNet and PINN-symplectic, four
seeds each -- are loaded frozen from experiments/external_arch/ckpt and are
never retrained.  For each of them:

  the one-step Jacobian in canonical coordinates at 64 points: a 3 x 8 grid of
  gyroradii and phases, 32 random points in a ball, and 8 points along the run
  the map itself produces.  A nonlinear map has no single spectrum, so the
  point dependence is the measurement, not a nuisance to be averaged away.

  from each Jacobian: |lambda| of all four eigenvalues, the argument of the
  gyration pair, ||J^T Omega J - Omega||_F, |det J - 1|, and the reciprocity
  residual max_i |a_i a_{3-i} - 1| of the sorted moduli, which is zero for a
  symplectic matrix and is an independent measure of how far from symplectic
  the map is.

  the finite-time Lyapunov spectrum over 4000 steps along the orbit, which is
  the quantity the pointwise spectral radius is usually mistaken for.

TWO FIELDS
----------
Everything is done twice: in the frozen field tau = infinity, where the map is
autonomous and its spectrum is a well-defined time-independent object, and in
the decaying field tau = 1.2e5 of Wave 9, which is the setting in which those
checkpoints were trained and scored.  The two agree to about 1e-5 in
max|lambda|, and the difference is reported rather than assumed away.

CROSS-CHECK AGAINST WAVE 9
--------------------------
Section 5.2 of the Wave 9 report gives the symplecticity residual of the seed-0
checkpoints at tau = 1.2e5, measured by `ea3_cost.symplecticity` with the same
stencil at eight states along the run: 5.060e-05 for the HNN, 4.426e-10 for the
SympNet, 0.2188 for the PINN and 0.1315 for the Boris scheme.  Those four
numbers are recomputed here through this file's own machinery and asserted
against the committed ea3_cost.json.  If the assertion fails, this directory is
measuring something other than what Wave 9 measured and nothing below is
comparable with it.
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

OUT = os.path.join(HERE, "sp2_spectra.json")
EA3 = os.path.normpath(os.path.join(HERE, "..", "external_arch", "ea3_cost.json"))

FIELDS = (("frozen", S.TAU_FROZEN), ("decaying", S.TAU_DECAY))


def _stats(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max())}


def _converged(short, long_):
    """Is the leading finite-time exponent an exponent, or a bounded transient?

    A real exponent is constant in the run length: lambda(n1) / lambda(n2) = 1.
    A bounded transient of size C contributes log(C)/n, so its ratio is
    n2 / n1 = 4 here.  `verdict` reports which of the two the cell is closer to
    on a logarithmic scale; nothing downstream branches on it, it is printed so
    that no exponent in this report is quoted without it.

    Every cell here, the Boris control included, comes out nearer the transient
    end than the exponent end, which means none of these exponents may be
    quoted as measured.  What can be quoted is the extrapolation.  Writing
    lambda(n) = lambda_inf + C/n and eliminating C between the two run lengths
    with n2 = 4 n1 gives

        lambda_inf = (4 lambda(n2) - lambda(n1)) / 3 ,

    which is Richardson's, at first order in 1/n.  It returns -6.2e-5 for the
    Boris scheme, whose true value is exactly zero, and that residue is the
    accuracy of the extrapolation and the number every extrapolated exponent
    below is to be read against.
    """
    if not short.get("n_steps") or not long_.get("n_steps"):
        return {"verdict": "not measured"}
    a = short["lambda_max_per_step"]
    b = long_["lambda_max_per_step"]
    ratio = float(a / b) if b != 0 else float("inf")
    transient = float(long_["n_steps"]) / float(short["n_steps"])
    verdict = "transient" if abs(np.log(max(ratio, 1e-12)) - np.log(transient)) \
        < abs(np.log(max(ratio, 1e-12))) else "exponent"
    extrap = (transient * b - a) / (transient - 1.0)
    return {"lambda_short": a, "lambda_long": b, "ratio_short_over_long": ratio,
            "ratio_if_pure_transient": transient,
            "ratio_if_real_exponent": 1.0, "verdict": verdict,
            "lambda_extrapolated": float(extrap)}


def _sweep(stepper, tau, points):
    """spec() at every point of a list of (label, time, canonical state).

    The time is carried with the point because in the decaying field the map
    depends on it; the grid and the cloud are linearised at t = 0 and the orbit
    points at the time the run reaches them, which is the convention
    `ea3_cost.symplecticity` uses and the reason the cross-check above agrees
    digit for digit.
    """
    recs = []
    for label, t, w in points:
        J, _ = S.jacobian(stepper, w, t, tau)
        if not np.all(np.isfinite(J)):
            recs.append({"point": label, "finite": False})
            continue
        d = S.spec(J)
        d["point"] = label
        d["finite"] = True
        recs.append(d)
    return recs


def _points_for(stepper, tau, rng):
    pts = []
    for rho, phi, w in S.grid_points():
        pts.append(("grid_rho%.1f_phi%.3f" % (rho, phi), 0.0, w))
    for i, w in enumerate(S.cloud_points(rng)):
        pts.append(("cloud%02d" % i, 0.0, w))
    for i, (t, w) in enumerate(S.orbit_points(stepper, tau)):
        pts.append(("orbit%d_t%.2f" % (i, t), t, w))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = {"setup": {
        "checkpoints": "experiments/external_arch/ckpt, frozen, never retrained",
        "n_reps": S.N_REP,
        "dt": S.DT,
        "fd": S.FD,
        "grid": "%d gyroradii x %d phases" % (len(S.RHO_GRID), S.N_PHASE),
        "rho_grid": list(S.RHO_GRID),
        "n_cloud": S.N_CLOUD,
        "cloud_radius": S.CLOUD_RADIUS,
        "n_orbit_points": 8,
        "n_lyapunov_steps": S.N_LYAP,
        "seed_block": S.SEED_BLOCK,
        "seed_rule": "13_000_000 + 1_000 * arch_slot + rep; the only draw in "
                     "this directory is the random cloud of linearisation "
                     "points; the generator is built once, outside every loop",
    }, "wave9_cross_check": {}, "cells": {}, "by_architecture": {},
        "structure": {}}

    # ------------------------------------------- cross-check against Wave 9
    ea3 = json.load(open(EA3, encoding="utf-8"))
    ref = ea3["symplecticity"]
    chk = {}
    bs = A.BorisStepper(S.TAU_DECAY)
    for name, st in [("boris", bs)] + [
            (arch, S.load(arch, 0, S.TAU_DECAY)) for arch in S.ARCHS]:
        pts = [("orbit%d" % i, t, w)
               for i, (t, w) in enumerate(S.orbit_points(st, S.TAU_DECAY,
                                                         n_pts=8, stride=50))]
        recs = _sweep(st, S.TAU_DECAY, pts)
        med = float(np.median([r["symplectic_defect"] for r in recs]))
        mx = float(np.max([r["symplectic_defect"] for r in recs]))
        chk[name] = {"here_median": med, "here_max": mx,
                     "wave9_median": ref[name]["residual_median"],
                     "wave9_max": ref[name]["residual_max"],
                     "ratio": float(med / ref[name]["residual_median"]),
                     "at_finite_difference_floor": bool(med < 1e-8)}
    out["wave9_cross_check"] = chk
    out["wave9_cross_check"]["note"] = (
        "ea3_cost.symplecticity samples eight states with stride 50 along the "
        "run at tau = 1.2e5; the same eight states are re-differenced here by "
        "this file's own Jacobian.  Boris, HNN and PINN agree to every digit "
        "printed in Section 5.2 of the Wave 9 report.  The SympNet row does "
        "not, and cannot: at 3e-10 it is the resolution of the stencil rather "
        "than a property of the map, and the two directories batch the eight "
        "legs of the difference differently, which moves a rounding artefact "
        "by a factor of 1.4.  That row is asserted to be at the floor, not to "
        "agree.")
    for name, v in chk.items():
        if name == "note":
            continue
        if v["at_finite_difference_floor"]:
            # A residual of a few times 1e-10 is the resolution of a central
            # difference at fd = 1e-6 in double precision, not a measurement of
            # the map.  Wave 9 says so in as many words about this row, and the
            # only thing that can be asserted about it is that it is still at
            # the floor.  Requiring the two directories to agree to a few per
            # cent on a rounding artefact would make the gate fail on a numpy
            # release rather than on a change to the science.
            assert v["wave9_median"] < 1e-8, \
                "%s used to be above the finite-difference floor" % name
            continue
        assert 0.99 < v["ratio"] < 1.01, \
            "the symplecticity residual for %s no longer reproduces Wave 9 " \
            "Section 5.2 (%g vs %g)" % (name, v["here_median"],
                                        v["wave9_median"])

    # ------------------------------------------------------------ the sweep
    for arch in S.ARCHS:
        for rep in range(S.N_REP):
            for fname, tau in FIELDS:
                st = S.load(arch, rep, tau)
                rng = np.random.default_rng(S.sp_seed(arch, rep))
                pts = _points_for(st, tau, rng)
                recs = _sweep(st, tau, pts)
                ok = [r for r in recs if r["finite"]]
                cell = {
                    "arch": arch, "rep": rep, "field": fname, "tau": None
                    if not np.isfinite(tau) else float(tau),
                    "seed": S.sp_seed(arch, rep),
                    "n_points": len(recs), "n_finite": len(ok),
                    "max_abs": _stats([r["max_abs"] for r in ok]),
                    "min_abs": _stats([r["min_abs"] for r in ok]),
                    "symplectic_defect": _stats(
                        [r["symplectic_defect"] for r in ok]),
                    "det_minus_one": _stats([r["det_minus_one"] for r in ok]),
                    "reciprocity": _stats([r["reciprocity"] for r in ok]),
                    "rotation_arg": _stats([r["rotation_arg"] for r in ok]),
                    "n_points_with_max_abs_below_1":
                        int(sum(1 for r in ok if r["max_abs"] < 1.0 - 1e-9)),
                    "per_point": recs,
                }
                if fname == "frozen":
                    w0 = [1.0, 0.0, 0.0, 0.5]
                    cell["lyapunov"] = S.lyapunov(st, w0, tau)
                    cell["lyapunov_short"] = S.lyapunov(st, w0, tau,
                                                        n=S.N_LYAP_SHORT)
                    cell["lyapunov_convergence"] = _converged(
                        cell["lyapunov_short"], cell["lyapunov"])
                out["cells"]["%s_r%d_%s" % (arch, rep, fname)] = cell

    # ------------------------------------- the classical control, same machinery
    # The Boris scheme in the frozen field is a constant matrix with its whole
    # spectrum on the unit circle, so its Lyapunov exponents must all be zero.
    # Measuring them through the same QR run is the calibration of that run: a
    # procedure that returned a positive exponent here would be measuring its
    # own arithmetic.  The exact flow map is added beside it for the same
    # reason, as the object every scheme is trying to be.
    bsf = A.BorisStepper(S.TAU_FROZEN)
    w0 = [1.0, 0.0, 0.0, 0.5]
    b_long = S.lyapunov(bsf, w0, S.TAU_FROZEN)
    b_short = S.lyapunov(bsf, w0, S.TAU_FROZEN, n=S.N_LYAP_SHORT)
    Mb = S.canonicalise_matrix(S.boris_matrix(S.DT))
    out["classical_control"] = {
        "boris_frozen": {
            "spectrum": S.spec(Mb),
            "lyapunov": b_long, "lyapunov_short": b_short,
            "lyapunov_convergence": _converged(b_short, b_long),
            "norm_of_matrix_power": {
                str(n): float(np.linalg.norm(np.linalg.matrix_power(Mb, n), 2))
                for n in (1, 10, 100, 1000, 4000)},
        },
        "exact_flow": {
            "spectrum": S.spec(S.canonicalise_matrix(S.exact_matrix(S.DT))),
            "lyapunov_analytic_per_step": [0.0, 0.0, 0.0, 0.0],
        },
        "note": "The true dynamics of the frozen field is an integrable "
                "rotation: every Lyapunov exponent is zero, and so are the "
                "Boris scheme's, since its one-step matrix is constant, "
                "diagonalisable and has its whole spectrum on the unit circle "
                "-- ||M^n|| merely wanders between 1.12 and 2.01 for ever.  "
                "The QR run nevertheless returns 1.2e-4 per step for it at "
                "n = 4000, because a finite-time exponent divides a bounded "
                "transient by n.  That is the floor of the estimator, and it "
                "is what every learned exponent below has to be read against: "
                "at 1.2e-3 the SympNet is ten times the floor, and its "
                "lambda(1000)/lambda(4000) is near one where the Boris row's "
                "is near four.",
    }
    assert out["classical_control"]["boris_frozen"]["lyapunov"][
        "lambda_max_per_step"] < 3e-4, \
        "the Boris floor of the Lyapunov estimator moved; every exponent in " \
        "this file is quoted against it"
    assert out["classical_control"]["boris_frozen"]["lyapunov_convergence"][
        "verdict"] == "transient", \
        "the Boris exponent stopped behaving like a bounded transient, so " \
        "the convergence test no longer separates the floor from a real " \
        "exponent"
    assert abs(out["classical_control"]["boris_frozen"]
               ["lyapunov_convergence"]["lambda_extrapolated"]) < 1e-4, \
        "the extrapolated Boris exponent, whose true value is exactly zero, " \
        "is no longer near zero; the extrapolation cannot be trusted on the " \
        "learned maps either"

    # ------------------------------------------- per architecture, and the test
    for arch in S.ARCHS:
        rows = [out["cells"]["%s_r%d_frozen" % (arch, r)] for r in range(S.N_REP)]
        dec = [out["cells"]["%s_r%d_decaying" % (arch, r)] for r in range(S.N_REP)]
        out["by_architecture"][arch] = {
            "max_abs_over_all_points": {
                "min": float(min(r["max_abs"]["min"] for r in rows)),
                "median_of_medians": float(np.median(
                    [r["max_abs"]["median"] for r in rows])),
                "max": float(max(r["max_abs"]["max"] for r in rows))},
            "points_below_unit_circle": int(sum(
                r["n_points_with_max_abs_below_1"] for r in rows)),
            "symplectic_defect_median_of_medians": float(np.median(
                [r["symplectic_defect"]["median"] for r in rows])),
            "reciprocity_median_of_medians": float(np.median(
                [r["reciprocity"]["median"] for r in rows])),
            "lyapunov_max_per_step": [r["lyapunov"]["lambda_max_per_step"]
                                      for r in rows],
            "lyapunov_sum_per_step": [r["lyapunov"]["sum_per_step"]
                                      for r in rows],
            "lyapunov_pairing_residual": [r["lyapunov"]["pairing_residual"]
                                          for r in rows],
            "lyapunov_short_over_long": [
                r["lyapunov_convergence"]["ratio_short_over_long"]
                for r in rows],
            "lyapunov_verdict": [r["lyapunov_convergence"]["verdict"]
                                 for r in rows],
            "lyapunov_max_extrapolated": [
                r["lyapunov_convergence"]["lambda_extrapolated"]
                for r in rows],
            "frozen_vs_decaying_max_abs_shift": float(max(
                abs(a["max_abs"]["median"] - b["max_abs"]["median"])
                for a, b in zip(rows, dec))),
        }

    out["structure"] = {
        "weak_claim": "det J = 1 means the four moduli multiply to one, so "
                      "rho(J) >= 1 for any volume-preserving map, symplectic "
                      "or not.  This is what forbids contraction, and the "
                      "Boris scheme has it without being symplectic.",
        "strong_claim": "a real symplectic matrix has spectrum closed under "
                        "lambda -> 1/lambda, so the moduli come in reciprocal "
                        "pairs: an eigenvalue can only leave the unit circle "
                        "together with its reciprocal, radially.  Volume "
                        "preservation alone does not give that.",
        "min_rho_over_every_point_and_seed": {
            arch: out["by_architecture"][arch]["max_abs_over_all_points"]["min"]
            for arch in S.ARCHS},
        "points_strictly_inside_the_unit_circle": {
            arch: out["by_architecture"][arch]["points_below_unit_circle"]
            for arch in S.ARCHS},
        "det_minus_one_median_of_medians": {
            arch: float(np.median([
                out["cells"]["%s_r%d_frozen" % (arch, r)]["det_minus_one"]
                ["median"] for r in range(S.N_REP)])) for arch in S.ARCHS},
        "note": "The SympNet is symplectic to 1e-10 and never once returns "
                "rho < 1 in 256 measurements.  The PINN neither preserves "
                "volume (|det J - 1| = 0.12) nor is symplectic, and returns "
                "rho < 1 at 76 of its 256 points: it contracts, which neither "
                "claim above permits.  The HNN sits between, because RK4 on a "
                "Hamiltonian field preserves volume only to O(h^5): six of its "
                "256 points fall inside the circle, the deepest at 1 - 5.3e-6, "
                "the same order as the 5.0e-6 by which sp1_calibration.py "
                "shows RK4 damps an *exact* Hamiltonian field at Omega h = 0.3 "
                "in closed form.  That much of the contraction is the "
                "integrator's and not the learned field's.",
    }

    return S.check_or_write(OUT, out, rtol=1e-6, atol=1e-12, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
