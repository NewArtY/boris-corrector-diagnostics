"""Validation of the two reference objects, before a single weight is trained.

Nothing about the reproduction of Drimalas et al. means anything if the
analytical solution they measure against, or the Boris pusher they compare to,
is wrong in our hands.  This script checks both against statements that were
published by somebody else.

The analytical transfer matrix, their Eq. (5), is checked five ways: it is
symplectic, it integrates the canonical equations of motion to DOP853's
tolerance, it forms a one-parameter group, it commutes with a simultaneous
rotation of positions and momenta (which is the symmetry their data
augmentation exploits), and its spectrum is exp(+-i theta), 1, 1 as their
Sec. III.A states.

The Boris variants are checked against Chin and Cator's closed-form errors in a
constant field, J. Comput. Phys. 466, 111422 (2022), Sects. III and IV:

    B2B          gyrocentre exact, gyroradius exact               (on-orbit)
    BLF stored   gyrocentre exact, gyroradius r sqrt(1 + theta^2/4)
    B1A          gyrocentre displaced by -(h/2) v_0, same radius
    B1B          gyrocentre displaced by +(h/2) v_0, same radius

The displacement +- ||v|| h / 2 is the same quantity Section 3.3 of the
manuscript locates by its two-handle scan.  Reproducing it here to twelve
digits is what licenses this directory to speak about which Boris was run.

Writes sm0_analytic.json; exits non-zero if a rerun no longer reproduces it.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_common as S                                        # noqa: E402


def _canonical_rhs(b):
    """Hamilton's equations for their Eq. (3) in the normalisation of Sec. III.C."""
    def rhs(_t, z):
        x, y, px, py = z
        vx = px + 0.5 * b * y
        vy = py - 0.5 * b * x
        return [vx, vy, 0.5 * b * vy, -0.5 * b * vx]
    return rhs


def _fit_circle(P):
    """Algebraic circle fit; exact for exactly circular data."""
    A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
    cx, cy, c = sol
    return np.array([cx, cy]), float(np.sqrt(c + cx * cx + cy * cy))


def main(force=False):
    out = {"declared": {"dt_ladder": [float(v) for v in S.DT_LADDER],
                        "b_eval": list(S.B_EVAL),
                        "n_ensemble": S.N_ENSEMBLE,
                        "n_seeds": S.N_SEEDS,
                        "n_pairs_per_b": S.N_PAIRS_PER_B,
                        "tf_main": S.TF_MAIN}}

    # ---- 1. symplecticity of the analytical matrix over a (b, tau) grid ----
    bs = np.linspace(0.5, 2.5, 9)
    taus = np.array([0.0078125, 0.125, 2.0, 8.0, 37.0])
    d = max(S.sympl_defect(S.analytic_M(b, t)) for b in bs for t in taus)
    out["max_symplectic_defect"] = d

    # ---- 2. against DOP853 on the canonical equations ----------------------
    rng = np.random.default_rng(S.seed_of("ensemble", 0, 7))
    worst = 0.0
    for b in (0.5, 1.0, 2.5):
        z0 = rng.uniform(-1.0, 1.0, size=4)
        for tf in (0.5, 2.0, 8.0):
            sol = solve_ivp(_canonical_rhs(b), (0.0, tf), z0, method="DOP853",
                            rtol=1e-13, atol=1e-15, t_eval=[tf])
            assert sol.success
            worst = max(worst, float(np.max(np.abs(
                S.analytic_M(b, tf) @ z0 - sol.y[:, -1]))))
    out["max_vs_dop853"] = worst

    # ---- 3. group property, 4. rotational symmetry, 5. spectrum ------------
    g = 0.0
    for b in bs:
        for t1, t2 in ((0.3, 1.7), (2.0, 2.0), (0.01, 7.99)):
            g = max(g, float(np.max(np.abs(
                S.analytic_M(b, t1) @ S.analytic_M(b, t2)
                - S.analytic_M(b, t1 + t2)))))
    out["max_group_defect"] = g

    r = 0.0
    for phi in (0.3, 1.1, 2.9):
        c, s = np.cos(phi), np.sin(phi)
        R2 = np.array([[c, -s], [s, c]])
        R = np.block([[R2, np.zeros((2, 2))], [np.zeros((2, 2)), R2]])
        for b in bs:
            M = S.analytic_M(b, 2.0)
            r = max(r, float(np.max(np.abs(R @ M - M @ R))))
    out["max_rotation_commutator"] = r

    sp = 0.0
    for b in bs:
        for tau in (0.7, 2.0):
            ev = np.sort_complex(np.linalg.eigvals(S.analytic_M(b, tau)))
            th = b * tau
            want = np.sort_complex(np.array([np.exp(1j * th), np.exp(-1j * th),
                                             1.0 + 0j, 1.0 + 0j]))
            sp = max(sp, float(np.max(np.abs(np.sort(np.abs(ev - 1.0)) -
                                             np.sort(np.abs(want - 1.0))))))
    out["max_spectrum_defect"] = sp

    # ---- 6. Chin and Cator's closed-form errors for the Boris family -------
    fam = {}
    for b in (0.5, 1.0, 2.5):
        for h in (0.25, 1.0):
            th = b * h
            v0 = np.array([0.0, 0.83])
            r_true = np.linalg.norm(v0) / b
            # gyrocentre at the origin:  c = r0 + (1/b)(v_y, -v_x) = 0
            r0 = -np.array([v0[1], -v0[0]]) / b
            z0 = np.array([r0[0], r0[1], v0[0], v0[1]])
            mats = S.boris_step_matrices(b, h)
            key = "b%g_h%g" % (b, h)
            fam[key] = {"theta": float(th), "r_true": float(r_true),
                        "Rg_predicted": float(r_true * np.sqrt(1 + th * th / 4)),
                        "offset_predicted": float(0.5 * h * np.linalg.norm(v0))}
            for name in ("B2B", "BLF_stored", "B1A", "B1B"):
                first, rep, read = mats[name]
                P = np.array([S.rollout_matrix(first, rep, read, n) @ z0
                              for n in range(1, 400)])[:, :2]
                c, R = _fit_circle(P)
                fam[key][name] = {"radius": float(R),
                                  "centre_offset": float(np.linalg.norm(c)),
                                  "offset_along_v0": float(c @ v0 / np.linalg.norm(v0)),
                                  "offset_perp_v0": float(
                                      c @ np.array([-v0[1], v0[0]]) / np.linalg.norm(v0))}
            # assertions against the published formulas
            e = fam[key]
            assert abs(e["B2B"]["radius"] - r_true) < 1e-10 * r_true, e
            assert e["B2B"]["centre_offset"] < 1e-10, e
            assert abs(e["BLF_stored"]["radius"] - e["Rg_predicted"]) < 1e-10, e
            assert e["BLF_stored"]["centre_offset"] < 1e-10, e
            for nm, sg in (("B1A", -1.0), ("B1B", +1.0)):
                assert abs(e[nm]["radius"] - e["Rg_predicted"]) < 1e-10, e
                assert abs(e[nm]["offset_along_v0"]
                           - sg * e["offset_predicted"]) < 1e-10, (nm, e)
                assert abs(e[nm]["offset_perp_v0"]) < 1e-10, (nm, e)
    out["boris_family"] = fam

    # ---- 7. BLF centred is B2B, in a uniform field ------------------------
    dm = 0.0
    for b in (0.5, 2.5):
        for h in (0.125, 2.0):
            for n in (1, 5, 37):
                dm = max(dm, float(np.max(np.abs(
                    S.boris_total_map(b, h, n, "BLF_centred")
                    - S.boris_total_map(b, h, n, "B2B")))))
    out["blf_centred_minus_b2b"] = dm

    # ---- 8. flop model ----------------------------------------------------
    out["flops"] = {"boris_step_uniform": S.flops_boris_step_uniform(),
                    "sympmat_step_uniform": S.flops_sympmat_step_uniform(),
                    "parametric_matrix_build": S.flops_parametric_build()}

    for k in ("max_symplectic_defect", "max_vs_dop853", "max_group_defect",
              "max_rotation_commutator", "max_spectrum_defect",
              "blf_centred_minus_b2b"):
        print("%-28s %.3e" % (k, out[k]))
    print("flops/step  Boris %d   SympMat %d   rebuild %d"
          % (out["flops"]["boris_step_uniform"],
             out["flops"]["sympmat_step_uniform"],
             out["flops"]["parametric_matrix_build"]))
    return S.check_or_write(S.outpath("sm0_analytic.json"), out,
                            rtol=1e-7, atol=1e-14, force=force)


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
