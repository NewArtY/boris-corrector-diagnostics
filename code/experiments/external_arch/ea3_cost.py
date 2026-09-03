"""EA3: the three external architectures against the classical family, on flops.

Writes ea3_cost.json.  Rerunning compares against the committed file and exits
non-zero on any disagreement.

    python ea3_cost.py [--force]

WHAT IS COMPARED AND ON WHAT AXIS
---------------------------------
The run is the one of Section 7: the decaying field at tau = 1.2e5,
Omega h = 0.3, t = 120, which is 19.1 gyro-orbits, scored against DOP853 at
rtol 1e-12 and atol 1e-14 on the same grid.  The trajectory error is the root
mean square of the position error in Larmor radii and the energy error is the
median relative error over the second half.

Cost is the flop count, not the elapsed time.  Section 7 gives the reason and
this directory reproduces it: a Boris step is 113 flops and takes 43.9 us in
this codebase, while a corrector step is 114,091 flops and takes 142.1 us, so
elapsed time understates the arithmetic by a factor near a thousand.  Elapsed
time is measured here as well, and printed beside the flop count, because the
gap between the two is the point.

CALIBRATION
-----------
Two checks run before any comparison, and both must hold or the script exits
non-zero.

  1. The flop model reproduces the 113,958 flops that Section 9 prints for one
     forward pass of the learned corrector.
  2. The planar Boris row computed here reproduces the three-dimensional row in
     experiments/classical/verdict.json, which is the row Table 1 prints:
     0.4167 Larmor radii and 1.250e-6.  The motion is planar, so the two must
     agree; if they stop agreeing, the planar reduction is wrong and every
     number below is wrong with it.

The classical rows are read from experiments/classical/verdict.json rather than
recomputed, so that the comparison sits on exactly the numbers Table 1 prints.

STEP REFINEMENT
---------------
Of the three architectures only the HNN can be refined.  It learns a vector
field and the step size belongs to the integrator wrapped around it, so RK4 at
h/2 is a meaningful question to ask.  A SympNet and this PINN learn a map at
one step size and have no h to refine; halving h would mean applying a map
trained for 0.3 twice, which is a different scheme.  The refinement sweep is
therefore run for the HNN alone, and what it measures is the floor set by the
error of the learned field rather than by the integrator.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ea_common as C          # noqa: E402
import ea_arch as A            # noqa: E402
import ea1_train as T          # noqa: E402

OUT = os.path.join(HERE, "ea3_cost.json")
CKPT = os.path.join(HERE, "ckpt")
CLASSICAL = os.path.normpath(os.path.join(HERE, "..", "classical", "verdict.json"))
TRAIN_JSON = os.path.join(HERE, "ea1_training.json")

REFINE_H = (0.3, 0.15, 0.075, 0.0375, 0.01875)


OMEGA4 = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                   [-1., 0., 0., 0.], [0., -1., 0., 0.]])


def symplecticity(stepper, tau, dt=C.DT, n_states=8, fd=1e-6):
    """How far the one-step map is from symplectic, measured rather than argued.

    The map is differenced in canonical coordinates (x, y, p_x, p_y) by central
    differences at states taken along the run, and the residual is
    ||J^T Omega J - Omega||_F together with |det J - 1|.  A central difference
    at fd = 1e-6 in double precision resolves the residual down to about
    1e-9, which separates a construction that is symplectic at any weights from
    one that is penalised toward symplecticity during training.
    """
    x = np.array([C.R0[0]]); y = np.array([C.R0[1]])
    vx = np.array([C.V0[0]]); vy = np.array([C.V0[1]])
    t = 0.0
    res, dets = [], []
    stride = max(1, int(round(C.T_FINAL / dt)) // n_states)
    for i in range(n_states * stride):
        if i % stride == 0 and np.isfinite(vx[0]):
            ax, ay = C.vecpot(x[0], y[0], t, tau)
            w0 = np.array([x[0], y[0], vx[0] - ax, vy[0] - ay])
            J = np.empty((4, 4))
            for k in range(4):
                col = []
                for s in (+1, -1):
                    w = w0.copy()
                    w[k] += s * fd
                    a1, a2 = C.vecpot(w[0], w[1], t, tau)
                    nx, ny, nvx, nvy = stepper.step(
                        np.array([w[0]]), np.array([w[1]]),
                        np.array([w[2] + a1]), np.array([w[3] + a2]), t, dt)
                    b1, b2 = C.vecpot(nx[0], ny[0], t + dt, tau)
                    col.append(np.array([nx[0], ny[0], nvx[0] - b1,
                                         nvy[0] - b2]))
                J[:, k] = (col[0] - col[1]) / (2 * fd)
            R = J.T @ OMEGA4 @ J - OMEGA4
            res.append(float(np.sqrt(np.sum(R ** 2))))
            dets.append(float(abs(np.linalg.det(J) - 1.0)))
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        if not np.isfinite(vx[0]):
            break
    if not res:
        return {"n_states": 0}
    return {"n_states": len(res), "fd_step": fd,
            "residual_median": float(np.median(res)),
            "residual_max": float(np.max(res)),
            "det_minus_one_median": float(np.median(dets)),
            "det_minus_one_max": float(np.max(dets))}


def wall_time(stepper, n=400, dt=C.DT, repeats=3):
    best = np.inf
    for _ in range(repeats):
        x = np.array([C.R0[0]]); y = np.array([C.R0[1]])
        vx = np.array([C.V0[0]]); vy = np.array([C.V0[1]])
        t = 0.0
        t0 = time.perf_counter()
        for _i in range(n):
            x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
            t += dt
        best = min(best, (time.perf_counter() - t0) / n)
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    assert C.mlp_forward_flops([13, 128, 128, 128, 128, 6]) == 113958, \
        "the flop model no longer reproduces the 113,958 of Section 9"

    cl = json.load(open(CLASSICAL, encoding="utf-8"))
    train = json.load(open(TRAIN_JSON, encoding="utf-8"))

    out = {"setup": {
        "run": "decaying field, tau = 1.2e5, Omega h = 0.3, t = 120 "
               "(19.1 gyro-orbits), DOP853 rtol 1e-12 atol 1e-14 on the grid",
        "physical_signal": cl["physical_signal"],
        "cost": "flops, one per arithmetic operation and twenty per "
                "transcendental (Section 9)",
        "flop_model_calibration_corrector_forward": 113958,
    }, "calibration": {}, "rows": {}, "against_classical": {},
        "symplecticity": {}, "hnn_step_refinement": {}}

    # ---- calibration: the planar Boris row against the row Table 1 prints
    bs = A.BorisStepper(C.TAU_PAPER)
    b_here = T.score_section7(bs)
    b_there = cl["schemes"]["shipped"]
    out["calibration"] = {
        "boris_planar_traj": b_here["pos_err_rms"],
        "boris_3d_traj_table1": b_there["traj"],
        "traj_relative_difference":
            abs(b_here["pos_err_rms"] - b_there["traj"]) / b_there["traj"],
        "boris_planar_energy": b_here["energy_err_median_2nd_half"],
        "boris_3d_energy_table1": b_there["energy"],
        "energy_relative_difference":
            abs(b_here["energy_err_median_2nd_half"] - b_there["energy"])
            / b_there["energy"],
        "boris_wall_s_per_step": wall_time(bs),
    }
    assert out["calibration"]["traj_relative_difference"] < 1e-9, \
        "the planar reduction no longer reproduces the Boris trajectory row"
    assert out["calibration"]["energy_relative_difference"] < 1e-5, \
        "the planar reduction no longer reproduces the Boris energy row"

    # ---- the classical rows, as Table 1 prints them
    for k, v in cl["schemes"].items():
        out["rows"][k] = {"traj": v["traj"], "energy": v["energy"],
                          "flops_run": v["flops"], "wall_s_run": v["wall_s"],
                          "source": "experiments/classical/verdict.json"}

    # ---- the three external architectures, four repetitions each
    n_steps = int(round(C.T_FINAL / C.DT))
    for arch in ("hnn", "sympnet", "pinn"):
        rows = []
        for rep in range(T.N_REP):
            p = os.path.join(CKPT, "%s_r%d.npz" % (arch, rep))
            if not os.path.exists(p):
                continue
            st = T.load_stepper(p, C.TAU_PAPER)
            sc = T.score_section7(st)
            sc["rep"] = rep
            sc["wall_s_per_step"] = wall_time(st)
            sc["wall_s_run"] = sc["wall_s_per_step"] * n_steps
            sc["n_parameters"] = train["runs"]["%s/rep%d" % (arch, rep)][
                "n_parameters"]
            rows.append(sc)
        if not rows:
            continue
        traj = np.array([r["pos_err_rms"] for r in rows])
        ener = np.array([r["energy_err_median_2nd_half"] for r in rows])
        # the row printed in the table is one seed, repetition 0, which is the
        # seed the probes are run on and was fixed before the runs.  The
        # comparisons below use the seed with the smallest trajectory error,
        # which is the reading most favourable to the architecture: a negative
        # result stated there is a negative result at the best of four.
        best = int(np.argmin(traj))
        out["rows"][arch] = {
            "traj": float(traj[0]),
            "energy": float(ener[0]),
            "row_is": "repetition 0, the seed the probes are run on",
            "traj_median_over_reps": float(np.median(traj)),
            "traj_best_of_reps": float(traj[best]),
            "traj_reps": [float(v) for v in traj],
            "traj_spread_factor": float(traj.max() / traj.min()),
            "energy_median_over_reps": float(np.median(ener)),
            "energy_best_of_reps": float(np.min(ener)),
            "energy_of_best_traj_rep": float(ener[best]),
            "energy_reps": [float(v) for v in ener],
            "energy_spread_factor": float(ener.max() / ener.min()),
            "best_traj_rep": best,
            "flops_per_step": rows[0]["flops_per_step"],
            "flops_run": rows[0]["flops_run"],
            "wall_s_per_step": float(np.median([r["wall_s_per_step"] for r in rows])),
            "wall_s_run": float(np.median([r["wall_s_run"] for r in rows])),
            "n_parameters": rows[0]["n_parameters"],
            "n_reps": len(rows),
            "reporting_rule": "median over the four repetitions; the best "
                              "repetition is given beside it, and the "
                              "comparisons below use the best one, which is the "
                              "reading most favourable to the architecture",
            "source": "ea1_training.json + this script"}

    # ---- the comparison Section 7 makes for the corrector, made for each
    hyb = cl["schemes"]["hybrid"]
    for arch in ("hnn", "sympnet", "pinn"):
        if arch not in out["rows"]:
            continue
        r = out["rows"][arch]
        tb, eb = r["traj_best_of_reps"], r["energy_of_best_traj_rep"]
        row = {"reading": "the repetition with the smallest trajectory error, "
                          "rep %d; ratios above 1 mean the classical scheme is "
                          "the more accurate or the cheaper of the two"
                          % r["best_traj_rep"]}
        for opp in ("vps4", "vps2", "gl4", "imr", "shipped"):
            o = cl["schemes"][opp]
            row[opp] = {
                "classical_more_accurate_in_traj_by": float(tb / o["traj"]),
                "classical_more_accurate_in_energy_by": float(eb / o["energy"]),
                "classical_cheaper_in_flops_by": float(r["flops_run"] / o["flops"]),
                "classical_cheaper_in_wall_by": float(r["wall_s_run"] / o["wall_s"])}
        row["learned_corrector"] = {
            "corrector_more_accurate_in_traj_by": float(tb / hyb["traj"]),
            "corrector_more_accurate_in_energy_by": float(eb / hyb["energy"]),
            "corrector_dearer_in_flops_by": float(hyb["flops"] / r["flops_run"])}
        row["below_physical_signal"] = bool(eb < cl["physical_signal"])
        row["signal_over_error"] = float(cl["physical_signal"] / eb)
        row["signal_over_error_rep0"] = float(cl["physical_signal"] / r["energy"])
        out["against_classical"][arch] = row

    # ---- how symplectic each map actually is
    out["symplecticity"]["boris"] = symplecticity(bs, C.TAU_PAPER)
    for arch in ("hnn", "sympnet", "pinn"):
        p = os.path.join(CKPT, "%s_r0.npz" % arch)
        if os.path.exists(p):
            out["symplecticity"][arch] = symplecticity(
                T.load_stepper(p, C.TAU_PAPER), C.TAU_PAPER)
    out["symplecticity"]["note"] = (
        "Central differences at 1e-6 in double precision resolve the residual "
        "to about 1e-9.  The G-SympNet is a composition of shears and is "
        "symplectic at any weights; the PINN is penalised toward symplecticity "
        "during training and is not; RK4 on a learned Hamiltonian field is not "
        "symplectic either, and its residual is the RK4 residual rather than a "
        "property of the learned field.")

    # ---- step refinement, for the one architecture that admits it
    p = os.path.join(CKPT, "hnn_r0.npz")
    if os.path.exists(p):
        for h in REFINE_H:
            st = T.load_stepper(p, C.TAU_PAPER)
            sc = T.score_section7(st, dt=h)
            sc["flops_run"] = int(st.flops_per_step() * round(C.T_FINAL / h))
            out["hnn_step_refinement"]["h=%g" % h] = sc
        ks = list(out["hnn_step_refinement"])
        tr = [out["hnn_step_refinement"][k]["pos_err_rms"] for k in ks]
        out["hnn_step_refinement"]["note"] = (
            "RK4 on the learned field.  A fourth-order integrator on an exact "
            "field would divide the trajectory error by 16 at each halving; "
            "what is measured is the floor of the learned field.")
        out["hnn_step_refinement"]["ratios_between_successive_h"] = \
            [float(tr[i] / tr[i + 1]) for i in range(len(tr) - 1)]

    return C.check_or_write(OUT, out, rtol=1e-4, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
