"""mp4_saturation.py -- why the map has the shape it has.

The grid of `mp2_grid.py` says where the learned corrector works.  This file
says why, and the answer is in the checkpoint rather than in the physics.

`training/train_corrector_b4.py` standardises the thirteen network inputs by
the mean and the standard deviation of the training set:

    model.x_std.copy_(Xt.std(0).clamp_min(1e-12))

Four of the thirteen inputs are **constant** over that training set --- B_x and
B_y (the training field is along z), E_z (the induced field is azimuthal) and
dt (one working step, 0.3).  Their standard deviation is exactly zero, the
clamp replaces it by 1e-12, and the standardised input becomes

    z = (x - x_mean) / 1e-12 ,

so a departure of 0.2 in the step size, or of half a tesla in B_x, arrives at
the first layer as 2e11.  Every hidden unit saturates, tanh returns +-1, and
the network's output stops depending on its input at all: it becomes one fixed
vector, added to the Boris step at every step of the run.

This is a property of the committed checkpoint, not an inference about learned
integrators in general, and it is measured here rather than argued:

  1. the standardisation itself, input by input, with the degenerate ones named
  2. the standardised input magnitude reached in each configuration and at each
     step size on the map's grid
  3. the fraction of first-layer units driven past |tanh| > 0.999
  4. the spread of the correction over eight initial conditions and along a
     run -- if the network has saturated, the correction is the same vector
     everywhere and that spread is zero
  5. the size of that fixed correction against the one-step Boris defect it is
     supposed to be correcting, which is what the training target was

Writes mp4_saturation.json; exits non-zero if a rerun stops reproducing it.
Usage: python mp4_saturation.py [--force]
"""
import json
import os
import sys

import numpy as np

import map_common as C
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mp4_saturation.json")

INPUTS = ["r_x", "r_y", "r_z", "v_x", "v_y", "v_z",
          "B_x", "B_y", "B_z", "E_x", "E_y", "E_z", "dt"]
CLAMP = 1e-12                     # train_corrector_b4.py: clamp_min(1e-12)


def pack(fld, R, V, t, dt):
    ex, ey, ez, bx, by, bz = fld.eb(R[:, 0], R[:, 1], R[:, 2], t)
    nb = R.shape[0]
    X = np.empty((13, nb))
    X[0:3] = R.T
    X[3:6] = V.T
    X[6] = bx; X[7] = by; X[8] = bz
    X[9] = ex; X[10] = ey; X[11] = ez
    X[12] = dt
    return X


def main():
    force = "--force" in sys.argv
    mlp = C.load_corrector_numpy()
    fields = C.make_fields()
    fast = C.make_fast_fields(fields)
    R0, V0 = C.initial_conditions(C.N_IC)

    out = {"meta": {
        "what": "the input standardisation of the committed corrector "
                "checkpoint and what it does off the training point",
        "checkpoint": "checkpoints/boris_corrector_b4.pt",
        "clamp_min": CLAMP,
        "source": "training/train_corrector_b4.py, "
                  "model.x_std.copy_(Xt.std(0).clamp_min(1e-12))",
        "one_checkpoint_only": True,
    }}

    # ------------------------- 0. the numpy lift is the torch checkpoint
    # Everything below is a statement about the committed checkpoint, so the
    # numpy evaluation it is read through has to be the torch model itself,
    # off the training point as well as on it.
    import torch
    tm = mlp.torch_model
    worst = 0.0
    for fname in C.FIELD_NAMES:
        for dt in C.DT_GRID:
            X = pack(fast[fname], R0, V0, 0.0, dt)
            with torch.no_grad():
                b = tm(torch.from_numpy(X.T.copy())).numpy().T
            worst = max(worst, float(np.abs(mlp.forward(X) - b).max()))
    out["numpy_lift_vs_torch_max_abs"] = worst
    print("the numpy lift reproduces the torch checkpoint to %.3e over "
          "5 configurations x 12 step sizes x %d initial conditions"
          % (worst, C.N_IC))

    # -------------------------------------------------- 1. standardisation
    xm = mlp.x_mean.ravel()
    xs = mlp.x_std.ravel()
    std = {}
    degenerate = []
    for i, n in enumerate(INPUTS):
        std[n] = {"x_mean": float(xm[i]), "x_std": float(xs[i]),
                  "at_clamp": bool(xs[i] <= CLAMP)}
        if xs[i] <= CLAMP:
            degenerate.append(n)
    out["standardisation"] = std
    out["degenerate_inputs"] = degenerate
    out["y_scale"] = mlp.y_scale.ravel()
    print("degenerate inputs (zero variance in training, standard deviation "
          "replaced by the clamp): %s" % ", ".join(degenerate))

    # ------------------------------- 2, 3. saturation over the map's own grid
    sat = {}
    W1 = mlp.Ws[0]
    b1 = mlp.bs[0]
    for fname in C.FIELD_NAMES:
        fld = fast[fname]
        for dt in C.DT_GRID:
            X = pack(fld, R0, V0, 0.0, dt)
            Z = (X - mlp.x_mean) / mlp.x_std
            A = np.tanh(W1 @ Z + b1)
            d = mlp.forward(X)
            sat["%s|%g" % (fname, dt)] = {
                "max_abs_standardised_input": float(np.abs(Z).max()),
                "which_input": INPUTS[int(np.argmax(np.abs(Z).max(axis=1)))],
                "frac_layer1_units_saturated": float(
                    np.mean(np.abs(A) > 0.999)),
                "correction_dr_norm_ic0": float(np.linalg.norm(d[0:3, 0])),
                "correction_dv_norm_ic0": float(np.linalg.norm(d[3:6, 0])),
                "correction_dr_spread_over_ics": float(
                    np.linalg.norm(d[0:3], axis=0).std()),
            }
    out["saturation_on_the_grid"] = sat
    print("\n%-12s %8s %10s %10s %12s %12s"
          % ("field", "Omega h", "|z|max", "sat.frac", "|dr|", "spread"))
    for fname in C.FIELD_NAMES:
        for dt in (0.001, 0.1, 0.3, 0.5):
            r = sat["%s|%g" % (fname, dt)]
            print("%-12s %8g %10.2e %10.3f %12.4e %12.2e"
                  % (fname, dt, r["max_abs_standardised_input"],
                     r["frac_layer1_units_saturated"],
                     r["correction_dr_norm_ic0"],
                     r["correction_dr_spread_over_ics"]))

    # ------------------------- 4. is the correction still a function of state?
    # Along a Boris run of 400 steps, at the training step and at two others,
    # in the training field and in one it never saw.
    state_dep = {}
    for fname in ("B4_decaying", "B3_tilted"):
        fld = fast[fname]
        for dt in (C.DT_TRAIN, 0.1):
            x = R0[:, 0].copy(); y = R0[:, 1].copy(); z = R0[:, 2].copy()
            vx = V0[:, 0].copy(); vy = V0[:, 1].copy(); vz = V0[:, 2].copy()
            ds = []
            for n in range(400):
                R = np.stack([x, y, z], axis=1)
                V = np.stack([vx, vy, vz], axis=1)
                ds.append(mlp.forward(pack(fld, R, V, n * dt, dt)))
                x, y, z, vx, vy, vz = C.step_boris(fld, x, y, z, vx, vy, vz,
                                                   n * dt, dt)
            D = np.stack(ds)                      # (400, 6, nb)
            nrm = np.linalg.norm(D[:, 0:3, :], axis=1)
            state_dep["%s|%g" % (fname, dt)] = {
                "correction_dr_norm_mean": float(nrm.mean()),
                "correction_dr_norm_std": float(nrm.std()),
                "relative_variation": float(nrm.std() / max(nrm.mean(), 1e-300)),
                "n_samples": int(nrm.size),
            }
    out["state_dependence"] = state_dep
    print("\nis the correction still a function of the state?")
    for k, v in state_dep.items():
        print("  %-22s |dr| = %.4e +- %.2e  (relative variation %.3e)"
              % (k, v["correction_dr_norm_mean"],
                 v["correction_dr_norm_std"], v["relative_variation"]))

    # --------- 5. the fixed correction against the defect it should correct
    # The training target was r_ref - r_boris over one step, the reference
    # being the same Boris scheme at a 150 times smaller step, which is how
    # `train_corrector_b4.py` defines the defect.  That is recomputed here at
    # each step size and compared with what the network actually adds.
    from models.boris import boris_step, integrate_boris
    defect = {}
    for fname in ("B4_decaying", "uniform"):
        f = fields[fname]
        fld = fast[fname]
        for dt in C.DT_GRID:
            r = R0[0].copy(); v = V0[0].copy()
            r_b, v_b = boris_step(r, v, 0.0, dt, f)
            rs, vs, _ = integrate_boris(r, v, 0.0, dt / 150.0, 150, f)
            true_dr = float(np.linalg.norm(rs[-1] - r_b))
            d = mlp.forward(pack(fld, R0[:1], V0[:1], 0.0, dt))
            got = float(np.linalg.norm(d[0:3, 0]))
            defect["%s|%g" % (fname, dt)] = {
                "true_one_step_defect_dr": true_dr,
                "correction_applied_dr": got,
                "ratio_applied_over_true": got / max(true_dr, 1e-300),
            }
    out["one_step_defect_vs_correction"] = defect
    print("\nthe fixed correction against the one-step defect it is meant to "
          "be (B4, canonical initial condition)")
    for dt in C.DT_GRID:
        r = defect["B4_decaying|%g" % dt]
        print("  Omega h = %-6g true defect %.3e   applied %.3e   "
              "ratio %.3e" % (dt, r["true_one_step_defect_dr"],
                              r["correction_applied_dr"],
                              r["ratio_applied_over_true"]))

    return check_or_write(OUT, json.loads(json.dumps(C.clean(out))),
                          force=force)


if __name__ == "__main__":
    raise SystemExit(main())
