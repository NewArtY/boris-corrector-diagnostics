"""AB4: the fallback gate, and its demonstration on the map of W14.

    python ab4_gate.py            measure, then check against the file
    python ab4_gate.py --force    overwrite ab4_gate.json

TASK 8 OF THE FIRST AUTHOR
--------------------------
"A diagnostic gate: if the corrector leaves the region it was trained in, the
scheme falls back to the plain Boris step.  Show it firing on a run outside the
training distribution."

THE SIGNAL, AND WHY IT IS NOT A HEURISTIC
------------------------------------------
W14 found the failure mechanism rather than guessing at it.  Four of the
network's thirteen inputs -- B_x, B_y, E_z and h -- had zero variance over the
training set, because every training trajectory lives in a field along z with a
planar induced E and every one of them is taken at h = 0.3.  `x_std` is filled
with `Xt.std(0).clamp_min(1e-12)`, so those four divisors are exactly 1e-12,
and the standardised input the first layer sees is 1e12 times the departure.
A step size of 0.2 instead of 0.3 is 1e11 standard deviations from the training
mean.  Every tanh in the first layer saturates and the correction collapses to
one constant vector.

The gate is therefore the network's own argument, read before the tanh flattens
it:

    g = max_i | (x_i - x_mean_i) / x_std_i |

and the scheme falls back to plain Boris on any step with g above the largest
value g attained anywhere in the training set.  It costs 40 flops against the
113,958 of the forward pass it decides whether to make, and it is evaluated
first, so a gated step does not pay for the network at all.

WHAT IS CHECKED
---------------
1  The mechanism: the four dead inputs are where W14 left them, and the
   network's output really does collapse to a constant outside.
2  The threshold: the largest standardised input over the committed training
   set, rebuilt by calling `training/train_corrector_b4.py:build_dataset()`
   unchanged, at the committed seed.  No new draw.
3  The map: the gated scheme is run over all 120 cells of W14 and the gate's
   verdict is compared with the map's own verdict A, "the corrector beats the
   Boris scheme on the trajectory", which holds in 4 cells of 120.
4  The demonstration: two runs at Omega h = 0.2 and 0.1 in the training field,
   where the corrector is known to be far worse than the Boris scheme, with
   the gate on and off and the errors against the closed form.

Writes ab4_gate.json.  Draws nothing new: `build_dataset` builds its one
generator from the committed SEED, exactly as the committed training did.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common as AB                                          # noqa: E402
import map_common as MC                                         # noqa: E402

OUT = AB.outpath("ab4_gate.json")
CACHE = os.path.join(AB.CKPT, "training_inputs.npz")
MAP_DIR = os.path.join(AB.EXP, "map")

#: the cells of W14 in which claim A holds -- the corrector beats the Boris
#: scheme on the trajectory.  Read from `../map/mp3_maps.json`, not retyped.
MP3 = os.path.join(MAP_DIR, "mp3_maps.json")


def training_inputs():
    """The committed training set's input matrix, (n, 13).

    `training/train_corrector_b4.py:build_dataset()` called unchanged.  It
    builds one generator from the committed SEED at the top of the function, so
    this is the same 6000 states the committed checkpoint was fitted on and not
    a new sample.  Cached, because it costs 900,000 Boris steps.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["X"], z["Y"]
    from training.train_corrector_b4 import build_dataset
    X, Y = build_dataset()
    np.savez_compressed(CACHE, X=X, Y=Y)
    return X, Y


def main():
    force = "--force" in sys.argv
    t_start = time.time()
    import torch
    torch.set_default_dtype(torch.float64)
    from fields import DecayingField

    AB.assert_committed_untouched()
    model = AB.load_torch(AB.COMMITTED_CORRECTOR)
    x_mean = model.x_mean.numpy().copy()
    x_std = model.x_std.numpy().copy()
    mlp = MC.load_corrector_numpy()

    out = {"meta": {
        "wave": "W17",
        "what": "the fallback gate of task 8, and its verification on the "
                "120 cells of W14",
        "signal": "g = max_i |(x_i - x_mean_i)/x_std_i| over the network's "
                  "thirteen inputs",
        "why_this_signal": "W14: four inputs had zero training variance, "
                           "clamp_min(1e-12) made their divisor 1e-12, and "
                           "outside the training point the first layer "
                           "saturates and the correction degenerates to one "
                           "constant vector",
        "n_random_draws": 0,
        "nothing_retrained": True,
    }}

    # ---- 1: the mechanism -------------------------------------------------
    dead = [i for i in range(13) if x_std[i] == 1e-12]
    out["mechanism"] = {
        "dead_inputs": {AB.INPUT_NAMES[i]: {"x_mean": float(x_mean[i]),
                                            "x_std": float(x_std[i])}
                        for i in dead},
        "n_dead_inputs": len(dead),
    }
    assert dead == sorted(AB.DEAD_INPUTS), "the dead inputs moved: %s" % dead

    # the collapse, measured: the network's output over the eight initial
    # conditions of the map, at the training step and at three others.
    field = DecayingField(B0=1.0, tau=AB.TAU)
    R0, V0 = MC.initial_conditions(MC.N_IC)
    collapse = {}
    for h in (0.3, 0.2, 0.1, 0.5, 0.30000000001):
        X = np.empty((13, MC.N_IC))
        for j in range(MC.N_IC):
            B = np.atleast_1d(field.B(R0[j], 0.0)).ravel()
            E = np.atleast_1d(field.E(R0[j], 0.0)).ravel()
            X[:, j] = np.concatenate([R0[j], V0[j], B, E, [h]])
        d = mlp.forward(X)
        g = AB.gate_signal(X, x_mean, x_std)
        spread = np.max(d, axis=1) - np.min(d, axis=1)
        scale = np.maximum(np.abs(d).max(axis=1), 1e-300)
        collapse["h=%r" % h] = {
            "gate_signal_max": float(g.max()),
            "gate_signal_min": float(g.min()),
            "output_spread_over_8_ic_relative":
                float(np.max(spread / scale)),
            "output_ic0": [float(v) for v in d[:, 0]],
        }
    out["mechanism"]["collapse"] = collapse

    # ---- 2: the threshold -------------------------------------------------
    X_tr, _ = training_inputs()
    g_tr = AB.gate_signal(X_tr.T, x_mean, x_std)
    thr = float(g_tr.max())
    per_input = np.abs((X_tr - x_mean) / x_std).max(axis=0)
    out["threshold"] = {
        "value": thr,
        "rule": "the largest standardised input attained anywhere in the "
                "committed training set; declared before the map was run and "
                "carried unchanged",
        "n_training_states": int(X_tr.shape[0]),
        "training_gate_signal": {
            "max": thr, "median": float(np.median(g_tr)),
            "p99": float(np.percentile(g_tr, 99.0)),
            "min": float(g_tr.min())},
        "per_input_max_over_training":
            {AB.INPUT_NAMES[i]: float(per_input[i]) for i in range(13)},
        "flops": AB.flops_gate(),
        "flops_of_the_forward_pass_it_decides": AB.FLOPS_NET_FORWARD,
    }
    print("threshold g_max over the committed training set: %.4f "
          "(median %.4f)" % (thr, np.median(g_tr)))

    # ---- what the threshold says the domain of validity is -----------------
    # The gate stays open only while every standardised input is below thr, so
    # thr and x_std together are a statement of the corrector's domain in the
    # units of the problem.  Two of those windows are worth printing.
    bz_lo = x_mean[8] - thr * x_std[8]
    bz_hi = x_mean[8] + thr * x_std[8]
    t_out = -AB.TAU * np.log(max(bz_lo, 1e-300) / 1.0)
    out["domain_of_validity"] = {
        "step_size": {
            "half_width": thr * float(x_std[12]),
            "window": [AB.DT - thr * float(x_std[12]),
                       AB.DT + thr * float(x_std[12])],
            "note": "h is one of the four inputs whose training variance was "
                    "zero, so the corrector's step-size domain is 0.3 plus or "
                    "minus three parts in 1e12"},
        "field_magnitude": {
            "window": [float(bz_lo), float(bz_hi)],
            "gyro_orbits_before_B_leaves_it":
                float(t_out / (2.0 * np.pi)),
            "time_before_B_leaves_it": float(t_out),
            "paper_window": AB.T_FINAL,
            "crossover_window": MC.T_LONG,
            "note": "B_z decays as exp(-t/tau), so the corrector leaves its "
                    "own training range in |B| at a time that is inside the "
                    "crossover horizon and outside the paper window"},
    }
    print("the domain of validity in the step is 0.3 +/- %.3e, and in time "
          "the field leaves the training range at t = %.1f (%.1f gyro-orbits)"
          % (thr * float(x_std[12]), t_out, t_out / (2 * np.pi)))

    # ---- 3: the map -------------------------------------------------------
    with open(MP3, encoding="utf-8") as fh:
        mp3 = json.load(fh)
    A_holds = set(mp3["summary"]["A_holds_where"])
    assert len(A_holds) == 4, "W14 held A in 4 cells, not %d" % len(A_holds)

    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    cells = {}
    print("\nrunning the gated scheme over the 120 cells of W14", flush=True)
    for fname in MC.FIELD_NAMES:
        t0 = time.time()
        for dt in MC.DT_GRID:
            n_long = int(round(MC.T_LONG / dt))
            n_mark = int(round(MC.T_SHORT / dt))
            idx = MC.sample_indices(n_long, n_mark)
            Rs, Vs, meta = AB.rollout_gated(fast[fname], R0, V0, dt, n_long,
                                            idx, mlp, x_mean, x_std, thr,
                                            mark=n_mark)
            # The two horizons share one run, as W14's do, but they do NOT
            # share one gate count: the counters are snapshotted at the mark,
            # so H_paper carries the firing over its own window and not over
            # the long one.
            for hname, m in (("H_paper", meta["at_mark"]),
                             ("H_crossover", meta)):
                f_ic = np.asarray(m["gated_fraction_per_ic"])
                key = "%s|%s|%.10g" % (hname, fname, dt)
                cells[key] = {
                    "gated_fraction": m["gated_fraction"],
                    "gated_fraction_ic0": float(f_ic[0]),
                    "gated_fraction_median_over_ic": float(np.median(f_ic)),
                    "gate_signal_max": m["gate_signal_max"],
                    "gate_signal_max_ic0":
                        float(m["gate_signal_max_per_ic"][0]),
                    "fired_on_the_first_step_ic0":
                        bool(meta["fired_first_step_per_ic"][0]),
                    "n_nonfinite": meta["n_nonfinite"],
                    # three readings of "the corrector is used in this cell",
                    # all three declared here rather than chosen afterwards
                    "corrector_used": m["gated_fraction"] < 1.0,
                    "corrector_used_majority":
                        float(np.median(f_ic)) < 0.5,
                    "corrector_used_at_ic0_first_step":
                        not bool(meta["fired_first_step_per_ic"][0]),
                    "A_holds_in_W14": key in A_holds,
                }
        print("  %-12s %.1fs" % (fname, time.time() - t0), flush=True)

    assert len(cells) == 120, "the map is 120 cells, got %d" % len(cells)

    open_cells = sorted(k for k in cells if cells[k]["corrector_used"])
    mixed = sorted(k for k in cells
                   if 0.0 < cells[k]["gated_fraction"] < 1.0)
    never = sorted(k for k in cells if cells[k]["gated_fraction"] == 0.0)
    out["map"] = {
        "n_cells": len(cells),
        "cells": cells,
        "gate_open_cells": open_cells,
        "n_gate_open": len(open_cells),
        "A_holds_cells": sorted(A_holds),
        "mixed_cells": mixed,
        "n_mixed": len(mixed),
        "never_gated_cells": never,
        "n_never_gated": len(never),
        "note": "a cell whose gated fraction is 1.0 runs the plain Boris step "
                "at every step and its error is W14's Boris row exactly; a "
                "cell whose gated fraction is 0.0 runs the corrector at every "
                "step and its error is W14's corrector row exactly.  Nothing "
                "is re-measured against a reference here, because in the "
                "absence of a mixed cell there is nothing new to measure.",
    }

    # ---- the confusion matrix, under all three readings --------------------
    out["P3"] = {
        "prediction": "the gate separates the 4 cells where the corrector "
                      "wins from the other 116, with no false negative among "
                      "the 4",
        "readings": {},
    }
    for reading in ("corrector_used", "corrector_used_majority",
                    "corrector_used_at_ic0_first_step"):
        tp = sorted(k for k in cells
                    if cells[k][reading] and cells[k]["A_holds_in_W14"])
        fp = sorted(k for k in cells
                    if cells[k][reading] and not cells[k]["A_holds_in_W14"])
        fn = sorted(k for k in cells
                    if not cells[k][reading] and cells[k]["A_holds_in_W14"])
        tn = [k for k in cells
              if not cells[k][reading] and not cells[k]["A_holds_in_W14"]]
        out["P3"]["readings"][reading] = {
            "gate_open_and_corrector_wins": {"n": len(tp), "cells": tp},
            "gate_open_and_corrector_loses": {"n": len(fp), "cells": fp},
            "gate_closed_but_corrector_would_have_won": {"n": len(fn),
                                                         "cells": fn},
            "gate_closed_and_corrector_loses": {"n": len(tn)},
            "no_false_negative": len(fn) == 0,
            "exact_separation": len(fn) == 0 and len(fp) == 0,
        }
        print("\n%-34s open %3d  A holds %d  false neg %d  false pos %d"
              % (reading, len(tp) + len(fp), len(A_holds), len(fn), len(fp)))
    out["P3"]["no_false_negative_under_every_reading"] = all(
        r["no_false_negative"] for r in out["P3"]["readings"].values())

    # ---- 4: the demonstration --------------------------------------------
    # Two runs in the training field at steps the corrector never saw, with
    # the gate on and off, scored against the closed form.  These are the
    # numbers the plan asked to be shown: at Omega h = 0.2 and 0.1 the
    # corrector's trajectory error is tens of Larmor radii.
    demo = {}
    ic = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    for dt in (0.2, 0.1, 0.3):
        n = int(round(MC.T_SHORT / dt))
        idx = np.arange(n + 1)
        ts = idx * dt
        Rr, Vr = MC.exact_b4(ts, ic[0][0], ic[1][0], tau=AB.TAU)
        row = {}
        for name, fn_ in (("boris", "boris"), ("corrector", "corrector")):
            Rs, Vs, _ = MC.rollout(fast["B4_decaying"], fn_, ic[0], ic[1],
                                   dt, n, idx, mlp=mlp)
            e = np.linalg.norm(Rs[:, 0, :] - Rr, axis=1)
            row[name] = {"pos_err_rms": float(np.sqrt(np.mean(e ** 2))),
                         "pos_err_final": float(e[-1])}
        Rs, Vs, meta = AB.rollout_gated(fast["B4_decaying"], ic[0], ic[1], dt,
                                        n, idx, mlp, x_mean, x_std, thr)
        e = np.linalg.norm(Rs[:, 0, :] - Rr, axis=1)
        row["gated"] = {"pos_err_rms": float(np.sqrt(np.mean(e ** 2))),
                        "pos_err_final": float(e[-1]),
                        "gated_fraction": meta["gated_fraction"],
                        "gate_signal_max": meta["gate_signal_max"]}
        row["gate_fires"] = meta["gated_fraction"] > 0.0
        row["gated_equals_boris"] = bool(
            abs(row["gated"]["pos_err_rms"] - row["boris"]["pos_err_rms"])
            <= 1e-12 * row["boris"]["pos_err_rms"])
        row["gated_equals_corrector"] = bool(
            abs(row["gated"]["pos_err_rms"] - row["corrector"]["pos_err_rms"])
            <= 1e-12 * row["corrector"]["pos_err_rms"])
        row["what_the_gate_saved"] = (row["corrector"]["pos_err_rms"]
                                      / row["gated"]["pos_err_rms"])
        row["flops_per_step_gated"] = (AB.FLOPS_BORIS + AB.flops_gate()
                                       if meta["gated_fraction"] == 1.0
                                       else None)
        demo["Omega_h=%g" % dt] = row
    out["demonstration"] = demo

    print("\n%-14s %13s %13s %13s %9s"
          % ("Omega h", "boris", "corrector", "gated", "gated %"))
    for k, r in demo.items():
        print("%-14s %13.4e %13.4e %13.4e %9.1f"
              % (k, r["boris"]["pos_err_rms"], r["corrector"]["pos_err_rms"],
                 r["gated"]["pos_err_rms"],
                 100 * r["gated"]["gated_fraction"]))

    out["meta"]["wall_s"] = time.time() - t_start
    AB.assert_committed_untouched()
    AB.assert_no_draws(0)
    return AB.write(OUT, out, force=force)


if __name__ == "__main__":
    raise SystemExit(main())
