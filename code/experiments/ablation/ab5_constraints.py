"""AB5: the three constraints of the loss, taken off one at a time, and the
volume property, measured rather than imposed.

    python ab5_constraints.py            train, measure, check against the file
    python ab5_constraints.py --force    overwrite ab5_constraints.json

THIS IS THE ONE SCRIPT IN THIS DIRECTORY THAT RETRAINS, AND WHY
---------------------------------------------------------------
The pre-registration says the ablations are done on the committed checkpoint
and on the twenty of W16, and that retraining is out of scope **except where an
ablation is a term removed from the loss**, in which case it is to be said
explicitly.  This is that case, and this is the saying.  A constraint that acts
through the loss cannot be taken off at inference time: there is nothing to
switch off in the shipped network, only a network that was fitted under one
penalty and not under another.

WHAT THE THREE CONSTRAINTS ACTUALLY ARE
---------------------------------------
`training/train_corrector_b4.py` writes them out as

    small = mean( ||delta||^2 )                       lambda = 1e-3
    ortho = mean( (delta_v . v_hat)^2 )               lambda = 1e-3
    ener  = mean( (delta_v . v)^2 )                   lambda = 1e-3

and the first thing to say about them is that **two of the three are the same
constraint**.  v_hat = v/||v||, so ener = ||v||^2 * ortho exactly, and on this
problem ||v|| stays within one part in a thousand of 1 over the whole training
set.  The energy penalty and the orthogonality penalty are one penalty at
double weight, and the manuscript's Eq. (2) lists them as two.  That is
measured here, not asserted: `equivalence` below reports the ratio.

There is **no volume constraint anywhere in the loss**, and the wave's task
list names one.  It is reported as what it is -- absent -- and the property it
would have imposed is measured instead: |det J - 1| of the one-step map, for
the corrector, for its ablations, for the non-learned corrector and for the
classical schemes, against the Boris row, which is volume preserving exactly.
The smallness penalty is the term that plays the volume role, since a
correction of size epsilon leaves the corrected map within O(epsilon) of the
volume-preserving one, and its ablation is one of the eight below.

THE LATTICE, AND THE SEEDS
--------------------------
Eight configurations, the three penalties on and off in every combination, at
three seeds each.  The seeds are **not new**: 42 is the committed one and
16_000_000 and 16_000_001 are the first two of W16's block.  A loss ablation is
only readable against the run it ablates, so it is run at that run's seed.  No
seed is formed in this directory.

The training procedure is `../stats/seed_sweep_b4.py:train_one`, which is
`training/train_corrector_b4.py:train` with the seed as an argument, with the
three lambdas redirected and nothing else.  The dataset does not depend on the
lambdas, so it is built once per seed and shared by all eight configurations of
that seed; that is not an approximation, it is the same array.

Writes ab5_constraints.json and checkpoints into ckpt/.  Never writes into the
bundle's own checkpoint directory; the committed file's md5 is checked before
and after.
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

OUT = AB.outpath("ab5_constraints.json")
LOSS_CKPT = os.path.join(AB.CKPT, "loss")
DATA_CACHE = os.path.join(AB.CKPT, "data")
os.makedirs(LOSS_CKPT, exist_ok=True)
os.makedirs(DATA_CACHE, exist_ok=True)

SEEDS = [42, 16_000_000, 16_000_001]
COMMITTED_SEED = 42

#: the eight corners.  `all_on` is the shipped configuration.
CONFIGS = {}
for _s in (True, False):
    for _o in (True, False):
        for _e in (True, False):
            _name = ("all_on" if (_s and _o and _e) else
                     "no_" + "_".join([n for n, b in
                                       (("small", _s), ("ortho", _o),
                                        ("energy", _e)) if not b]))
            CONFIGS[_name] = {"small": _s, "ortho": _o, "energy": _e}


def dataset(seed):
    p = os.path.join(DATA_CACHE, "ds_%d.npz" % seed)
    if os.path.exists(p):
        z = np.load(p)
        return z["X"], z["Y"]
    sys.path.insert(0, os.path.join(AB.EXP, "stats"))
    import seed_sweep_b4 as SS
    X, Y = SS.build_dataset(seed)
    np.savez_compressed(p, X=X, Y=Y)
    return X, Y


def train_one(seed, cfg, X, Y):
    """`../stats/seed_sweep_b4.py:train_one` with the three lambdas switched.

    Copied rather than imported because the lambdas are read from module
    globals there; every other line is that function's, and the dataset is
    passed in instead of rebuilt so that the eight configurations of one seed
    see the same states in the same order.
    """
    import torch
    from common import set_global_seed
    from training.train_corrector_b4 import (
        DefectNet, EPOCHS, BATCH, LR, LAMBDA_SMALL, LAMBDA_ORTHO,
        LAMBDA_ENERGY)

    torch.set_default_dtype(torch.float64)
    set_global_seed(seed)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)

    model = DefectNet(n_in=X.shape[1])
    model.x_mean.copy_(Xt.mean(0))
    model.x_std.copy_(Xt.std(0).clamp_min(1e-12))
    model.y_scale.copy_(Yt.abs().mean(0).clamp_min(1e-16))

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = Xt.shape[0]
    idx_all = torch.randperm(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]
    v_dir = Xt[:, 3:6] / Xt[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-12)

    ls = LAMBDA_SMALL if cfg["small"] else 0.0
    lo = LAMBDA_ORTHO if cfg["ortho"] else 0.0
    le = LAMBDA_ENERGY if cfg["energy"] else 0.0

    for ep in range(EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(tr_idx.numel())]
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            pred = model(Xt[b])
            data = ((pred - Yt[b]) ** 2).mean()
            small = (pred ** 2).mean()
            ortho = ((pred[:, 3:] * v_dir[b]).sum(1) ** 2).mean()
            ener = ((pred[:, 3:] * Xt[b, 3:6]).sum(1) ** 2).mean()
            loss = (data / model.y_scale.pow(2).mean()
                    + ls * small + lo * ortho + le * ener)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pv = model(Xt[val_idx])
        val_rel = ((pv - Yt[val_idx]).norm(dim=1)
                   / Yt[val_idx].norm(dim=1).clamp_min(1e-30)).mean().item()
    return model, float(val_rel)


def main():
    force = "--force" in sys.argv
    t_start = time.time()
    import torch
    torch.set_default_dtype(torch.float64)
    from fields import DecayingField

    AB.assert_committed_untouched()
    field = DecayingField(B0=1.0, tau=AB.TAU)
    ref = AB.closed_form_ref()

    out = {"meta": {
        "wave": "W17",
        "what": "the three penalties of the loss taken off one at a time, and "
                "the volume property measured",
        "retraining_is_declared": "the pre-registration allows retraining only "
                                  "where an ablation IS a term removed from "
                                  "the loss; this is that case",
        "training": "../stats/seed_sweep_b4.py:train_one with the three "
                    "lambdas redirected and nothing else",
        "seeds": SEEDS,
        "seeds_are_not_new": "42 is the committed seed; 16000000 and 16000001 "
                             "are the first two of W16's block",
        "n_configurations": len(CONFIGS),
        "configurations": CONFIGS,
        "reference": "the closed form of B4",
    }}

    # ---- the two penalties that are one -----------------------------------
    X42, Y42 = dataset(COMMITTED_SEED)
    v = X42[:, 3:6]
    speed2 = np.sum(v * v, axis=1)
    out["equivalence"] = {
        "claim": "ener = ||v||^2 * ortho exactly, so the energy penalty and "
                 "the orthogonality penalty are one penalty at double weight",
        "speed_squared_over_training_set": {
            "min": float(speed2.min()), "max": float(speed2.max()),
            "median": float(np.median(speed2)),
            "max_departure_from_1": float(np.abs(speed2 - 1.0).max())},
        "effective_lambda_if_merged": 2e-3,
        "manuscript_lists_them_as_two": True,
    }

    # ---- the eight corners, at three seeds --------------------------------
    runs = {}
    variants = {"full": AB.Variant("net", ortho=True, rescale=True),
                "raw": AB.Variant("net", ortho=False, rescale=False)}
    for seed in SEEDS:
        X, Y = dataset(seed)
        for name, cfg in CONFIGS.items():
            tag = "%s|s%d" % (name, seed)
            path = os.path.join(LOSS_CKPT, "corrector_%s_s%d.pt" % (name, seed))
            t0 = time.time()
            if os.path.exists(path):
                model = AB.load_torch(path)
                val_rel = None
            else:
                model, val_rel = train_one(seed, cfg, X, Y)
                torch.save(model.state_dict(), path)
            # The validation split is drawn inside the training run, so it can
            # only be reported on the run that made it.  The fit itself is
            # reported by a statistic that depends on the checkpoint and the
            # dataset alone, so that a rerun off `ckpt/loss/` reproduces the
            # file rather than filling this field with nulls.
            with torch.no_grad():
                pv = model(torch.tensor(X))
                rel_all = float(
                    ((pv - torch.tensor(Y)).norm(dim=1)
                     / torch.tensor(Y).norm(dim=1).clamp_min(1e-30))
                    .mean().item())
            res = AB.evaluate_variant(variants, model=model, ref=ref)
            runs[tag] = {
                "config": cfg, "seed": seed, "md5": AB.md5(path),
                "rel_defect_error_over_the_whole_dataset": rel_all,
                "val_rel_defect_error_if_trained_here": val_rel,
                "full": {k: res["full"][k] for k in
                         ("energy_separation", "traj_gain_over_boris",
                          "pos_err_rms", "energy_err_median_2nd_half",
                          "constraint_max_abs")},
                "raw": {k: res["raw"][k] for k in
                        ("energy_separation", "traj_gain_over_boris",
                         "pos_err_rms", "energy_err_median_2nd_half",
                         "constraint_max_abs")},
                "wall_s": time.time() - t0,
            }
            print("  %-28s E-sep %8.4f (raw %8.4f)  gain %8.2f  (%.0fs)"
                  % (tag, runs[tag]["full"]["energy_separation"],
                     runs[tag]["raw"]["energy_separation"],
                     runs[tag]["full"]["traj_gain_over_boris"],
                     runs[tag]["wall_s"]), flush=True)
    out["runs"] = runs

    # ---- the table, medians over the three seeds --------------------------
    table = {}
    for name in CONFIGS:
        row = {"config": CONFIGS[name]}
        for proj in ("full", "raw"):
            for q in ("energy_separation", "traj_gain_over_boris",
                      "pos_err_rms", "energy_err_median_2nd_half"):
                vals = [runs["%s|s%d" % (name, s)][proj][q] for s in SEEDS]
                row["%s.%s" % (proj, q)] = {
                    "per_seed": {str(s): runs["%s|s%d" % (name, s)][proj][q]
                                 for s in SEEDS},
                    "median": float(np.median(vals)),
                    "min": float(min(vals)), "max": float(max(vals))}
        table[name] = row
    out["table"] = table

    # ---- the reproduction check -------------------------------------------
    # all_on at the committed seed is a retraining of the committed run.  It is
    # not bit-identical -- W16 already found that -- and the distance is the
    # scale against which a penalty's ablation has to be read.
    base = table["all_on"]
    committed = {"energy_separation": 45.752058584917684,
                 "traj_gain_over_boris": 120.11205620944715}
    out["reproduction_of_the_committed_run"] = {
        "all_on_at_seed_42": {
            k: runs["all_on|s42"]["full"][k] for k in committed},
        "committed_checkpoint": committed,
        "note": "the committed checkpoint is a different object from a "
                "retraining at the same seed (W16); the difference here is "
                "the floor below which no penalty ablation can be read",
        "ratio": {k: runs["all_on|s42"]["full"][k] / committed[k]
                  for k in committed},
    }

    # ---- what each penalty is worth ---------------------------------------
    worth = {}
    for name in CONFIGS:
        if name == "all_on":
            continue
        w = {}
        for q in ("energy_separation", "traj_gain_over_boris"):
            a = base["full.%s" % q]["median"]
            b = table[name]["full.%s" % q]["median"]
            w[q] = {"all_on_median": a, "ablated_median": b,
                    "ratio": (b / a) if a else None}
        for q in ("energy_separation",):
            a = base["raw.%s" % q]["median"]
            b = table[name]["raw.%s" % q]["median"]
            w["unprojected_" + q] = {"all_on_median": a, "ablated_median": b,
                                     "ratio": (b / a) if a else None}
        worth[name] = w
    out["what_each_penalty_is_worth"] = worth

    # ---- the volume property, measured ------------------------------------
    fast = MC.FastField("B4_decaying", field)
    model_c = AB.load_torch(AB.COMMITTED_CORRECTOR)
    vol = {}
    for name, var, mdl in (
            ("boris", AB.Variant("none", ortho=True, rescale=True), None),
            ("corrector_full", AB.Variant("net", ortho=True, rescale=True),
             model_c),
            ("corrector_raw", AB.Variant("net", ortho=False, rescale=False),
             model_c),
            ("corrector_dr0", AB.Variant("net", ortho=True, rescale=True,
                                         zero_dr=True), model_c),
            ("corrector_dv0", AB.Variant("net", ortho=True, rescale=True,
                                         zero_dv=True), model_c),
            ("analytic", AB.Variant("analytic", ortho=True, rescale=True),
             None),
            ("analytic_order3", AB.Variant("analytic", ortho=True,
                                           rescale=True,
                                           analytic_kind="order3"), None),
            ("trapezoid", AB.Variant("analytic", ortho=True, rescale=True,
                                     analytic_kind="trapezoid"), None)):
        vol[name] = AB.volume_defect(field, var, mdl)
    # the classical splittings, through the map's own steppers
    for s in ("vps2", "vps4"):
        J = np.empty((6, 6))
        w = np.concatenate([AB.R0, AB.V0])
        fd = 1e-7
        step = MC.step_vps2 if s == "vps2" else MC.step_vps4
        for k in range(6):
            leg = []
            for sg in (+1.0, -1.0):
                wp = w.copy()
                wp[k] += sg * fd
                o = step(fast, np.array([wp[0]]), np.array([wp[1]]),
                         np.array([wp[2]]), np.array([wp[3]]),
                         np.array([wp[4]]), np.array([wp[5]]), 0.0, AB.DT)
                leg.append(np.array([float(c[0]) for c in o]))
            J[:, k] = (leg[0] - leg[1]) / (2.0 * fd)
        vol[s] = {"det_minus_one": float(abs(np.linalg.det(J) - 1.0)),
                  "det": float(np.linalg.det(J)), "fd": fd}
    out["volume"] = {
        "what": "|det J - 1| of the one-step map at the canonical initial "
                "condition, central differences at fd = 1e-7",
        "there_is_no_volume_penalty_in_the_loss": True,
        "why_the_boris_row_is_the_scale": "the Boris map preserves phase-space "
                                          "volume exactly (Qin et al. 2013), "
                                          "so its row is the resolution of "
                                          "the difference",
        "per_scheme": vol,
    }

    # a fit statistic that does depend on the run, reported per configuration
    fit = {}
    for name in CONFIGS:
        vals = [runs["%s|s%d" % (name, s)]
                ["rel_defect_error_over_the_whole_dataset"] for s in SEEDS]
        fit[name] = {"per_seed": {str(s): v for s, v in zip(SEEDS, vals)},
                     "median": float(np.median(vals))}
    out["fit_quality"] = {
        "what": "mean over the dataset of ||pred - target|| / ||target||, "
                "which depends on the checkpoint and the dataset alone",
        "per_configuration": fit}

    out["meta"]["wall_s"] = time.time() - t_start
    AB.assert_committed_untouched()
    rc = AB.write(OUT, out, force=force,
                  ignore=("wall_s", "val_rel_defect_error_if_trained_here"))

    print("\n%-22s %14s %14s %13s"
          % ("loss configuration", "E-sep (median)", "raw E-sep", "traj gain"))
    for name in CONFIGS:
        r = table[name]
        print("%-22s %14.4f %14.4f %13.2f"
              % (name, r["full.energy_separation"]["median"],
                 r["raw.energy_separation"]["median"],
                 r["full.traj_gain_over_boris"]["median"]))
    print("\n%-20s %16s" % ("scheme", "|det J - 1|"))
    for k, d in vol.items():
        print("%-20s %16.3e" % (k, d["det_minus_one"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
