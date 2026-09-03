"""SD2: retrain the corrector at the declared seeds.

    python sd2_train.py                     all declared seeds, in order
    python sd2_train.py --shard 0 --of 4    seeds 0, 4, 8, ... of the list
    python sd2_train.py --seed42            the reproduction check, see below
    python sd2_train.py --force             overwrite sd2_training.json

Writes ckpt/corrector_b4_s<seed>.pt, ckpt/params_s<seed>.json and, once every
declared seed exists, sd2_training.json.

WHAT IS CHANGED AND WHAT IS NOT
-------------------------------
Nothing in `training/train_corrector_b4.py` is edited.  Its own
`build_dataset()` and `train()` are called, with exactly two module globals
redirected before the call:

    SEED             the single place the procedure consumes randomness --
                     `build_dataset` builds `numpy.random.default_rng(SEED)`
                     once at the top of the function, outside its loops, and
                     `train()` draws the weight initialisation, the
                     train/validation split and the epoch shuffling from the
                     torch generator that `set_global_seed(SEED)` fixes
    CHECKPOINT_DIR   so that `checkpoints/boris_corrector_b4.pt` is never the
                     file `train()` writes to

The architecture (13-128-128-128-128-6, tanh), the optimiser (Adam, lr 1e-3,
cosine annealing), the 400 epochs at batch 512, the five decay times, the three
trajectories each, the working step, the fine reference at h/150, the three
constraint weights and float64 throughout are untouched.  Every one of them is
read back out of the produced params file and asserted equal to the committed
`checkpoints/corrector_b4_params.json`, field by field, so that "no
hyper-parameter was changed" is a check and not a claim.

THE COMMITTED CHECKPOINT IS NOT TOUCHED
---------------------------------------
`--seed42` reruns the procedure at the seed the shipped checkpoint was trained
at, writes it to this directory under a different name, and compares the two
state dictionaries parameter by parameter.  That is the reproduction check: it
says whether the committed checkpoint is the one this machine's torch produces
from the shipped script, and therefore whether the ensemble and the committed
run are draws from one procedure.  The md5 of the committed file is asserted
before and after every invocation.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sd_common as SD                                        # noqa: E402

OUT = os.path.join(HERE, "sd2_training.json")

#: the hyper-parameters that must not move, read from the committed params file
FROZEN = ("dt_work", "dt_fine", "t_final", "tau_train", "tau_main", "hidden",
          "n_layers", "epochs", "batch", "lr", "dtype", "lambda_small",
          "lambda_ortho", "lambda_energy", "n_samples")


def committed_params():
    return json.load(open(os.path.join(SD.BUNDLE_CHECKPOINTS,
                                       "corrector_b4_params.json"),
                          encoding="utf-8"))


def train_one(seed, tag=None):
    """One retraining.  Returns the params dict the shipped script writes."""
    import torch
    from common import set_global_seed
    import training.train_corrector_b4 as TC

    tag = tag if tag is not None else "s%d" % seed
    tmp = os.path.join(SD.CKPT, "_tmp_%s" % tag)
    os.makedirs(tmp, exist_ok=True)
    if os.path.abspath(tmp) == os.path.abspath(SD.BUNDLE_CHECKPOINTS):
        raise SystemExit("refusing to write into the bundle's checkpoints")

    TC.SEED = int(seed)
    TC.CHECKPOINT_DIR = tmp
    torch.set_default_dtype(torch.float64)
    set_global_seed(int(seed))

    t0 = time.time()
    TC.train()
    wall = time.time() - t0

    src_pt = os.path.join(tmp, "boris_corrector_b4.pt")
    src_js = os.path.join(tmp, "corrector_b4_params.json")
    dst_pt = os.path.join(SD.CKPT, "corrector_b4_%s.pt" % tag)
    dst_js = os.path.join(SD.CKPT, "params_%s.json" % tag)
    shutil.move(src_pt, dst_pt)
    shutil.move(src_js, dst_js)
    shutil.rmtree(tmp, ignore_errors=True)

    p = json.load(open(dst_js, encoding="utf-8"))
    ref = committed_params()
    bad = [k for k in FROZEN if p.get(k) != ref.get(k)]
    if bad:
        raise SystemExit("hyper-parameters moved at seed %d: %s" % (seed, bad))
    p["_wall_s"] = wall
    p["_checkpoint"] = os.path.relpath(dst_pt, HERE).replace("\\", "/")
    p["_md5"] = SD.md5(dst_pt)
    return p


def compare_state_dicts(path_a, path_b):
    """Parameter-by-parameter comparison of two checkpoints."""
    import torch
    a = torch.load(path_a, map_location="cpu")
    b = torch.load(path_b, map_location="cpu")
    keys = sorted(set(a) | set(b))
    out = {"keys": len(keys), "identical_bitwise": True,
           "max_abs_diff": 0.0, "max_rel_diff": 0.0, "per_key": {}}
    for k in keys:
        if k not in a or k not in b:
            out["identical_bitwise"] = False
            out["per_key"][k] = "missing"
            continue
        x = a[k].detach().double().numpy().ravel()
        y = b[k].detach().double().numpy().ravel()
        same = bool(np.array_equal(x, y))
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        scale = float(np.max(np.abs(x))) if x.size else 1.0
        out["per_key"][k] = {"n": int(x.size), "bitwise_equal": same,
                             "max_abs_diff": d,
                             "max_rel_diff": d / max(scale, 1e-300)}
        out["identical_bitwise"] &= same
        out["max_abs_diff"] = max(out["max_abs_diff"], d)
        out["max_rel_diff"] = max(out["max_rel_diff"],
                                  d / max(scale, 1e-300))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--seed42", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    SD.assert_committed_untouched()

    if a.seed42:
        tag = "reproduce42"
        dst = os.path.join(SD.CKPT, "corrector_b4_%s.pt" % tag)
        if not os.path.exists(dst):
            train_one(SD.COMMITTED_SEED, tag=tag)
        cmp_ = compare_state_dicts(SD.COMMITTED_CORRECTOR, dst)
        cmp_["md5_committed"] = SD.md5(SD.COMMITTED_CORRECTOR)
        cmp_["md5_reproduced"] = SD.md5(dst)
        print(json.dumps({k: v for k, v in cmp_.items() if k != "per_key"},
                         indent=1))
        with open(os.path.join(HERE, "sd2_reproduce42.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(cmp_, fh, indent=1)
        SD.assert_committed_untouched()
        return 0

    todo = [s for i, s in enumerate(SD.CORRECTOR_SEEDS) if i % a.of == a.shard]
    for s in todo:
        dst = os.path.join(SD.CKPT, "corrector_b4_s%d.pt" % s)
        if os.path.exists(dst):
            print("seed %d already trained" % s, flush=True)
            continue
        t0 = time.time()
        p = train_one(s)
        print("seed %d  val_rel_defect_error %.6f  %.1fs"
              % (s, p["history"][-1]["val_rel_defect_error"],
                 time.time() - t0), flush=True)

    SD.assert_committed_untouched()

    have = [s for s in SD.CORRECTOR_SEEDS
            if os.path.exists(os.path.join(SD.CKPT, "corrector_b4_s%d.pt" % s))]
    if len(have) < SD.N_CORRECTOR_SEEDS:
        print("%d/%d seeds trained; run the remaining shards before the "
              "summary" % (len(have), SD.N_CORRECTOR_SEEDS))
        return 0

    runs = {}
    for s in SD.CORRECTOR_SEEDS:
        p = json.load(open(os.path.join(SD.CKPT, "params_s%d.json" % s),
                           encoding="utf-8"))
        runs["s%d" % s] = {
            "seed": s,
            "md5": SD.md5(os.path.join(SD.CKPT, "corrector_b4_s%d.pt" % s)),
            "n_samples": p["n_samples"],
            "final_train_loss": p["history"][-1]["train_loss"],
            "final_val_rel_defect_error": p["history"][-1]["val_rel_defect_error"],
            "history": p["history"],
        }
    ref = committed_params()
    payload = {
        "meta": {
            "wave": "W16",
            "what": "retrainings of the committed corrector at the declared "
                    "seeds, architecture and hyper-parameters unchanged",
            "script_called": "training/train_corrector_b4.py:train(), with "
                             "SEED and CHECKPOINT_DIR redirected and nothing "
                             "else",
            "frozen_hyper_parameters": {k: ref[k] for k in FROZEN},
            "seed_block": SD.SEED_BLOCK,
            "seeds": SD.CORRECTOR_SEEDS,
            "n_seeds": SD.N_CORRECTOR_SEEDS,
            "committed_seed": SD.COMMITTED_SEED,
            "committed_md5": SD.COMMITTED_MD5,
            "committed_final_val_rel_defect_error":
                ref["history"][-1]["val_rel_defect_error"],
            "committed_final_train_loss": ref["history"][-1]["train_loss"],
        },
        "runs": runs,
        "training_cost_flops": SD.corrector_training_flops(),
        "val_rel_defect_error": SD.quartiles(
            [runs["s%d" % s]["final_val_rel_defect_error"]
             for s in SD.CORRECTOR_SEEDS]),
        "committed_in_ensemble_val_rel_defect_error": SD.percentile_of(
            ref["history"][-1]["val_rel_defect_error"],
            [runs["s%d" % s]["final_val_rel_defect_error"]
             for s in SD.CORRECTOR_SEEDS]),
    }
    return SD.write(OUT, payload, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
