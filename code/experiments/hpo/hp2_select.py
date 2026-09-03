"""HP2: read the phase-one grid, select a configuration, and ask whether the
quantity the training minimises predicts the quantity the rollout is scored on.

    python hp2_select.py [--force]

Writes `hp2_selection.json`.  Rerunning recomputes from `runs/*.json` and exits
non-zero if the file no longer reproduces.

SELECTION RULE, DECLARED BEFORE THE RUN
---------------------------------------
Within each architecture, the configuration with the lowest median validation
loss over its four seeds.  The validation loss is that architecture's own loss
functional on the held-out draw of fifteen trajectories described in
`hp_common`, so selection never compares one architecture with another and
never looks at the rollout error.

The configuration with the lowest median *rollout* error is recorded beside it
under `oracle_by_traj`.  That is not a selection anybody could make -- it reads
the test metric -- and it is here for one purpose: if even the oracle choice
does not close the gap, no selection rule could have.

DOES THE LOSS PREDICT THE ROLLOUT
---------------------------------
W9.1 measured one instance of a mismatch: the HNN's field-matching loss fell by
a factor of twenty at four times the budget while its trajectory error rose from
0.088 to 0.142.  Whether that is a property of that architecture or of all three
is a question about rank correlation, and it is asked here two ways.

  across    Spearman rank correlation between the final validation loss and the
            rollout trajectory error over every (configuration, seed) of the
            grid, within an architecture.  A correlation near +1 means the
            validation loss is a usable model-selection signal; near zero or
            negative means selecting on it is selecting on noise or worse.
  within    the same correlation over the six checkpoints of a single training
            run, then the median over runs, and the fraction of runs in which
            it is negative -- that is, in which descending the loss moved the
            rollout error the wrong way.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hp_common as H          # noqa: E402
import ea_common as C          # noqa: E402

OUT = os.path.join(HERE, "hp2_selection.json")


def load_runs(mult=None):
    """The grid and ladder jobs.  A job of the data sweep carries a `data_key`
    and is excluded here: it shares the base budget with the grid but not the
    training set, and the selection is over the grid."""
    out = []
    for p in sorted(glob.glob(os.path.join(H.RUNS, "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if "data_key" in r:
            continue
        if mult is None or r["budget_multiplier"] == mult:
            out.append(r)
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or len(set(a[m])) < 2 or len(set(b[m])) < 2:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def stats(v):
    v = np.asarray([x for x in v], float)
    fin = v[np.isfinite(v)]
    return {"n": int(v.size), "n_finite": int(fin.size),
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "min": float(fin.min()) if fin.size else float("nan"),
            "max": float(fin.max()) if fin.size else float("nan"),
            "spread": float(fin.max() / fin.min())
            if fin.size and fin.min() > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    runs = load_runs(mult=1)
    want = len(H.grid_jobs())
    if len(runs) != want:
        print("expected %d grid jobs, found %d -- run hp1_grid.py first"
              % (want, len(runs)))
        return 2

    sch, signal = H.classical_rows()
    out = {"meta": {
        "n_grid_jobs": len(runs),
        "n_seeds": H.N_SEEDS,
        "base_adam_steps": C.ADAM_STEPS, "batch": C.BATCH,
        "data_rep_train": list(H.DATA_REP_TRAIN),
        "data_rep_val": H.DATA_REP_VAL,
        "selection_rule": "lowest median validation loss over the four seeds, "
                          "within an architecture",
        "vps4": {"traj": sch["vps4"]["traj"], "energy": sch["vps4"]["energy"],
                 "flops": sch["vps4"]["flops"]},
        "vps2": {"traj": sch["vps2"]["traj"], "energy": sch["vps2"]["energy"],
                 "flops": sch["vps2"]["flops"]},
        "corrector": {"traj": sch["hybrid"]["traj"],
                      "energy": sch["hybrid"]["energy"],
                      "flops": sch["hybrid"]["flops"]},
        "physical_signal": signal,
    }, "by_config": {}, "selection": {}, "loss_vs_rollout": {}}

    for arch in H.ARCHS:
        per_cfg = {}
        for ci, (name, cfg) in enumerate(H.GRID[arch]):
            rr = [r for r in runs if r["arch"] == arch and r["cfg_index"] == ci]
            assert len(rr) == H.N_SEEDS, (arch, name, len(rr))
            rr.sort(key=lambda r: r["rep"])
            per_cfg[name] = {
                "cfg_index": ci,
                "hyper": rr[0]["hyper"],
                "n_parameters": rr[0]["n_parameters"],
                "flops_per_step": rr[0]["flops_per_step"],
                "flops_run": rr[0]["flops_run"],
                "adam_steps": rr[0]["adam_steps"],
                "val_loss": stats([r["val_loss"] for r in rr]),
                "train_loss": stats([r["final_train_loss_mean50"] for r in rr]),
                "traj": stats([r["traj"] for r in rr]),
                "energy": stats([r["energy"] for r in rr]),
                "omega_h": stats([r["omega_h"] for r in rr]),
                "seeds": [r["seed_init"] for r in rr],
                "traj_by_seed": [r["traj"] for r in rr],
                "energy_by_seed": [r["energy"] for r in rr],
                "val_by_seed": [r["val_loss"] for r in rr],
            }
        out["by_config"][arch] = per_cfg

        names = list(per_cfg)
        pick = min(names, key=lambda n: per_cfg[n]["val_loss"]["median"])
        oracle = min(names, key=lambda n: per_cfg[n]["traj"]["median"])
        oracle_seed = min(names, key=lambda n: per_cfg[n]["traj"]["min"])
        out["selection"][arch] = {
            "cfg": pick, "cfg_index": per_cfg[pick]["cfg_index"],
            "val_loss_median": per_cfg[pick]["val_loss"]["median"],
            "traj_median": per_cfg[pick]["traj"]["median"],
            "traj_min": per_cfg[pick]["traj"]["min"],
            "energy_median": per_cfg[pick]["energy"]["median"],
            "anchor_cfg": H.GRID[arch][0][0],
            "is_anchor": bool(per_cfg[pick]["cfg_index"] == 0),
            "oracle_by_traj": {
                "cfg_median": oracle,
                "traj_median": per_cfg[oracle]["traj"]["median"],
                "cfg_best_seed": oracle_seed,
                "traj_min": per_cfg[oracle_seed]["traj"]["min"],
                "energy_at_that_seed": per_cfg[oracle_seed]["energy_by_seed"][
                    int(np.argmin(per_cfg[oracle_seed]["traj_by_seed"]))],
            },
        }

        rr = [r for r in runs if r["arch"] == arch]
        across_traj = spearman([r["val_loss"] for r in rr],
                               [r["traj"] for r in rr])
        across_en = spearman([r["val_loss"] for r in rr],
                             [r["energy"] for r in rr])
        within = [spearman([p["val_loss"] for p in r["trace"]],
                           [p["traj"] for p in r["trace"]]) for r in rr]
        within = [w for w in within if np.isfinite(w)]
        # over the last three checkpoints only: the late part of training, where
        # the question "does more of the same help" actually lives
        late = [spearman([p["val_loss"] for p in r["trace"][-3:]],
                         [p["traj"] for p in r["trace"][-3:]]) for r in rr]
        late = [w for w in late if np.isfinite(w)]
        out["loss_vs_rollout"][arch] = {
            "n_runs": len(rr),
            "across_configs_val_vs_traj": across_traj,
            "across_configs_val_vs_energy": across_en,
            "within_run_median": float(np.median(within)) if within else float("nan"),
            "within_run_negative_fraction":
                float(np.mean([w < 0 for w in within])) if within else float("nan"),
            "within_run_late_median":
                float(np.median(late)) if late else float("nan"),
            "within_run_late_negative_fraction":
                float(np.mean([w < 0 for w in late])) if late else float("nan"),
        }

    return C.check_or_write(OUT, out, rtol=1e-9, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
