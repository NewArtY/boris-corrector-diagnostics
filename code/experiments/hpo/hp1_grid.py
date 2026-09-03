"""HP1: phase one of W12.1 -- the hyper-parameter grid at the base budget.

Trains every configuration of `hp_common.GRID` at the corrector's own budget
(4400 Adam steps, batch 512) for four seeds, scores each on the run of
Section 7, and writes one JSON per job under `runs/`.

    python hp1_grid.py                       run every job in this process
    python hp1_grid.py --shard 0 --nshards 8 run one eighth of them
    python hp1_grid.py --list                what would be run, and its order
    python hp1_grid.py --force               overwrite the committed job files

Sharding is scheduling only.  A job's seeds are a function of
(arch, cfg, rep, role) alone, so the file a job writes does not depend on which
shard ran it or on how many shards there were.  Jobs are sorted by a static
cost hint before being dealt round-robin, which balances the shards; the hint
enters no number.

Every job file is gated: the first pass writes it, every later pass recomputes
and compares at rtol 1e-6, and a disagreement exits non-zero.  One thread per
process, because torch on this problem gains almost nothing from more (a
measured 1.75x for the HNN and 0.93x for the PINN at ten threads) while thread
count is one more thing that could move a number between runs.
"""
import argparse
import os
import sys

import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hp_common as H          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="one short job per architecture, writes nothing")
    a = ap.parse_args()

    jobs = H.grid_jobs()
    if a.list:
        bins, load = H.deal(jobs, a.nshards)
        for k, b in enumerate(bins):
            print("shard %2d  ~%6.0f s  %s"
                  % (k, load[k], ", ".join(H.job_id(*j) for j in b)))
        print("%d jobs, ~%.0f s of work, longest shard ~%.0f s"
              % (len(jobs), sum(load), max(load)))
        return 0

    if a.quick:
        import hp_common
        hp_common.TRACE_FRACTIONS = (0.5, 1.0)
        for arch in H.ARCHS:
            rec = H.run_job(arch, 0, 0, 1.0 / 44.0)   # 100 Adam steps
            print("  %-8s traj %.4e  energy %.4e  val %.4e  params %d"
                  % (arch, rec["traj"], rec["energy"], rec["val_loss"],
                     rec["n_parameters"]))
        return 0

    return H.run_shard(jobs, a.shard, a.nshards, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
