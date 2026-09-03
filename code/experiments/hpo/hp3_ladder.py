"""HP3: phase two of W12.1 -- the budget ladder on the selected configuration.

Retrains the anchor configuration of each architecture --- entry 0 of its grid,
the one W9.1 trained and the manuscript prints --- at multiples of the
corrector's budget.  Four seeds at every rung, the same four data draws as
phase one.  The x1 rung is phase one and is not repeated.

    python hp3_ladder.py --shard 0 --nshards 8 [--with-data]
    python hp3_ladder.py --list
    python hp3_ladder.py --force

Each rung is a complete training run of its own length, not a checkpoint taken
out of a longer one: the cosine schedule anneals over whatever budget it is
given, so a truncated long run is not the same object as a short run.  How
different they are is measured in `hp4_report.py` -- a SympNet run of 35,200
steps read at its 4400th is worse than a completed 4400-step run by a factor of
37.  The x4 controls of W9.1 were done the same way, so they are comparable and
`hp4_report.py` prints them beside these.

WHICH RUNGS, AND WHY THESE.  The declared ladder and the ladder that was run
differ, and both are in `hp_common.LADDER` and `hp_common.LADDER_CFG_INDEX`
with the reason.  In short: the machine turned out to deliver a quarter of the
planned throughput, the ladder was shortened rather than thinned, and the four
seeds at every rung were kept.  `--with-data` deals the sweep of `hp5_data.py`
together with the ladder so that one scheduler balances both.
"""
import argparse
import json
import os
import sys

import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hp_common as H          # noqa: E402

SEL = os.path.join(HERE, "hp2_selection.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--with-data", action="store_true",
                    help="deal the hp5 data sweep together with the ladder")
    a = ap.parse_args()

    if not os.path.exists(SEL):
        print("hp2_selection.json is missing -- run hp2_select.py first")
        return 2
    sel = json.load(open(SEL, encoding="utf-8"))["selection"]
    jobs = H.ladder_jobs(sel)
    if a.with_data:
        # dealt together with the ladder so that one scheduler balances both
        jobs = jobs + H.data_jobs(sel)

    if a.list:
        bins, load = H.deal(jobs, a.nshards)
        for k, b in enumerate(bins):
            print("shard %2d  ~%6.0f s  %s"
                  % (k, load[k], ", ".join(H.job_id(*j) for j in b)))
        print("%d jobs, ~%.0f s of work, longest shard ~%.0f s, longest job "
              "~%.0f s" % (len(jobs), sum(load), max(load),
                           max(H.cost_hint(j) for j in jobs)))
        return 0

    return H.run_shard(jobs, a.shard, a.nshards, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
