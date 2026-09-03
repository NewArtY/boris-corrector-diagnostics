"""HP5: phase three of W12.1 -- the third resource, how much training data.

Budget and capacity are not the only things a referee can say were too small.
This sweep holds the architecture, the configuration, the optimiser and the
4400 Adam steps fixed and moves only the size of the training set: 2000, 6000
and 24000 states, cut at the level of the trajectory, on the anchor
configuration of the SympNet.  Four seeds at each point, from the sweep's own
pair of seed roles.  Why the SympNet alone, and why the anchor, is in
`hp_common.data_jobs` and in the report.

    python hp5_data.py --shard 0 --nshards 8
    python hp5_data.py --list
    python hp5_data.py --force

Note what is *not* varied with the data.  The number of gradient steps is held
fixed, so the large point sees each of its states a quarter as often as the
full point does.  That is the honest way to isolate the data axis from the
budget axis, which the ladder measures separately; a sweep that grew both at
once could not say which of the two had moved the error.
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
    a = ap.parse_args()

    if not os.path.exists(SEL):
        print("hp2_selection.json is missing -- run hp2_select.py first")
        return 2
    sel = json.load(open(SEL, encoding="utf-8"))["selection"]
    jobs = H.data_jobs(sel)

    if a.list:
        bins, load = H.deal(jobs, a.nshards)
        for k, b in enumerate(bins):
            print("shard %2d  ~%6.0f s  %s"
                  % (k, load[k], ", ".join(H.job_id(*j) for j in b)))
        print("%d jobs, ~%.0f s of work, longest shard ~%.0f s"
              % (len(jobs), sum(load), max(load)))
        return 0

    return H.run_shard(jobs, a.shard, a.nshards, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
