"""mp5_equalcost.py -- the corrector's own flop budget, spent on Boris and on
vps2 instead.

Claim B, which the pre-registration is about, is the corrector against vps4 at
equal total flops, and `mp2_grid.py` settles it.  This file answers the
question a referee asks next, and the first author asks first: the corrector
costs 114,091 flops a step against the Boris scheme's 113.  What does the
Boris scheme do with that budget?

One corrector step buys 1009 Boris steps or 1253 vps2 steps, from the committed
flop model of `../classical/schemes.py`.  Those runs are made here at the three
step sizes `mp2_grid.py` declared for its own equal-cost run, in all five
configurations, and read out at both horizons.  Nothing else is new: the
corrector's own numbers are read from the grid, not recomputed.

    python mp5_equalcost.py --field B4_decaying
    python mp5_equalcost.py

Writes mp5_equalcost__<field>.json; exits non-zero on a non-reproducing rerun.
"""
import argparse
import json
import os
import time

import numpy as np

import map_common as C
from ea_common import check_or_write
from mp2_grid import (EQUAL_COST_DT, N_OUT, build_solutions, eval_reference,
                      metrics_block)

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = ["boris", "vps2"]

#: the five files this script writes, spelled out so that
#: `../audit_numbers/an6_orphan_data.py` can find each of them named in the
#: source of its own directory.  The names are built at run time below.
OUTPUTS = ["mp5_equalcost__uniform.json", "mp5_equalcost__B1_radial.json",
           "mp5_equalcost__B2_wave.json", "mp5_equalcost__B3_tilted.json",
           "mp5_equalcost__B4_decaying.json"]


def run_field(fname, force=False):
    t_start = time.time()
    fields = C.make_fields()
    field = fields[fname]
    fld = C.make_fast_fields(fields)[fname]
    R0, V0 = C.initial_conditions(C.N_IC)
    r_L = C.larmor_radii(field, R0, V0)

    subs = {s: C.equal_cost_substeps(s) for s in SCHEMES}
    out = {"meta": {
        "field": fname,
        "what": "one corrector step's flop budget spent on a classical scheme",
        "substeps_per_corrector_step": subs,
        "flops_per_step": {s: C.FLOPS[s] for s in SCHEMES},
        "corrector_flops_per_step": C.FLOPS["corrector"],
        "dt_grid": EQUAL_COST_DT, "horizons": C.HORIZONS,
        "reference_best": ("closed form" if C.CLOSED_FORM[fname]
                           else "DOP853 rtol 3e-14 atol 1e-16"),
        "larmor_radii": r_L,
    }}
    sols = build_solutions(field, R0, V0, C.T_LONG)

    res = {}
    for dt in EQUAL_COST_DT:
        n_long = int(round(C.T_LONG / dt))
        n_short = int(round(C.T_SHORT / dt))
        idx = C.sample_indices(n_long, n_short, N_OUT)
        ts = idx * dt
        j_short = int(np.searchsorted(idx, n_short))
        _, best = eval_reference(fname, field, sols, R0, V0, ts)
        cuts = {"H_paper": j_short + 1, "H_crossover": len(idx)}
        for s in SCHEMES:
            m = subs[s]
            t0 = time.time()
            Rs, Vs, meta = C.rollout(fld, s, R0, V0, dt / m, n_long * m,
                                     idx * m)
            wall = time.time() - t0
            for hname, cut in cuts.items():
                ch = C.channels(Rs[:cut], Vs[:cut], best[0][:cut],
                                best[1][:cut], r_L)
                rec = metrics_block(ch["position"], ch["energy"])
                n_used = int(round(C.HORIZONS[hname] / dt))
                rec["substeps_per_corrector_step"] = m
                rec["h"] = dt / m
                rec["n_steps"] = n_used * m
                rec["total_flops"] = C.FLOPS[s] * n_used * m
                rec["corrector_total_flops"] = C.FLOPS["corrector"] * n_used
                res["%s|%g|%s" % (hname, dt, s)] = rec
            print("  dt=%-5g %-6s x%-5d h=%.3e  %6.1f s  "
                  "pos_rms(H_crossover, ic0)=%.3e"
                  % (dt, s, m, dt / m, wall,
                     res["H_crossover|%g|%s" % (dt, s)]["position"]["rms"][0]),
                  flush=True)
    out["equal_cost"] = res
    out["meta"]["wall_s"] = time.time() - t_start
    path = os.path.join(HERE, "mp5_equalcost__%s.json" % fname)
    return check_or_write(path, json.loads(json.dumps(C.clean(out))),
                          force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, choices=C.FIELD_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rc = 0
    for n in ([a.field] if a.field else C.FIELD_NAMES):
        print("[%s]" % n, flush=True)
        rc |= run_field(n, force=a.force)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
