"""mp2_grid.py -- the cross: five schemes x twelve steps x two horizons, in
one field configuration per invocation.

    python mp2_grid.py --field B4_decaying      one configuration
    python mp2_grid.py                          all five, in sequence
    python mp2_grid.py --field B1_radial --force  overwrite deliberately

Writes mp2_grid__<field>.json beside this file, one per configuration, and
exits non-zero if a rerun no longer reproduces a committed one.  Sharding by
configuration is scheduling only: the five are independent and no number
depends on how they are spread over processes.

WHAT IS MEASURED
----------------
For every (scheme, step, horizon) the two residual channels of the
pre-registration -- position in Larmor radii and relative kinetic energy --
under the four metrics declared there: maximum, root mean square, final value
and the running-maximum envelope.  Nothing is selected after the fact; the
whole cross is written out and `mp3_maps.py` reads it.

THE REFERENCE IS CHOSEN PER CONFIGURATION
-----------------------------------------
W13 established that DOP853 at rtol 1e-12 -- the manuscript's reference --
carries its own position error of 6.196e-12 Larmor radii, and that four of its
eleven runs were limited by the reference rather than by the scheme.  Every
residual here is therefore measured twice: against the manuscript's reference,
so that the numbers connect to the ones it prints, and against the best
reference available in that configuration, which is

    uniform, B3   the closed form (both are static, spatially uniform fields)
    B4            the closed form (Bessel of order zero, from W13)
    B1, B2        DOP853 at rtol 3e-14, there being no closed form

`mp1_calibration.py` measures what each of those is worth, and `mp3_maps.py`
marks every cell whose residual has fallen to within a factor of ten of it.

THE EQUAL-COST RUNS
-------------------
Claim B is about equal total flops, and one corrector step buys 417 vps4
steps.  Running that at every step size on the grid would be 4e8 vps4 steps
and is not the way to decide it: where vps4 already beats the corrector *at the
same step*, it beats it at equal cost too, since equal cost buys it a step 417
times smaller and it is fourth order.  The explicit equal-cost run is therefore
made at a set declared here before the grid --- Omega h in {0.2, 0.3, 0.5}, in
every configuration, at both horizons --- which is where that monotonicity is
checked rather than assumed, plus at any (field, step) where the corrector does
beat same-step vps4, since that is the only place claim B could hold.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import map_common as C
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))

#: declared before the grid: where the equal-cost vps4 run is made explicitly.
EQUAL_COST_DT = [0.2, 0.3, 0.5]
N_OUT = 1200

#: the five files this script writes, spelled out so that
#: `../audit_numbers/an6_orphan_data.py` can find each of them named in the
#: source of its own directory.  The names are built at run time below.
OUTPUTS = ["mp2_grid__uniform.json", "mp2_grid__B1_radial.json",
           "mp2_grid__B2_wave.json", "mp2_grid__B3_tilted.json",
           "mp2_grid__B4_decaying.json"]


def build_solutions(field, R0, V0, t_end):
    """One DOP853 solution per initial condition per tolerance, with dense
    output, so that every step grid is read off the same integration."""
    ts = np.array([0.0, t_end])
    out = {}
    for tag, kw in (("paper", dict(rtol=1e-12, atol=1e-14)),
                    ("tight", dict(rtol=3e-14, atol=1e-16))):
        out[tag] = [C.dop853(field, ts, R0[i], V0[i], **kw)
                    for i in range(R0.shape[0])]
    return out


def eval_reference(fname, field, sols, R0, V0, ts):
    nb = R0.shape[0]
    n = len(ts)
    paper = np.empty((n, nb, 3)), np.empty((n, nb, 3))
    best = np.empty((n, nb, 3)), np.empty((n, nb, 3))
    for i in range(nb):
        r, v = C.dop853_at(sols["paper"][i], ts)
        paper[0][:, i], paper[1][:, i] = r, v
        ex = C.exact(fname, field, ts, R0[i], V0[i])
        if ex is None:
            r, v = C.dop853_at(sols["tight"][i], ts)
        else:
            r, v = ex
        best[0][:, i], best[1][:, i] = r, v
    return paper, best


def metrics_block(ch_pos, ch_en):
    p = C.time_metrics(ch_pos)
    e = C.time_metrics(ch_en)
    return {
        "position": {"max": p["max"], "rms": p["rms"], "final": p["final"],
                     "envelope_deciles_ic0": p["envelope_deciles"][:, 0]},
        "energy": {"max": e["max"], "rms": e["rms"], "final": e["final"],
                   "median_2nd_half": C.median_second_half(ch_en),
                   "envelope_deciles_ic0": e["envelope_deciles"][:, 0]},
    }


def run_field(fname, force=False):
    t_start = time.time()
    fields = C.make_fields()
    fast = C.make_fast_fields(fields)
    field = fields[fname]
    fld = fast[fname]
    R0, V0 = C.initial_conditions(C.N_IC)
    r_L = C.larmor_radii(field, R0, V0)
    mlp = C.load_corrector_numpy()

    out = {"meta": {
        "field": fname,
        "field_class": type(field).__name__,
        "field_description": getattr(field, "description", ""),
        "corrector_saw_this_field_in_training": C.TRAINED_ON[fname],
        "closed_form": C.CLOSED_FORM[fname],
        "reference_best": ("closed form" if C.CLOSED_FORM[fname]
                           else "DOP853 rtol 3e-14 atol 1e-16"),
        "reference_paper": "DOP853 rtol 1e-12 atol 1e-14",
        "dt_grid": C.DT_GRID, "schemes": C.SCHEMES,
        "horizons": C.HORIZONS,
        "n_initial_conditions": C.N_IC,
        "map_seed": C.MAP_SEED, "n_random_draws": C.N_RANDOM_DRAWS,
        "larmor_radii": r_L,
        "corrector_checkpoint": "boris_corrector_b4.pt, one checkpoint, "
                                "trained on B4 at Omega h = 0.3",
        "equal_cost_dt_declared": EQUAL_COST_DT,
        "n_out_samples": N_OUT,
    }}

    print("[%s] building the references" % fname, flush=True)
    sols = build_solutions(field, R0, V0, C.T_LONG)

    # the physical energy signal of this configuration, on the paper window
    ts_w = np.arange(int(round(C.T_SHORT / C.DT_TRAIN)) + 1) * C.DT_TRAIN
    _, best_w = eval_reference(fname, field, sols, R0, V0, ts_w)
    Ew = 0.5 * np.sum(best_w[1] ** 2, axis=-1)
    half = len(ts_w) // 2
    out["meta"]["physical_signal_median_2nd_half"] = np.median(
        np.abs(Ew - Ew[0])[half:] / Ew[0], axis=0)

    grid = {}
    equal_cost = {}
    for dt in C.DT_GRID:
        n_long = int(round(C.T_LONG / dt))
        n_short = int(round(C.T_SHORT / dt))
        assert abs(n_long * dt - C.T_LONG) < 1e-9
        assert abs(n_short * dt - C.T_SHORT) < 1e-9
        idx = C.sample_indices(n_long, n_short, N_OUT)
        ts = idx * dt
        j_short = int(np.searchsorted(idx, n_short))
        paper, best = eval_reference(fname, field, sols, R0, V0, ts)
        cuts = {"H_paper": j_short + 1, "H_crossover": len(idx)}

        for scheme in C.SCHEMES:
            t0 = time.time()
            Rs, Vs, meta = C.rollout(fld, scheme, R0, V0, dt, n_long, idx,
                                     mlp=mlp)
            wall = time.time() - t0
            fps = C.flops_per_step(scheme, meta.get("mean_iters"))
            for hname, cut in cuts.items():
                chb = C.channels(Rs[:cut], Vs[:cut], best[0][:cut],
                                 best[1][:cut], r_L)
                chp = C.channels(Rs[:cut], Vs[:cut], paper[0][:cut],
                                 paper[1][:cut], r_L)
                rec = metrics_block(chb["position"], chb["energy"])
                rec["position_rms_vs_paper_reference"] = np.sqrt(
                    np.mean(chp["position"] ** 2, axis=0))
                rec["energy_median_vs_paper_reference"] = \
                    C.median_second_half(chp["energy"])
                n_used = int(round(C.HORIZONS[hname] / dt))
                rec["n_steps"] = n_used
                rec["flops_per_step"] = fps
                rec["total_flops"] = fps * n_used
                rec["gyro_orbits"] = C.HORIZONS[hname] / C.TWO_PI
                if "mean_iters" in meta:
                    rec["mean_iters"] = meta["mean_iters"]
                    rec["max_iters"] = meta["max_iters"]
                rec["n_nonfinite"] = meta["n_nonfinite"]
                grid["%s|%s|%s" % (hname, "%g" % dt, scheme)] = rec
            print("  dt=%-6g %-10s %6.1f s  pos_rms(H_crossover, ic0)=%.3e"
                  % (dt, scheme, wall,
                     grid["H_crossover|%g|%s" % (dt, scheme)]
                     ["position"]["rms"][0]), flush=True)

        # ------------------------------------------------ the equal-cost run
        m = C.equal_cost_substeps("vps4")
        corr_ok = grid["H_crossover|%g|corrector" % dt]["position"]["rms"][0]
        vps4_ok = grid["H_crossover|%g|vps4" % dt]["position"]["rms"][0]
        corr_s = grid["H_paper|%g|corrector" % dt]["position"]["rms"][0]
        vps4_s = grid["H_paper|%g|vps4" % dt]["position"]["rms"][0]
        trigger = (corr_ok < vps4_ok) or (corr_s < vps4_s)
        if dt in EQUAL_COST_DT or trigger:
            t0 = time.time()
            Rs, Vs, meta = C.rollout(fld, "vps4", R0, V0, dt / m, n_long * m,
                                     idx * m, mlp=None)
            wall = time.time() - t0
            for hname, cut in cuts.items():
                chb = C.channels(Rs[:cut], Vs[:cut], best[0][:cut],
                                 best[1][:cut], r_L)
                rec = metrics_block(chb["position"], chb["energy"])
                n_used = int(round(C.HORIZONS[hname] / dt))
                rec["substeps_per_corrector_step"] = m
                rec["h"] = dt / m
                rec["n_steps"] = n_used * m
                rec["flops_per_step"] = C.FLOPS["vps4"]
                rec["total_flops"] = C.FLOPS["vps4"] * n_used * m
                rec["corrector_total_flops"] = C.FLOPS["corrector"] * n_used
                rec["triggered_by_corrector_win"] = bool(trigger)
                equal_cost["%s|%g" % (hname, dt)] = rec
            print("  dt=%-6g %-10s %6.1f s  equal-cost vps4 at h=%.3e"
                  % (dt, "vps4x%d" % m, wall, dt / m), flush=True)

    out["grid"] = grid
    out["equal_cost_vps4"] = equal_cost
    out["meta"]["wall_s"] = time.time() - t_start
    path = os.path.join(HERE, "mp2_grid__%s.json" % fname)
    return check_or_write(path, json.loads(json.dumps(C.clean(out))),
                          force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, choices=C.FIELD_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = [a.field] if a.field else C.FIELD_NAMES
    rc = 0
    for n in names:
        rc |= run_field(n, force=a.force)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
