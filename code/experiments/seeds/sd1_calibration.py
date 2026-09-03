"""SD1: calibrate the stand before anything is retrained.

    python sd1_calibration.py            measure, then check against the file
    python sd1_calibration.py --force    overwrite the committed file

Writes sd1_calibration.json.  Three checks, all on the COMMITTED checkpoint,
all before a single retraining exists:

  1  All five rows of Table~\\ref{tab:family} reproduced on this stand, scored
     exactly as the table is -- B4, Omega h = 0.3, t = 120, 401 samples,
     against DOP853 at rtol 1e-12 -- and compared both with the three printed
     digits and with the full-precision values in `../map/mp1_calibration.json`.
     If this stand cannot put back the manuscript's own row, nothing it says
     about an ensemble is worth reading.

  2  The committed checkpoint put through the whole W16 measurement -- five
     configurations, four channels, two horizons, eight initial conditions --
     and compared leaf by leaf with the corrector entries of the committed
     `../gtable/gt2_channels__*.json`.  The two are the same code on the same
     checkpoint, so the required agreement is exact zero, and a non-zero
     difference means this directory has changed something it claims to import.

  3  The md5 of `checkpoints/boris_corrector_b4.pt`, recorded here so that
     every later script in this directory can assert it has not moved.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sd_common as SD                                        # noqa: E402
import map_common as MC                                       # noqa: E402
import gt_common as G                                         # noqa: E402

OUT = os.path.join(HERE, "sd1_calibration.json")

#: Table~\ref{tab:family} as the manuscript prints it, three digits.
TAB_FAMILY_PRINTED = {
    "boris":     {"trajectory": 4.17e-1, "energy": 1.25e-6},
    "corrector": {"trajectory": 3.47e-3, "energy": 1.64e-5},
    "vps2":      {"trajectory": 1.06e-2, "energy": 5.57e-6},
    "vps4":      {"trajectory": 5.35e-5, "energy": 2.64e-7},
    "gl4":       {"trajectory": 7.75e-4, "energy": 4.18e-8},
}


def part1_tab_family():
    """The five rows of Table~\\ref{tab:family}, on this stand."""
    from gt1_calibration import reference_orbit
    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    fname = "B4_decaying"
    field, fld = fields[fname], fast[fname]
    R0, V0 = MC.initial_conditions(MC.N_IC)
    r_L = MC.larmor_radii(field, R0, V0)
    n = G.HORIZONS["H_paper"]                      # 400 steps, 401 samples
    ts = np.arange(n + 1) * SD.DT
    idx = np.arange(n + 1)
    Rr, Vr = reference_orbit(fname, field, R0, V0, ts, tag="paper")
    mlp = SD.load_corrector(SD.COMMITTED_CORRECTOR)

    rows = {}
    for s in MC.SCHEMES:
        Rs, Vs, meta = MC.rollout(fld, s, R0, V0, SD.DT, n, idx, mlp=mlp)
        ch = G.channel_series(Rs, Vs, Rr, Vr, r_L)
        half = (n + 1) // 2
        traj = float(np.sqrt(np.mean(ch["trajectory"][:, 0] ** 2)))
        ener = float(np.median(np.abs(ch["energy"][half:, 0])))
        p = TAB_FAMILY_PRINTED[s]
        rows[s] = {
            "trajectory": traj, "trajectory_printed": p["trajectory"],
            "trajectory_rel_diff": abs(traj - p["trajectory"]) / p["trajectory"],
            "energy": ener, "energy_printed": p["energy"],
            "energy_rel_diff": abs(ener - p["energy"]) / p["energy"],
        }
    worst = max(max(r["trajectory_rel_diff"], r["energy_rel_diff"])
                for r in rows.values())
    return {"setup": "B4_decaying, Omega h = 0.3, t = 120, 401 samples, "
                     "DOP853 rtol 1e-12 atol 1e-14, canonical initial "
                     "condition r0=(1,0,0) v0=(0,1,0), r_L = 1",
            "rows": rows,
            "worst_relative_difference_vs_printed": worst,
            "note": "the printed table carries three significant digits, so "
                    "agreement at 2e-3 is agreement to the printed precision"}


def part2_against_gt2():
    """The committed checkpoint through the whole W16 stand, against W15."""
    import json
    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    R0, V0 = MC.initial_conditions(MC.N_IC)
    mlp = SD.load_corrector(SD.COMMITTED_CORRECTOR)
    N = max(SD.HORIZONS.values())
    ts = np.arange(N) * SD.DT

    out = {}
    worst = 0.0
    for fname in SD.FIELD_NAMES:
        field = fields[fname]
        r_L = MC.larmor_radii(field, R0, V0)
        w0 = G.reference_gyrofrequency(field, R0)
        Rr, Vr = SD.reference(fname, field, R0, V0, ts)
        t0 = time.time()
        rec = SD.measure_run(fast[fname], "corrector", R0, V0, Rr, Vr, r_L, w0,
                             mlp=mlp)
        gt2 = json.load(open(os.path.join(SD.EXP, "gtable",
                                          "gt2_channels__%s.json" % fname),
                             encoding="utf-8"))
        cmp_ = {}
        for hname in SD.HORIZONS:
            for c in SD.CHANNELS:
                mine = float(rec[hname][c]["primary"][0])
                theirs = float(gt2["runs"]["%s|corrector" % hname][c]["primary"][0])
                d = abs(mine - theirs) / max(abs(theirs), 1e-300)
                worst = max(worst, d)
                cmp_["%s|%s" % (hname, c)] = {"sd1": mine, "gt2": theirs,
                                              "rel_diff": d}
        out[fname] = {"cells": cmp_, "wall_s": time.time() - t0}
        print("  %-12s worst rel diff so far %.3e (%.1fs)"
              % (fname, worst, time.time() - t0), flush=True)
    return {"per_field": out, "worst_relative_difference": worst,
            "requirement": "exact zero: the same code on the same checkpoint"}


def part3_reference_floor():
    """What the reference of `../horizon/` and `../stats/` is worth.

    The trajectory-advantage numbers of Section~\\ref{sec:family} -- 117.8 at
    19.1 gyro-orbits, the crossover at 101, and the five-seed ensemble whose
    median is 135 and whose range is 50 to 447 -- are all scored by
    `../horizon/crossover.py` and `../stats/seed_sweep_b4.py` against a Boris
    run at h/150.  W13 established the discipline: before a residual is
    attributed to a scheme, the reference is put through the same channel.
    That was done there for DOP853 and it has never been done for this one.

    It matters here and not elsewhere because the corrector is TRAINED on that
    reference: its target is the h/150 Boris propagation of the state over one
    working step.  A corrector that learns its target perfectly reproduces the
    h/150 Boris trajectory, not the true one, so the h/150 Boris trajectory is
    both the target and the ruler.

    The closed form is the arbiter.  `../spectral/sw_common.py` solves this
    configuration exactly in Bessel functions and `../map/map_common.py:exact`
    evaluates it; W13 and W15 both price it, and this function prices DOP853
    against it again so that the two floors are read on one scale.
    """
    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    fname = "B4_decaying"
    field, fld = fields[fname], fast[fname]
    R0 = np.array([[1.0, 0.0, 0.0]])
    V0 = np.array([[0.0, 1.0, 0.0]])
    n = G.HORIZONS["H_paper"]
    ts = np.arange(n + 1) * SD.DT
    Rex, _ = MC.exact(fname, field, ts, R0[0], V0[0])

    #: the step of the reference, as a divisor of the working step.  150 is the
    #: one `training/train_corrector_b4.py:DT_FINE`, `../horizon/crossover.py`
    #: and `../stats/seed_sweep_b4.py` all use.
    ladder = [1, 10, 50, 150, 750, 1500]
    rows = {}
    prev = None
    for m in ladder:
        h = SD.DT / m
        nn = n * m
        idx = np.arange(0, nn + 1, m)
        Rs, _, _ = MC.rollout(fld, "boris", R0, V0, h, nn, idx)
        d = np.linalg.norm(Rs[:, 0] - Rex, axis=1)
        rms = float(np.sqrt(np.mean(d ** 2)))
        row = {"h": h, "refinement": m, "rms_vs_closed_form": rms,
               "final_vs_closed_form": float(d[-1])}
        if prev is not None:
            row["fitted_order"] = float(np.log(prev[0] / rms)
                                        / np.log(prev[1] / h))
        rows["h_over_%d" % m] = row
        prev = (rms, h)

    # DOP853 at the reference setting of Table~\ref{tab:family}, on one scale
    from gt1_calibration import reference_orbit
    R0b, V0b = MC.initial_conditions(MC.N_IC)
    Rd, _ = reference_orbit(fname, field, R0b, V0b, ts, tag="paper")
    dd = np.linalg.norm(Rd[:, 0] - Rex, axis=1)

    floor = rows["h_over_150"]["rms_vs_closed_form"]
    boris_work = rows["h_over_1"]["rms_vs_closed_form"]
    return {
        "what": "the trajectory error of the reference itself, in Larmor "
                "radii, over the window of Table tab:family (t = 120, 401 "
                "samples), against the closed form of ../spectral/sw_common.py",
        "boris_ladder": rows,
        "dop853_rtol_1e-12": {"rms_vs_closed_form": float(np.sqrt(np.mean(dd ** 2))),
                              "final_vs_closed_form": float(dd[-1])},
        "reference_of_horizon_and_stats": "Boris at h/150",
        "its_own_rms_error": floor,
        "boris_at_the_working_step": boris_work,
        "gain_at_which_a_report_falls_below_this_floor": boris_work / floor,
        "gain_within_a_factor_of_ten_of_the_floor":
            boris_work / (10.0 * floor),
        "why_it_is_first_order":
            "the shipped Boris step drifts the position over the full step "
            "with the NEW velocity (models/boris.py:boris_step, "
            "r_new = r + v_new dt), which is the synchronized reading.  The "
            "manuscript already reports the fitted order of that reading as "
            "1.31 over the whole step grid and 0.96 over its fine half; the "
            "consequence for a reference built at h/150 is what is measured "
            "here.",
        "consequence": "a corrector trained on the h/150 Boris propagation is "
                       "trained on a trajectory that is this far from the "
                       "true one, and is then scored against the same "
                       "trajectory.  A reported advantage above "
                       "boris_at_the_working_step / its_own_rms_error is a "
                       "corrector error below the error of the ruler.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    md5_before = SD.assert_committed_untouched()
    print("committed checkpoint md5 %s" % md5_before, flush=True)

    print("part 1: Table tab:family on this stand", flush=True)
    p1 = part1_tab_family()
    for s, r in p1["rows"].items():
        print("  %-10s traj %.10e (printed %.2e, rel %.1e)   "
              "energy %.7e (printed %.2e, rel %.1e)"
              % (s, r["trajectory"], r["trajectory_printed"],
                 r["trajectory_rel_diff"], r["energy"], r["energy_printed"],
                 r["energy_rel_diff"]), flush=True)

    print("part 2: the committed checkpoint against ../gtable/", flush=True)
    p2 = part2_against_gt2()

    print("part 3: what the h/150 Boris reference is worth", flush=True)
    p3 = part3_reference_floor()
    for k, r in p3["boris_ladder"].items():
        print("  %-12s h=%.6f  rms %.6e  order %s"
              % (k, r["h"], r["rms_vs_closed_form"],
                 ("%.3f" % r["fitted_order"]) if "fitted_order" in r else "-"),
              flush=True)
    print("  DOP853 rtol 1e-12: rms %.4e"
          % p3["dop853_rtol_1e-12"]["rms_vs_closed_form"], flush=True)
    print("  a reported gain above %.1f is a corrector error below the ruler"
          % p3["gain_at_which_a_report_falls_below_this_floor"], flush=True)

    payload = {
        "meta": {
            "wave": "W16",
            "what": "calibration of the seed-ensemble stand, before any "
                    "retraining exists",
            "committed_checkpoint": os.path.relpath(SD.COMMITTED_CORRECTOR,
                                                    SD.ROOT).replace("\\", "/"),
            "committed_checkpoint_md5": md5_before,
            "committed_seed": SD.COMMITTED_SEED,
            "seed_block": SD.SEED_BLOCK,
            "n_corrector_seeds_declared": SD.N_CORRECTOR_SEEDS,
            "corrector_seeds_declared": SD.CORRECTOR_SEEDS,
            "external_reps_committed": list(SD.EA_REPS_COMMITTED),
            "external_reps_added": list(SD.EA_REPS_NEW),
            "dt": SD.DT, "horizons": SD.HORIZONS,
            "channels": SD.CHANNELS, "fields": SD.FIELD_NAMES,
            "n_initial_conditions": SD.N_IC,
            "n_random_draws_in_this_script": 0,
            "wall_s": time.time() - t0,
        },
        "tab_family": p1,
        "against_gtable": p2,
        "reference_floor": p3,
        "training_cost": SD.corrector_training_flops(),
    }
    SD.assert_committed_untouched()
    return SD.write(OUT, payload, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
