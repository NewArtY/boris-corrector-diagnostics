"""rc3_seeds.py -- the twenty retrainings of W16, read against the closed form.

Table~\\ref{tab:seeds} reports the trajectory advantage of the corrector over
twenty independent retrainings -- committed 117.8, median 152.8, interquartile
range 95.7 to 252.2, the committed run at the fortieth percentile.  Every one
of those numbers is `../stats/seed_sweep_b4.py:evaluate` scored against a Boris
reference at h/150, which is the surface the corrector was trained to reach.

This script puts the same twenty checkpoints, plus the committed one, through
the same function with the reference replaced by the closed form.  Nothing is
retrained: the checkpoints are the committed file and the twenty of W16 and
I1.3, read and not written, with their md5 checked.

`evaluate` is called unchanged.  It takes its reference as an argument, so
swapping the ruler needs no edit to any committed file:

    evaluate(model, ref=(rs, vs, ts))

with (rs, vs, ts) the closed form on the working grid.  `evaluate` interpolates
the reference onto the working times; on the working grid that interpolation is
the identity, so no interpolation error is introduced.  The old ruler is run
alongside, from the same function, and must reproduce `../seeds/sd3_ensemble.json`
to 1e-9 before any new number is reported.

Writes rc3_seeds.json.  Draws nothing: every member is a committed checkpoint.
Usage: python rc3_seeds.py [--force]
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import rc_common as RC
from rc_common import check_or_write

torch.set_default_dtype(torch.float64)

SEEDS_DIR = os.path.join(RC.EXP, "seeds")
STATS_DIR = os.path.join(RC.EXP, "stats")
for _p in (SEEDS_DIR, STATS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sd_common as SD                                   # noqa: E402
import seed_sweep_b4 as SS                               # noqa: E402
from fields import DecayingField                         # noqa: E402
from models.boris import integrate_boris                 # noqa: E402
from training.train_corrector_b4 import DefectNet        # noqa: E402

OUT = RC.outpath("rc3_seeds.json")
COMMITTED_SD3 = os.path.join(SEEDS_DIR, "sd3_ensemble.json")
COMMITTED_SD5 = os.path.join(SEEDS_DIR, "sd5_summary.json")

I13_SEEDS = (1, 7, 123, 2026)
I13_REPRODUCTION_SEED = 42

#: the same statistic W16 declared before its own first run
STATS = ("median", "q1", "q3", "min", "max")


def members():
    """(tag, seed, source, path) -- `sd3_measure.py:members`, same order."""
    out = [("committed", SD.COMMITTED_SEED, "committed", SD.COMMITTED_CORRECTOR)]
    for s in SD.CORRECTOR_SEEDS:
        p = SD.seed_ckpt(s)
        if os.path.exists(p):
            out.append(("w16_s%d" % s, s, "W16", p))
    for s in I13_SEEDS:
        p = os.path.join(STATS_DIR, "checkpoints", "corrector_b4_seed%d.pt" % s)
        if os.path.exists(p):
            out.append(("i13_s%d" % s, s, "I1.3", p))
    p = os.path.join(STATS_DIR, "checkpoints",
                     "corrector_b4_seed%d.pt" % I13_REPRODUCTION_SEED)
    if os.path.exists(p):
        out.append(("i13_reproduce42", I13_REPRODUCTION_SEED,
                    "I1.3 reproduction of the committed seed", p))
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def summarise(values):
    v = np.asarray(sorted(values), dtype=float)
    return {"n": int(v.size),
            "median": float(np.percentile(v, 50)),
            "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max()),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "ratio_max_min": float(v.max() / v.min()) if v.min() > 0 else None,
            "iqr_ratio_q3_q1": float(np.percentile(v, 75)
                                     / np.percentile(v, 25))
            if np.percentile(v, 25) > 0 else None}


def place(committed_value, values):
    """W16's convention: counted strictly below, never a member of the sample."""
    v = np.asarray(values, dtype=float)
    n_below = int((v < committed_value).sum())
    return {"n": int(v.size), "value": float(committed_value),
            "n_below": n_below, "n_above": int((v > committed_value).sum()),
            "percentile_below": 100.0 * n_below / v.size,
            "rank_best_is_1": int((v > committed_value).sum()) + 1}


def main():
    force = "--force" in sys.argv
    with open(COMMITTED_SD3, encoding="utf-8") as fh:
        sd3 = json.load(fh)
    with open(COMMITTED_SD5, encoding="utf-8") as fh:
        sd5 = json.load(fh)

    out = {"meta": {
        "what": "Table tab:seeds, the twenty retrainings, on the closed form",
        "n_random_draws": 0,
        "nothing_retrained": True,
        "evaluate": "../stats/seed_sweep_b4.py:evaluate, unchanged, with its "
                    "`ref` argument set to the closed form",
        "old_ruler": "Boris at h/150 through models/boris.py:integrate_boris",
        "new_ruler": "../spectral/sw_common.py closed form on the working grid",
    }}

    # ------------------------------------------------------------- the rulers
    field = DecayingField(B0=1.0, tau=SS.TAU_MAIN)
    n_fine = int(round(SS.T_FINAL / SS.DT_FINE))
    n_work = int(round(SS.T_FINAL / SS.DT_WORK))
    t0 = time.time()
    ref_old = integrate_boris(np.array([1.0, 0.0, 0.0]),
                              np.array([0.0, 1.0, 0.0]), 0.0,
                              SS.DT_FINE, n_fine, field)
    t_old = time.time() - t0

    ts_work = np.arange(n_work + 1) * SS.DT_WORK
    t0 = time.time()
    R_ex, V_ex = RC.closed_form(ts_work)
    ref_new = (R_ex, V_ex, ts_work)
    t_new = time.time() - t0

    d = np.linalg.norm(ref_old[0] - RC.closed_form(ref_old[2])[0], axis=1)
    out["ruler"] = {
        "old_seconds": t_old, "new_seconds": t_new,
        "old_n_steps": n_fine, "new_n_samples": int(n_work + 1),
        "old_own_rms_error_vs_closed_form": RC.rms(d),
        "old_own_max_error_vs_closed_form": float(d.max()),
        "w16_reference_floor": sd5["reference_floor"]["its_own_rms_error"],
        "cost_flops_old": RC.flops_boris_reference(n_work),
        "cost_flops_new": RC.flops_closed_form(n_work + 1),
    }
    out["ruler"]["cost_ratio"] = (out["ruler"]["cost_flops_old"]
                                  / out["ruler"]["cost_flops_new"])
    print("ruler: Boris h/150 carries %.4e Larmor radii of its own error over "
          "the window (W16: %.4e); the closed form costs %.0fx fewer flops"
          % (out["ruler"]["old_own_rms_error_vs_closed_form"],
             out["ruler"]["w16_reference_floor"], out["ruler"]["cost_ratio"]))

    # ------------------------------------------------------------- the members
    rows = {}
    bad = []
    print("\n%-20s %10s %12s %12s %9s" % ("member", "seed", "gain old",
                                          "gain new", "shift"))
    for tag, seed, source, path in members():
        m = load(path)
        r_old = SS.evaluate(m, ref=ref_old)
        r_new = SS.evaluate(m, ref=ref_new)
        want = sd3["runs"][tag]["paper_recipe"]
        rel = abs(r_old["traj_gain_projected"] - want["traj_gain_projected"]) \
            / abs(want["traj_gain_projected"])
        if rel > 1e-9:
            bad.append((tag, rel))
        rows[tag] = {
            "seed": seed, "source": source, "md5": md5(path),
            "old": r_old, "new": r_new,
            "committed_file_old": want,
            "calibration_rel_diff": float(rel),
            "traj_gain_shift": RC.rel(r_new["traj_gain_projected"],
                                      r_old["traj_gain_projected"]),
            "corrector_error_shift": RC.rel(
                r_new["corrector_projected"]["pos_err_rms"],
                r_old["corrector_projected"]["pos_err_rms"]),
        }
        print("%-20s %10d %12.4f %12.4f %9.4f"
              % (tag, seed, r_old["traj_gain_projected"],
                 r_new["traj_gain_projected"],
                 r_new["traj_gain_projected"] / r_old["traj_gain_projected"]))
    out["members"] = rows
    out["calibration_failures"] = [{"member": t, "rel_diff": r} for t, r in bad]
    if bad:
        print("\nCALIBRATION FAILED: %d members do not reproduce "
              "sd3_ensemble.json" % len(bad))
        return 1
    print("\ncalibration: all %d members reproduce sd3_ensemble.json to 1e-9 "
          "on the old ruler" % len(rows))

    # ------------------------------------------------------------ the ensemble
    ens_tags = [t for t in rows if t not in ("committed", "i13_reproduce42")]
    assert len(ens_tags) == 20, "the ensemble is twenty, got %d" % len(ens_tags)

    stat = {}
    for field_name, getter in (
            ("traj_gain", lambda r: r["traj_gain_projected"]),
            ("corrector_traj_error",
             lambda r: r["corrector_projected"]["pos_err_rms"]),
            ("energy_separation_hybrid",
             lambda r: r["energy_separation_hybrid"])):
        block = {}
        for ruler in ("old", "new"):
            vals = [getter(rows[t][ruler]) for t in ens_tags]
            block[ruler] = {"ensemble": summarise(vals),
                            "committed": place(getter(rows["committed"][ruler]),
                                               vals),
                            "values": {t: getter(rows[t][ruler])
                                       for t in ens_tags}}
        block["shift"] = {
            k: RC.rel(block["new"]["ensemble"][k], block["old"]["ensemble"][k])
            for k in STATS}
        block["shift"]["committed"] = RC.rel(
            block["new"]["committed"]["value"],
            block["old"]["committed"]["value"])
        block["shift"]["percentile_below"] = RC.rel(
            block["new"]["committed"]["percentile_below"],
            block["old"]["committed"]["percentile_below"])
        stat[field_name] = block
    out["ensemble"] = stat

    # what the manuscript prints, next to what it becomes
    tg = stat["traj_gain"]
    out["manuscript_row"] = {
        "table": "tab:seeds, row 'trajectory advantage over Boris'",
        "printed": {"committed": 117.8, "median": 152.8,
                    "q1": 95.7, "q3": 252.2, "n_below": 8},
        "old_ruler_recomputed": {
            "committed": tg["old"]["committed"]["value"],
            "median": tg["old"]["ensemble"]["median"],
            "q1": tg["old"]["ensemble"]["q1"],
            "q3": tg["old"]["ensemble"]["q3"],
            "n_below": tg["old"]["committed"]["n_below"]},
        "closed_form": {
            "committed": tg["new"]["committed"]["value"],
            "median": tg["new"]["ensemble"]["median"],
            "q1": tg["new"]["ensemble"]["q1"],
            "q3": tg["new"]["ensemble"]["q3"],
            "n_below": tg["new"]["committed"]["n_below"]},
    }
    out["manuscript_row"]["committed_sd5"] = sd5["paper_recipe"]["traj_gain"]

    print("\n=== tab:seeds, trajectory advantage: was and now ===")
    print("%-14s %14s %14s %10s" % ("statistic", "old ruler", "closed form",
                                    "x shift"))
    for k in ("committed",) + STATS:
        s = tg["shift"][k]
        print("%-14s %14.4f %14.4f %10.4f" % (k, s["old"], s["new"],
                                              s["ratio"]))
    print("%-14s %14d %14d" % ("n below", tg["old"]["committed"]["n_below"],
                               tg["new"]["committed"]["n_below"]))
    print("%-14s %14.1f %14.1f" % ("percentile",
                                   tg["old"]["committed"]["percentile_below"],
                                   tg["new"]["committed"]["percentile_below"]))

    ce = stat["corrector_traj_error"]
    print("\n=== the corrector's own trajectory error over the twenty ===")
    for k in STATS:
        s = ce["shift"][k]
        print("%-14s %14.6e %14.6e %10.4f" % (k, s["old"], s["new"],
                                              s["ratio"]))
    print("%-14s %14.6e %14.6e %10.4f"
          % ("committed", ce["shift"]["committed"]["old"],
             ce["shift"]["committed"]["new"], ce["shift"]["committed"]["ratio"]))
    print("\nthe manuscript's independent-ruler row of the same table "
          "(DOP853 rtol 1e-12) prints median 2.91e-3, range 2.29e-3 to "
          "4.02e-3; the closed form gives median %.3e, min %.3e, max %.3e"
          % (ce["new"]["ensemble"]["median"], ce["new"]["ensemble"]["min"],
             ce["new"]["ensemble"]["max"]))

    # how many members sit below the old ruler's own error
    floor = out["ruler"]["old_own_rms_error_vs_closed_form"]
    below = {r: sum(1 for t in ens_tags
                    if rows[t][r]["corrector_projected"]["pos_err_rms"] < floor)
             for r in ("old", "new")}
    out["members_below_the_ruler_floor"] = {
        "floor": floor, "old_ruler": below["old"], "new_ruler": below["new"],
        "w16_reported": sd5["paper_recipe"]["members_reported_below_the_ruler"]}
    print("\nmembers whose reported error is below the ruler's own error: "
          "%d on the old ruler (W16 said %d), %d on the closed form"
          % (below["old"],
             sd5["paper_recipe"]["members_reported_below_the_ruler"],
             below["new"]))

    RC.assert_no_draws(0)
    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
