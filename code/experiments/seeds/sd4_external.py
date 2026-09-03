"""SD4: the external architectures, from four repetitions to ten.

    python sd4_external.py                     repetitions 4..9, in order
    python sd4_external.py --shard 0 --of 3    every third repetition
    python sd4_external.py --force             overwrite sd4_external.json

Writes ckpt_ea/<arch>_r<rep>.npz and sd4_external.json.

WHY
---
W9.1 trained the HNN, the SympNet and the physics-informed network at four
repetitions and said so in the caption of Table~\\ref{tab:external}: "each of
the three upper rows is one seed of four".  W12 measured the spread over those
four at a factor of 2.3 to 3.6 in the trajectory channel and observed in its
own Section 6 that a design with four repetitions cannot resolve a difference
below about three -- that is, it cannot resolve its own spread.  Four
repetitions also cannot see a tail: the largest and the smallest of four are
the 12th and the 88th percentile in expectation, so a run twice as bad as the
worst of four is not a surprise, it is unobserved.

WHAT IS RUN
-----------
`external_arch/ea1_train.py` unchanged: `build_states`, `canonicalise`,
`train_one`, `to_stepper`, `score_section7`, at the committed budget of 4400
Adam steps, batch 512, lr 1e-3 under cosine annealing, at the committed widths.
Repetitions 4 to 9 of the same ledger W9.1 used for 0 to 3, so the ten are one
ensemble and not two.  The committed `ea1_training.json` and `ckpt/` of W9.1
are read and never written.

The two capacity controls of W9.1 are NOT rerun.  They answer a different
question -- whether the budget was too small -- and W12 answered it at length
over a seventeen-configuration grid.  This wave buys repetitions, which is the
axis W9.1 itself named as under-resolved.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sd_common as SD                                        # noqa: E402

EXT = os.path.join(SD.EXP, "external_arch")
sys.path.insert(0, EXT)

import ea_common as C                                         # noqa: E402
import ea_arch as A                                           # noqa: E402
import ea1_train as T                                         # noqa: E402

OUT = os.path.join(HERE, "sd4_external.json")
COMMITTED = os.path.join(EXT, "ea1_training.json")

CHANNELS = ("pos_err_rms", "energy_err_median_2nd_half")


def run_rep(rep):
    """One repetition of all three architectures, W9.1's own procedure."""
    t0 = time.time()
    data = T.canonicalise(T.build_states(rep))
    print("data rep%d  %d states  %.1fs" % (rep, data["x"].size,
                                            time.time() - t0), flush=True)
    out = {}
    for arch in SD.EA_ARCHS:
        path = os.path.join(SD.CKPT_EA, "%s_r%d.npz" % (arch, rep))
        info_path = os.path.join(SD.CKPT_EA, "%s_r%d.json" % (arch, rep))
        if os.path.exists(info_path):
            out[arch] = json.load(open(info_path, encoding="utf-8"))
            print("  %-8s rep%d already trained" % (arch, rep), flush=True)
            continue
        model, info = T.train_one(arch, rep, data, verbose=False)
        st = T.to_stepper(arch, model, C.TAU_PAPER)
        info.update(T.score_section7(st))
        info["omega_h_measured"] = A.measure_scheme_frequency(st)
        T.save_stepper(path, arch, model)
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1)
        out[arch] = info
        print("  -> %-8s rep%d  traj %.4e  energy %.4e  %.1fs"
              % (arch, rep, info["pos_err_rms"],
                 info["energy_err_median_2nd_half"], time.time() - t0),
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    # the flop model must still reproduce the figure Section 9 prints
    assert C.mlp_forward_flops([13, 128, 128, 128, 128, 6]) == 113958

    todo = [r for i, r in enumerate(SD.EA_REPS_NEW) if i % a.of == a.shard]
    for rep in todo:
        run_rep(rep)

    missing = [(arch, rep) for rep in SD.EA_REPS_NEW for arch in SD.EA_ARCHS
               if not os.path.exists(os.path.join(SD.CKPT_EA,
                                                  "%s_r%d.json" % (arch, rep)))]
    if missing:
        print("still missing: %s" % missing)
        return 0

    # ---- assemble the ten, four committed and six new -------------------
    com = json.load(open(COMMITTED, encoding="utf-8"))
    runs = {}
    for arch in SD.EA_ARCHS:
        for rep in SD.EA_REPS_COMMITTED:
            r = com["runs"]["%s/rep%d" % (arch, rep)]
            runs["%s/rep%d" % (arch, rep)] = {
                "source": "W9.1 committed (external_arch/ea1_training.json)",
                "seed_init": r["seed_init"], "seed_shuffle": r["seed_shuffle"],
                "n_parameters": r["n_parameters"],
                "flops_per_step": r["flops_per_step"],
                "final_loss": r["final_loss"],
                **{c: r[c] for c in CHANNELS}}
        for rep in SD.EA_REPS_NEW:
            r = json.load(open(os.path.join(SD.CKPT_EA,
                                            "%s_r%d.json" % (arch, rep)),
                               encoding="utf-8"))
            runs["%s/rep%d" % (arch, rep)] = {
                "source": "W16",
                "seed_init": r["seed_init"], "seed_shuffle": r["seed_shuffle"],
                "n_parameters": r["n_parameters"],
                "flops_per_step": r["flops_per_step"],
                "final_loss": r["final_loss"],
                **{c: r[c] for c in CHANNELS}}

    seeds = sorted(runs[k]["seed_init"] for k in runs)
    assert len(set(seeds)) == len(seeds), "a seed is used twice"

    summary = {}
    for arch in SD.EA_ARCHS:
        s = {}
        for c in CHANNELS:
            four = [runs["%s/rep%d" % (arch, r)][c]
                    for r in SD.EA_REPS_COMMITTED]
            ten = [runs["%s/rep%d" % (arch, r)][c]
                   for r in list(SD.EA_REPS_COMMITTED) + list(SD.EA_REPS_NEW)]
            s[c] = {"four_committed": SD.quartiles(four),
                    "ten": SD.quartiles(ten),
                    "values_ten": ten,
                    "spread_widened_by":
                        (SD.quartiles(ten)["ratio_max_min"]
                         / SD.quartiles(four)["ratio_max_min"]),
                    "quoted_row_is_rep0": runs["%s/rep0" % arch][c],
                    "rep0_in_ten": SD.percentile_of(
                        runs["%s/rep0" % arch][c],
                        [v for r, v in zip(
                            list(SD.EA_REPS_COMMITTED) + list(SD.EA_REPS_NEW),
                            ten) if r != 0])}
        summary[arch] = s

    payload = {
        "meta": {
            "wave": "W16",
            "what": "the external architectures of Section sec:external at ten "
                    "repetitions instead of four",
            "procedure": "external_arch/ea1_train.py unchanged: build_states, "
                         "canonicalise, train_one, to_stepper, score_section7",
            "budget": {"adam_steps": C.ADAM_STEPS, "batch": C.BATCH,
                       "lr": C.LR, "schedule": "cosine annealing"},
            "seed_ledger": "external_arch/ea_common.py:seed_of, unchanged; "
                           "W9.1 took repetitions 0..3, W12 took data draws "
                           "10..13, 30..45 and 90, W16 takes repetitions 4..9",
            "reps_committed": list(SD.EA_REPS_COMMITTED),
            "reps_added": list(SD.EA_REPS_NEW),
            "n_repetitions": len(SD.EA_REPS_COMMITTED) + len(SD.EA_REPS_NEW),
            "controls_not_rerun": "the two capacity controls of W9.1 answer "
                                  "the budget question, which W12 answered "
                                  "over a 17-configuration grid; this wave "
                                  "buys repetitions",
            "readout": "score_section7: trajectory root mean square in Larmor "
                       "radii and median relative energy error over the second "
                       "half, against DOP853 rtol 1e-12 atol 1e-14, "
                       "Omega h = 0.3, t = 120",
            "all_seeds_distinct": True,
        },
        "runs": runs,
        "summary": summary,
    }
    return SD.write(OUT, payload, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
