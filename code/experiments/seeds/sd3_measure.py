"""SD3: put every checkpoint of the ensemble through the same measurements.

    python sd3_measure.py            measure, then check against the file
    python sd3_measure.py --force    overwrite sd3_ensemble.json

Writes sd3_ensemble.json.

WHO IS IN THE ENSEMBLE
----------------------
    committed   `checkpoints/boris_corrector_b4.pt`, seed 42 -- the object
                every corrector number in the manuscript stands on.  It is the
                thing being PLACED and is never counted as a member of the
                sample it is placed in.
    W16         sixteen retrainings at seeds 16_000_000 .. 16_000_015
    I1.3        the four retrainings at seeds 1, 7, 123 and 2026 that
                `../stats/seed_sweep_b4.py` already committed, whose
                checkpoints are in `../stats/checkpoints/`.  Its fifth, seed
                42, is a retraining of the committed run at the committed seed
                and is reported as the reproduction check rather than counted
                as an independent draw.

Twenty independent retrainings in all.  Section 7 of the manuscript quotes an
ensemble of five; this is that ensemble continued, in the same ledger, with
the same procedure, not a second one started beside it.

WHAT IS MEASURED, THREE READOUTS
--------------------------------
1.  `paper_recipe` -- `../stats/seed_sweep_b4.py:evaluate` unchanged.  This is
    the statistic Section 7 prints: the trajectory advantage over the Boris
    scheme and the separation between the corrector's energy error and the
    physical signal, both scored against a Boris reference at h/150.  It is
    used because the numbers 117.8, 45.8, "median 135" and "50 to 447" are
    that function's output, and an ensemble that placed them under a different
    convention would be answering a different question.  The fine reference is
    built once and passed to every member.

2.  `tab_family` -- the corrector row of Table~\\ref{tab:family}: B4,
    Omega h = 0.3, 401 samples, against DOP853 at rtol 1e-12.

3.  `channels` and `G` -- the four channels of W15 on all five configurations
    and both horizons, and G = log10(E_Boris / E_corrector) formed from them.
    This is the column of Table~\\ref{tab:gtable} whose caption says it is a
    single committed checkpoint.

THE CLASSICAL SCHEMES ARE MEASURED ONCE
---------------------------------------
Boris, vps2, vps4 and gl4 contain no random draw.  They are run once, their
numbers are recorded under `deterministic`, and their spread over seeds is not
zero-by-measurement but undefined-by-construction.  The report says so in
words; this file says so in a field.
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
import map_common as MC                                       # noqa: E402
import gt_common as G                                         # noqa: E402

STATS = os.path.join(SD.EXP, "stats")
sys.path.insert(0, STATS)

OUT = os.path.join(HERE, "sd3_ensemble.json")

I13_SEEDS = (1, 7, 123, 2026)          # the four independent ones of I1.3
I13_REPRODUCTION_SEED = 42             # its retraining of the committed run


def members():
    """(tag, seed, source, path) for everything that gets measured."""
    out = [("committed", SD.COMMITTED_SEED, "committed",
            SD.COMMITTED_CORRECTOR)]
    for s in SD.CORRECTOR_SEEDS:
        p = SD.seed_ckpt(s)
        if os.path.exists(p):
            out.append(("w16_s%d" % s, s, "W16", p))
    for s in I13_SEEDS:
        p = os.path.join(STATS, "checkpoints", "corrector_b4_seed%d.pt" % s)
        if os.path.exists(p):
            out.append(("i13_s%d" % s, s, "I1.3", p))
    p = os.path.join(STATS, "checkpoints",
                     "corrector_b4_seed%d.pt" % I13_REPRODUCTION_SEED)
    if os.path.exists(p):
        out.append(("i13_reproduce42", I13_REPRODUCTION_SEED,
                    "I1.3 reproduction of the committed seed", p))
    return out


# ---------------------------------------------------------------- readout 1 --
_FINE_REF = None


def paper_recipe(torch_model):
    """`../stats/seed_sweep_b4.py:evaluate`, unchanged, on one model."""
    global _FINE_REF
    import seed_sweep_b4 as SS
    from fields import DecayingField
    from models.boris import integrate_boris
    if _FINE_REF is None:
        field = DecayingField(B0=1.0, tau=SS.TAU_MAIN)
        n_fine = int(round(SS.T_FINAL / SS.DT_FINE))
        _FINE_REF = integrate_boris(np.array([1.0, 0.0, 0.0]),
                                    np.array([0.0, 1.0, 0.0]), 0.0,
                                    SS.DT_FINE, n_fine, field)
    return SS.evaluate(torch_model, ref=_FINE_REF)


# ---------------------------------------------------------------- readout 2 --
def tab_family_row(fld, mlp, R0, V0, r_L, Rr401, Vr401, scheme="corrector"):
    n = G.HORIZONS["H_paper"]
    idx = np.arange(n + 1)
    Rs, Vs, _ = MC.rollout(fld, scheme, R0, V0, SD.DT, n, idx, mlp=mlp)
    ch = G.channel_series(Rs, Vs, Rr401, Vr401, r_L)
    half = (n + 1) // 2
    return {"trajectory": float(np.sqrt(np.mean(ch["trajectory"][:, 0] ** 2))),
            "energy": float(np.median(np.abs(ch["energy"][half:, 0])))}


# ---------------------------------------------------------------- readout 3 --
def channels_of(rec):
    """ic0 and the median over the eight initial conditions, per channel."""
    out = {}
    for h in SD.HORIZONS:
        for c in SD.CHANNELS:
            v = np.asarray(rec[h][c]["primary"], dtype=float)
            out["%s|%s" % (h, c)] = {"ic0": float(v[0]),
                                     "median_over_ic": float(np.median(v))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    SD.assert_committed_untouched()

    import torch
    from training.train_corrector_b4 import DefectNet
    from gt1_calibration import reference_orbit

    mem = members()
    n_ens = sum(1 for m in mem if m[2] in ("W16", "I1.3"))
    print("%d checkpoints, %d of them independent ensemble members"
          % (len(mem), n_ens), flush=True)

    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    R0, V0 = MC.initial_conditions(MC.N_IC)
    N = max(SD.HORIZONS.values())
    ts = np.arange(N) * SD.DT
    ts401 = np.arange(G.HORIZONS["H_paper"] + 1) * SD.DT

    per_field = {}
    for fname in SD.FIELD_NAMES:
        field = fields[fname]
        per_field[fname] = {
            "r_L": MC.larmor_radii(field, R0, V0),
            "w0": G.reference_gyrofrequency(field, R0),
            "ref": SD.reference(fname, field, R0, V0, ts),
        }
    Rr401, Vr401 = reference_orbit("B4_decaying", fields["B4_decaying"],
                                   R0, V0, ts401, tag="paper")
    r_L_b4 = per_field["B4_decaying"]["r_L"]

    # ---- the classical schemes, once ------------------------------------
    print("the deterministic schemes, measured once", flush=True)
    deterministic = {}
    for s in ("boris", "vps2", "vps4", "gl4"):
        d = {}
        for fname in SD.FIELD_NAMES:
            pf = per_field[fname]
            rec = SD.measure_run(fast[fname], s, R0, V0, pf["ref"][0],
                                 pf["ref"][1], pf["r_L"], pf["w0"])
            d[fname] = channels_of(rec)
        deterministic[s] = {
            "channels": d,
            "tab_family": tab_family_row(fast["B4_decaying"], None, R0, V0,
                                         r_L_b4, Rr401, Vr401, scheme=s),
            "spread_over_seeds": "none: the scheme contains no random draw. "
                                 "This is not a measured zero, it is "
                                 "undefined by construction.",
            "flops_per_step": MC.flops_per_step(
                s, 15.0 if s == "gl4" else None),
        }
    boris = deterministic["boris"]["channels"]

    # ---- every checkpoint ------------------------------------------------
    runs = {}
    for tag, seed, source, path in mem:
        t0 = time.time()
        m = DefectNet(n_in=13)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        mlp = SD.load_corrector(path)

        rec = {"seed": seed, "source": source,
               "checkpoint": os.path.relpath(path, SD.ROOT).replace("\\", "/"),
               "md5": SD.md5(path),
               "x_std_dt_input": float(m.x_std.numpy()[12]),
               "paper_recipe": paper_recipe(m),
               "tab_family": tab_family_row(fast["B4_decaying"], mlp, R0, V0,
                                            r_L_b4, Rr401, Vr401)}
        ch = {}
        gg = {}
        for fname in SD.FIELD_NAMES:
            pf = per_field[fname]
            r = SD.measure_run(fast[fname], "corrector", R0, V0, pf["ref"][0],
                               pf["ref"][1], pf["r_L"], pf["w0"], mlp=mlp)
            ch[fname] = channels_of(r)
            gg[fname] = {k: G.G(boris[fname][k]["ic0"], ch[fname][k]["ic0"])
                         for k in ch[fname]}
        rec["channels"] = ch
        rec["G"] = gg
        runs[tag] = rec
        print("  %-18s seed %-9d gain %8.2f  E-sep %7.4f  traj %.4e  (%.1fs)"
              % (tag, seed, rec["paper_recipe"]["traj_gain_projected"],
                 rec["paper_recipe"]["energy_separation_hybrid"],
                 rec["tab_family"]["trajectory"], time.time() - t0),
              flush=True)

    SD.assert_committed_untouched()
    payload = {
        "meta": {
            "wave": "W16",
            "what": "every corrector checkpoint of the ensemble through the "
                    "same three readouts",
            "n_checkpoints": len(mem),
            "n_independent_members": n_ens,
            "committed_is_placed_not_counted": True,
            "readout_1": "../stats/seed_sweep_b4.py:evaluate, unchanged -- the "
                         "statistic Section 7 prints (Boris reference at "
                         "h/150)",
            "readout_2": "the corrector row of Table tab:family (DOP853 rtol "
                         "1e-12, 401 samples)",
            "readout_3": "the four channels of W15 on five configurations and "
                         "two horizons, and G against the Boris scheme",
            "dt": SD.DT, "horizons": SD.HORIZONS, "fields": SD.FIELD_NAMES,
            "channels": SD.CHANNELS, "n_initial_conditions": SD.N_IC,
            "n_random_draws_in_this_script": 0,
            "wall_s": time.time() - t_start,
        },
        "deterministic": deterministic,
        "runs": runs,
    }
    return SD.write(OUT, payload, force=a.force)


if __name__ == "__main__":
    raise SystemExit(main())
