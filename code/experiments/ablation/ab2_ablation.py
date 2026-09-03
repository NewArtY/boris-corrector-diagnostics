"""AB2: the ablation lattice, on the committed checkpoint and on the twenty.

    python ab2_ablation.py            measure, then check against the file
    python ab2_ablation.py --force    overwrite ab2_ablation.json

WHAT IS ABLATED
---------------
The shipped corrector is a Boris step, a network, and a hard constraint on the
network's output.  Six pieces come off, one at a time, and nothing is
retrained -- these are the *inference-time* ablations, on the committed
checkpoint and on the twenty retrainings of W16 and I1.3.  The ablations that
require retraining (a term removed from the loss) are `ab5_loss_ablation.py`,
and the pre-registration says so.

  full          net, orthogonal projection, rescaling            (as shipped)
  no_rescale    the speed-magnitude constraint off, orthogonality kept
  no_ortho      orthogonality off, the speed-magnitude constraint kept
  raw           both off -- the "no projection" row of tab:seeds
  dr0           the position half of the correction set to zero
  dv0           the velocity half set to zero
  net_off       no correction at all; with a zero correction the projection is
                the identity, so this row is exactly the Boris scheme, and
                that is the point: ablating the network *while keeping the
                projection* leaves plain Boris

WHAT IS MEASURED
----------------
`ab_common.evaluate_variant`, which is `../stats/seed_sweep_b4.py:evaluate`
generalised over the lattice and checked against it in `ab1_calibration.py`,
against the **closed form** of B4 -- never the h/150 Boris ruler the corrector
was trained on.  Beside it, the four-channel readout of W15 on the same record,
with the phase taken through `atan2` and never `arccos`.

The classical schemes are run once on the same stand and the same reference,
so that P4 -- "no ablation makes the corrector better than vps4" -- is decided
against a measured vps4 and not against a quoted one.

Writes ab2_ablation.json.  Draws nothing; retrains nothing.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common as AB                                          # noqa: E402
import map_common as MC                                         # noqa: E402
import gt_common as G                                           # noqa: E402

OUT = AB.outpath("ab2_ablation.json")

LATTICE = {
    "full":       AB.Variant("net", ortho=True,  rescale=True),
    "no_rescale": AB.Variant("net", ortho=True,  rescale=False),
    "no_ortho":   AB.Variant("net", ortho=False, rescale=True),
    "raw":        AB.Variant("net", ortho=False, rescale=False),
    "dr0":        AB.Variant("net", ortho=True,  rescale=True, zero_dr=True),
    "dv0":        AB.Variant("net", ortho=True,  rescale=True, zero_dv=True),
    "net_off":    AB.Variant("none", ortho=True, rescale=True),
}

WHAT = {
    "full": "the corrector as shipped",
    "no_rescale": "the speed-magnitude constraint removed, the orthogonal "
                  "projection kept",
    "no_ortho": "the orthogonal projection removed, the speed-magnitude "
                "rescaling kept",
    "raw": "the whole symmetric projection removed -- the 'no projection' row "
           "of tab:seeds",
    "dr0": "the position half of the learned correction set to zero",
    "dv0": "the velocity half of the learned correction set to zero",
    "net_off": "the network removed and the projection kept; a zero "
               "correction passes through the projection unchanged, so this "
               "row is the Boris scheme exactly",
}


def channels_of(rs, vs, Rr, Vr, omega_c):
    """The four-channel readout of W15 on one record, one initial condition."""
    Rs = rs[:, None, :]
    Vs = vs[:, None, :]
    ch = G.channel_series(Rs, Vs, Rr[:, None, :], Vr[:, None, :],
                          np.array([1.0]))
    sm = G.summarise(ch, AB.DT, np.array([float(omega_c)]))
    return {c: {"primary": float(np.asarray(sm[c]["primary"]).ravel()[0]),
                "rms": float(np.asarray(sm[c]["rms"]).ravel()[0])}
            for c in G.CHANNELS}


def main():
    force = "--force" in sys.argv
    t_start = time.time()
    import torch
    torch.set_default_dtype(torch.float64)
    from fields import DecayingField

    AB.assert_committed_untouched()
    field = DecayingField(B0=1.0, tau=AB.TAU)
    fast = MC.FastField("B4_decaying", field)
    ref = AB.closed_form_ref()
    Rr, Vr, ts = ref
    omega_c = float(G.reference_gyrofrequency(field, AB.R0[None, :])[0])

    out = {"meta": {
        "wave": "W17",
        "what": "the inference-time ablation lattice on the committed "
                "checkpoint and the twenty retrainings",
        "reference": "the closed form of B4 on the working grid",
        "field": "B4_decaying, tau=1.2e5", "dt": AB.DT,
        "n_steps": AB.N_WORK, "horizon": AB.T_FINAL,
        "n_random_draws": 0, "nothing_retrained": True,
        "lattice": {k: dict(WHAT[k] and {"what": WHAT[k]},
                            **LATTICE[k].asdict()) for k in LATTICE},
    }}

    # ---- the classical schemes, once -------------------------------------
    idx = np.arange(AB.N_WORK + 1)
    R0b = AB.R0[None, :]
    V0b = AB.V0[None, :]
    E0 = 0.5 * float(np.sum(np.asarray(AB.V0) ** 2))
    E_ref = 0.5 * np.sum(Vr ** 2, axis=1)
    half = AB.N_WORK // 2
    classical = {}
    for s in ("boris", "vps2", "vps4", "gl4"):
        Rs, Vs, meta = MC.rollout(fast, s, R0b, V0b, AB.DT, AB.N_WORK, idx)
        rs, vs = Rs[:, 0, :], Vs[:, 0, :]
        pos = np.linalg.norm(rs - Rr, axis=1)
        e = 0.5 * np.sum(vs ** 2, axis=1)
        e_err = np.abs(e - E_ref) / E0
        classical[s] = {
            "pos_err_rms": float(np.sqrt(np.mean(pos ** 2))),
            "pos_err_final": float(pos[-1]),
            "energy_err_median_2nd_half": float(np.median(e_err[half:])),
            "channels": channels_of(rs, vs, Rr, Vr, omega_c),
            "flops_per_step": MC.flops_per_step(s, meta.get("mean_iters")),
        }
    phys = float(np.median(np.abs((E_ref - E_ref[0]) / E0)[half:]))
    for s in classical:
        classical[s]["energy_separation"] = (
            phys / classical[s]["energy_err_median_2nd_half"])
        classical[s]["traj_gain_over_boris"] = (
            classical["boris"]["pos_err_rms"] / classical[s]["pos_err_rms"])
    out["classical"] = classical
    out["physical_signal_median"] = phys

    # the Boris row has to be the same object on both stands
    b_a = classical["boris"]["pos_err_rms"]
    b_b = AB.evaluate_variant({}, model=None, ref=ref)["boris"]["pos_err_rms"]
    out["meta"]["boris_agreement_between_stands"] = {
        "map_common_rollout": b_a, "models_boris_integrate": b_b,
        "rel_diff": float(abs(b_a - b_b) / b_b)}
    if abs(b_a - b_b) / b_b > 1e-12:
        print("the two Boris implementations disagree: %.3e" % (abs(b_a - b_b) / b_b))
        return 1

    # ---- the lattice, on every checkpoint ---------------------------------
    members = AB.ensemble_members()
    n_ens = sum(1 for m in members if m[2] in ("W16", "I1.3"))
    print("%d checkpoints, %d of them independent ensemble members"
          % (len(members), n_ens), flush=True)

    runs = {}
    for tag, seed, source, path in members:
        t0 = time.time()
        model = AB.load_torch(path)
        res = AB.evaluate_variant(LATTICE, model=model, ref=ref)
        rec = {"seed": seed, "source": source, "md5": AB.md5(path),
               "variants": {}}
        for name, var in LATTICE.items():
            rr, vv, tt, _ = AB.integrate_variant(field, AB.R0, AB.V0, AB.DT,
                                                 AB.N_WORK, model, var)
            rec["variants"][name] = dict(res[name])
            rec["variants"][name]["channels"] = channels_of(rr, vv, Rr, Vr,
                                                            omega_c)
        runs[tag] = rec
        print("  %-18s  full E-sep %7.4f  raw %7.4f  gain %8.2f  (%.1fs)"
              % (tag, rec["variants"]["full"]["energy_separation"],
                 rec["variants"]["raw"]["energy_separation"],
                 rec["variants"]["full"]["traj_gain_over_boris"],
                 time.time() - t0), flush=True)
    out["runs"] = runs

    # ---- the table --------------------------------------------------------
    ens = [t for t in runs if t != "committed"]
    assert len(ens) == 20, "the ensemble is twenty, got %d" % len(ens)
    table = {}
    for name in LATTICE:
        row = {"what": WHAT[name], "variant": LATTICE[name].asdict()}
        for q in ("energy_separation", "traj_gain_over_boris", "pos_err_rms",
                  "energy_err_median_2nd_half", "constraint_max_abs",
                  "constraint_max_rel"):
            vals = [runs[t]["variants"][name][q] for t in ens]
            row[q] = {"committed": runs["committed"]["variants"][name][q],
                      "ensemble": AB.summarise(vals),
                      "placed": AB.place(
                          runs["committed"]["variants"][name][q], vals)}
        for c in G.CHANNELS:
            vals = [runs[t]["variants"][name]["channels"][c]["primary"]
                    for t in ens]
            row["channel_" + c] = {
                "committed":
                    runs["committed"]["variants"][name]["channels"][c]["primary"],
                "ensemble": AB.summarise(vals)}
        table[name] = row
    out["table"] = table

    # ---- what each ablation costs, relative to the shipped corrector ------
    base = table["full"]
    loss = {}
    for name in LATTICE:
        r = {}
        for q, better in (("energy_separation", "higher"),
                          ("traj_gain_over_boris", "higher"),
                          ("pos_err_rms", "lower"),
                          ("energy_err_median_2nd_half", "lower")):
            c0 = base[q]["committed"]
            c1 = table[name][q]["committed"]
            m0 = base[q]["ensemble"]["median"]
            m1 = table[name][q]["ensemble"]["median"]
            r[q] = {"better_is": better,
                    "committed_ratio_to_full": (c1 / c0) if c0 else None,
                    "ensemble_median_ratio_to_full": (m1 / m0) if m0 else None,
                    "committed": c1, "full": c0}
        loss[name] = r
    out["loss_relative_to_full"] = loss

    # ---- P1: the network against the projection ---------------------------
    # "ablate the network, keep the projection" is `net_off`; "ablate the
    # projection, keep the network" is `raw`.  The comparison is made on both
    # channels and stated whichever way it falls.
    p1 = {}
    for q, better in (("energy_separation", "higher"),
                      ("traj_gain_over_boris", "higher")):
        full = base[q]["committed"]
        net_off = table["net_off"][q]["committed"]
        proj_off = table["raw"][q]["committed"]
        p1[q] = {
            "full": full,
            "network_ablated_projection_kept": net_off,
            "projection_ablated_network_kept": proj_off,
            "loss_from_ablating_the_network": (full / net_off) if net_off else None,
            "loss_from_ablating_the_projection": (full / proj_off) if proj_off else None,
            "ensemble_median": {
                "full": base[q]["ensemble"]["median"],
                "network_ablated": table["net_off"][q]["ensemble"]["median"],
                "projection_ablated": table["raw"][q]["ensemble"]["median"]},
            "reproducibility_sd_over_20": {
                "full": base[q]["ensemble"]["sd"],
                "network_ablated": table["net_off"][q]["ensemble"]["sd"],
                "projection_ablated": table["raw"][q]["ensemble"]["sd"]},
        }
    out["P1"] = p1

    # ---- P4: does any ablation beat vps4 ----------------------------------
    p4 = {}
    for name in LATTICE:
        p4[name] = {
            "traj_ratio_to_vps4":
                table[name]["pos_err_rms"]["committed"]
                / classical["vps4"]["pos_err_rms"],
            "traj_ratio_to_vps4_best_of_ensemble":
                table[name]["pos_err_rms"]["ensemble"]["min"]
                / classical["vps4"]["pos_err_rms"],
            "energy_ratio_to_vps4":
                table[name]["energy_err_median_2nd_half"]["committed"]
                / classical["vps4"]["energy_err_median_2nd_half"],
            "beats_vps4_on_trajectory_at_equal_step":
                bool(table[name]["pos_err_rms"]["committed"]
                     < classical["vps4"]["pos_err_rms"]),
        }
    out["P4"] = {"vps4": classical["vps4"], "per_variant": p4,
                 "any_variant_beats_vps4": bool(
                     any(v["beats_vps4_on_trajectory_at_equal_step"]
                         for v in p4.values())),
                 "any_member_of_any_variant_beats_vps4": bool(
                     any(v["traj_ratio_to_vps4_best_of_ensemble"] < 1.0
                         for v in p4.values()))}

    out["meta"]["wall_s"] = time.time() - t_start
    AB.assert_committed_untouched()
    AB.assert_no_draws(0)
    rc = AB.write(OUT, out, force=force)

    # ---- the printed table ------------------------------------------------
    print("\n%-12s %14s %14s %13s %13s"
          % ("variant", "E-separation", "traj gain", "traj rms",
             "constraint"))
    for name in LATTICE:
        r = table[name]
        print("%-12s %14.4f %14.2f %13.4e %13.2e"
              % (name, r["energy_separation"]["committed"],
                 r["traj_gain_over_boris"]["committed"],
                 r["pos_err_rms"]["committed"],
                 r["constraint_max_abs"]["committed"]))
    print("%-12s %14.4f %14.2f %13.4e" % ("vps4",
                                          classical["vps4"]["energy_separation"],
                                          classical["vps4"]["traj_gain_over_boris"],
                                          classical["vps4"]["pos_err_rms"]))
    print("%-12s %14.4f %14.2f %13.4e" % ("vps2",
                                          classical["vps2"]["energy_separation"],
                                          classical["vps2"]["traj_gain_over_boris"],
                                          classical["vps2"]["pos_err_rms"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
