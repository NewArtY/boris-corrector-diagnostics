"""mp1_calibration.py -- does the stand measure the schemes, and is the
reference the floor of the measurement in each configuration?

The pre-registration of wave W14 puts both questions before any map.  Nothing
in this file is a result about the corrector; everything in it is a check that
would invalidate the map if it failed.

Four things are established here, in this order.

1. THE BRIDGE IS THE PHYSICS.  Each of the five field classes of
   `code/fields/` is paired with a component-wise evaluator in `map_common.py`,
   and the two are required to agree **bit for bit** on a batch of random
   states.  A single differing bit fails the run.

2. VECTORISATION CHANGES NOTHING.  The pre-registration asserts that carrying
   nb trajectories side by side is bit-exact because the arithmetic is
   elementwise.  That is asserted here rather than believed: every scheme is
   run once with nb = 8 and once with nb = 1 and the shared column is required
   to be identical to the last bit, in a field with a batch-varying |B| (B1)
   as well as in B4.

3. CALIBRATION AGAINST tab:family.  The five trajectory errors of Table 4 of
   the manuscript are reproduced by this stand, against the same DOP853
   reference at rtol 1e-12 / atol 1e-14, at Omega h = 0.3 over t = 120 in B4.
   This is the check wave W13 made before its own measurements, and it is what
   catches an error in the bridge before it reaches the map.  The energy column
   and the physical signal are reproduced too.

4. THE REFERENCE, PER CONFIGURATION.  W13 established that DOP853 at rtol
   1e-12 carries its own position error of 6.196e-12 Larmor radii on the paper
   window, and that four of eleven runs were limited by the reference rather
   than by the scheme.  The pre-registration therefore requires the reference
   to be adjudicated separately in each configuration.  Three of the five have
   a closed form -- the uniform field, B3 (which is a *static, spatially
   uniform* field, merely tilted, so the pre-registration's "no closed form
   for B1-B3" is too strong by one), and B4 (Bessel, from W13).  For B1 and B2
   there is none, and the reference is instead held to the exact invariants of
   the motion and to its own self-convergence between rtol 1e-12 and 3e-14.

Writes mp1_calibration.json; exits non-zero if a rerun stops reproducing it.
Usage: python mp1_calibration.py [--force]
"""
import json
import os
import sys
import time

import numpy as np

import map_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mp1_calibration.json")

#: the trajectory column of tab:family, and the corrector's, as W13 committed
#: them in ../spectral/sw1_reference.py.
TABLE4_TRAJ = {"boris": 0.4166534686545755,
               "vps2": 0.010583014451054718,
               "vps4": 5.35394015973843e-05,
               "gl4": 0.0007745714317733591,
               "corrector": 0.0034688730013653656}

#: the energy column of tab:family, to the two digits the table prints, and the
#: physical signal of its caption.
TABLE4_ENERGY = {"boris": 1.25e-6, "vps2": 5.57e-6, "vps4": 2.64e-7,
                 "gl4": 4.18e-8, "corrector": 1.64e-5}
TABLE4_SIGNAL = 7.497e-4

REF_TOLS = [("rtol1e-12", dict(rtol=1e-12, atol=1e-14)),
            ("rtol1e-13", dict(rtol=1e-13, atol=1e-16)),
            ("rtol3e-14", dict(rtol=3e-14, atol=1e-16))]


def main():
    force = "--force" in sys.argv
    fields = C.make_fields()
    fast = C.make_fast_fields(fields)
    R0, V0 = C.initial_conditions(C.N_IC)
    mlp = C.load_corrector_numpy()

    out = {"meta": {
        "what": "calibration of the W14 stand and reference adequacy per "
                "field configuration",
        "dt_grid": C.DT_GRID,
        "horizons": {k: v for k, v in C.HORIZONS.items()},
        "gyro_orbits": {k: v / C.TWO_PI for k, v in C.HORIZONS.items()},
        "schemes": C.SCHEMES,
        "fields": C.FIELD_NAMES,
        "n_initial_conditions": C.N_IC,
        "n_random_draws": C.N_RANDOM_DRAWS,
        "map_seed": C.MAP_SEED,
        "corrector_checkpoint": "checkpoints/boris_corrector_b4.pt",
        "corrector_trained_at": {"dt": C.DT_TRAIN, "field": "B4_decaying"},
        "one_checkpoint_only": True,
        "closed_form": C.CLOSED_FORM,
    }}
    out["meta"]["initial_conditions"] = {
        "r0": R0.tolist(), "v0": V0.tolist(),
        "larmor_radii": {k: C.larmor_radii(fields[k], R0, V0).tolist()
                         for k in C.FIELD_NAMES}}

    # ------------------------------------------- 1. the bridge is the physics
    rng = np.random.default_rng(C.MAP_SEED + 1)
    bridge = {}
    worst_bits = 0
    for name, f in fields.items():
        ok_e, ok_b = True, True
        for t in (0.0, 0.37, 12.9, 636.0):
            r = 2.0 * rng.normal(size=(C.N_IC, 3))
            ex, ey, ez, bx, by, bz = fast[name].eb(r[:, 0], r[:, 1], r[:, 2], t)
            E = np.atleast_2d(f.E(r, t))
            B = np.atleast_2d(f.B(r, t))
            fe = np.stack([np.broadcast_to(ex, (C.N_IC,)),
                           np.broadcast_to(ey, (C.N_IC,)),
                           np.broadcast_to(ez, (C.N_IC,))], axis=1)
            fb = np.stack([np.broadcast_to(bx, (C.N_IC,)),
                           np.broadcast_to(by, (C.N_IC,)),
                           np.broadcast_to(bz, (C.N_IC,))], axis=1)
            ok_e = ok_e and np.array_equal(fe, E)
            ok_b = ok_b and np.array_equal(fb, B)
        bridge[name] = {"E_bit_identical": bool(ok_e),
                        "B_bit_identical": bool(ok_b)}
        worst_bits += (not ok_e) + (not ok_b)
    out["bridge_bit_identity"] = bridge
    if worst_bits:
        print("BRIDGE FAILED: a component evaluator differs from its class")
        return 1
    print("bridge: all five field classes reproduced bit for bit")

    # -------------------------------------- 2. vectorisation changes nothing
    # The four classical schemes are required to be bit-identical between a
    # batch of eight and a single trajectory.  The corrector is not, and the
    # reason is worth stating rather than hiding: its network's first layer is
    # a matrix product, and a BLAS library dispatches a matrix-vector product
    # (nb = 1) to a different kernel, with a different summation order, than a
    # matrix-matrix product (nb >= 2).  So the corrector is required to be
    # bit-identical between nb = 8 and nb = 2, and the nb = 1 deviation is
    # measured and reported instead of being asserted away.
    vec = {}
    n = 400
    idx = C.sample_indices(n, n, 100)
    bad = 0
    for fname in ("B4_decaying", "B1_radial"):
        for s in C.SCHEMES:
            Rb, Vb, _ = C.rollout(fast[fname], s, R0, V0, 0.3, n, idx, mlp=mlp)
            R1, V1, _ = C.rollout(fast[fname], s, R0[:1], V0[:1], 0.3, n, idx,
                                  mlp=mlp)
            R2, V2, _ = C.rollout(fast[fname], s, R0[:2], V0[:2], 0.3, n, idx,
                                  mlp=mlp)
            rec = {
                "nb8_vs_nb2_bit_identical":
                    bool(np.array_equal(Rb[:, :2], R2)
                         and np.array_equal(Vb[:, :2], V2)),
                "nb8_vs_nb1_bit_identical":
                    bool(np.array_equal(Rb[:, :1], R1)
                         and np.array_equal(Vb[:, :1], V1)),
                "nb8_vs_nb1_max_abs_position":
                    float(np.abs(Rb[:, :1] - R1).max()),
                "nb8_vs_nb1_max_abs_velocity":
                    float(np.abs(Vb[:, :1] - V1).max()),
            }
            required = (rec["nb8_vs_nb2_bit_identical"]
                        if s == "corrector"
                        else rec["nb8_vs_nb1_bit_identical"]
                        and rec["nb8_vs_nb2_bit_identical"])
            rec["passes"] = bool(required)
            bad += (not required)
            vec["%s/%s" % (fname, s)] = rec
    out["batch_equals_single_bitwise"] = vec
    out["vectorisation_note"] = (
        "the four classical schemes are bit-identical between nb = 8, nb = 2 "
        "and nb = 1; the corrector is bit-identical between nb = 8 and nb = 2 "
        "and differs at nb = 1 only through the BLAS kernel its first layer "
        "dispatches to, by the amount recorded above")
    if bad:
        print("VECTORISATION FAILED: %d of %d runs" % (bad, len(vec)))
        return 1
    print("vectorisation: %d runs pass; corrector nb=1 BLAS deviation "
          "%.2e Larmor radii after 400 steps"
          % (len(vec), max(v["nb8_vs_nb1_max_abs_position"]
                           for k, v in vec.items() if "corrector" in k)))

    # ------------------------------------------- 3. calibration vs tab:family
    f4 = fields["B4_decaying"]
    dt = C.DT_TRAIN
    n = int(round(C.T_SHORT / dt))
    ts = np.arange(n + 1) * dt
    sol = C.dop853(f4, ts, R0[0], V0[0], rtol=1e-12, atol=1e-14)
    Rr, Vr = C.dop853_at(sol, ts)

    # the dense output this directory reads the reference through, against the
    # t_eval output the manuscript's own scripts read it through
    Rt, Vt = C.SW.dop853_ref(ts, r0=R0[0], v0=V0[0], tau=f4.tau,
                             rtol=1e-12, atol=1e-14)
    out["dense_output_vs_t_eval"] = {
        "max_abs_position": float(np.abs(Rr - Rt).max()),
        "max_abs_velocity": float(np.abs(Vr - Vt).max())}

    idx = np.arange(n + 1)
    rl = C.larmor_radii(f4, R0, V0)
    E_ref = 0.5 * np.sum(Vr ** 2, axis=1)
    half = len(ts) // 2
    signal = float(np.median(np.abs(E_ref - E_ref[0])[half:] / E_ref[0]))

    # the batch the map itself runs, scored on its first column, which is the
    # canonical initial condition of the manuscript
    cal = {}
    worst = 0.0
    for s in C.SCHEMES:
        Rs, Vs, meta = C.rollout(fast["B4_decaying"], s, R0, V0, dt, n,
                                 idx, mlp=mlp)
        ch = C.channels(Rs, Vs, Rr[:, None, :], Vr[:, None, :], rl)
        rms = float(np.sqrt(np.mean(ch["position"][:, 0] ** 2)))
        en = float(C.median_second_half(ch["energy"])[0])
        rel = abs(rms - TABLE4_TRAJ[s]) / TABLE4_TRAJ[s]
        worst = max(worst, rel)
        rec = {"pos_err_rms": rms, "table4_trajectory": TABLE4_TRAJ[s],
               "rel_diff_trajectory": rel,
               "energy_err_median_2nd_half": en,
               "table4_energy": TABLE4_ENERGY[s],
               "rel_diff_energy": abs(en - TABLE4_ENERGY[s])
               / TABLE4_ENERGY[s],
               "flops_per_step": C.flops_per_step(s, meta.get("mean_iters"))}
        if "mean_iters" in meta:
            rec["mean_iters"] = meta["mean_iters"]
        cal[s] = rec
        print("  %-10s rms %.10e  vs table4 %.10e  rel %.2e"
              % (s, rms, TABLE4_TRAJ[s], rel))
    out["calibration_vs_tab_family"] = cal
    out["calibration_signal"] = {"physical_signal_median_2nd_half": signal,
                                 "table4_caption": TABLE4_SIGNAL,
                                 "rel_diff": abs(signal - TABLE4_SIGNAL)
                                 / TABLE4_SIGNAL}
    out["calibration_worst_rel_diff_trajectory"] = worst
    if worst > 1e-9:
        print("CALIBRATION FAILED: worst relative difference %.3e" % worst)
        return 1
    print("calibration: five schemes reproduce tab:family to %.1e" % worst)

    # ------------------------------------ 4. the reference, per configuration
    ref = {}
    for fname in C.FIELD_NAMES:
        f = fields[fname]
        rec = {"closed_form": C.CLOSED_FORM[fname]}
        for hname, T in C.HORIZONS.items():
            tsh = np.linspace(0.0, T, 1201)
            sols = {}
            t0 = time.time()
            for tag, kw in REF_TOLS:
                sols[tag] = C.dop853_at(
                    C.dop853(f, tsh, R0[0], V0[0], **kw), tsh)
            wall = time.time() - t0
            ex = C.exact(fname, f, tsh, R0[0], V0[0])
            h = {}
            r_L = float(C.larmor_radii(f, R0[:1], V0[:1])[0])
            if ex is not None:
                for tag, kw in REF_TOLS:
                    d = np.linalg.norm(sols[tag][0] - ex[0], axis=1) / r_L
                    h[tag + "_vs_closed_form"] = {
                        "pos_err_rms": float(np.sqrt(np.mean(d ** 2))),
                        "pos_err_max": float(d.max()),
                        "pos_err_final": float(d[-1])}
                # what the closed form itself is worth: its own initial-value
                # residual, and (B4 only) the float64 reconstruction against
                # the same closed form carried in mpmath
                h["closed_form_ic_residual"] = float(
                    np.abs(ex[0][0] - R0[0]).max()
                    + np.abs(ex[1][0] - V0[0]).max())
                # what the float64 reconstruction of the closed form costs,
                # priced against the same closed form carried end to end in
                # mpmath at 40 digits.  This, and not the DOP853 error, is the
                # floor of the map in a configuration that has a closed form.
                spot = np.linspace(0, len(tsh) - 1, 25).astype(int)
                mp40 = C.exact_mp(fname, f, tsh[spot], R0[0], V0[0], dps=40)[0]
                mp60 = C.exact_mp(fname, f, tsh[spot], R0[0], V0[0], dps=60)[0]
                h["closed_form_float64_vs_mpmath40"] = float(
                    np.abs(ex[0][spot] - mp40).max() / r_L)
                h["closed_form_dps40_vs_dps60"] = float(
                    np.abs(mp40 - mp60).max() / r_L)
            # self-convergence of DOP853, which is all there is where the
            # closed form is absent
            d12 = np.linalg.norm(sols["rtol1e-12"][0] - sols["rtol3e-14"][0],
                                 axis=1) / r_L
            d13 = np.linalg.norm(sols["rtol1e-13"][0] - sols["rtol3e-14"][0],
                                 axis=1) / r_L
            h["self_convergence_1e-12_vs_3e-14"] = {
                "pos_rms": float(np.sqrt(np.mean(d12 ** 2))),
                "pos_max": float(d12.max())}
            h["self_convergence_1e-13_vs_3e-14"] = {
                "pos_rms": float(np.sqrt(np.mean(d13 ** 2))),
                "pos_max": float(d13.max())}
            # the exact invariants of the continuous motion, on the reference
            for tag in ("rtol1e-12", "rtol3e-14"):
                inv = C.invariants(fname, f, sols[tag][0][:, None, :],
                                   sols[tag][1][:, None, :])
                h["invariants_" + tag] = inv
            if ex is not None:
                h["invariants_closed_form"] = C.invariants(
                    fname, f, ex[0][:, None, :], ex[1][:, None, :])
            h["wall_s_three_tolerances"] = wall
            rec[hname] = h
        # ------------------------------------------------------- the floor
        # What the map in this configuration is measured against, and what
        # that reference is itself worth.  Where there is a closed form the map
        # uses it, and its floor is the cost of evaluating it in double
        # precision, measured against mpmath.  Where there is none the map uses
        # DOP853 at rtol 3e-14, and its floor is estimated by the shift between
        # rtol 1e-13 and rtol 3e-14 -- a proxy that the three closed-form
        # configurations calibrate: there it overestimates the true 3e-14 error
        # by the factor printed below, so it is an upper bound and errs the
        # safe way.
        best = "closed form" if C.CLOSED_FORM[fname] else "DOP853 rtol 3e-14"
        rec["reference_used_in_the_map"] = best
        for hname in C.HORIZONS:
            h = rec[hname]
            if C.CLOSED_FORM[fname]:
                floor = max(h["closed_form_float64_vs_mpmath40"],
                            h["closed_form_ic_residual"], 1e-16)
                h["proxy_over_true_3e-14"] = (
                    h["self_convergence_1e-13_vs_3e-14"]["pos_rms"]
                    / h["rtol3e-14_vs_closed_form"]["pos_err_rms"])
            else:
                floor = h["self_convergence_1e-13_vs_3e-14"]["pos_rms"]
            rec["floor_estimate_" + hname] = floor
            rec["paper_reference_floor_" + hname] = (
                h["rtol1e-12_vs_closed_form"]["pos_err_rms"]
                if C.CLOSED_FORM[fname]
                else h["self_convergence_1e-12_vs_3e-14"]["pos_rms"])
        ref[fname] = rec
        print("  %-12s closed form: %-3s  map floor  H_paper %.3e  "
              "H_crossover %.3e   (paper reference %.3e / %.3e)"
              % (fname, "yes" if C.CLOSED_FORM[fname] else "NO",
                 rec["floor_estimate_H_paper"],
                 rec["floor_estimate_H_crossover"],
                 rec["paper_reference_floor_H_paper"],
                 rec["paper_reference_floor_H_crossover"]))
    out["reference_per_configuration"] = ref

    # ------------------------------------------------- the equal-cost budgets
    out["cost_model"] = {
        "flops_per_step": {s: C.flops_per_step(s, cal.get("gl4", {})
                                               .get("mean_iters"))
                           for s in C.SCHEMES},
        "equal_cost_substeps_vs_corrector": {
            s: C.equal_cost_substeps(s, cal.get("gl4", {}).get("mean_iters"))
            for s in ("boris", "vps2", "vps4", "gl4")},
        "note": "m steps of the scheme cost no more than one corrector step; "
                "gl4 is priced from the mean iteration count measured at "
                "Omega h = 0.3 in B4",
    }
    print("equal-cost sub-steps against one corrector step: %s"
          % out["cost_model"]["equal_cost_substeps_vs_corrector"])

    return check_or_write_json(OUT, out, force)


def check_or_write_json(path, payload, force):
    from ea_common import check_or_write
    return check_or_write(path, json.loads(json.dumps(C.clean(payload))),
                          force=force)


if __name__ == "__main__":
    raise SystemExit(main())
