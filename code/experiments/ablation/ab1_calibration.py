"""AB1: the stand, before any ablation is reported.

    python ab1_calibration.py            measure, then check against the file
    python ab1_calibration.py --force    overwrite ab1_calibration.json

Four things are checked, and every one of them has to pass before anything else
in this directory is allowed to mean something.

  1  The committed checkpoint is the committed checkpoint (md5), and its
     standardisation carries exactly the four dead inputs W14 found.

  2  `ab_common.integrate_variant` with every switch in its shipped position is
     `../stats/seed_sweep_b4.py:integrate_corrected` **to the last bit**, on
     both of the two settings that function has -- projected and raw.  A
     variant machinery that is not the committed integrator at its own centre
     measures a different object at every other point of the lattice.

  3  The two numbers the manuscript prints are reproduced on this stand against
     the closed form: the energy separation with the projection, $45.75$, and
     without it, $2.95$.  Both come out of `evaluate_variant`, which is the
     generalisation of the committed `evaluate`, so the generalisation is
     checked on the two points where the two functions overlap.

  4  The old ruler -- Boris at h/150 -- is run alongside and must reproduce
     `../seeds/sd3_ensemble.json`, so that the change of reference is visible
     as a shift rather than as an unexplained difference.

Writes ab1_calibration.json.  Draws nothing.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_common as AB                                          # noqa: E402
import rc_common as RC                                          # noqa: E402

OUT = AB.outpath("ab1_calibration.json")

#: what the manuscript prints, and where.  Declared here so that the check is
#: against the text and not against whatever the code happens to return.
MANUSCRIPT = {
    "energy_separation_projected": 45.75,       # tab:seeds, DOP853/closed form
    "energy_separation_unprojected": 2.95,      # tab:seeds, same row block
    "traj_advantage_closed_form": 120.1,        # tab:seeds
    "traj_advantage_h150": 117.8,               # tab:seeds, the old ruler
}

#: the full-precision values `../refcheck/rc3_seeds.json` committed for the
#: committed checkpoint.  This directory has to land on them exactly.
RC3 = {
    "new": {"energy_separation_hybrid": 45.752058584917684,
            "energy_separation_raw": 2.9462773703544824,
            "traj_gain_projected": 120.11205620944715,
            "corrector_projected_pos_err_rms": 0.003468872999164907,
            "corrector_raw_pos_err_rms": 0.003538661738741027,
            "boris_pos_err_rms": 0.41665346865912883,
            "physical_signal_median": 0.0007497187722405041,
            "corrector_projected_energy": 1.6386558232106552e-05},
    "old": {"energy_separation_hybrid": 45.78433113345744,
            "energy_separation_raw": 2.946256882688746,
            "traj_gain_projected": 117.81413380788808},
}


def main():
    force = "--force" in sys.argv
    t_start = time.time()
    import torch
    torch.set_default_dtype(torch.float64)
    sys.path.insert(0, os.path.join(AB.EXP, "stats"))
    import seed_sweep_b4 as SS
    from fields import DecayingField
    from models.boris import integrate_boris

    out = {"meta": {
        "wave": "W17",
        "what": "the ablation stand, calibrated against the committed numbers",
        "reference": "the closed form of B4 (../spectral/sw_common.py through "
                     "../refcheck/rc_common.py), never a Boris run at h/150",
        "n_random_draws": 0,
        "nothing_retrained": True,
    }}

    # ---- 1: the checkpoint, and the four dead inputs ---------------------
    out["checkpoint"] = {"path": "checkpoints/boris_corrector_b4.pt",
                         "md5": AB.assert_committed_untouched()}
    model = AB.load_torch(AB.COMMITTED_CORRECTOR)
    x_mean = model.x_mean.numpy().copy()
    x_std = model.x_std.numpy().copy()
    y_scale = model.y_scale.numpy().copy()
    dead = [i for i in range(13) if x_std[i] == 1e-12]
    out["standardisation"] = {
        "x_mean": {AB.INPUT_NAMES[i]: float(x_mean[i]) for i in range(13)},
        "x_std": {AB.INPUT_NAMES[i]: float(x_std[i]) for i in range(13)},
        "y_scale": [float(s) for s in y_scale],
        "dead_inputs": [AB.INPUT_NAMES[i] for i in dead],
        "n_dead_inputs": len(dead),
        "note": "an input whose training variance was zero; clamp_min(1e-12) "
                "made its divisor 1e-12, so a departure of 1e-12 in that "
                "input is one standard deviation to the first layer (W14)",
    }
    assert dead == sorted(AB.DEAD_INPUTS), \
        "the four dead inputs of W14 are not where W14 left them: %s" % dead

    # ---- 2: the variant machinery is the committed integrator ------------
    field = DecayingField(B0=1.0, tau=AB.TAU)
    ident = {}
    for name, project, var in (
            ("projected", True,
             AB.Variant(source="net", ortho=True, rescale=True)),
            ("raw", False,
             AB.Variant(source="net", ortho=False, rescale=False))):
        rs_a, vs_a, ts_a = SS.integrate_corrected(field, AB.R0, AB.V0, AB.DT,
                                                  AB.N_WORK, model,
                                                  project=project)
        rs_b, vs_b, ts_b, _ = AB.integrate_variant(field, AB.R0, AB.V0, AB.DT,
                                                   AB.N_WORK, model, var)
        ident[name] = {
            "variant": var.asdict(),
            "position_max_abs_diff": float(np.abs(rs_a - rs_b).max()),
            "velocity_max_abs_diff": float(np.abs(vs_a - vs_b).max()),
            "bit_identical": bool(np.array_equal(rs_a, rs_b)
                                  and np.array_equal(vs_a, vs_b)),
        }
        if not ident[name]["bit_identical"]:
            print("VARIANT MACHINERY IS NOT THE COMMITTED INTEGRATOR (%s)"
                  % name)
            print(json.dumps(ident[name], indent=1))
            return 1
    out["variant_is_the_committed_integrator"] = ident
    print("the variant integrator reproduces integrate_corrected bit for bit "
          "on both of its settings")

    # ---- 3 and 4: the two rulers ----------------------------------------
    n_fine = int(round(AB.T_FINAL / SS.DT_FINE))
    ref_old = integrate_boris(AB.R0, AB.V0, 0.0, SS.DT_FINE, n_fine, field)
    ref_new = AB.closed_form_ref()

    variants = {
        "corrector_projected": AB.Variant(source="net", ortho=True,
                                          rescale=True),
        "corrector_raw": AB.Variant(source="net", ortho=False, rescale=False),
    }
    res = {}
    for ruler, ref in (("old_h150", ref_old), ("closed_form", ref_new)):
        res[ruler] = AB.evaluate_variant(variants, model=model, ref=ref)

    # the committed function, on the same two rulers, for the overlap check
    committed = {"old_h150": SS.evaluate(model, ref=ref_old),
                 "closed_form": SS.evaluate(model, ref=ref_new)}

    overlap = {}
    for ruler in res:
        a, b = res[ruler], committed[ruler]
        pairs = [
            ("energy_separation_projected",
             a["corrector_projected"]["energy_separation"],
             b["energy_separation_hybrid"]),
            ("energy_separation_unprojected",
             a["corrector_raw"]["energy_separation"],
             b["energy_separation_raw"]),
            ("traj_gain_projected",
             a["corrector_projected"]["traj_gain_over_boris"],
             b["traj_gain_projected"]),
            ("boris_pos_err_rms",
             a["boris"]["pos_err_rms"], b["boris"]["pos_err_rms"]),
        ]
        overlap[ruler] = {
            k: {"ab": float(x), "committed": float(y),
                "rel_diff": float(abs(x - y) / abs(y))}
            for k, x, y in pairs}
    out["overlap_with_the_committed_evaluate"] = overlap
    worst = max(v["rel_diff"] for r in overlap.values() for v in r.values())
    out["overlap_worst_rel_diff"] = worst
    if worst > 1e-12:
        print("the generalised evaluate does not agree with the committed one "
              "(worst relative difference %.3e)" % worst)
        return 1
    print("evaluate_variant agrees with seed_sweep_b4.evaluate to %.1e on the "
          "four quantities they share, on both rulers" % worst)

    # ---- the calibration proper ------------------------------------------
    cal = {}
    for ruler, want in (("closed_form", RC3["new"]), ("old_h150", RC3["old"])):
        r = res[ruler]
        got = {"energy_separation_hybrid":
               r["corrector_projected"]["energy_separation"],
               "energy_separation_raw":
               r["corrector_raw"]["energy_separation"],
               "traj_gain_projected":
               r["corrector_projected"]["traj_gain_over_boris"]}
        if ruler == "closed_form":
            got.update({
                "corrector_projected_pos_err_rms":
                    r["corrector_projected"]["pos_err_rms"],
                "corrector_raw_pos_err_rms": r["corrector_raw"]["pos_err_rms"],
                "boris_pos_err_rms": r["boris"]["pos_err_rms"],
                "physical_signal_median": r["physical_signal_median"],
                "corrector_projected_energy":
                    r["corrector_projected"]["energy_err_median_2nd_half"]})
        cal[ruler] = {k: {"got": float(got[k]), "rc3": float(want[k]),
                          "rel_diff": float(abs(got[k] - want[k])
                                            / abs(want[k]))}
                      for k in want}
    out["calibration_against_rc3"] = cal
    worst_cal = max(v["rel_diff"] for r in cal.values() for v in r.values())
    out["calibration_worst_rel_diff"] = worst_cal
    if worst_cal > 1e-12:
        print("CALIBRATION FAILED: worst relative difference %.3e against "
              "../refcheck/rc3_seeds.json" % worst_cal)
        print(json.dumps(cal, indent=1))
        return 1

    out["manuscript"] = {
        "printed": MANUSCRIPT,
        "reproduced": {
            "energy_separation_projected":
                res["closed_form"]["corrector_projected"]["energy_separation"],
            "energy_separation_unprojected":
                res["closed_form"]["corrector_raw"]["energy_separation"],
            "traj_advantage_closed_form":
                res["closed_form"]["corrector_projected"]["traj_gain_over_boris"],
            "traj_advantage_h150":
                res["old_h150"]["corrector_projected"]["traj_gain_over_boris"],
        },
    }
    print("\ncalibration, against the closed form:")
    print("  energy separation with the projection   %.4f   (text: 45.75)"
          % out["manuscript"]["reproduced"]["energy_separation_projected"])
    print("  energy separation without it            %.4f   (text: 2.95)"
          % out["manuscript"]["reproduced"]["energy_separation_unprojected"])
    print("  trajectory advantage over Boris         %.2f    (text: 120.1)"
          % out["manuscript"]["reproduced"]["traj_advantage_closed_form"])
    print("  the same on the old h/150 ruler         %.2f    (text: 117.8)"
          % out["manuscript"]["reproduced"]["traj_advantage_h150"])

    # ---- what the projection is worth on its own -------------------------
    out["runs"] = res

    # ---- the reference itself --------------------------------------------
    d = np.linalg.norm(ref_old[0] - RC.closed_form(ref_old[2])[0], axis=1)
    out["reference"] = {
        "closed_form_basis_agreement": RC.basis_agreement(ref_new[2]),
        "old_ruler_own_rms_error_vs_closed_form": RC.rms(d),
        "why_not_h150": "W18: the corrector's training target is a Boris run "
                        "at h/150, so ruler and pupil are the same object",
    }

    out["meta"]["wall_s"] = time.time() - t_start
    AB.assert_committed_untouched()
    AB.assert_no_draws(0)
    return AB.write(OUT, out, force=force)


if __name__ == "__main__":
    raise SystemExit(main())
