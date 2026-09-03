"""AB3: the non-learned corrector -- the baseline the campaign never had.

    python ab3_baseline.py            measure, then check against the file
    python ab3_baseline.py --force    overwrite ab3_baseline.json

WHAT THIS IS
------------
The first author's list of required experiments contains a "simple non-ML
corrector" and it had never been run.  This script writes the correction out by
hand, from the Taylor expansion of the one-step error of the shipped Boris map,
with no training, no data, no fit and no seed, and puts it through exactly the
measurements the learned corrector is put through in `ab2_ablation.py`.

The derivation is in `ab_common.analytic_defect`.  In one line: the shipped map
updates the position with the *new* velocity over the whole step, which is a
first-order rule and carries a defect of -(h^2/2)a; and it rotates the velocity
by 2 arctan(|B|h/2) where the exact rotation is by |B|h, so it is short by
(|B|h)^3/12.  Adding both back is thirteen lines of arithmetic and uses only
the thirteen numbers the network is given.

THREE FORMS ARE MEASURED
------------------------
  analytic          both terms; the rotation completion done as an exact
                    rotation, so the velocity correction is exactly
                    norm-preserving
  analytic_linear   the same, with the rotation replaced by its first-order
                    form and the angle by its series (|B|h)^3/12 -- no
                    transcendental at all
  analytic_dr       the position term alone
  analytic_dv       the rotation completion alone

each with the shipped projection and, for the first, without it, so that the
question "does the projection matter for a correction that is already exactly
energy neutral" is asked of this corrector too.

COST
----
Counted operation by operation in `ab_common.flops_analytic_defect`, on the
flop model of Section 9.  The equal-cost comparison of W14 is repeated: how
many steps of the non-learned scheme fit in the budget of one learned step, and
what error it reaches there.

Writes ab3_baseline.json.  Draws nothing; trains nothing; loads no checkpoint
except to place the result beside it.
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
from ab2_ablation import channels_of                            # noqa: E402

OUT = AB.outpath("ab3_baseline.json")

VARIANTS = {
    "analytic": AB.Variant("analytic", ortho=True, rescale=True,
                           analytic_kind="rotation"),
    "analytic_raw": AB.Variant("analytic", ortho=False, rescale=False,
                               analytic_kind="rotation"),
    "analytic_linear": AB.Variant("analytic", ortho=True, rescale=True,
                                  analytic_kind="linear"),
    "analytic_dr": AB.Variant("analytic", ortho=True, rescale=True,
                              zero_dv=True, analytic_kind="rotation"),
    "analytic_dv": AB.Variant("analytic", ortho=True, rescale=True,
                              zero_dr=True, analytic_kind="rotation"),
    "analytic_order3": AB.Variant("analytic", ortho=True, rescale=True,
                                  analytic_kind="order3"),
    "analytic_order3_raw": AB.Variant("analytic", ortho=False, rescale=False,
                                      analytic_kind="order3"),
    "trapezoid": AB.Variant("analytic", ortho=True, rescale=True,
                            analytic_kind="trapezoid"),
    "trapezoid_rot": AB.Variant("analytic", ortho=True, rescale=True,
                                analytic_kind="trapezoid_rot"),
}

WHAT = {
    "analytic": "the hand-written defect, both terms, with the shipped "
                "projection",
    "analytic_raw": "the same, with the projection removed",
    "analytic_linear": "the same, with the rotation completion linearised and "
                       "its angle taken from the series -- no transcendental",
    "analytic_dr": "the position term alone",
    "analytic_dv": "the rotation completion alone",
    "analytic_order3": "the position half carried one order further, to "
                       "-(h^3/3) a'; 1/tau is inverted out of E and the "
                       "inputs, so this still uses nothing the network is not "
                       "given",
    "analytic_order3_raw": "the same, with the projection removed",
    "trapezoid": "no expansion at all: the position update made symmetric, "
                 "r_{n+1} = r_n + (h/2)(v_n + v_{n+1}), written as a "
                 "correction to the shipped rule -- six flops",
    "trapezoid_rot": "the symmetric position update together with the "
                     "rotation completion",
}


def defect_comparison(field, model, n=None):
    """What the network predicts against what the expansion predicts, on the
    states the network actually sees.

    For each state of the reference orbit at the working step, three defects
    are formed against the same one-step map:

      true       the exact flow over one step minus the Boris step.  The
                 states are taken on the reference orbit itself, so the exact
                 propagation of the state at t_n is the reference at t_{n+1}
                 and costs nothing.

                 (The obvious alternative -- re-anchoring the closed form at
                 each state by shifting B_0 to B_0 e^{-t_n/tau} -- is wrong on
                 the shipped code and was tried first.  `fast_basis` takes a
                 `b0` argument, but `sw_common.exact_from_basis` forms
                 zeta'(0) = w(0) - i (B_0/2) z(0) from the *module-level* B_0,
                 so a basis built at any other B_0 is silently mis-anchored;
                 the one-step propagation then drifts by t_n/(2 tau) in the
                 velocity, which is 5e-4 by the end of this window.  Nothing
                 committed is affected -- every committed call anchors at
                 t = 0 with B_0 = 1 -- but the argument is a trap and this note
                 is the report of it.)
      trained    the target the network was actually fitted to: the same thing
                 with the exact flow replaced by 150 Boris steps of h/150,
                 which is what `training/train_corrector_b4.py` builds.
      network    the committed checkpoint's own output at that state.
      analytic   `ab_common.analytic_defect` and the rotation completion.

    This is the measurement that says whether the network learned the leading
    terms of the expansion or something else.  It integrates nothing.
    """
    import torch
    from models.boris import boris_step, integrate_boris

    n = AB.N_WORK if n is None else n
    Rr, Vr, ts = AB.closed_form_ref()
    h = AB.DT
    rows = {"true": [], "trained": [], "network": [], "analytic": [],
            "analytic_order3": []}
    for i in range(n):
        r, v, t = Rr[i], Vr[i], float(ts[i])
        r_b, v_b = boris_step(r, v, t, h, field)
        B = np.atleast_1d(field.B(r, t)).ravel()
        E = np.atleast_1d(field.E(r, t)).ravel()

        rows["true"].append(np.concatenate([Rr[i + 1] - r_b,
                                            Vr[i + 1] - v_b]))

        rs_f, vs_f, _ = integrate_boris(r, v, t, h / 150.0, 150, field)
        rows["trained"].append(np.concatenate([rs_f[-1] - r_b,
                                               vs_f[-1] - v_b]))

        with torch.no_grad():
            x = torch.tensor(np.concatenate([r, v, B, E, [h]]))[None, :]
            rows["network"].append(model(x).numpy()[0].copy())

        dr = AB.analytic_defect(r, v, E, B, h)
        dv = AB._rotation_completion(v_b, B, h)
        rows["analytic"].append(np.concatenate([dr, dv]))

        dr3 = AB.analytic_defect(r, v, E, B, h, kind="order3")
        rows["analytic_order3"].append(np.concatenate([dr3, dv]))

    D = {k: np.asarray(v) for k, v in rows.items()}

    def rel(a, b, sl):
        num = np.linalg.norm(D[a][:, sl] - D[b][:, sl], axis=1)
        den = np.linalg.norm(D[b][:, sl], axis=1)
        q = num / np.maximum(den, 1e-300)
        return {"median": float(np.median(q)), "mean": float(q.mean()),
                "max": float(q.max())}

    pos, vel, both = slice(0, 3), slice(3, 6), slice(0, 6)
    out = {"n_states": int(n),
           "magnitudes": {k: {"position_median": float(np.median(
               np.linalg.norm(D[k][:, pos], axis=1))),
               "velocity_median": float(np.median(
                   np.linalg.norm(D[k][:, vel], axis=1)))} for k in D}}
    for a, b in (("network", "true"), ("analytic", "true"),
                 ("analytic_order3", "true"),
                 ("network", "trained"), ("analytic", "trained"),
                 ("analytic_order3", "trained"),
                 ("trained", "true"), ("network", "analytic"),
                 ("network", "analytic_order3")):
        out["%s_vs_%s" % (a, b)] = {
            "position": rel(a, b, pos), "velocity": rel(a, b, vel),
            "both": rel(a, b, both)}
    # the correlation of the two predictions, component by component
    corr = []
    for j in range(6):
        a, b = D["network"][:, j], D["analytic"][:, j]
        if a.std() == 0.0 or b.std() == 0.0:
            corr.append(None)          # both identically zero: the z channel
        else:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    out["network_analytic_correlation"] = corr
    return out


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
        "what": "the non-learned corrector: the one-step Boris defect written "
                "out by hand and put through the same measurements",
        "derivation": "ab_common.analytic_defect",
        "reference": "the closed form of B4 on the working grid",
        "n_random_draws": 0, "nothing_trained": True,
        "uses_only_the_networks_own_inputs": True,
        "variants": {k: dict({"what": WHAT[k]}, **VARIANTS[k].asdict())
                     for k in VARIANTS},
    }}

    # ---- the runs ---------------------------------------------------------
    res = AB.evaluate_variant(VARIANTS, model=None, ref=ref)
    for name, var in VARIANTS.items():
        rr, vv, tt, _ = AB.integrate_variant(field, AB.R0, AB.V0, AB.DT,
                                             AB.N_WORK, None, var)
        res[name]["channels"] = channels_of(rr, vv, Rr, Vr, omega_c)
    out["runs"] = res

    # ---- what the learned corrector and the classical schemes did ---------
    with open(AB.outpath("ab2_ablation.json"), encoding="utf-8") as fh:
        ab2 = json.load(fh)
    learned = ab2["table"]["full"]
    classical = ab2["classical"]

    def row(name, d, flops):
        return {"energy_separation": d["energy_separation"],
                "traj_gain_over_boris": d["traj_gain_over_boris"],
                "pos_err_rms": d["pos_err_rms"],
                "energy_err_median_2nd_half": d["energy_err_median_2nd_half"],
                "constraint_max_abs": d.get("constraint_max_abs"),
                "flops_per_step": flops}

    f_analytic = AB.FLOPS_BORIS + AB.flops_analytic_defect("rotation") \
        + AB.flops_projection()
    f_linear = AB.FLOPS_BORIS + AB.flops_analytic_defect("linear") \
        + AB.flops_projection()
    flops = {"analytic": f_analytic, "analytic_raw":
             AB.FLOPS_BORIS + AB.flops_analytic_defect("rotation"),
             "analytic_linear": f_linear,
             "analytic_dr": f_analytic, "analytic_dv": f_analytic,
             "analytic_order3": AB.FLOPS_BORIS
             + AB.flops_analytic_defect("order3") + AB.flops_projection(),
             "analytic_order3_raw": AB.FLOPS_BORIS
             + AB.flops_analytic_defect("order3"),
             "trapezoid": AB.FLOPS_BORIS
             + AB.flops_analytic_defect("trapezoid") + AB.flops_projection(),
             "trapezoid_rot": AB.FLOPS_BORIS
             + AB.flops_analytic_defect("trapezoid_rot")
             + AB.flops_projection()}
    out["flops"] = {
        "model": "one flop per arithmetic operation, twenty per "
                 "transcendental (Section 9), counted operation by operation "
                 "in ab_common.flops_analytic_defect",
        "boris_step": AB.FLOPS_BORIS,
        "network_forward": AB.FLOPS_NET_FORWARD,
        "corrector_step": AB.FLOPS_CORRECTOR,
        "projection": AB.flops_projection(),
        "analytic_defect_rotation": AB.flops_analytic_defect("rotation"),
        "analytic_defect_linear": AB.flops_analytic_defect("linear"),
        "per_step": flops,
        "corrector_over_analytic": AB.FLOPS_CORRECTOR / f_analytic,
        "corrector_over_analytic_linear": AB.FLOPS_CORRECTOR / f_linear,
        "vps4_step": AB.FLOPS_VPS4,
    }

    table = {"learned_corrector": {
        "energy_separation": learned["energy_separation"]["committed"],
        "traj_gain_over_boris": learned["traj_gain_over_boris"]["committed"],
        "pos_err_rms": learned["pos_err_rms"]["committed"],
        "energy_err_median_2nd_half":
            learned["energy_err_median_2nd_half"]["committed"],
        "constraint_max_abs": learned["constraint_max_abs"]["committed"],
        "flops_per_step": AB.FLOPS_CORRECTOR,
        "ensemble_median_traj_gain":
            learned["traj_gain_over_boris"]["ensemble"]["median"],
        "ensemble_range_traj_gain":
            [learned["traj_gain_over_boris"]["ensemble"]["min"],
             learned["traj_gain_over_boris"]["ensemble"]["max"]],
    }}
    for name in VARIANTS:
        table[name] = row(name, res[name], flops[name])
    for s in ("boris", "vps2", "vps4", "gl4"):
        table[s] = row(s, classical[s], classical[s]["flops_per_step"])
    out["table"] = table

    # ---- P2: what share of the advantage does the non-learned form take ----
    # The headline form is `analytic_order3`, the best of the hand-written
    # ones that still uses nothing outside the network's own thirteen inputs;
    # the two-term `analytic` is reported beside it so that the value of the
    # third term is visible rather than folded in.
    L = table["learned_corrector"]
    ens_gains = ab2["table"]["full"]["traj_gain_over_boris"]
    ens_err = ab2["table"]["full"]["pos_err_rms"]
    ens_vals = [ab2["runs"][t]["variants"]["full"]["pos_err_rms"]
                for t in ab2["runs"] if t != "committed"]
    p2 = {"note": "the non-learned corrector against the learned one, on the "
                  "same reference, the same field, the same step and the same "
                  "statistic"}
    for form in ("analytic_order3", "analytic", "trapezoid_rot"):
        A = table[form]
        p2[form] = {
            "trajectory": {
                "learned_gain_over_boris": L["traj_gain_over_boris"],
                "non_learned_gain_over_boris": A["traj_gain_over_boris"],
                "share_of_the_learned_gain": (A["traj_gain_over_boris"]
                                              / L["traj_gain_over_boris"]),
                "error_ratio_to_learned": (A["pos_err_rms"]
                                           / L["pos_err_rms"]),
                "beats_the_committed_checkpoint":
                    bool(A["pos_err_rms"] < L["pos_err_rms"]),
                "beats_n_of_the_twenty_retrainings":
                    int(sum(1 for e in ens_vals if A["pos_err_rms"] < e)),
                "learned_ensemble_median_gain": ens_gains["ensemble"]["median"],
                "learned_ensemble_gain_range":
                    [ens_gains["ensemble"]["min"],
                     ens_gains["ensemble"]["max"]],
                "inside_the_learned_ensemble_range":
                    bool(ens_gains["ensemble"]["min"]
                         <= A["traj_gain_over_boris"]
                         <= ens_gains["ensemble"]["max"]),
            },
            "energy": {
                "learned_separation": L["energy_separation"],
                "non_learned_separation": A["energy_separation"],
                "share_of_the_learned_separation": (A["energy_separation"]
                                                    / L["energy_separation"]),
                "boris_separation": table["boris"]["energy_separation"],
            },
            "cost": {
                "learned_flops_per_step": L["flops_per_step"],
                "non_learned_flops_per_step": A["flops_per_step"],
                "ratio": L["flops_per_step"] / A["flops_per_step"],
            },
            "accuracy_per_flop_ratio":
                (L["pos_err_rms"] / A["pos_err_rms"])
                * (L["flops_per_step"] / A["flops_per_step"]),
            "training_cost_not_paid_by_the_non_learned_form":
                "one training of the corrector is 7.71e11 flops (W16), which "
                "the cost column does not contain; the non-learned form has "
                "no training term at all",
        }
    p2["ensemble_error_min"] = ens_err["ensemble"]["min"]
    out["P2"] = p2

    # ---- the equal-cost comparison, W14's ---------------------------------
    f3 = flops["analytic_order3"]
    m = int(AB.FLOPS_CORRECTOR // f3)
    sub_dt = AB.DT / m
    n_sub = AB.N_WORK * m
    var = VARIANTS["analytic_order3"]
    rr, vv, tt, _ = AB.integrate_variant(field, AB.R0, AB.V0, sub_dt, n_sub,
                                         None, var)
    take = np.arange(0, n_sub + 1, m)
    pos = np.linalg.norm(rr[take] - Rr, axis=1)
    E_ref = 0.5 * np.sum(Vr ** 2, axis=1)
    E0 = E_ref[0]
    e = 0.5 * np.sum(vv[take] ** 2, axis=1)
    half = AB.N_WORK // 2
    e_err = np.abs(e - E_ref) / E0
    out["equal_cost"] = {
        "form": "analytic_order3",
        "substeps_per_corrector_step": m,
        "sub_dt": sub_dt,
        "omega_h": sub_dt,
        "pos_err_rms": float(np.sqrt(np.mean(pos ** 2))),
        "energy_err_median_2nd_half": float(np.median(e_err[half:])),
        "energy_separation": (ab2["physical_signal_median"]
                              / float(np.median(e_err[half:]))),
        "learned_pos_err_rms": L["pos_err_rms"],
        "ratio_to_learned": (float(np.sqrt(np.mean(pos ** 2)))
                             / L["pos_err_rms"]),
        "vps4_pos_err_rms": classical["vps4"]["pos_err_rms"],
        "note": "the largest integer m for which m steps of the non-learned "
                "scheme cost no more than one step of the learned one, and "
                "what it reaches on that budget",
    }

    # ---- what is left out -------------------------------------------------
    # The map evaluates E at the left end of the step.  Price that term rather
    # than assert it is small: run the same non-learned corrector with the
    # electric impulse taken at the midpoint of the reference orbit, which is
    # information the network does not have either, and report the change.
    out["omitted_term"] = {
        "what": "the map takes E at the left end of the step; the exact "
                "impulse is h E(r_mid, t_mid), a difference of O(h^2 dE/dt)",
        "E_scale": float(np.linalg.norm(np.atleast_1d(
            field.E(AB.R0, 0.0)).ravel())),
        "estimated_per_step_velocity_defect":
            float(0.5 * AB.DT ** 2 * np.linalg.norm(np.atleast_1d(
                field.E(AB.R0, 0.0)).ravel()) / AB.TAU * AB.TAU),
        "why_not_corrected": "it cannot be formed from the thirteen inputs the "
                             "network is given, and correcting it here would "
                             "make the comparison unequal",
    }

    # ---- what the network learned, against what the expansion says --------
    out["defect_comparison"] = defect_comparison(
        field, AB.load_torch(AB.COMMITTED_CORRECTOR))

    out["meta"]["wall_s"] = time.time() - t_start
    AB.assert_committed_untouched()
    AB.assert_no_draws(0)
    rc = AB.write(OUT, out, force=force)

    print("\n%-20s %14s %12s %13s %13s"
          % ("scheme", "E-separation", "traj gain", "traj rms", "flops/step"))
    order = ["boris", "learned_corrector", "analytic", "analytic_linear",
             "analytic_order3", "analytic_raw", "analytic_dr",
             "analytic_dv", "trapezoid", "trapezoid_rot", "vps2", "vps4",
             "gl4"]
    for k in order:
        r = table[k]
        print("%-20s %14.4f %12.2f %13.4e %13.0f"
              % (k, r["energy_separation"], r["traj_gain_over_boris"],
                 r["pos_err_rms"], r["flops_per_step"]))
    q = p2["analytic_order3"]
    print("\nthe non-learned corrector reaches %.0f%% of the learned "
          "trajectory gain and %.0f%% of its energy separation at %.0fx fewer "
          "flops per step, and beats %d of the twenty retrainings outright"
          % (100 * q["trajectory"]["share_of_the_learned_gain"],
             100 * q["energy"]["share_of_the_learned_separation"],
             q["cost"]["ratio"],
             q["trajectory"]["beats_n_of_the_twenty_retrainings"]))
    print("at equal cost (%d substeps, Omega h = %.4f) it reaches %.4e "
          "Larmor radii against the learned %.4e"
          % (m, sub_dt, out["equal_cost"]["pos_err_rms"], L["pos_err_rms"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
