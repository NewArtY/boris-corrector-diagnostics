"""
s1_s2_s4.py -- the remaining three settings of the F0.1 elimination.
====================================================================
S1  field known only on a coarse time grid   -> attack: cubic spline + vps4
S2  deterministic quasi-periodic dB          -> attack: fit the tones + vps4
S4  large step, field fully known            -> attack: subcycled vps4 at
                                                equal flops

For S1 and S2 the truth is the quasi-periodic field (three incommensurate
tones near the gyrofrequency), because the smooth B4 field alone is nearly
linear over T=120 (tau = 1.2e5) and any interpolant would nail it trivially --
that would be a rigged test, not an attack.
"""
import json
import os
import numpy as np
from scipy.optimize import least_squares

import harness as H
from pfields import PerturbedDecaying, QuasiPeriodic, SplineField

HERE = os.path.dirname(os.path.abspath(__file__))

TONES = QuasiPeriodic(amps=[2.0e-3, 1.3e-3, 0.8e-3],
                      freqs=[0.9137, 1.6180, 2.7183],   # incommensurate
                      phases=[0.3, 1.7, 2.9])


def truth_field():
    return PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN, a=TONES.a, adot=TONES.adot)


def score_against(field_used, r_ref, v_ref, dt, n):
    rs, vs, ts, fl = H.run_scheme("vps4", field_used, dt, n)
    sc = H.score(rs, vs, ts, r_ref, v_ref)
    sc["flops"] = fl
    return sc


# ------------------------------------------------------------------- S1
def run_s1(r_ref, v_ref, n, sig):
    res = {"description": "field known only on a coarse time grid",
           "attack": "cubic spline through the same samples + vps4",
           "nodes": []}
    tf = truth_field()
    for h_node in (2.0, 1.0, 0.5, 0.25):
        t_nodes = np.arange(0.0, H.T_FINAL + h_node, h_node)
        bz = np.array([tf.Bz_of_t(t) for t in t_nodes])
        sp = SplineField(t_nodes, bz)
        sc = score_against(sp, r_ref, v_ref, H.DT_WORK, n)
        sc["h_node"] = h_node
        sc["n_nodes"] = len(t_nodes)
        res["nodes"].append(sc)
        print(f"  S1 h_node={h_node:.2f} ({len(t_nodes):3d} nodes): "
              f"pos={sc['pos_err_rms']:.3e} E={sc['energy_err_median_2nd_half']:.3e} "
              f"flops={sc['flops']:.3e}")
    best = min(res["nodes"], key=lambda d: d["pos_err_rms"])
    res["best"] = best
    return res


# ------------------------------------------------------------------- S2
def run_s2(r_ref, v_ref, n, sig):
    """Attacker sees B_z(t) samples and fits three tones -- 9 parameters."""
    tf = truth_field()
    h_node = 0.25
    t_nodes = np.arange(0.0, H.T_FINAL + h_node, h_node)
    bz = np.array([tf.Bz_of_t(t) for t in t_nodes])
    smooth = 1.0 * np.exp(-t_nodes / H.TAU_MAIN)
    resid = bz - smooth

    def model(p, t):
        A, w, ph = p[0:3], p[3:6], p[6:9]
        return np.sum(A * np.sin(np.outer(t, w) + ph), axis=-1)

    p0 = np.array([1e-3, 1e-3, 1e-3, 0.9, 1.6, 2.7, 0.0, 0.0, 0.0])
    fit = least_squares(lambda p: model(p, t_nodes) - resid, p0,
                        xtol=1e-14, ftol=1e-14, max_nfev=20000)
    p = fit.x
    rec = QuasiPeriodic(amps=p[0:3], freqs=p[3:6], phases=p[6:9])
    f_rec = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN, a=rec.a, adot=rec.adot)
    sc = score_against(f_rec, r_ref, v_ref, H.DT_WORK, n)
    rms_fit = float(np.sqrt(np.mean((model(p, t_nodes) - resid) ** 2)))
    print(f"  S2 tone fit: residual rms={rms_fit:.3e}  "
          f"pos={sc['pos_err_rms']:.3e} E={sc['energy_err_median_2nd_half']:.3e}")
    return {"description": "deterministic quasi-periodic dB, 9 free parameters",
            "attack": "least-squares identification of the tones + vps4",
            "fit_residual_rms": rms_fit, "n_params": 9,
            "fitted": {"amps": p[0:3].tolist(), "freqs": p[3:6].tolist(),
                       "phases": p[6:9].tolist()},
            "true": {"amps": TONES.amps.tolist(), "freqs": TONES.freqs.tolist(),
                     "phases": TONES.phases.tolist()},
            "score": sc}


# ------------------------------------------------------------------- S4
def run_s4():
    """Large outer step, field fully known. Give vps4 the hybrid's flop budget
    and subcycle; compare at matched cost."""
    field = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
    rows = []
    for dt_outer in (1.0, 2.0, 3.0):
        n_outer = int(round(H.T_FINAL / dt_outer))
        ts = np.linspace(0.0, H.T_FINAL, n_outer + 1)
        r_ref, v_ref = H.dop853(field, ts, H.T_FINAL)
        budget = H.HYBRID_FLOPS_PER_STEP * n_outer      # one hybrid run
        per_sub = 273.0                                  # vps4 flops/substep
        n_sub = max(1, int(budget / (per_sub * n_outer)))
        dt_sub = dt_outer / n_sub
        n_tot = n_outer * n_sub
        ts_f = np.linspace(0.0, H.T_FINAL, n_tot + 1)
        r_ref_f, v_ref_f = H.dop853(field, ts_f, H.T_FINAL)
        rs, vs, tt, fl = H.run_scheme("vps4", field, dt_sub, n_tot)
        sc = H.score(rs, vs, tt, r_ref_f, v_ref_f)
        sc.update(dt_outer=dt_outer, subcycles=n_sub, dt_sub=dt_sub,
                  flops=fl, hybrid_budget=budget)
        rows.append(sc)
        print(f"  S4 dt_outer={dt_outer:.1f}: vps4 subcycled x{n_sub} "
              f"(dt={dt_sub:.4f}) pos={sc['pos_err_rms']:.3e} "
              f"E={sc['energy_err_median_2nd_half']:.3e} "
              f"flops={fl:.3e} vs hybrid {budget:.3e}")
    return {"description": "large step, field fully known",
            "attack": "vps4 subcycled to the hybrid's flop budget",
            "runs": rows,
            "known_competitor": {
                "ref": "Bigi, Spies, Ceriotti, arXiv:2508.01068",
                "what": "learned generating function for large-step molecular "
                        "dynamics; trained from scratch on autonomous systems",
                "why_it_matters": "direct precedent for a learned construction "
                                  "in exactly the large-step regime S4 probes. "
                                  "Not reimplemented (out of budget, and S4 is "
                                  "expected to die on flops regardless), but "
                                  "any wall-clock-only survival of S4 must be "
                                  "argued against this work, not only against "
                                  "filtered Boris."}}


def main():
    field = truth_field()
    n = int(round(H.T_FINAL / H.DT_WORK))
    ts = np.linspace(0.0, H.T_FINAL, n + 1)
    r_ref, v_ref = H.dop853(field, ts, H.T_FINAL)
    sig = H.physical_signal(v_ref)
    print(f"physical signal (quasi-periodic truth) = {sig:.6e}\n")

    # what the integrator gets if it simply ignores dB
    f_smooth = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
    sc_ign = score_against(f_smooth, r_ref, v_ref, H.DT_WORK, n)
    print(f"  ignoring dB entirely: pos={sc_ign['pos_err_rms']:.3e} "
          f"E={sc_ign['energy_err_median_2nd_half']:.3e}\n")

    out = {"meta": {"t_final": H.T_FINAL, "dt_work": H.DT_WORK,
                    "physical_signal": sig,
                    "hybrid_ref": {"pos_err_rms": 3.469e-3,
                                   "energy_err_median_2nd_half": 1.639e-5,
                                   "flops": 4.56e7,
                                   "source": "experiments/classical/verdict.json"}},
           "ignoring_perturbation": sc_ign}
    out["S1"] = run_s1(r_ref, v_ref, n, sig)
    out["S2"] = run_s2(r_ref, v_ref, n, sig)
    out["S4"] = run_s4()

    with open(os.path.join(HERE, "s1_s2_s4.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote s1_s2_s4.json")


if __name__ == "__main__":
    main()
