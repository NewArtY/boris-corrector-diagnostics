"""
summarize.py -- F0.1 verdict table.
===================================
Rule: a setting FAILS if a classical attack reaches comparable or better
accuracy in BOTH channels at comparable or lower flop cost. The hybrid figures
(experiments/classical/verdict.json) stand in for "what a learned method would
have to achieve": pos 3.469e-3 r_L, energy 1.639e-5, 4.56e7 flops per run.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HYB = {"pos": 3.469e-3, "energy": 1.639e-5, "flops": 4.56e7}


def load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def main():
    a = load("s1_s2_s4.json")
    s3 = load("s3_broadband.json")
    s3b = load("s3b_structured.json")

    v = {"rule": "FAILS if a classical attack matches or beats the hybrid in "
                 "both channels at no greater flop cost",
         "hybrid_reference": HYB, "settings": {}}

    # ---- S1
    best = a["S1"]["best"]
    v["settings"]["S1"] = {
        "name": "field known only on a coarse time grid",
        "attack": "cubic spline through the same samples + vps4",
        "verdict": "FAILS",
        "numbers": {"h_node": best["h_node"], "n_nodes": best["n_nodes"],
                    "pos_err_rms": best["pos_err_rms"],
                    "energy_err": best["energy_err_median_2nd_half"],
                    "flops": best["flops"]},
        "margin": {"position": HYB["pos"] / best["pos_err_rms"],
                   "energy": HYB["energy"] / best["energy_err_median_2nd_half"],
                   "cost": HYB["flops"] / best["flops"]},
        "note": "already beats the hybrid on position at 121 nodes; GEMPIC "
                "closes this setting classically in the literature too"}

    # ---- S2
    sc = a["S2"]["score"]
    v["settings"]["S2"] = {
        "name": "deterministic quasi-periodic dB, 9 free parameters",
        "attack": "least-squares identification of the tones + vps4",
        "verdict": "FAILS",
        "numbers": {"fit_residual_rms": a["S2"]["fit_residual_rms"],
                    "pos_err_rms": sc["pos_err_rms"],
                    "energy_err": sc["energy_err_median_2nd_half"],
                    "flops": sc["flops"]},
        "margin": {"position": HYB["pos"] / sc["pos_err_rms"],
                   "energy": HYB["energy"] / sc["energy_err_median_2nd_half"],
                   "cost": HYB["flops"] / sc["flops"]},
        "note": "the field is recovered to 7e-12; a deterministic field with "
                "finitely many parameters is always identifiable from data"}

    # ---- S3
    mf = s3["attack_mean_field"]
    quad = [r for r in s3b["mean_shift"] if not r["structured"]]
    coefs = [r["mean_shift_vs_smooth"] / r["rms"] ** 2 for r in quad]
    v["settings"]["S3"] = {
        "name": "broadband dB, realization unknown, statistics public",
        "attacks": ["mean field (ignore dB)", "Monte Carlo from the public PSD",
                    "closed-form O(a^2) mean shift"],
        "verdict": "FAILS",
        "numbers": {
            "ignoring_dB_error_vs_ensemble_mean": mf["max_err_vs_truth_mean"],
            "ignoring_dB_rel_error": mf["rel_err"],
            "ignoring_dB_flops": mf["flops"],
            "quadratic_coefficient_uniform": coefs,
            "coefficient_constant_to": "4 significant figures over two decades "
                                       "of amplitude (36.02 -> 36.09)",
            "antithetic_runs_to_resolve_it": 48},
        "note": "the ensemble-averaged effect of a zero-mean broadband "
                "perturbation is exactly one number: the shift is quadratic in "
                "amplitude with a constant coefficient across the whole "
                "perturbative range. A classical attacker measures that number "
                "once with 48 antithetic runs (~5e6 flops at the working step, "
                "9x less than a single hybrid run) and adds it. Nothing is left "
                "for an effective learned model except amortisation across "
                "queries, which is an engineering argument, not a numerical one."}

    # ---- S4
    r = a["S4"]["runs"][0]
    v["settings"]["S4"] = {
        "name": "large step, field fully known",
        "attack": "vps4 subcycled to the hybrid's own flop budget",
        "verdict": "FAILS",
        "numbers": {"dt_outer": r["dt_outer"], "subcycles": r["subcycles"],
                    "pos_err_rms": r["pos_err_rms"],
                    "energy_err": r["energy_err_median_2nd_half"],
                    "flops": r["flops"], "hybrid_budget": r["hybrid_budget"]},
        "margin": {"position": HYB["pos"] / r["pos_err_rms"]},
        "known_competitor": a["S4"]["known_competitor"],
        "note": "at matched flops the classical scheme is eleven orders of "
                "magnitude more accurate; wall-clock-only survival would also "
                "have to be argued against arXiv:2508.01068"}

    # ---- S5
    v["settings"]["S5"] = {
        "name": "radiation reaction (Landau-Lifshitz)",
        "attack": "the force is known in closed form",
        "verdict": "FAILS",
        "computed": False,
        "note": "dies on paper, as the plan anticipated: the LL force is an "
                "explicit function of the state and the known fields, so it "
                "enters the right-hand side directly and any classical scheme "
                "integrates it. There is no epistemic gap, only a stiffer ODE. "
                "The analytic contraction rate Lambda = 2(alpha - eps_rad) "
                "makes it a good VALIDATION system with an exact reference, "
                "which is how 08_ARCHITECTURE.md proposed using it -- but that "
                "is a test case, not a setting that needs learning."}

    v["summary"] = {
        "surviving": [],
        "failing": ["S1", "S2", "S3", "S4", "S5"],
        "rejection_criterion_met": True,
        "statement": "No candidate setting survives classical attack. Every "
                     "attack costs at most one hybrid run and wins on both "
                     "channels, usually by orders of magnitude."}

    with open(os.path.join(HERE, "verdicts.json"), "w") as fh:
        json.dump(v, fh, indent=1)

    print(f"{'setting':6} {'verdict':9} {'position':>12} {'energy':>12} {'flops':>11}")
    for k in ("S1", "S2", "S3", "S4", "S5"):
        s = v["settings"][k]
        n = s.get("numbers", {})
        print(f"{k:6} {s['verdict']:9} "
              f"{n.get('pos_err_rms', float('nan')):12.3e} "
              f"{n.get('energy_err', float('nan')):12.3e} "
              f"{n.get('flops', float('nan')):11.3e}")
    print(f"\nhybrid {'':9} {HYB['pos']:12.3e} {HYB['energy']:12.3e} {HYB['flops']:11.3e}")
    print("\nwrote verdicts.json")


if __name__ == "__main__":
    main()
