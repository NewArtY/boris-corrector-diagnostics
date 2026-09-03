"""Ф0.2 step 3 -- the decisive structural test.

Three branches on long horizons, identical envelope methodology to
experiments/horizon/long_runs.py:

    base      clean variational midpoint (control)
    varnet    same map with a frozen random dL_d INSIDE the discrete Lagrangian
    additive  same map with a frozen random dv added to the velocity OUTSIDE

Both perturbed branches are calibrated to the same relative per-step velocity
increment (calibration.json). 3 amplitudes x 5 seeds each.

Acceptance: varnet exponent <= 0.1 up to the working amplitude,
            additive exponent > 0.5 (contrast).
Rejection : varnet secular at working amplitude -> programme B stops.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from training.train_corrector_b4 import DT_WORK, TAU_MAIN
import varint as V
from calibrate import SEEDS, T_SCALE, TARGET

TWO_PI = 2.0 * np.pi
H = DT_WORK
N_GYR = float(os.environ.get("N_GYR", 1e5))
N_STEPS = int(round(N_GYR * TWO_PI / H))
AMP_FACTORS = [float(x) for x in
               os.environ.get("AMPS", "0.1,1.0,10.0").split(",")]
TAG = os.environ.get("TAG", "")
CONFIGS = {"paper_tau1.2e5": TAU_MAIN, "quasistatic_tau1.2e8": 1.2e8}
HORIZONS = [1e3, 1e4, 1e5]

cal = json.load(open(os.path.join(HERE, "calibration.json")))

# ensemble: amp factor x seed
ens_seeds, ens_amp_var, ens_amp_add, ens_tag = [], [], [], []
for f in AMP_FACTORS:
    for i, s in enumerate(SEEDS):
        ens_seeds.append(s)
        ens_amp_var.append(cal["varnet"]["amp_calibrated"][i] * f)
        ens_amp_add.append(cal["additive"]["amp_calibrated"][i] * f)
        ens_tag.append({"amp_factor": f, "seed": s,
                        "target_rel_dv": TARGET * f})
N_ENS = len(ens_seeds)

net_var = V.DeltaLNet(N_ENS, ens_seeds, ens_amp_var, hidden=32, t_scale=T_SCALE)
net_add = V.DeltaLNet(N_ENS, ens_seeds, ens_amp_add, hidden=32, t_scale=T_SCALE)

print(f"horizon {N_GYR:.0e} gyrations = {N_STEPS} steps, ensemble {N_ENS}")

store, summary = {}, {"config": {"h": H, "n_gyrations": N_GYR,
                                 "n_steps": N_STEPS, "taus": CONFIGS,
                                 "amp_factors": AMP_FACTORS,
                                 "target_rel_dv_per_step": TARGET,
                                 "ensemble": ens_tag},
                      "results": {}}

for cname, tau in CONFIGS.items():
    summary["results"][cname] = {}
    for mode in ["base", "varnet", "additive"]:
        t0 = time.time()
        n_ens = 1 if mode == "base" else N_ENS
        d = V.integrate(mode, tau, H, N_STEPS,
                        net=net_var if mode == "varnet" else None,
                        dv_net=net_add if mode == "additive" else None,
                        n_samples=4000, n_ens=n_ens)
        el = time.time() - t0
        t = d["t"]
        recs = []
        for j in range(n_ens):
            expo, env = V.envelope_exponent(t, d["env"][j])
            hz = {}
            for Hg in HORIZONS:
                if Hg > N_GYR * 1.0001:
                    continue
                m = t <= Hg * TWO_PI
                if m.sum() < 5:
                    continue
                half = m.sum() // 2
                phys = 1.0 - np.exp(-Hg * TWO_PI / tau)
                emax = float(env[m][-1])
                emed = float(np.median(d["e_err"][j][m][half:]))
                hz[f"{Hg:.0e}"] = {"energy_err_max": emax,
                                   "energy_err_median_2nd_half": emed,
                                   "physical_signal": phys,
                                   "signal_over_err_max": phys / max(emax, 1e-300)}
            rec = {"envelope_powerlaw_exponent": expo, "horizons": hz}
            if mode != "base":
                rec.update(ens_tag[j])
            recs.append(rec)
            store[f"{cname}/{mode}/{j}/env"] = env
        store[f"{cname}/{mode}/t"] = t
        summary["results"][cname][mode] = {
            "seconds": el, "us_per_step": el / N_STEPS * 1e6,
            "newton_iters_mean": d.get("newton_iters_mean"),
            "newton_iters_max": d.get("newton_iters_max"),
            "members": recs}
        ex = np.array([r["envelope_powerlaw_exponent"] for r in recs])
        print(f"{cname:22s} {mode:9s} {el:7.1f}s  exponent: "
              f"min={np.nanmin(ex):6.3f} med={np.nanmedian(ex):6.3f} "
              f"max={np.nanmax(ex):6.3f}")

np.savez_compressed(os.path.join(HERE, f"f0_runs{TAG}.npz"), **store)
json.dump(summary, open(os.path.join(HERE, f"f0_summary{TAG}.json"), "w"), indent=2)
print("\nsaved f0_runs.npz, f0_summary.json")
