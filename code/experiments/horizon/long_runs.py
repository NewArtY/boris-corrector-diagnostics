"""Long-horizon energy / adiabatic-invariant study, up to 1e5 gyrations."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import fast as F

TWO_PI = 2.0 * np.pi
DT = F.DT_WORK
N_GYR = 100000
T_END = N_GYR * TWO_PI
N_STEPS = int(round(T_END / DT))
HORIZONS = [1e3, 1e4, 1e5]
CONFIGS = {"paper_tau1.2e5": F.TAU_MAIN, "quasistatic_tau1.2e8": 1.2e8}
MODES = ["boris", "raw", "proj"]

Ws, bs, xm, xs, ysc, _ = F.load_net_numpy()
fwd = F.make_forward(Ws, bs, xm, xs, ysc)

store, summary = {}, {"config": {"dt": DT, "n_gyrations": N_GYR, "n_steps": N_STEPS,
                                 "taus": CONFIGS, "net_params": 52102}, "results": {}}
for cname, tau in CONFIGS.items():
    summary["results"][cname] = {}
    for mode in MODES:
        t0 = time.time()
        d = F.run(mode, tau, DT, N_STEPS, n_samples=4000, fwd=fwd)
        el = time.time() - t0
        t = d["t"]; env = np.maximum.accumulate(d["env"])
        store[f"{cname}/{mode}/t"] = t
        store[f"{cname}/{mode}/e_err"] = d["e_err"]
        store[f"{cname}/{mode}/mu_err"] = d["mu_err"]
        store[f"{cname}/{mode}/env"] = env
        res = {"seconds": el, "us_per_step": el / N_STEPS * 1e6, "horizons": {}}
        for H in HORIZONS:
            tH = H * TWO_PI
            m = t <= tH
            if m.sum() < 5: continue
            half = m.sum() // 2
            phys = 1.0 - np.exp(-tH / tau)
            emax = float(env[m][-1]); emed = float(np.median(d["e_err"][m][half:]))
            res["horizons"][f"{H:.0e}"] = {
                "t_end": tH,
                "energy_err_max": emax,
                "energy_err_median_2nd_half": emed,
                "mu_err_at_end": float(d["mu_err"][m][-1]),
                "mu_err_max": float(np.max(d["mu_err"][m])),
                "physical_signal": phys,
                "signal_over_err_max": phys / max(emax, 1e-300),
                "signal_over_err_median": phys / max(emed, 1e-300)}
        # power-law fit of the envelope over the last two decades of time
        sel = (t > t[-1] / 100) & (env > 0)
        if sel.sum() > 10:
            p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
            res["envelope_powerlaw_exponent"] = float(p[0])
        summary["results"][cname][mode] = res
        print(f"{cname:22s} {mode:6s} {el:6.1f}s  "
              f"exp={res.get('envelope_powerlaw_exponent', float('nan')):6.3f}  "
              f"E_err(1e5)={res['horizons']['1e+05']['energy_err_max']:.3e}  "
              f"mu_err={res['horizons']['1e+05']['mu_err_max']:.3e}")

np.savez_compressed(os.path.join(HERE, "long_runs.npz"), **store)
json.dump(summary, open(os.path.join(HERE, "long_runs_summary.json"), "w"), indent=2)
print("\nсохранено: long_runs.npz, long_runs_summary.json")
