"""Envelope growth: one-sided vs symmetric projection. Usage: main.py <config> <base>"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import symproj as S

TWO_PI = 2*np.pi
DT = S.DT_WORK
N_GYR = 100000
N_STEPS = int(round(N_GYR*TWO_PI/DT))
HORIZONS = [1e3, 1e4, 1e5]
CONFIGS = {"paper": S.TAU_MAIN, "quasistatic": 1.2e8}
MODES = {"shipped": ["boris","raw","proj","sym"], "staggered": ["boris","proj","sym"]}

cname, base = sys.argv[1], sys.argv[2]
tau = CONFIGS[cname]
fwd = S.load_forward()
jf = os.path.join(HERE, "envelope_growth.json")
summary = json.load(open(jf)) if os.path.exists(jf) else {
    "config": {"dt": DT, "n_gyrations": N_GYR, "n_steps": N_STEPS, "taus": CONFIGS},
    "results": {}}
summary["results"].setdefault(cname, {})
store = {}
nf = os.path.join(HERE, f"env_{cname}_{base}.npz")

for mode in MODES[base]:
    key = f"{base}/{mode}"
    t0 = time.time()
    d = S.run(mode, tau, DT, N_STEPS, fwd=fwd, base=base, n_samples=4000)
    el = time.time()-t0
    t = d["t"]; env = np.maximum.accumulate(d["env"])
    store[f"{key}/t"] = t; store[f"{key}/env"] = env; store[f"{key}/e_err"] = d["e_err"]
    res = {"seconds": el, "us_per_step": el/N_STEPS*1e6,
           "net_evals_per_step": d["nfev_net"]/N_STEPS,
           "boris_evals_per_step": d["nfev_boris"]/N_STEPS,
           "envelope_exponent": S.envelope_exponent(t, env), "horizons": {}}
    for H in HORIZONS:
        m = t <= H*TWO_PI
        if m.sum() < 5: continue
        phys = 1.0-np.exp(-H*TWO_PI/tau)
        emax = float(env[m][-1])
        res["horizons"][f"{H:.0e}"] = {"energy_err_max": emax, "physical_signal": phys,
                                       "signal_over_err": phys/max(emax,1e-300)}
    summary["results"][cname][key] = res
    print(f"{cname:12s} {key:18s} {el:6.1f}s  exp={res['envelope_exponent']:7.3f}  "
          f"E(1e5)={res['horizons']['1e+05']['energy_err_max']:.3e}  "
          f"S/E={res['horizons']['1e+05']['signal_over_err']:.3e}", flush=True)

np.savez_compressed(nf, **store)
json.dump(summary, open(jf,"w"), indent=2)
