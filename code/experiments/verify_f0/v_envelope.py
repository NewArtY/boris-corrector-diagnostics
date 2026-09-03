"""Recompute envelope exponents and horizon Emax from the raw npz stores,
using the horizon/long_runs.py methodology re-implemented here (not imported):
    env = maximum.accumulate(win_env); sel = t > t[-1]/100 & env>0;
    exponent = slope of log10(env) vs log10(t); Emax(H) = env[t<=H*2pi][-1].
Cross-check against f0_summary*.json and the report tables.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FV = os.path.abspath(os.path.join(HERE, '..', 'f0_variational'))
TWO_PI = 2*np.pi

def expo_and_emax(t, env_win):
    env = np.maximum.accumulate(env_win)
    sel = (t > t[-1]/100.0) & (env > 0)
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)[0] if sel.sum() > 10 else np.nan
    emax = {}
    for Hg in (1e3, 1e4, 1e5):
        m = t <= Hg*TWO_PI
        if m.sum() >= 5:
            emax[Hg] = float(env[m][-1])
    return float(p), emax

for tag, fname, sname in (("large", "f0_runs.npz", "f0_summary.json"),
                          ("small", "f0_runs_small.npz", "f0_summary_small.json")):
    z = np.load(os.path.join(FV, fname))
    summ = json.load(open(os.path.join(FV, sname)))
    ens = summ["config"]["ensemble"]
    print(f"===== {tag} =====")
    worst = 0.0
    for cname in ("paper_tau1.2e5", "quasistatic_tau1.2e8"):
        t = z[f"{cname}/base/t"] if f"{cname}/base/t" in z else z[f"{cname}/varnet/t"]
        for mode in ("base", "varnet", "additive"):
            tm = z[f"{cname}/{mode}/t"]
            n_ens = 1 if mode == "base" else len(ens)
            for j in range(n_ens):
                e, em = expo_and_emax(tm, z[f"{cname}/{mode}/{j}/env"])
                rec = summ["results"][cname][mode]["members"][j]
                de = abs(e - rec["envelope_powerlaw_exponent"])
                worst = max(worst, de)
                for Hg, v in em.items():
                    claimed = rec["horizons"][f"{Hg:.0e}"]["energy_err_max"]
                    if claimed:
                        worst = max(worst, abs(v/claimed - 1))
    print(f"  max |recomputed - summary| over exponents & Emax ratios: {worst:.2e}")

    # aggregated table like analysis.json / the report
    for cname in ("paper_tau1.2e5", "quasistatic_tau1.2e8"):
        for mode in ("varnet", "additive"):
            tm = z[f"{cname}/{mode}/t"]
            byfac = {}
            for j, tagd in enumerate(ens):
                e, em = expo_and_emax(tm, z[f"{cname}/{mode}/{j}/env"])
                byfac.setdefault(tagd["amp_factor"], []).append((e, em))
            for fac, lst in sorted(byfac.items()):
                ex = np.array([x[0] for x in lst])
                e3 = np.median([x[1][1e3] for x in lst])
                e4 = np.median([x[1][1e4] for x in lst])
                e5 = np.median([x[1][1e5] for x in lst])
                print(f"  {cname[:5]} {mode:8s} fac={fac:g}: exp med={np.median(ex):.3f} "
                      f"max={ex.max():.3f}  Emax(med) {e3:.4g} -> {e4:.4g} -> {e5:.4g} "
                      f"(1e4/1e3={e4/e3:.3f}, 1e5/1e4={e5/e4:.4f})")
