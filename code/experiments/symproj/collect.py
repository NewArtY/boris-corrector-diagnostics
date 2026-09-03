"""Rebuild the consolidated summary from the per-run npz files."""
import os, sys, json
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import symproj as S
TWO_PI=2*np.pi
TAUS={"paper":S.TAU_MAIN,"quasistatic":1.2e8}
out={"note":"envelope exponent fitted over the last 2 decades of t","results":{}}
for cname in ["paper","quasistatic"]:
    out["results"][cname]={}
    for base in ["shipped","staggered"]:
        f=os.path.join(HERE,f"env_{cname}_{base}.npz")
        if not os.path.exists(f): continue
        z=np.load(f)
        modes=sorted({k.split("/")[1] for k in z.files})
        for mode in modes:
            t=z[f"{base}/{mode}/t"]; env=np.maximum.accumulate(z[f"{base}/{mode}/env"])
            row={"envelope_exponent":S.envelope_exponent(t,env)}
            for H in [1e3,1e4,1e5]:
                m=t<=H*TWO_PI
                if m.sum()<5: continue
                phys=1.0-np.exp(-H*TWO_PI/TAUS[cname]); emax=float(env[m][-1])
                row[f"E_err_{H:.0e}"]=emax
                row[f"signal_over_err_{H:.0e}"]=phys/max(emax,1e-300)
            out["results"][cname][f"{base}/{mode}"]=row
json.dump(out,open(os.path.join(HERE,"summary.json"),"w"),indent=2)
hdr=f"{'конфигурация':14s}{'схема':22s}{'показатель':>11s}{'E(1e3)':>11s}{'E(1e4)':>11s}{'E(1e5)':>11s}"
print(hdr); print("-"*len(hdr))
for c,v in out["results"].items():
    for k,r in v.items():
        print(f"{c:14s}{k:22s}{r['envelope_exponent']:>11.3f}"
              f"{r.get('E_err_1e+03',float('nan')):>11.3e}"
              f"{r.get('E_err_1e+04',float('nan')):>11.3e}"
              f"{r.get('E_err_1e+05',float('nan')):>11.3e}")
