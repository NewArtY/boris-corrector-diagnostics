"""Validate symproj.py against the fast implementation used in I1.2."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "horizon"))
import symproj as S
import fast as F

TWO_PI = 2*np.pi
DT = F.DT_WORK; TAU = F.TAU_MAIN
N = int(round(1000*TWO_PI/DT))          # 1000 gyrations
fwd_new = S.load_forward()
Ws, bs, xm, xs, ysc, _ = F.load_net_numpy()
fwd_old = F.make_forward(Ws, bs, xm, xs, ysc)

out = {}
for mode in ["boris", "raw", "proj"]:
    a = F.run(mode, TAU, DT, N, n_samples=500, fwd=fwd_old)
    b = S.run(mode, TAU, DT, N, fwd=fwd_new, base="shipped", n_samples=500)
    rel = np.max(np.abs(a["e_err"] - b["e_err"]) / np.maximum(np.abs(a["e_err"]), 1e-300))
    out[mode] = {"max_rel_diff_vs_fast": float(rel),
                 "e_err_end_fast": float(a["e_err"][-1]),
                 "e_err_end_new": float(b["e_err"][-1])}
    print(f"{mode:6s} max rel diff vs fast.py = {rel:.3e}")

# constraint residual for symmetric projection
d = S.run("sym", TAU, DT, 20000, fwd=fwd_new, base="shipped", n_samples=500,
          collect_mu=True, freeze_net=True)
out["sym_mu_stats"] = {"mean_abs": float(np.mean(np.abs(d["mu"]))),
                       "max_abs": float(np.max(np.abs(d["mu"]))),
                       "net_evals_per_step": d["nfev_net"]/20000,
                       "boris_evals_per_step": d["nfev_boris"]/20000}
print(f"\nsym: |mu| mean={out['sym_mu_stats']['mean_abs']:.3e} "
      f"max={out['sym_mu_stats']['max_abs']:.3e}")
print(f"sym: net evals/step={out['sym_mu_stats']['net_evals_per_step']:.2f} "
      f"boris evals/step={out['sym_mu_stats']['boris_evals_per_step']:.2f}")

json.dump(out, open(os.path.join(HERE, "validation.json"), "w"), indent=2)
