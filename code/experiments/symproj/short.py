"""Short-horizon check (19.1 gyr, article config) + crossover for symmetric projection."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "horizon"))
import symproj as S, fast as F

TWO_PI = 2*np.pi; DT = S.DT_WORK; TAU = S.TAU_MAIN
fwd = S.load_forward()
out = {}

# ---------- 1. article horizon, t_final = 120 (19.1 gyrations) ----------
N = int(round(120.0/DT))
refine = 150
_, Rref, _, _ = F.fine_reference(TAU, DT/refine, N*refine, refine)
E0 = 0.5
res = {}
for mode in ["boris", "raw", "proj", "sym"]:
    d = S.run(mode, TAU, DT, N, fwd=fwd, base="shipped", n_samples=N, keep_traj=True)
    err = np.linalg.norm(d["traj"] - Rref[:N], axis=1)
    half = N//2
    res[mode] = {"pos_err_rms": float(np.sqrt(np.mean(err**2))),
                 "pos_err_final": float(err[-1]),
                 "energy_err_median_2nd_half": float(np.median(d["e_err"][half:]))}
phys = 1.0-np.exp(-90.0/TAU)
for m in res:
    res[m]["signal_over_energy_err"] = phys/max(res[m]["energy_err_median_2nd_half"],1e-300)
    res[m]["traj_gain_over_boris"] = res["boris"]["pos_err_rms"]/res[m]["pos_err_rms"]
out["article_horizon_19.1gyr"] = {"physical_signal_median": phys, "schemes": res}
print(f"{'схема':8s}{'поз.ошибка rms':>16s}{'энерг.ошибка':>15s}{'сигнал/ошибка':>15s}{'выигрыш':>10s}")
for m in ["boris","raw","proj","sym"]:
    r = res[m]
    print(f"{m:8s}{r['pos_err_rms']:>16.4e}{r['energy_err_median_2nd_half']:>15.4e}"
          f"{r['signal_over_energy_err']:>15.2f}{r['traj_gain_over_boris']:>10.1f}")

# ---------- 2. crossover: where does the hybrid stop beating Boris ----------
H = 1000
n_w = int(round(H*TWO_PI/DT))
_, Rr, _, _ = F.fine_reference(TAU, DT/refine, n_w*refine, refine)
tg = np.arange(1, n_w+1)*DT/TWO_PI
crms = {}
for mode in ["boris","proj","sym"]:
    d = S.run(mode, TAU, DT, n_w, fwd=fwd, base="shipped", n_samples=100, keep_traj=True)
    e = np.linalg.norm(d["traj"] - Rr[:n_w], axis=1)
    crms[mode] = np.sqrt(np.cumsum(e**2)/np.arange(1, n_w+1))
cr = {}
for mode in ["proj","sym"]:
    ratio = crms["boris"]/crms[mode]
    i = np.where(ratio < 1)[0]
    cr[mode] = float(tg[i[0]]) if len(i) else None
    k = np.where(crms[mode] > 1.0)[0]
    cr[mode+"_reaches_1_larmor_gyr"] = float(tg[k[0]]) if len(k) else None
k = np.where(crms["boris"] > 1.0)[0]
cr["boris_reaches_1_larmor_gyr"] = float(tg[k[0]]) if len(k) else None
out["crossover"] = cr
print(f"\nинверсия выигрыша: односторонняя {cr['proj']}, симметричная {cr['sym']} гир.")
print(f"порог 1 r_L: boris {cr['boris_reaches_1_larmor_gyr']:.1f}, "
      f"одност. {cr['proj_reaches_1_larmor_gyr']:.1f}, симм. {cr['sym_reaches_1_larmor_gyr']:.1f} гир.")

json.dump(out, open(os.path.join(HERE,"short_horizon.json"),"w"), indent=2)
