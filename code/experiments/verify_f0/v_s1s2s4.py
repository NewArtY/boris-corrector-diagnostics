"""Independent rerun of S1/S2/S4 (deterministic), output kept in verify_f0.
Replicates s1_s2_s4.py logic via its own modules, plus one extra sanity check:
the DOP853 reference is cross-checked against vps4 at dt=0.01.
"""
import sys, os, json, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_settings')))
import harness as H
from pfields import PerturbedDecaying, QuasiPeriodic, SplineField
from s1_s2_s4 import TONES, truth_field, score_against

t0 = time.time()
out = {}
field = truth_field()
n = int(round(H.T_FINAL / H.DT_WORK))
ts = np.linspace(0.0, H.T_FINAL, n + 1)
r_ref, v_ref = H.dop853(field, ts, H.T_FINAL)
sig = H.physical_signal(v_ref)
print(f"signal={sig:.6e}  ({time.time()-t0:.0f}s)")
out["signal"] = sig

# sanity: reference vs fine vps4
rs, vs, tt, fl = H.run_scheme("vps4", field, 0.01, 12000)
idx = np.searchsorted(tt, ts)
d = np.linalg.norm(rs[idx] - r_ref, axis=1).max()
print(f"reference cross-check |vps4(dt=0.01) - DOP853|max = {d:.3e}")
out["ref_crosscheck"] = float(d)

# ignoring dB
f_smooth = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
sc = score_against(f_smooth, r_ref, v_ref, H.DT_WORK, n)
out["ignoring"] = sc
print("ignoring:", {k: '%.4g' % sc[k] for k in ('pos_err_rms', 'energy_err_median_2nd_half')})

# S1
tf = truth_field()
s1 = []
for h_node in (2.0, 1.0, 0.5, 0.25):
    t_nodes = np.arange(0.0, H.T_FINAL + h_node, h_node)
    bz = np.array([tf.Bz_of_t(t) for t in t_nodes])
    sp = SplineField(t_nodes, bz)
    sc = score_against(sp, r_ref, v_ref, H.DT_WORK, n)
    s1.append({"h_node": h_node, "pos": sc["pos_err_rms"],
               "e": sc["energy_err_median_2nd_half"], "flops": sc["flops"]})
    print(f"S1 h={h_node}: pos={sc['pos_err_rms']:.4e} e={sc['energy_err_median_2nd_half']:.4e}")
out["S1"] = s1

# S2
h_node = 0.25
t_nodes = np.arange(0.0, H.T_FINAL + h_node, h_node)
bz = np.array([tf.Bz_of_t(t) for t in t_nodes])
resid = bz - 1.0*np.exp(-t_nodes/H.TAU_MAIN)
def model(p, t):
    return np.sum(p[0:3]*np.sin(np.outer(t, p[3:6]) + p[6:9]), axis=-1)
fit = least_squares(lambda p: model(p, t_nodes)-resid,
                    np.array([1e-3,1e-3,1e-3,0.9,1.6,2.7,0,0,0]),
                    xtol=1e-14, ftol=1e-14, max_nfev=20000)
rec = QuasiPeriodic(amps=fit.x[0:3], freqs=fit.x[3:6], phases=fit.x[6:9])
f_rec = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN, a=rec.a, adot=rec.adot)
sc = score_against(f_rec, r_ref, v_ref, H.DT_WORK, n)
rms_fit = float(np.sqrt(np.mean((model(fit.x, t_nodes)-resid)**2)))
out["S2"] = {"fit_rms": rms_fit, "pos": sc["pos_err_rms"],
             "e": sc["energy_err_median_2nd_half"]}
print(f"S2: fit_rms={rms_fit:.3e} pos={sc['pos_err_rms']:.4e} e={sc['energy_err_median_2nd_half']:.4e}")

# S4
f4 = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
s4 = []
for dt_outer in (1.0, 3.0):
    n_outer = int(round(H.T_FINAL/dt_outer))
    budget = H.HYBRID_FLOPS_PER_STEP * n_outer
    n_sub = max(1, int(budget/(273.0*n_outer)))
    dt_sub = dt_outer/n_sub
    n_tot = n_outer*n_sub
    ts_f = np.linspace(0.0, H.T_FINAL, n_tot+1)
    r_f, v_f = H.dop853(f4, ts_f, H.T_FINAL)
    rs, vs, tt, fl = H.run_scheme("vps4", f4, dt_sub, n_tot)
    sc = H.score(rs, vs, tt, r_f, v_f)
    s4.append({"dt_outer": dt_outer, "n_sub": n_sub, "pos": sc["pos_err_rms"],
               "e": sc["energy_err_median_2nd_half"], "flops": fl, "budget": budget})
    print(f"S4 dt={dt_outer}: x{n_sub} pos={sc['pos_err_rms']:.4e} "
          f"e={sc['energy_err_median_2nd_half']:.4e} fl={fl:.4g} vs {budget:.4g} "
          f"({time.time()-t0:.0f}s)")
out["S4"] = s4

json.dump(out, open(os.path.join(HERE, "v_s1s2s4.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
