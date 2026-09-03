"""Refinement: drive the inside (symplectic) defect at the integrator's OWN
numerical gyrofrequency omega_num = 2*atan(Omega h/2)/h. If the envelope then
grows secularly, symplecticity does not protect against coherent defects and
the F0.2 'flat envelope' rests on the defect being incoherent/transient."""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_variational')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import varint as V

TAU_Q = 1.2e8
H = 0.3
TWO_PI = 2*np.pi
AMP = 3.494e-07          # calibrated in v_caveat.py to per-step dv 2.2e-7
t00 = time.time()

def run_sin(omega, n_gyr=1e4, track=False):
    n_steps = int(round(n_gyr*TWO_PI/H))
    q = np.array([[1.0, 0.0, 0.0]]); v = np.array([[0.0, 1.0, 0.0]])
    t = 0.0; p = v - V.A_of(q, t, TAU_Q)
    E0 = 0.5
    class SinNet:
        def grads(self, qk, qk1, tt):
            g = np.zeros((qk.shape[0], 3))
            g[:, 0] = 0.5*AMP*np.sin(omega*(tt + 0.5*H))
            return g, g.copy()
    sn = SinNet()
    stride = max(1, n_steps//4000)
    ts, envs = [], []; run_max = 0.0
    for i in range(1, n_steps+1):
        q, p, _ = V.var_step(q, p, t, H, TAU_Q, sn, tol=1e-14)
        t += H
        v1 = p + V.A_of(q, t, TAU_Q)
        dev = abs(0.5*np.sum(v1*v1) - E0*np.exp(-t/TAU_Q))/E0
        run_max = max(run_max, dev)
        if i % stride == 0 or i == n_steps:
            ts.append(t); envs.append(run_max); run_max = 0.0
    ts = np.array(ts); env = np.maximum.accumulate(np.array(envs))
    sel = (ts > ts[-1]/100.0) & (env > 0)
    expo = float(np.polyfit(np.log10(ts[sel]), np.log10(env[sel]), 1)[0])
    emax = {f"{Hg:.0e}": float(env[ts <= Hg*TWO_PI][-1]) for Hg in (1e2, 1e3, 1e4)}
    return expo, emax

om_num = 2*np.arctan(H/2)/H     # Omega=B=1 initially
print(f"omega_num = {om_num:.7f}")
out = {"omega_num": om_num, "runs": []}
for om in (om_num, om_num*(1+2e-4), 1.0):
    e, em = run_sin(om)
    out["runs"].append({"omega": om, "exponent": e, "emax": em})
    print(f"omega={om:.7f}: exponent={e:.3f}  Emax={em}  ({time.time()-t00:.0f}s)")
json.dump(out, open(os.path.join(HERE, "v_caveat2.json"), "w"), indent=1)
