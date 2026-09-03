"""Section B of the verification: weight of the F0.2 caveat.

 1. Do the frozen random nets saturate in their t-feature (t/120 >> 1 on the
    long runs), i.e. is the tested perturbation effectively STATIC in time?
 2. Systematic ADDITIVE control: dv = eps*|v|*v_hat per step, same calibrated
    per-step amplitude. Does it reproduce the trained hybrid's secular growth
    (exponent ~1)? If yes, the missing control was constructible.
 3. Systematic INSIDE defect: dL_d = amp*sin(t_m)*x_m -- exactly symplectic by
    construction, resonant with the gyrofrequency. Does the variational branch
    still show a flat envelope? (If not, symplecticity alone does not bound the
    energy error for time-dependent defects, and the F0.2 gate is contingent on
    the perturbation class.)
 4. Same inside defect off resonance (omega=0.37) as contrast.
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_variational')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import varint as V

TAU_Q = 1.2e8
H = 0.3
TWO_PI = 2*np.pi
out = {}
t00 = time.time()

# ------------------------------------------------ 1. t-feature saturation
cal = json.load(open(os.path.join(HERE, '..', 'f0_variational', 'calibration.json')))
net = V.DeltaLNet(1, [11], [cal["varnet"]["amp_calibrated"][0]], hidden=32,
                  t_scale=120.0)
q = np.array([[1.0, 0.0, 0.0]]); q1 = np.array([[1.05, 0.02, 0.0]])
vals = {}
for t in (0.0, 60.0, 120.0, 1000.0, 1e4, 1e5, 6e5):
    d1, d2 = net.grads(q, q1, t)
    vals[t] = float(np.linalg.norm(d1) + np.linalg.norm(d2))
rel_change_late = abs(vals[6e5] - vals[1e4]) / max(vals[1e4], 1e-300)
out["t_saturation"] = {"grad_norm_by_t": {str(k): v for k, v in vals.items()},
                       "rel_change_1e4_to_6e5": rel_change_late}
print("1. |grad dL_d| vs t:", {k: '%.3e' % v for k, v in vals.items()})
print(f"   relative change t=1e4 -> 6e5: {rel_change_late:.2e}")

# ------------------------------------------------ helpers
def expo_emax(t, env_win):
    env = np.maximum.accumulate(env_win)
    sel = (t > t[-1]/100.0) & (env > 0)
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)[0] if sel.sum() > 10 else np.nan
    emax = {}
    for Hg in (1e2, 1e3, 1e4):
        m = t <= Hg*TWO_PI
        if m.sum() >= 5:
            emax[f"{Hg:.0e}"] = float(env[m][-1])
    return float(p), emax

N_GYR = 1e4
N_STEPS = int(round(N_GYR*TWO_PI/H))

def run_custom(kind, amp, omega=1.0):
    """kind: 'add_valigned' | 'inside_sin'. Returns (exponent, emax dict)."""
    q = np.array([[1.0, 0.0, 0.0]]); v = np.array([[0.0, 1.0, 0.0]])
    t = 0.0
    p = v - V.A_of(q, t, TAU_Q)
    E0 = 0.5*np.sum(v[0]*v[0])
    stride = max(1, N_STEPS//4000)
    ts, envs = [], []
    run_max = 0.0

    class SinNet:
        """dL_d = amp*sin(omega*t_m)*(qk_x+q1_x)/2 -> D1=D2=(amp/2) sin(.) x_hat"""
        def grads(self, qk, qk1, tt):
            g = np.zeros((qk.shape[0], 3))
            g[:, 0] = 0.5*amp*np.sin(omega*(tt + 0.5*H))
            return g, g.copy()
    sn = SinNet()

    for i in range(1, N_STEPS+1):
        if kind == 'add_valigned':
            q1, p1, _ = V.var_step(q, p, t, H, TAU_Q, None)
            v1 = p1 + V.A_of(q1, t+H, TAU_Q)
            nv = np.linalg.norm(v1, axis=-1, keepdims=True)
            v1 = v1 + amp*v1              # dv = amp*v  (|dv|/|v| = amp)
            p1 = v1 - V.A_of(q1, t+H, TAU_Q)
        else:
            q1, p1, _ = V.var_step(q, p, t, H, TAU_Q, sn, tol=1e-14)
            v1 = p1 + V.A_of(q1, t+H, TAU_Q)
        q, p = q1, p1
        t += H
        Ecur = 0.5*np.sum(v1*v1, axis=-1)[0]
        dev = abs(Ecur - E0*np.exp(-t/TAU_Q))/E0
        run_max = max(run_max, dev)
        if i % stride == 0 or i == N_STEPS:
            ts.append(t); envs.append(run_max); run_max = 0.0
    return expo_emax(np.array(ts), np.array(envs))

# calibrate the per-step |dv| of the sin-defect empirically to 2.2e-7
def one_step_dv_sin(amp, omega=1.0, n=2000):
    q = np.array([[1.0, 0.0, 0.0]]); v = np.array([[0.0, 1.0, 0.0]])
    t = 0.0; p = v - V.A_of(q, t, TAU_Q)
    class SinNet:
        def grads(self, qk, qk1, tt):
            g = np.zeros((qk.shape[0], 3))
            g[:, 0] = 0.5*amp*np.sin(omega*(tt + 0.5*H))
            return g, g.copy()
    sn = SinNet(); acc = 0.0
    for i in range(n):
        qc, pc, _ = V.var_step(q, p, t, H, TAU_Q, None)
        vc = pc + V.A_of(qc, t+H, TAU_Q)
        qp, pp, _ = V.var_step(q, p, t, H, TAU_Q, sn, tol=1e-15)
        vp = pp + V.A_of(qp, t+H, TAU_Q)
        acc += np.linalg.norm(vp-vc)/np.linalg.norm(vc)
        q, p, t = qc, pc, t+H
    return acc/n

TARGET = 2.2e-7          # the informative small amplitude of section 6.2
amp0 = 1e-6
got = one_step_dv_sin(amp0)
amp_sin = amp0*TARGET/got
print(f"2. sin-defect calibration: unit response {got:.3e} -> amp={amp_sin:.3e}")
got2 = one_step_dv_sin(amp_sin)
print(f"   achieved per-step dv = {got2:.3e} (target {TARGET:.1e})")

rows = []
for kind, amp, om, label in (
        ("add_valigned", TARGET, None, "systematic additive dv || v"),
        ("inside_sin", amp_sin, 1.0, "inside dL_d, resonant omega=1"),
        ("inside_sin", amp_sin, 0.37, "inside dL_d, off-resonant omega=0.37")):
    e, em = run_custom(kind, amp, omega=om if om else 1.0)
    rows.append({"kind": kind, "label": label, "amp": amp, "omega": om,
                 "exponent": e, "emax": em})
    print(f"3. {label:34s}: exponent={e:6.3f}  Emax={em}  ({time.time()-t00:.0f}s)")
out["systematic"] = rows

# symplecticity of the resonant inside defect (must stay ~1e-10)
OM = np.block([[np.zeros((3,3)), np.eye(3)], [-np.eye(3), np.zeros((3,3))]])
class SinNet2:
    def grads(self, qk, qk1, tt):
        g = np.zeros((qk.shape[0], 3))
        g[:, 0] = 0.5*amp_sin*np.sin(1.0*(tt + 0.5*H))
        return g, g.copy()
sn2 = SinNet2()
def stepz(z, t):
    q1, p1, _ = V.var_step(z[None,:3], z[None,3:], t, H, TAU_Q, sn2, tol=1e-15)
    return np.concatenate([q1[0], p1[0]])
z = np.concatenate([[1.0,0,0], [0,1.0,0] - V.A_of(np.array([[1.0,0,0]]), 0.0, TAU_Q)[0]])
w = 0.0; t = 0.0
for i in range(30):
    J = np.zeros((6,6))
    for j in range(6):
        dz = np.zeros(6); dz[j] = 1e-6
        J[:, j] = (stepz(z+dz, t) - stepz(z-dz, t))/(2e-6)
    w = max(w, float(np.max(np.abs(J.T@OM@J - OM))))
    z = stepz(z, t); t += H
out["sin_defect_symplecticity"] = w
print(f"4. resonant inside defect symplecticity residual = {w:.2e}")

json.dump(out, open(os.path.join(HERE, "v_caveat.json"), "w"), indent=1)
print(f"done {time.time()-t00:.0f}s")
