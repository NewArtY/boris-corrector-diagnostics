"""Ф0.3 -- non-degeneracy of D1D2 L_d and the conjugate-point question.

Analytic result for the midpoint discretisation of the magnetic Lagrangian:

    -D1 L_d = M D + b ,  M = [[1/h, a, 0], [-a, 1/h, 0], [0, 0, 1/h]] ,  a = Bz/2
    =>  D1D2 L_d = -M    (constant in q, exactly)

    singular values of M : sqrt(1/h^2 + a^2) (x2) and 1/h
    cond(M) = sqrt(1 + (a h)^2) = sqrt(1 + (Omega_c h / 2)^2)

So the map is ALGEBRAICALLY non-degenerate for every step size: the midpoint
L_d has no conjugate points at all. What degrades near Omega_c h = 2 pi k is
FIDELITY, not invertibility -- the exact discrete Lagrangian is singular there,
the midpoint approximation simply stops representing it. Both are measured.

With a learned defect the requirement becomes
    || D1D2 dL_d ||  <  sigma_min(M) = 1/h
which is checked numerically for the calibrated nets of Ф0.2.
"""
import os, sys, json
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from fields import DecayingField
from training.train_corrector_b4 import DT_WORK, T_FINAL, TAU_MAIN
import varint as V
from calibrate import SEEDS, T_SCALE

R0 = np.array([1.0, 0.0, 0.0]); V0 = np.array([0.0, 1.0, 0.0])
out = {}

# ---------------------------------------------------- 1. cond(D1D2 L_d0)
hs = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 2.0, np.pi, 2 * np.pi,
               2 * np.pi + 0.1, 4 * np.pi, 10.0, 30.0])
rows = []
for h in hs:
    a = 0.5 * V.Bz_of(0.0, TAU_MAIN)          # Bz ~ 1 over the run
    M = np.array([[1 / h, a, 0.0], [-a, 1 / h, 0.0], [0.0, 0.0, 1 / h]])
    sv = np.linalg.svd(M, compute_uv=False)
    # numerical D1D2 L_d along the trajectory, finite differences
    q = R0.copy()[None]; p = (V0[None] - V.A_of(R0[None], 0.0, TAU_MAIN))
    t = 0.0; worst = 0.0
    for i in range(min(200, int(T_FINAL / h))):
        eps = 1e-6
        Jn = np.zeros((3, 3))
        for j in range(3):
            dq = np.zeros((1, 3)); dq[0, j] = eps
            # -D1 L_d as a function of q_{k+1}: M(q1-q)+b
            aa = 0.5 * V.Bz_of(t + 0.5 * h, TAU_MAIN)
            Mn = np.array([[1 / h, aa, 0], [-aa, 1 / h, 0], [0, 0, 1 / h]])
            Jn[:, j] = (Mn @ (q[0] + dq[0] - q[0]) - Mn @ (q[0] - dq[0] - q[0])) / (2 * eps)
        worst = max(worst, float(np.linalg.cond(Jn)))
        q, p, _ = V.var_step(q, p, t, h, TAU_MAIN, None)
        t += h
    rows.append({"h": float(h), "omega_c_h": float(h),
                 "cond_analytic": float(np.sqrt(1 + (a * h) ** 2)),
                 "cond_svd": float(sv[0] / sv[-1]),
                 "cond_numeric_traj_max": worst,
                 "sigma_min": float(sv[-1])})
out["conditioning"] = {"formula": "cond = sqrt(1 + (Omega_c h / 2)^2)", "rows": rows}
print("1. conditioning of D1D2 L_d0")
for r in rows:
    print(f"   Om*h={r['omega_c_h']:8.4f}  cond={r['cond_analytic']:8.4f} "
          f"(svd {r['cond_svd']:8.4f}, numeric {r['cond_numeric_traj_max']:8.4f})")

# ---------------------------------------------------- 2. fidelity vs h
fld = DecayingField(B0=1.0, tau=TAU_MAIN)
Q, M_ = -1.0, 1.0
def rhs(t, y):
    r, v = y[:3], y[3:]
    E = np.atleast_1d(fld.E(r, t)).ravel(); B = np.atleast_1d(fld.B(r, t)).ravel()
    return np.concatenate([v, (Q / M_) * (E + np.cross(v, B))])
sol = solve_ivp(rhs, (0.0, T_FINAL), np.concatenate([R0, V0]), method="DOP853",
                rtol=1e-13, atol=1e-15, t_eval=[T_FINAL])
r_ref = sol.y[:3, -1]

fid = []
for h in [0.05, 0.1, 0.3, 1.0, 2.0, 3.0, np.pi, 2 * np.pi, 2 * np.pi + 0.3]:
    n = max(1, int(round(T_FINAL / h)))
    q = R0.copy()[None]; p = (V0[None] - V.A_of(R0[None], 0.0, TAU_MAIN)); t = 0.0
    ok = True
    for i in range(n):
        q, p, _ = V.var_step(q, p, t, h, TAU_MAIN, None); t += h
        if not np.all(np.isfinite(q)):
            ok = False; break
    err = float(np.linalg.norm(q[0] - r_ref)) if ok else float('inf')
    fid.append({"h": float(h), "omega_c_h": float(h), "err_r_larmor": err})
    print(f"2. Om*h={h:7.4f}  |dr|={err:.4e} r_L")
out["fidelity"] = {"note": "position error at t=120 vs DOP853", "rows": fid}

# ---------------------------------------------------- 3. margin with dL_d
cal = json.load(open(os.path.join(HERE, "calibration.json")))
marg = []
for fac in [0.1, 1.0, 10.0]:
    amps = np.array(cal["varnet"]["amp_calibrated"]) * fac
    net = V.DeltaLNet(len(SEEDS), SEEDS, amps, hidden=32, t_scale=T_SCALE)
    q = np.tile(R0, (len(SEEDS), 1)); p = np.tile(V0, (len(SEEDS), 1)) - V.A_of(
        np.tile(R0, (len(SEEDS), 1)), 0.0, TAU_MAIN)
    t = 0.0; worst = 0.0
    eps = 1e-5
    for i in range(300):
        # numerical D1D2 dL_d : d/dq_k1 of D1 dL_d
        for j in range(3):
            dq = np.zeros((len(SEEDS), 3)); dq[:, j] = eps
            gp, _ = net.grads(q, q + dq, t)
            gm, _ = net.grads(q, q - dq, t)
            worst = max(worst, float(np.max(np.abs((gp - gm) / (2 * eps)))))
        q, p, _ = V.var_step(q, p, t, DT_WORK, TAU_MAIN, net)
        t += DT_WORK
    marg.append({"amp_factor": fac, "max_abs_D1D2_dLd": worst,
                 "sigma_min_M": 1.0 / DT_WORK,
                 "margin_ratio": worst / (1.0 / DT_WORK)})
    print(f"3. amp={fac:5.1f}x  |D1D2 dL_d|max={worst:.3e}  "
          f"vs sigma_min={1/DT_WORK:.3f}  ratio={worst*DT_WORK:.3e}")
out["defect_margin"] = {"h": DT_WORK, "rows": marg}

json.dump(out, open(os.path.join(HERE, "degeneracy.json"), "w"), indent=2)
print("\nsaved degeneracy.json")
