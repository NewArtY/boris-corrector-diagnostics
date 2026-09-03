"""Ф0.2 step 1 -- validation of the variational integrator before any use.

Three checks:
  1. curl A == B and -dA/dt == E of fields/decaying_field.py  (gauge is right)
  2. the DEL map reproduces DOP853 and converges at 2nd order  (physics is right)
  3. symplecticity of the base map                            (structure is right)
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

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
Q, M = -1.0, 1.0
out = {}

# --------------------------------------------------------------- 1. gauge
fld = DecayingField(B0=1.0, tau=TAU_MAIN)
rng = np.random.default_rng(0)
max_E, max_curl = 0.0, 0.0
for _ in range(200):
    q = rng.normal(0, 2.0, 3)
    t = rng.uniform(0, T_FINAL)
    max_E = max(max_E, np.max(np.abs(V.E_of(q[None], t, TAU_MAIN)[0] - fld.E(q, t))))
    # numerical curl of A
    eps = 1e-6
    J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = eps
        J[:, j] = (V.A_of((q + dq)[None], t, TAU_MAIN)[0]
                   - V.A_of((q - dq)[None], t, TAU_MAIN)[0]) / (2 * eps)
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    max_curl = max(max_curl, np.max(np.abs(curl - fld.B(q, t))))
out["gauge"] = {"max_abs_E_mismatch_vs_shipped_field": float(max_E),
                "max_abs_curlA_minus_B": float(max_curl)}
print(f"1. gauge:  |E-E_shipped|max={max_E:.3e}   |curlA-B|max={max_curl:.3e}")

# --------------------------------------------------------------- 2. order
def rhs(t, y):
    r, v = y[:3], y[3:]
    E = np.atleast_1d(fld.E(r, t)).ravel()
    B = np.atleast_1d(fld.B(r, t)).ravel()
    return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])

sol = solve_ivp(rhs, (0.0, T_FINAL), np.concatenate([R0, V0]), method="DOP853",
                rtol=1e-13, atol=1e-15, t_eval=[T_FINAL])
assert sol.success
r_ref, v_ref = sol.y[:3, -1], sol.y[3:, -1]

conv = []
for frac in [1, 2, 4, 8, 16, 32]:
    h = DT_WORK / frac
    n = int(round(T_FINAL / h))
    d = V.integrate('base', TAU_MAIN, h, n, n_samples=4, n_ens=1)
    # re-run keeping the final state
    q = R0.copy()[None]; t = 0.0
    p = (V0 - V.A_of(R0[None], 0.0, TAU_MAIN))[None][0]
    q, p = R0.copy()[None], (V0[None] - V.A_of(R0[None], 0.0, TAU_MAIN))
    for i in range(n):
        q, p, _ = V.var_step(q, p, t, h, TAU_MAIN, None)
        t += h
    v_end = (p + V.A_of(q, t, TAU_MAIN))[0]
    er = float(np.linalg.norm(q[0] - r_ref))
    ev = float(np.linalg.norm(v_end - v_ref))
    conv.append({"h": h, "omega_dt": h, "n_steps": n, "err_r": er, "err_v": ev})
    print(f"2. h={h:.5f}  err_r={er:.3e}  err_v={ev:.3e}")

hs = np.array([c["h"] for c in conv]); ers = np.array([c["err_r"] for c in conv])
ordr = float(np.polyfit(np.log10(hs), np.log10(ers), 1)[0])
out["convergence"] = {"points": conv, "measured_order_position": ordr,
                      "reference": "DOP853 rtol=1e-13 atol=1e-15"}
print(f"2. measured order (position) = {ordr:.3f}")

# --------------------------------------------------------------- 3. symplecticity
h = DT_WORK
q0 = R0.copy(); p0 = V0 - V.A_of(R0[None], 0.0, TAU_MAIN)[0]
eps = 1e-7
def step_qp(q, p, t):
    q1, p1, _ = V.var_step(q[None], p[None], t, h, TAU_MAIN, None)
    return q1[0], p1[0]
Jm = np.zeros((6, 6))
z0 = np.concatenate([q0, p0])
for j in range(6):
    dz = np.zeros(6); dz[j] = eps
    zp = np.concatenate(step_qp((z0 + dz)[:3], (z0 + dz)[3:], 0.0))
    zm = np.concatenate(step_qp((z0 - dz)[:3], (z0 - dz)[3:], 0.0))
    Jm[:, j] = (zp - zm) / (2 * eps)
Om = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
resid = float(np.max(np.abs(Jm.T @ Om @ Jm - Om)))
out["symplecticity"] = {"max_abs_JtOmegaJ_minus_Omega": resid,
                        "note": "finite-difference Jacobian, eps=1e-7"}
print(f"3. symplecticity residual = {resid:.3e}")

json.dump(out, open(os.path.join(HERE, "validation.json"), "w"), indent=2)
print("\nsaved validation.json")
