"""Independent recheck of the F0.2/F0.3 variational-integrator claims.

  (5) gauge: E_of == shipped E, curl A == B (own random points, own FD)
  (6) convergence order vs DOP853 (own rerun at 3 resolutions + refit of
      their validation.json points), base symplecticity (own FD, 2 epsilons)
  (9) perturbed symplecticity inside vs outside (own loop, seed 11, fac 0.1/1/10)
  (10) numeric cond of D1D2 Ld from the ACTUAL code var_step (genuine FD of the
       momentum equation, not the analytic formula), + defect margin spot check
"""
import sys, os, json
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_variational')))

from fields import DecayingField
import varint as V

TAU = 1.2e5
H = 0.3
R0 = np.array([1.0, 0.0, 0.0]); V0 = np.array([0.0, 1.0, 0.0])
out = {}

# ---------------------------------------------------------------- (5) gauge
fld = DecayingField(B0=1.0, tau=TAU)
rng = np.random.default_rng(12345)          # different points than validate.py
mE = mC = 0.0
for _ in range(300):
    q = rng.normal(0, 3.0, 3); t = rng.uniform(0, 120.0)
    mE = max(mE, np.max(np.abs(V.E_of(q[None], t, TAU)[0] - fld.E(q, t))))
    eps = 1e-6; J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = eps
        J[:, j] = (V.A_of((q+dq)[None], t, TAU)[0] - V.A_of((q-dq)[None], t, TAU)[0])/(2*eps)
    curl = np.array([J[2,1]-J[1,2], J[0,2]-J[2,0], J[1,0]-J[0,1]])
    mC = max(mC, np.max(np.abs(curl - fld.B(q, t))))
out["gauge"] = {"E_mismatch": float(mE), "curlA_minus_B": float(mC)}
print(f"(5) gauge: |E-E_shipped|={mE:.3e}  |curlA-B|={mC:.3e}")

# ---------------------------------------------------------------- (6) order
def rhs(t, y):
    r, v = y[:3], y[3:]
    E = np.atleast_1d(fld.E(r, t)).ravel(); B = np.atleast_1d(fld.B(r, t)).ravel()
    return np.concatenate([v, -1.0*(E + np.cross(v, B))])
sol = solve_ivp(rhs, (0, 120.0), np.concatenate([R0, V0]), method="DOP853",
                rtol=1e-13, atol=1e-15, t_eval=[120.0])
r_ref = sol.y[:3, -1]

errs = []
for h in (0.3, 0.075, 0.01875):
    n = int(round(120.0/h))
    q = R0[None].copy(); p = V0[None] - V.A_of(R0[None], 0.0, TAU); t = 0.0
    for _ in range(n):
        q, p, _ = V.var_step(q, p, t, h, TAU, None); t += h
    e = float(np.linalg.norm(q[0]-r_ref)); errs.append((h, e))
    print(f"(6) h={h:.5f} err_r={e:.4e}")
hs = np.array([x[0] for x in errs]); es = np.array([x[1] for x in errs])
order_own = float(np.polyfit(np.log10(hs), np.log10(es), 1)[0])
val = json.load(open(os.path.join(HERE, '..', 'f0_variational', 'validation.json')))
pts = val["convergence"]["points"]
hj = np.array([c["h"] for c in pts]); ej = np.array([c["err_r"] for c in pts])
order_refit = float(np.polyfit(np.log10(hj), np.log10(ej), 1)[0])
# pairwise orders (the honest local slope)
pw = list(np.log(ej[:-1]/ej[1:])/np.log(hj[:-1]/hj[1:]))
out["order"] = {"own_3pt": order_own, "refit_their_6pt": order_refit,
                "pairwise": [float(x) for x in pw]}
print(f"(6) order: own={order_own:.3f} refit={order_refit:.3f} pairwise={['%.3f'%x for x in pw]}")

# ------------------------------------------------- base symplecticity, own FD
OM = np.block([[np.zeros((3,3)), np.eye(3)], [-np.eye(3), np.zeros((3,3))]])
def stepz(z, t, net):
    q1, p1, _ = V.var_step(z[None,:3], z[None,3:], t, H, TAU, net, tol=1e-15)
    return np.concatenate([q1[0], p1[0]])
def sympl_resid(z, t, net, eps):
    J = np.zeros((6,6))
    for j in range(6):
        dz = np.zeros(6); dz[j] = eps
        J[:, j] = (stepz(z+dz, t, net) - stepz(z-dz, t, net))/(2*eps)
    return float(np.max(np.abs(J.T @ OM @ J - OM)))
z0 = np.concatenate([R0, V0 - V.A_of(R0[None], 0.0, TAU)[0]])
base = {f"eps={e:g}": sympl_resid(z0, 0.0, None, e) for e in (1e-7, 1e-6, 1e-5)}
out["base_symplecticity"] = base
print("(6) base symplecticity residual:", {k: '%.2e'%v for k,v in base.items()})

# -------------------------------------- (9) perturbed symplecticity, own loop
cal = json.load(open(os.path.join(HERE, '..', 'f0_variational', 'calibration.json')))
def step_add(z, t, net):
    q1, p1, _ = V.var_step(z[None,:3], z[None,3:], t, H, TAU, None)
    v1 = p1 + V.A_of(q1, t+H, TAU)
    d1, d2 = net.grads(z[None,:3], q1, t)
    v1 = v1 + (d1 + d2)
    p1 = v1 - V.A_of(q1, t+H, TAU)
    return np.concatenate([q1[0], p1[0]])
def sympl_resid_fn(fn, z, t, net, eps=1e-6):
    J = np.zeros((6,6))
    for j in range(6):
        dz = np.zeros(6); dz[j] = eps
        J[:, j] = (fn(z+dz, t, net) - fn(z-dz, t, net))/(2*eps)
    return float(np.max(np.abs(J.T @ OM @ J - OM)))
rows = []
for fac in (0.1, 1.0, 10.0):
    av = np.array([cal["varnet"]["amp_calibrated"][0]*fac])
    aa = np.array([cal["additive"]["amp_calibrated"][0]*fac])
    nv = V.DeltaLNet(1, [11], av, hidden=32, t_scale=120.0)
    na = V.DeltaLNet(1, [11], aa, hidden=32, t_scale=120.0)
    z = z0.copy(); t = 0.0; wv = wa = 0.0
    for i in range(40):
        wv = max(wv, sympl_resid_fn(stepz, z, t, nv))
        wa = max(wa, sympl_resid_fn(step_add, z, t, na))
        z = stepz(z, t, nv); t += H
    rows.append({"fac": fac, "varnet": wv, "additive": wa})
    print(f"(9) fac={fac:4.1f} varnet={wv:.2e} additive={wa:.2e}")
out["perturbed_symplecticity"] = rows

# ---------------- (10) genuine numeric D1D2 Ld from the code + defect margin
# The code path: var_step solves p = M D + b. Momentum p as a function of
# (q_k fixed, q_k1) is p(q1) = -D1 Ld(qk, q1). Differentiate THE CODE's inverse:
# invert var_step: given qk and q1 = qk + D, p = M D + b (per _solve_M). Build p
# from the code by requesting the step that maps to q1 and reading back p:
def p_of_q1(qk, q1, t, h):
    # use the code's own algebra: solve for p that produces q1 in one step
    # var_step: D = M^{-1}(p - b) -> p = M D + b; M,b from the code's internals
    tm = t + 0.5*h; a = 0.5*V.Bz_of(tm, TAU)
    b = np.array([a*qk[1], -a*qk[0], 0.0])
    D = q1 - qk
    # apply M from _solve_M's definition by inverting _solve_M numerically:
    # _solve_M(rhs) = M^{-1} rhs; find p-b s.t. _solve_M(p-b)=D via linear solve
    # build M^{-1} columns from the code, then invert
    Minv = np.column_stack([V._solve_M(e[None], a, h)[0] for e in np.eye(3)])
    M = np.linalg.inv(Minv)
    return M @ D + b
rows10 = []
for h in (0.3, 2*np.pi):
    qk = R0.copy(); t = 0.0
    eps = 1e-6; Jn = np.zeros((3,3))
    q1c = qk + np.array([0.05, 0.02, 0.01])
    for j in range(3):
        dq = np.zeros(3); dq[j] = eps
        Jn[:, j] = (p_of_q1(qk, q1c+dq, t, h) - p_of_q1(qk, q1c-dq, t, h))/(2*eps)
    c = float(np.linalg.cond(Jn))
    a = 0.5*V.Bz_of(t+0.5*h, TAU)
    rows10.append({"h": float(h), "cond_numeric_from_code": c,
                   "cond_analytic": float(np.sqrt(1+(a*h)**2))})
    print(f"(10) h={h:.4f} cond(code)={c:.4f} analytic={np.sqrt(1+(a*h)**2):.4f}")
out["cond_from_code"] = rows10

# defect margin, fac=1, 100 steps
amps = np.array(cal["varnet"]["amp_calibrated"])
net = V.DeltaLNet(5, [11,22,33,44,55], amps, hidden=32, t_scale=120.0)
q = np.tile(R0, (5,1)); p = np.tile(V0, (5,1)) - V.A_of(np.tile(R0,(5,1)), 0.0, TAU)
t = 0.0; worst = 0.0; eps = 1e-5
for i in range(100):
    for j in range(3):
        dq = np.zeros((5,3)); dq[:, j] = eps
        gp, _ = net.grads(q, q+dq, t); gm, _ = net.grads(q, q-dq, t)
        worst = max(worst, float(np.max(np.abs((gp-gm)/(2*eps)))))
    q, p, _ = V.var_step(q, p, t, H, TAU, net); t += H
out["defect_margin"] = {"max_abs_D1D2_dLd": worst, "sigma_min": 1/H,
                        "ratio": worst*H}
print(f"(10) |D1D2 dLd|max={worst:.3e} vs sigma_min={1/H:.3f} ratio={worst*H:.2e}")

json.dump(out, open(os.path.join(HERE, "v_varint.json"), "w"), indent=1)
print("saved v_varint.json")
