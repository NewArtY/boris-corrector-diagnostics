"""vf3: T5 discrete Noether -- my own independent DEL implementation.

Discrete Lagrangian (midpoint, Marsden-West), same physics as f0_variational:
  L_d = |D|^2/(2h) - A(qm, tm).D + dL_d(q_k, q_{k+1}, t_k),
  A(q,t) = (Bz/2)(-q_y, q_x, 0),  q_c = -1 absorbed as in varint (L = |qd|^2/2 - A.qd)
DEL: p_k = -D1 L_d, p_{k+1} = D2 L_d.  My own fixed-point solve on D.

Tests:
  (1) invariant resonant defect  A sin(om_h tm) * (z.(q_k x q_{k+1})):
      J = x p_y - y p_x drift vs horizon (does it grow?), energy pumping.
  (2) 'any weights': random time-dependent combination of rotational invariants.
  (3) non-invariant control (force along x).
  (4) tolerance dependence: tol 1e-13 vs 1e-15.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_Q = 1.2e8
H = 0.3
TWO_PI = 2 * np.pi
OM_H = 2 * np.arctan(H / 2) / H
AMP = 3.5e-5
out = {}

def Bz(t): return np.exp(-t / TAU_Q)
def A_of(q, t):
    a = 0.5 * Bz(t)
    return np.array([-a * q[1], a * q[0], 0.0])

# ---- defects: return (D1, D2) = gradients of dL_d wrt qk, qk1 --------------
def inv_resonant(qk, qk1, t):
    s = AMP * np.sin(OM_H * (t + 0.5 * H))
    d1 = np.array([s * qk1[1], -s * qk1[0], 0.0])
    d2 = np.array([-s * qk[1], s * qk[0], 0.0])
    return d1, d2

RNG = np.random.default_rng(7)
C = RNG.normal(size=8)
def inv_random(qk, qk1, t):
    """dL_d = AMP*[c0 sin(om_h t+1) I_dot + c1 cos(2.31t) I_cross
              + c2 sin(0.77t) |qk_perp|^2 + c3 cos(om_h t) |qk1_perp|^2],
    all rotational invariants, arbitrary time dependence."""
    s0 = AMP * C[0] * np.sin(OM_H * t + 1.0)
    s1 = AMP * C[1] * np.cos(2.31 * t)
    s2 = AMP * C[2] * np.sin(0.77 * t)
    s3 = AMP * C[3] * np.cos(OM_H * t)
    # I_dot = qk_x qk1_x + qk_y qk1_y ; I_cross = qk_x qk1_y - qk_y qk1_x
    d1 = np.array([s0 * qk1[0] + s1 * qk1[1] + 2 * s2 * qk[0],
                   s0 * qk1[1] - s1 * qk1[0] + 2 * s2 * qk[1], 0.0])
    d2 = np.array([s0 * qk[0] - s1 * qk[1] + 2 * s3 * qk1[0],
                   s0 * qk[1] + s1 * qk[0] + 2 * s3 * qk1[1], 0.0])
    return d1, d2

def noninv(qk, qk1, t):
    g = np.array([0.5 * AMP * np.sin(OM_H * (t + 0.5 * H)), 0.0, 0.0])
    return g, g.copy()

def solve_D(p, q, t, defect, tol):
    """p_k = M D + b - D1dL(q, q+D):  M=[[1/h,a],[-a,1/h]] block, b = a(q_y,-q_x)."""
    tm = t + 0.5 * H
    a = 0.5 * Bz(tm)
    b = np.array([a * q[1], -a * q[0], 0.0])
    det = 1.0 / H ** 2 + a * a
    def Minv(rhs):
        return np.array([(rhs[0] / H - a * rhs[1]) / det,
                         (a * rhs[0] + rhs[1] / H) / det, H * rhs[2]])
    D = Minv(p - b)
    if defect is None:
        return D, b, 0
    for it in range(80):
        d1, _ = defect(q, q + D, t)
        Dn = Minv(p - b + d1)
        e = np.max(np.abs(Dn - D)); D = Dn
        if e < tol:
            break
    return D, b, it + 1

def run(defect, n_gyr, tol=1e-15):
    n = int(round(n_gyr * TWO_PI / H))
    q = np.array([1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0]); t = 0.0
    p = v - A_of(q, t)
    J0 = q[0] * p[1] - q[1] * p[0]
    E0 = 0.5
    Jmax = 0.0; Edev = 0.0
    checkpoints = {}
    marks = sorted(set([int(x * TWO_PI / H) for x in
                        [n_gyr / 30, n_gyr / 10, n_gyr / 3, n_gyr]]))
    for i in range(1, n + 1):
        D, b, _ = solve_D(p, q, t, defect, tol)
        q1 = q + D
        p1 = D / H + b
        if defect is not None:
            _, d2 = defect(q, q1, t)
            p1 = p1 + d2
        q, p = q1, p1
        t += H
        J = q[0] * p[1] - q[1] * p[0]
        Jmax = max(Jmax, abs(J - J0))
        v1 = p + A_of(q, t)
        Edev = max(Edev, abs(0.5 * v1 @ v1 - E0 * np.exp(-t / TAU_Q)) / E0)
        if i in marks:
            checkpoints[f"{i * H / TWO_PI:.0f}gyr"] = {
                "max_J_drift": float(Jmax), "max_E_dev": float(Edev)}
    return checkpoints

out["base_1000gyr"] = run(None, 1000)
out["invariant_resonant_1000gyr_tol1e-15"] = run(inv_resonant, 1000, tol=1e-15)
out["invariant_resonant_1000gyr_tol1e-13"] = run(inv_resonant, 1000, tol=1e-13)
out["invariant_random_1000gyr"] = run(inv_random, 1000)
out["noninvariant_1000gyr"] = run(noninv, 1000)
out["invariant_resonant_4000gyr"] = run(inv_resonant, 4000)

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf3_noether.json"), "w"), indent=1)
