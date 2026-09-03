"""t4: discrete Noether for rotation-invariant dL_d with explicit time
dependence (K5), on the actual variational integrator of f0_variational.

Claim: if dL_d(q_k, q_{k+1}, t) is invariant under simultaneous rotation of
q_k, q_{k+1} about z (arbitrary time dependence allowed), the discrete
angular momentum J = p . (z x q) = x p_y - y p_x is conserved EXACTLY
(up to the fixed-point solver tolerance), for ANY 'weights' -- here even for
a resonant coherent defect that pumps energy secularly.

Control: the non-invariant resonant defect of v_caveat2 (force along x)
breaks J at the size of the defect.
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_variational')))
import varint as V

TAU_Q = 1.2e8
H = 0.3
TWO_PI = 2 * np.pi
OM_H = 2 * np.arctan(H / 2) / H
AMP = 3.5e-5                       # 100x the v_caveat2 amplitude: clear signal


class InvariantNet:
    """dL_d = AMP * sin(om_h * tm) * (z . (q_k x q_{k+1}))  -- rotation
    invariant, resonantly time dependent."""
    def grads(self, qk, qk1, t):
        s = AMP * np.sin(OM_H * (t + 0.5 * H))
        d1 = np.zeros_like(qk); d2 = np.zeros_like(qk)
        d1[:, 0] = s * qk1[:, 1]; d1[:, 1] = -s * qk1[:, 0]
        d2[:, 0] = -s * qk[:, 1]; d2[:, 1] = s * qk[:, 0]
        return d1, d2


class NonInvariantNet:
    """dL_d = AMP * sin(om_h * tm) * x_m  (v_caveat2 defect, scaled)."""
    def grads(self, qk, qk1, t):
        g = np.zeros_like(qk)
        g[:, 0] = 0.5 * AMP * np.sin(OM_H * (t + 0.5 * H))
        return g, g.copy()


def run(net, n_gyr=1000):
    n = int(round(n_gyr * TWO_PI / H))
    q = np.array([[1.0, 0.0, 0.0]]); v = np.array([[0.0, 1.0, 0.0]])
    t = 0.0
    p = v - V.A_of(q, t, TAU_Q)
    J0 = q[0, 0] * p[0, 1] - q[0, 1] * p[0, 0]
    E0 = 0.5
    Jdrift = 0.0; Edev = 0.0
    for i in range(n):
        q, p, _ = V.var_step(q, p, t, H, TAU_Q, net, tol=1e-15)
        t += H
        J = q[0, 0] * p[0, 1] - q[0, 1] * p[0, 0]
        Jdrift = max(Jdrift, abs(J - J0))
        v1 = p + V.A_of(q, t, TAU_Q)
        Edev = max(Edev, abs(0.5 * np.sum(v1 * v1) - E0 * np.exp(-t / TAU_Q)) / E0)
    return {"max_J_drift": float(Jdrift), "max_energy_dev": float(Edev)}


out = {"base": run(None), "invariant_resonant": run(InvariantNet()),
       "noninvariant_resonant": run(NonInvariantNet())}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "t4_noether.json"), "w"), indent=1)
