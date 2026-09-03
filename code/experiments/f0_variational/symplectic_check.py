"""Ф0.2 supplement -- is the perturbed map actually symplectic?

The whole premise of variant B is: a defect living INSIDE L_d leaves the map
symplectic for any weights, whereas a correction added OUTSIDE does not.
That premise is checkable directly, and it must be checked -- otherwise a null
result in the drift test cannot be attributed to structure at all.

Measures max |J^T Omega J - Omega| along the trajectory for both branches.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from training.train_corrector_b4 import DT_WORK, TAU_MAIN
import varint as V
from calibrate import SEEDS, T_SCALE

H = DT_WORK
OM = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
EPS = 1e-6
cal = json.load(open(os.path.join(HERE, "calibration.json")))


def step_var(z, t, net1):
    q1, p1, _ = V.var_step(z[None, :3], z[None, 3:], t, H, TAU_MAIN, net1, tol=1e-15)
    return np.concatenate([q1[0], p1[0]])


def step_add(z, t, net1):
    q1, p1, _ = V.var_step(z[None, :3], z[None, 3:], t, H, TAU_MAIN, None)
    v1 = p1 + V.A_of(q1, t + H, TAU_MAIN)
    d1, d2 = net1.grads(z[None, :3], q1, t)
    v1 = v1 + (d1 + d2)
    p1 = v1 - V.A_of(q1, t + H, TAU_MAIN)
    return np.concatenate([q1[0], p1[0]])


def jac(fn, z, t, net1):
    J = np.zeros((6, 6))
    for j in range(6):
        dz = np.zeros(6); dz[j] = EPS
        J[:, j] = (fn(z + dz, t, net1) - fn(z - dz, t, net1)) / (2 * EPS)
    return J


out = {"eps": EPS, "h": H, "note": "finite-difference Jacobian; base scheme "
                                   "residual is the noise floor of the method",
       "rows": []}
for fac in [0.1, 1.0, 10.0]:
    for si, s in enumerate(SEEDS[:3]):
        amp_v = np.array([cal["varnet"]["amp_calibrated"][si] * fac])
        amp_a = np.array([cal["additive"]["amp_calibrated"][si] * fac])
        nv = V.DeltaLNet(1, [s], amp_v, hidden=32, t_scale=T_SCALE)
        na = V.DeltaLNet(1, [s], amp_a, hidden=32, t_scale=T_SCALE)
        q = np.array([[1.0, 0.0, 0.0]]); v = np.array([[0.0, 1.0, 0.0]])
        p = v - V.A_of(q, 0.0, TAU_MAIN)
        z = np.concatenate([q[0], p[0]]); t = 0.0
        wv, wa, wb = 0.0, 0.0, 0.0
        for i in range(60):
            for fn, key in ((step_var, 'v'), (step_add, 'a')):
                J = jac(fn, z, t, nv if key == 'v' else na)
                r = float(np.max(np.abs(J.T @ OM @ J - OM)))
                if key == 'v':
                    wv = max(wv, r)
                else:
                    wa = max(wa, r)
            Jb = jac(lambda zz, tt, _n: step_var(zz, tt, None), z, t, None)
            wb = max(wb, float(np.max(np.abs(Jb.T @ OM @ Jb - OM))))
            z = step_var(z, t, nv); t += H
        out["rows"].append({"amp_factor": fac, "seed": s,
                            "symplectic_resid_base": wb,
                            "symplectic_resid_varnet": wv,
                            "symplectic_resid_additive": wa})
        print(f"amp={fac:5.1f}x seed={s:3d}  base={wb:.2e}  "
              f"varnet={wv:.2e}  additive={wa:.2e}")

json.dump(out, open(os.path.join(HERE, "symplectic.json"), "w"), indent=2)
print("\nsaved symplectic.json")
