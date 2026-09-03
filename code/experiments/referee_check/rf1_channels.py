"""Referee check 1: Section 2 numbers of the manuscript.

Independent re-implementation (no reuse of authors' loops).
Checks: median/final angle theta, median relative speed error (2nd half),
identity residuals, RMS position error, floor median, r0-shift factor.
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp

TAU, H, TF, B0, Q = 1.2e5, 0.3, 120.0, 1.0, -1.0

def Bz(t): return B0 * np.exp(-t / TAU)
def fac(t): return 0.5 * Bz(t) / TAU

def boris_run(r0, v0, h=H, tf=TF):
    n = int(round(tf / h))
    r = np.array(r0, float); v = np.array(v0, float)
    rs = np.zeros((n + 1, 3)); vs = np.zeros((n + 1, 3)); ts = np.zeros(n + 1)
    rs[0], vs[0] = r, v
    t = 0.0
    for i in range(1, n + 1):
        f = fac(t); E = np.array([-f * r[1], f * r[0], 0.0])
        k = 0.5 * Q * h
        vm = v + k * E
        tz = k * Bz(t)
        sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vm[0] + vm[1] * tz
        vpy = vm[1] - vm[0] * tz
        vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])
        v = vp + k * E
        r = r + v * h
        t += h
        rs[i], vs[i], ts[i] = r, v, t
    return ts, rs, vs

def reference(r0, v0, t_eval, rtol=1e-13, atol=1e-14):
    def f(t, y):
        r, v = y[:3], y[3:]
        fc = fac(t)
        E = np.array([-fc * r[1], fc * r[0], 0.0])
        Bv = np.array([0.0, 0.0, Bz(t)])
        return np.concatenate([v, Q * (E + np.cross(v, Bv))])
    sol = solve_ivp(f, (0.0, t_eval[-1]), np.concatenate([r0, v0]),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:3].T, sol.y[3:].T

out = {}
r0 = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
ts, rs, vs = boris_run(r0, v0)
rref, vref = reference(r0, v0, ts)

sp = np.linalg.norm(vs, axis=1); spr = np.linalg.norm(vref, axis=1)
cosang = np.clip(np.sum(vs * vref, axis=1) / (sp * spr), -1, 1)
theta = np.degrees(np.arccos(cosang))
half = len(ts) // 2
speed_rel = np.abs(sp - spr) / spr

out["theta_median_2nd_half_deg"] = float(np.median(theta[half:]))
out["theta_final_deg"] = float(theta[-1])
out["speed_rel_err_median_2nd_half"] = float(np.median(speed_rel[half:]))
out["pos_err_rms_larmor"] = float(np.sqrt(np.mean(
    np.sum((rs - rref) ** 2, axis=1))))  # Larmor radius = 1 initially

# identity residuals
lhs = np.sum((vs - vref) ** 2, axis=1)
rhs = (sp - spr) ** 2 + 2 * sp * spr * (1 - cosang)
out["polar_identity_max_resid"] = float(np.max(np.abs(lhs - rhs)))
dE = 0.5 * (sp ** 2 - spr ** 2)
rhsE = 0.5 * (sp + spr) * (sp - spr)
out["energy_identity_max_resid"] = float(np.max(np.abs(dE - rhsE)))

# energy floor (vs exact law) and r0 shift
def floor_median(r0v):
    ts2, rs2, vs2 = boris_run(np.array(r0v, float), v0)
    sp2 = np.sum(vs2 * vs2, axis=1)
    d = np.abs(sp2 / sp2[0] - np.exp(-ts2 / TAU))
    return float(np.median(d[len(ts2) // 2:]))

f_base = floor_median([1, 0, 0]); f_shift = floor_median([1, 0.15, 0])
out["floor_base"] = f_base
out["floor_shifted"] = f_shift
out["ratio"] = f_base / f_shift

# lag formula
from scipy.integrate import quad
def lag(t):
    def integrand(s):
        Om = np.abs(Q) * Bz(s)
        return Om - (2 / H) * np.arctan(H * Om / 2)
    val, _ = quad(integrand, 0, t, limit=200)
    return np.degrees(val)
out["lag_pred_final_deg"] = lag(TF)
out["lag_pred_median_2nd_half_deg"] = float(np.median(
    [lag(t) for t in ts[half:][::10]]))

print(json.dumps(out, indent=1))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rf1_channels.json"), "w") as fjson:
    json.dump(out, fjson, indent=1)
