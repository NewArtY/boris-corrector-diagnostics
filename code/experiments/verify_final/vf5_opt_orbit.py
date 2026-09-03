"""vf5: (a) true (50-digit) optimal r0 floor for the shipped Boris energy
diagnostic -- is the claimed 2.73e7 collapse a real quantity?
(b) distance from the Boris trajectory to the exact orbit AS A CURVE
(claimed 0.1837 r_L median, 21.8% of the simultaneous error)."""
import json, os
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ------------------------------------------------------------------- (a)
mp.mp.dps = 50
TAU = mp.mpf("1.2e5"); H = mp.mpf("0.3"); Q = mp.mpf(-1)

def floor_mp(x0, y0, n=400):
    r = [mp.mpf(x0), mp.mpf(y0)]; v = [mp.mpf(0), mp.mpf(1)]; t = mp.mpf(0)
    devs = []
    for i in range(n):
        Bz = mp.e ** (-t / TAU); f = Bz / (2 * TAU)
        E = [-f * r[1], f * r[0]]
        k = Q * H / 2
        vm = [v[0] + k * E[0], v[1] + k * E[1]]
        tz = k * Bz; sz = 2 * tz / (1 + tz * tz)
        vpx = vm[0] + vm[1] * tz; vpy = vm[1] - vm[0] * tz
        v = [vm[0] + vpy * sz + k * E[0], vm[1] - vpx * sz + k * E[1]]
        r = [r[0] + v[0] * H, r[1] + v[1] * H]
        t += H
        devs.append(abs(v[0] * v[0] + v[1] * v[1] - mp.e ** (-t / TAU)))
    s = sorted(devs[n // 2:])
    return s[len(s) // 2]

# Nelder-Mead in float on the mp objective (few dozen evals)
from scipy.optimize import minimize
calls = []
def obj(p):
    val = floor_mp(repr(float(p[0])), repr(float(p[1])))
    calls.append((p[0], p[1], float(val)))
    return float(mp.log(val, 10))

o = minimize(obj, np.array([0.9999987, 0.1500041]), method="Nelder-Mead",
             options={"xatol": 1e-10, "fatol": 1e-3, "maxiter": 120})
out["mp_true_optimum"] = {"r0": [float(o.x[0]), float(o.x[1])],
                          "floor": float(10 ** o.fun),
                          "n_evals": len(calls),
                          "ratio_base_over_opt": float(1.2499e-6 / 10 ** o.fun)}

# ------------------------------------------------------------------- (b)
TAUf, Hf, TF = 1.2e5, 0.3, 120.0
def Bzf(t): return np.exp(-t / TAUf)
def facf(t): return 0.5 * Bzf(t) / TAUf

def boris_run(r0, v0, h=Hf, tf=TF):
    n = int(round(tf / h))
    r = np.array(r0, float); v = np.array(v0, float)
    rs = np.zeros((n + 1, 2)); ts = np.zeros(n + 1); rs[0] = r[:2]
    t = 0.0
    for i in range(1, n + 1):
        f = facf(t); E = np.array([-f * r[1], f * r[0], 0.0])
        k = -0.5 * h
        vm = v + k * E
        tz = k * Bzf(t); sz = 2 * tz / (1 + tz * tz)
        vpx = vm[0] + vm[1] * tz; vpy = vm[1] - vm[0] * tz
        v = np.array([vm[0] + vpy * sz + k * E[0], vm[1] - vpx * sz + k * E[1], vm[2]])
        r = r + v * h
        t += h
        rs[i] = r[:2]; ts[i] = t
    return ts, rs

ts, rs = boris_run((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
def rhs(t, y):
    r, v = y[:2], y[2:]
    fc = facf(t)
    E = np.array([-fc * r[1], fc * r[0]])
    Bz = Bzf(t)
    # q=-1: a = -(E + v x B); v x B (2D, B=Bz z) = (v_y Bz, -v_x Bz)
    return np.concatenate([v, -(E + np.array([v[1] * Bz, -v[0] * Bz]))])
sol = solve_ivp(rhs, (0.0, TF * 1.02), np.array([1.0, 0.0, 0.0, 1.0]),
                method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)

hh = len(ts) // 2
dist_curve = []; dist_simul = []
for i in range(hh, len(ts)):
    p = rs[i]
    sim = sol.sol(ts[i])[:2]
    dist_simul.append(np.hypot(*(p - sim)))
    g = lambda s: np.hypot(*(p - sol.sol(s)[:2]))
    lo, hi = max(0.0, ts[i] - 4.0), min(TF * 1.02, ts[i] + 4.0)
    m = minimize_scalar(g, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-10})
    dist_curve.append(m.fun)
dist_curve = np.array(dist_curve); dist_simul = np.array(dist_simul)
out["orbit_distance"] = {
    "median_dist_to_curve": float(np.median(dist_curve)),
    "median_simultaneous_err": float(np.median(dist_simul)),
    "ratio_percent": float(np.median(dist_curve) / np.median(dist_simul) * 100)}

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf5_opt_orbit.json"), "w"), indent=1)
