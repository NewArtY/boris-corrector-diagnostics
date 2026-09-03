"""vf1: independent verification (final verifier).

Covers:
  P1.1  r0 shift by h/2: energy floor collapse 35 800x / 2.73e7x, trajectory
        error unchanged ("bitwise"?)  -- recomputed from scratch.
  T1    polar identities, phase drift formula, 38.11 / 50.8 deg.
  T7    velocity-channel identity + underestimation factor 1.07e6.
  T4    rotation+shift map: linear growth, exact symplecticity, chirp question.
  P3.1  general floor law (1-f)/(delta |f'|): 2t/h vs t/h.

Everything below is computed with my OWN loops (no reuse of predecessors'
code). Conventions pinned to models/boris.py + fields/decaying_field.py:
q=-1, m=1, B=B0 e^{-t/tau} zhat, E = (Bz/2tau)(-y, x, 0), fields at (r_n,t_n),
v: half-E kick, exact-tangent Boris rotation, half-E kick; r += v_new h.
"""
import json, os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TAU, H, TF, B0 = 1.2e5, 0.3, 120.0, 1.0
Q = -1.0
out = {}

def Bz(t): return B0 * np.exp(-t / TAU)
def fac(t): return 0.5 * Bz(t) / TAU          # E = fac * (zhat x r)

# ---------------------------------------------------------------- my Boris
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
        tz = k * Bz(t)                       # t-vector (z only)
        sz = 2.0 * tz / (1.0 + tz * tz)
        # v' = vm + vm x t ; v+ = vm + v' x s   (t = tz zhat)
        vpx = vm[0] + vm[1] * tz
        vpy = vm[1] - vm[0] * tz
        vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])
        v = vp + k * E
        r = r + v * h
        t += h
        rs[i], vs[i], ts[i] = r, v, t
    return ts, rs, vs

# --------------------------------------------- high-accuracy reference (ODE)
def reference(r0, v0, t_eval, rtol=1e-12, atol=1e-14):
    def f(t, y):
        r, v = y[:3], y[3:]
        fc = fac(t)
        E = np.array([-fc * r[1], fc * r[0], 0.0])
        Bv = np.array([0.0, 0.0, Bz(t)])
        return np.concatenate([v, Q * (E + np.cross(v, Bv))])
    sol = solve_ivp(f, (0.0, t_eval[-1]), np.concatenate([r0, v0]),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:3].T, sol.y[3:].T

def dev_series(ts, vs):
    sp2 = np.sum(vs * vs, axis=1)
    return np.abs(sp2 / sp2[0] - np.exp(-ts / TAU))

def floor_median(r0):
    ts, rs, vs = boris_run(np.array([r0[0], r0[1], 0.0]), (0.0, 1.0, 0.0))
    d = dev_series(ts, vs)
    return float(np.median(d[len(ts) // 2:]))

# ============================================================== P1.1
res = {}
cases = {"base r0=(1,0,0)": (1.0, 0.0), "shifted r0=(1,0.15,0)": (1.0, 0.15)}
traj = {}
for name, (x0, y0) in cases.items():
    r0 = np.array([x0, y0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
    ts, rs, vs = boris_run(r0, v0)
    d = dev_series(ts, vs)
    hh = len(ts) // 2
    rr, vr = reference(r0, v0, ts)
    perr = np.linalg.norm(rs - rr, axis=1)
    traj[name] = perr
    res[name] = {
        "floor_median_2nd_half": float(np.median(d[hh:])),
        "floor_max": float(np.max(d)),
        "pos_err_rms": float(np.sqrt(np.mean(perr ** 2))),
        "pos_err_max": float(np.max(perr)),
        "pos_err_final": float(perr[-1]),
    }
k1, k2 = list(cases)
res["ratio_base/shifted"] = res[k1]["floor_median_2nd_half"] / res[k2]["floor_median_2nd_half"]
dtraj = traj[k1] - traj[k2]
res["traj_err_series_max_abs_diff"] = float(np.max(np.abs(dtraj)))
res["traj_err_series_max_rel_diff"] = float(np.max(np.abs(dtraj[1:]) / traj[k1][1:]))
res["traj_err_rms_rel_diff"] = abs(res[k1]["pos_err_rms"] - res[k2]["pos_err_rms"]) / res[k1]["pos_err_rms"]

# optimum over r0 (2-D), Nelder-Mead on log floor
opt = minimize(lambda p: np.log10(floor_median(p) + 1e-300), np.array([1.0, 0.15]),
               method="Nelder-Mead",
               options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 4000})
res["optimum_r0"] = [float(opt.x[0]), float(opt.x[1])]
res["optimum_floor"] = float(10 ** opt.fun)
res["ratio_base/optimum"] = res[k1]["floor_median_2nd_half"] / res["optimum_floor"]

# my own guiding-centre prediction: r0 = M v0, M = h R (R-I)^{-1}
# determine rotation direction from the map itself (pure B, one step from v0)
th = 2.0 * np.arctan(H * B0 / 2.0)
_, _, vs1 = boris_run(np.array([0.0, 0.0, 0.0]), (0.0, 1.0, 0.0), h=H, tf=H)
v1 = vs1[1][:2]
ang = np.arctan2(v1[1], v1[0]) - np.pi / 2          # rotation applied to (0,1)
Rm = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
M = H * Rm @ np.linalg.inv(Rm - np.eye(2))
res["rotation_angle_one_step_deg"] = float(np.degrees(ang))
res["M_v0_prediction"] = [float(x) for x in (M @ np.array([0.0, 1.0]))]
out["P1_1_r0_shift"] = res

# ============================================================== T1 / T7
r0 = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
ts, rs, vs = boris_run(r0, v0)
rr, vr = reference(r0, v0, ts, rtol=1e-13, atol=1e-15)
sp = np.linalg.norm(vs, axis=1); spr = np.linalg.norm(vr, axis=1)
dot = np.sum(vs * vr, axis=1)
cth = np.clip(dot / (sp * spr), -1, 1)
theta = np.arccos(cth)
hh = len(ts) // 2
# my drift formula: integral of Omega(s) - (2/h) atan(h Omega(s)/2)
tt = np.linspace(0, TF, 20001)
Om = Bz(tt)
drift_rate = Om - (2.0 / H) * np.arctan(H * Om / 2.0)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (drift_rate[1:] + drift_rate[:-1]) * np.diff(tt))])
th_pred = np.interp(ts, tt, cum)
t1 = {
    "theta_median_2nd_half_deg": float(np.degrees(np.median(theta[hh:]))),
    "theta_pred_median_2nd_half_deg": float(np.degrees(np.median(th_pred[hh:]))),
    "theta_final_deg": float(np.degrees(theta[-1])),
    "theta_pred_final_deg": float(np.degrees(th_pred[-1])),
    "theta_fixed_Omega1_final_deg": float(np.degrees((1 - (2 / H) * np.arctan(H / 2)) * TF)),
    "ratio_meas_over_pred_final": float(theta[-1] / th_pred[-1]),
}
# identities (pointwise, machine precision?)
dv = vs - vr
lhs = np.sum(dv * dv, axis=1)
rhs = (sp - spr) ** 2 + 2 * sp * spr * (1 - cth)
t1["polar_identity_max_resid"] = float(np.max(np.abs(lhs - rhs)))
dE = 0.5 * (sp ** 2 - spr ** 2)
t1["dE_identity_max_resid"] = float(np.max(np.abs(dE - 0.5 * (sp + spr) * (sp - spr))))
rhs7 = (2 * dE / (sp + spr)) ** 2 + 4 * sp * spr * np.sin(theta / 2) ** 2
t1["T7_identity_max_resid"] = float(np.max(np.abs(lhs - rhs7)))
a_ch = 2 * np.abs(dE) / (sp + spr)
nz = lhs > 0
t1["T7_underestimation_median_2nd_half"] = float(
    np.median((np.sqrt(lhs[nz]) / a_ch[nz])[hh:]))
out["T1_T7"] = t1

# ============================================================== T4 map
def t4_map(n_steps, kappa=3.5e-7, chirp=False, drive="fixed", h=H):
    """z_{n+1} = e^{-i th_h(t_n)} z_n + kappa * sin(phase_n); dev vs |z0|^2."""
    z0 = 1j
    z = z0
    t = 0.0
    Phi = 0.0
    devs = np.zeros(n_steps); tsl = np.zeros(n_steps)
    for n in range(n_steps):
        Omt = Bz(t) if chirp else 1.0
        thh = 2.0 * np.arctan(h * Omt / 2.0)
        om_h0 = (2.0 / h) * np.arctan(h / 2.0)     # fixed initial omega_h
        if drive == "fixed":
            kick = kappa * np.sin(om_h0 * t)
        else:                                       # phase-tracking
            kick = kappa * np.sin(Phi)
        z = np.exp(-1j * thh) * z + kick
        Phi += thh
        t += h
        devs[n] = abs(abs(z) ** 2 - abs(z0) ** 2) / abs(z0) ** 2
        tsl[n] = t
    return tsl, devs

def env_exponent(t, dev):
    env = np.maximum.accumulate(dev)
    sel = (t > t[-1] / 100) & (env > 0)
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
    return float(p[0]), env

TWO_PI = 2 * np.pi
n4 = int(round(1e4 * TWO_PI / H))
t4res = {}
tsl, dv4 = t4_map(n4, chirp=False, drive="fixed")
p, env = env_exponent(tsl, dv4)
# linearity check: dev(N)/N constant over decades (no chirp)
i1, i2, i3 = n4 // 100, n4 // 10, n4 - 1
t4res["no_chirp_fixed_drive"] = {
    "exponent": p, "env_ratios_decades": [float(env[i2] / env[i1]), float(env[i3] / env[i2])]}
tsl, dv4 = t4_map(n4, chirp=True, drive="fixed")
p, env = env_exponent(tsl, dv4)
t4res["chirp_fixed_drive"] = {
    "exponent": p, "emax_1e2": float(env[n4 // 100]), "emax_1e3": float(env[n4 // 10]),
    "emax_1e4": float(env[-1]),
    "decade1_growth": float(env[n4 // 10] / env[n4 // 100]),
    "decade2_growth": float(env[-1] / env[n4 // 10])}
tsl, dv4 = t4_map(n4, chirp=True, drive="track")
p, env = env_exponent(tsl, dv4)
t4res["chirp_tracking_drive"] = {"exponent": p, "emax_1e4": float(env[-1])}
# symplecticity: the map is affine, Jacobian = rotation, det == 1 exactly
th_ = 2.0 * np.arctan(H / 2.0)
Jac = np.array([[np.cos(th_), np.sin(th_)], [-np.sin(th_), np.cos(th_)]])
t4res["jacobian_det_minus_1"] = float(abs(np.linalg.det(Jac) - 1.0))
out["T4_map"] = t4res

# ============================================================== P3.1
p31 = {}
for law, f, fp in [
    ("exp", lambda t: np.exp(-t / TAU), lambda t: -np.exp(-t / TAU) / TAU),
    ("cos", lambda t: 1 - 0.5 * (1 - np.cos(t / TAU)),
     lambda t: -0.5 * np.sin(t / TAU) / TAU),
    ("gauss", lambda t: np.exp(-(t / TAU) ** 2),
     lambda t: -(2 * t / TAU ** 2) * np.exp(-(t / TAU) ** 2)),
]:
    tg = np.arange(0, TF + H / 2, H)
    hh = len(tg) // 2
    A_ref = np.median(np.abs(f(tg[hh:] + H / 2) - f(tg[hh:])))
    t_med = np.median(tg[hh:])
    sig = 1.0 - f(t_med)
    ratio = sig / A_ref / (2 * t_med / H)
    closed = (1 - f(t_med)) / ((H / 2) * abs(fp(t_med))) / (2 * t_med / H)
    p31[law] = {"R_art/(2t/h)": float(ratio), "closed_form/(2t/h)": float(closed)}
out["P3_1_floor_law"] = p31

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf1_magnetic.json"), "w"), indent=1)
