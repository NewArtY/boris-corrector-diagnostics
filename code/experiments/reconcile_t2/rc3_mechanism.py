"""RC3: the mechanism, in closed form, in each system.

A.  LL.  The phase space of dtheta/dt = f(theta) is ONE-dimensional, so the
    numerical point theta_n always lies ON the true orbit: there is an exact
    time lag s_n with  theta_n = theta_exact(t_n + s_n),  s_n = t(theta_n) - t_n
    (t(.) known in closed form).  The whole global error is a time
    re-parametrisation.  Claim, derived here and tested:

        ds/dt = h^p * f_p(theta)/f(theta)   (modified-equation defect f_p),
        Euler (p=1, f_p = -f f'/2):  s(t) = -(h/2) ln[ f(theta(t)) / f(theta_0) ]
        dev(t) = |Q(t+s) - Q(t)| ~= |s(t)| * |dQ/dt|
        dev0 / A_ref = 2|s|/h  ~  h^{p-1}.

B.  MAGNETIC.  Phase space is four-dimensional.  Test the same statement:
    how much of the state error can ANY single global time offset remove?
    Also locate the initial position that annihilates the energy floor and
    compare it with the closed-form discrete guiding centre r0 = M v0,
    M = h R (R - I)^{-1}.

Output: rc3_mechanism.json
"""
import json, math, os
import numpy as np
import rc_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# =============================================================== A.  LL ======
r_e, c_l = 2.8179403262e-15, 2.99792458e8
omega = 2 * math.pi * c_l / 1.0e-6
EPS_PHYS = (2 * r_e * omega / (3 * c_l)) * 857.0
SETS = {"physical": dict(alpha=0.36, eps=EPS_PHYS, theta0=0.8, T=8.0),
        "generic": dict(alpha=1.0, eps=0.1, theta0=0.3, T=2.0)}
ORDER = {"euler": 1, "ieuler": 1, "midpoint": 2, "trapezoid": 2, "rk4": 4}

llout = {}
for pname, S in SETS.items():
    sysL = rc.LL(S["alpha"], S["eps"], S["theta0"])
    rec = {"theta_star": sysL.theta_star, "Lambda": sysL.Lam, "eps": S["eps"]}

    # --- A0: the numerical point lies on the true orbit (exactness of t(theta))
    t, thn, Q = sysL.run("euler", 0.05, S["T"])
    back = sysL.theta_exact(sysL.t_of_theta(thn))
    rec["A0_orbit_exactness_max_abs_roundtrip_err"] = float(np.max(np.abs(back - thn)))

    # --- A1: Euler closed form  s(t) = -(h/2) ln[f(theta(t))/f(theta0)] ------
    a1 = {}
    for h in (0.1, 0.05, 0.025, 0.0125):
        t, thn, Q = sysL.run("euler", h, S["T"])
        s_meas = sysL.t_of_theta(thn) - t
        f0 = sysL.f(S["theta0"])
        s_pred = np.array([-(h / 2.0) * math.log(sysL.f(x) / f0) for x in thn])
        half = len(t) // 2
        sel = slice(half, None)
        a1[f"h={h}"] = {
            "s_measured_at_tmed": float(np.median(s_meas[sel])),
            "s_closed_form_at_tmed": float(np.median(s_pred[sel])),
            "ratio": float(np.median(s_meas[sel]) / np.median(s_pred[sel])),
            "max_rel_err_over_2nd_half": float(np.max(np.abs(s_meas[sel] / s_pred[sel] - 1.0))),
            "s_over_h_at_tmed": float(np.median(s_meas[sel]) / h)}
    rec["A1_euler_time_lag_closed_form"] = a1

    # --- A2: dev(t) = |s| * |dQ/dt| ----------------------------------------
    a2 = {}
    for sch in rc.LL_SCHEMES:
        h = 0.05
        t, thn, Q = sysL.run(sch, h, S["T"])
        s_meas = sysL.t_of_theta(thn) - t
        half = len(t) // 2
        sel = slice(half, None)
        dev_pt = np.abs(Q[sel] - sysL.Qref(t[sel]))
        pred_pt = np.abs(s_meas[sel]) * np.abs(sysL.dQref_dt(t[sel]))
        a2[sch] = {"median_dev_measured": float(np.median(dev_pt)),
                   "median_dev_predicted_|s|*|dQ/dt|": float(np.median(pred_pt)),
                   "median_ratio": float(np.median(pred_pt / dev_pt)),
                   "max_rel_err": float(np.max(np.abs(pred_pt / dev_pt - 1.0)))}
    rec["A2_dev_equals_lag_times_slope"] = a2

    # --- A3: floor/A_ref ~ h^{p-1} ------------------------------------------
    a3 = {}
    for sch in rc.LL_SCHEMES:
        hs = [0.1, 0.05, 0.025, 0.0125]
        vs = []
        for h in hs:
            t, thn, Q = sysL.run(sch, h, S["T"])
            m = rc.metrics(t, Q, sysL.Qref, sysL.dQref_dt, h)
            vs.append(m["floor_in_units_of_artefact"])
        lo = [float(math.log(vs[i] / vs[i + 1]) / math.log(2.0)) for i in range(len(vs) - 1)]
        a3[sch] = {"h": hs, "floor_over_A_ref": [float(x) for x in vs],
                   "local_orders_in_h": lo, "p": ORDER[sch], "expected_p_minus_1": ORDER[sch] - 1}
    rec["A3_floor_over_artefact_scaling"] = a3

    # --- A4: how much of the STATE error a single global offset removes -----
    a4 = {}
    for sch in rc.LL_SCHEMES:
        h = 0.05
        t, thn, Q = sysL.run(sch, h, S["T"])
        half = len(t) // 2
        sel = slice(half, None)
        def obj(d):
            return float(np.sqrt(np.mean((thn[sel] - sysL.theta_exact(t[sel] + d)) ** 2)))
        base = obj(0.0)
        lo, hi = -h, h
        best, bd = base, 0.0
        for _ in range(9):
            grid = np.linspace(lo, hi, 401)
            vv = np.array([obj(d) for d in grid])
            k = int(vv.argmin())
            if vv[k] < best:
                best, bd = float(vv[k]), float(grid[k])
            st = grid[1] - grid[0]
            lo, hi = grid[k] - 2 * st, grid[k] + 2 * st
        a4[sch] = {"rms_theta_err_delta0": base, "rms_theta_err_best_delta": best,
                   "best_delta_over_h": bd / h,
                   "removable_fraction": float(1.0 - best / base)}
    rec["A4_state_error_removable_by_one_offset"] = a4
    llout[pname] = rec
OUT["LL"] = llout

# =========================================================== B.  MAGNETIC ====
TAU, TF, H = 1.2e5, 120.0, 0.3
fB, ffac = rc.mk_field("exp", TAU)


def Qref_mag(t):
    return np.exp(-np.asarray(t, float) / TAU)


def mag_states(scheme, h, T, r0, v0=(0.0, 1.0, 0.0)):
    """Full state trajectory of the shipped Boris map."""
    n = int(round(T / h))
    r = np.array(r0, float); v = np.array(v0, float)
    R = np.zeros((n + 1, 3)); V = np.zeros((n + 1, 3))
    R[0] = r; V[0] = v
    tt = 0.0
    for i in range(1, n + 1):
        Bz = float(fB(tt)); fac = float(ffac(tt))
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        v = rc.kick_boris(v, E, Bz, h)
        r = r + v * h
        tt += h
        R[i] = r; V[i] = v
    return np.arange(n + 1) * h, R, V


# reference orbit: same map at h/500 (converged), interpolated
HF = H / 500.0
tf, Rf, Vf = mag_states("boris_shipped", HF, TF * 1.02, (1.0, 0.0, 0.0))
# use a genuinely convergent reference: RK4 at h/500
def rk4_states(h, T, r0, v0=(0.0, 1.0, 0.0), q=-1.0):
    n = int(round(T / h))
    y = np.concatenate([np.array(r0, float), np.array(v0, float)])
    Y = np.zeros((n + 1, 6)); Y[0] = y
    tt = 0.0

    def fode(tt, y):
        r = y[:3]; v = y[3:]
        fac = float(ffac(tt)); Bz = float(fB(tt))
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        return np.concatenate([v, q * (E + np.cross(v, np.array([0.0, 0.0, Bz])))])
    for i in range(1, n + 1):
        k1 = fode(tt, y); k2 = fode(tt + h / 2, y + h / 2 * k1)
        k3 = fode(tt + h / 2, y + h / 2 * k2); k4 = fode(tt + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tt += h
        Y[i] = y
    return np.arange(n + 1) * h, Y


tf, Yf = rk4_states(H / 400.0, TF * 1.02, (1.0, 0.0, 0.0))


def ref_state(t):
    t = np.asarray(t, float)
    return np.stack([np.interp(t, tf, Yf[:, k]) for k in range(6)], axis=1)


t, R, V = mag_states("boris_shipped", H, TF, (1.0, 0.0, 0.0))
S = np.concatenate([R, V], axis=1)
half = len(t) // 2
sel = slice(half, None)


def rms_state(d):
    ref = ref_state(t[sel] + d)
    return float(np.sqrt(np.mean(np.sum((S[sel] - ref) ** 2, axis=1))))


lo, hi = -8 * H, 8 * H
bestv, bestd = rms_state(0.0), 0.0
for _ in range(8):
    grid = np.linspace(lo, hi, 801)
    vals = np.array([rms_state(d) for d in grid])
    j = int(vals.argmin())
    if vals[j] < bestv:
        bestv, bestd = float(vals[j]), float(grid[j])
    st = grid[1] - grid[0]
    lo, hi = grid[j] - 2 * st, grid[j] + 2 * st
OUT["M_state_error_removable_by_one_offset"] = {
    "rms_state_err_delta0": rms_state(0.0),
    "rms_state_err_best_delta": bestv,
    "best_delta_over_h": bestd / H,
    "gyrophase_error_at_T_rad": float((2 * math.atan(H / 2) / H - 1.0) * TF),
    "removable_fraction": float(1.0 - bestv / rms_state(0.0)),
    "note": "reference = RK4 at h/400 from the same initial condition; "
            "gyroradius r_L = 1"}

# --- discrete guiding centre: brute-force optimum vs closed form -------------
th_h = 2 * math.atan(H / 2)
c_, s_ = math.cos(th_h), math.sin(th_h)
res = {}
for lbl, Rmat in (("R = clockwise (verifier's sign)", np.array([[c_, s_], [-s_, c_]])),
                  ("R = counter-clockwise", np.array([[c_, -s_], [s_, c_]]))):
    Mm = H * Rmat @ np.linalg.inv(Rmat - np.eye(2))
    mv = Mm @ np.array([0.0, 1.0])
    tt, Q, rho = rc.run_mag("boris_shipped", fB, ffac, H, TF,
                            r0=(float(mv[0]), float(mv[1]), 0.0))
    m = rc.metrics(tt, Q, Qref_mag, None, H)
    res[lbl] = {"r0 = M v0": [float(mv[0]), float(mv[1])], "dev0": m["dev0"],
                "floor_over_A_ref": m["floor_in_units_of_artefact"]}

# brute force 2-D refine around (1, 0.15)
best = (None, 1e300)
ax, ay = 1.0, 0.15
step = 0.05
for _ in range(9):
    for dx in np.linspace(-step, step, 21):
        for dy in np.linspace(-step, step, 21):
            r0 = (ax + dx, ay + dy, 0.0)
            tt, Q, rho = rc.run_mag("boris_shipped", fB, ffac, H, TF, r0=r0)
            d = float(np.median(np.abs(Q[len(tt) // 2:] - Qref_mag(tt[len(tt) // 2:]))))
            if d < best[1]:
                best = (r0, d)
    ax, ay = best[0][0], best[0][1]
    step /= 5.0
res["brute_force_optimum"] = {"r0": [float(best[0][0]), float(best[0][1])],
                              "dev0": best[1],
                              "dev0_baseline": 1.2498530289617449e-06}
tt, Q, rho = rc.run_mag("boris_shipped", fB, ffac, H, TF, r0=(1.0, H / 2.0, 0.0))
m = rc.metrics(tt, Q, Qref_mag, None, H)
res["r0=(1,h/2,0)"] = {"dev0": m["dev0"], "floor_over_A_ref": m["floor_in_units_of_artefact"]}
OUT["M_guiding_centre_placement"] = res

with open(os.path.join(HERE, "rc3_mechanism.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)

for pname, rec in OUT["LL"].items():
    print(f"===== LL {pname} =====")
    print(" orbit round-trip err:", rec["A0_orbit_exactness_max_abs_roundtrip_err"])
    print(" A1 Euler s closed form:")
    for k, v in rec["A1_euler_time_lag_closed_form"].items():
        print(f"   {k:10s} s_meas={v['s_measured_at_tmed']:+.6e} "
              f"s_pred={v['s_closed_form_at_tmed']:+.6e} ratio={v['ratio']:.6f} "
              f"maxrel={v['max_rel_err_over_2nd_half']:.3e} s/h={v['s_over_h_at_tmed']:+.5f}")
    print(" A2 dev = |s|*|dQ/dt|:")
    for k, v in rec["A2_dev_equals_lag_times_slope"].items():
        print(f"   {k:11s} meas={v['median_dev_measured']:.5e} "
              f"pred={v['median_dev_predicted_|s|*|dQ/dt|']:.5e} "
              f"ratio={v['median_ratio']:.6f} maxrel={v['max_rel_err']:.3e}")
    print(" A3 floor/A_ref vs h:")
    for k, v in rec["A3_floor_over_artefact_scaling"].items():
        print(f"   {k:11s} p={v['p']} orders={['%.3f' % x for x in v['local_orders_in_h']]} "
              f"expected {v['expected_p_minus_1']}")
    print(" A4 state error removable by one offset:")
    for k, v in rec["A4_state_error_removable_by_one_offset"].items():
        print(f"   {k:11s} rms0={v['rms_theta_err_delta0']:.4e} "
              f"best={v['rms_theta_err_best_delta']:.4e} removable={v['removable_fraction']:.6f}")
print("===== MAGNETIC =====")
print(json.dumps(OUT["M_state_error_removable_by_one_offset"], indent=1))
print(json.dumps(OUT["M_guiding_centre_placement"], indent=1))
print("wrote rc3_mechanism.json")
