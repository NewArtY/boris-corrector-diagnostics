"""RC4: two audits.

C1  Off-orbit test.  For each sample n, the smallest distance from the numerical
    state to the EXACT ORBIT REGARDED AS A CURVE (minimise over the reference
    time).  If the numerical point lies on the true orbit, this is zero and the
    whole error is a time re-parametrisation ("read-out convention").  If not,
    a time re-labelling can never remove it.

C2  The "scheme independence to five significant figures" claim.  A_signed =
    dev(h/2) - dev(0) equals +A_ref exactly whenever the scheme's own error is
    NEGATIVE (the scheme lags the reference) and equals A_ref - 2*dev0 when it
    is positive.  So the agreement across schemes is an identity of absolute
    values, not a property of the schemes.  Demonstrated by adding one scheme
    with the opposite error sign (implicit Euler).

Output: rc4_audit.json
"""
import json, math, os
import numpy as np
import rc_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ------------------------------------------------------------------- C1 -----
TAU, TF, H = 1.2e5, 120.0, 0.3
fB, ffac = rc.mk_field("exp", TAU)


def mag_states(h, T, r0, v0=(0.0, 1.0, 0.0)):
    n = int(round(T / h))
    r = np.array(r0, float); v = np.array(v0, float)
    S = np.zeros((n + 1, 6)); S[0] = np.concatenate([r, v])
    tt = 0.0
    for i in range(1, n + 1):
        Bz = float(fB(tt)); fac = float(ffac(tt))
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        v = rc.kick_boris(v, E, Bz, h)
        r = r + v * h
        tt += h
        S[i] = np.concatenate([r, v])
    return np.arange(n + 1) * h, S


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


tref, Yref = rk4_states(H / 400.0, TF * 1.05, (1.0, 0.0, 0.0))


def ref_at(t):
    t = np.atleast_1d(np.asarray(t, float))
    return np.stack([np.interp(t, tref, Yref[:, k]) for k in range(6)], axis=1)


t, S = mag_states(H, TF, (1.0, 0.0, 0.0))
half = len(t) // 2
dists = []
for i in range(half, len(t)):
    # local search over the reference time within +/- one gyroperiod
    lo, hi = t[i] - 2 * math.pi, t[i] + 2 * math.pi
    best = 1e300
    for _ in range(7):
        g = np.linspace(lo, hi, 401)
        rr = ref_at(g)
        dd = np.sqrt(np.sum((rr - S[i]) ** 2, axis=1))
        k = int(dd.argmin())
        if dd[k] < best:
            best = float(dd[k]); bt = float(g[k])
        st = g[1] - g[0]
        lo, hi = g[k] - 2 * st, g[k] + 2 * st
    dists.append(best)
dists = np.array(dists)
same_time = np.sqrt(np.sum((S[half:] - ref_at(t[half:])) ** 2, axis=1))
OUT["C1_magnetic_off_orbit_distance"] = {
    "median_distance_to_exact_orbit_as_curve": float(np.median(dists)),
    "max_distance_to_exact_orbit_as_curve": float(np.max(dists)),
    "median_same_time_state_error": float(np.median(same_time)),
    "fraction_of_error_that_is_NOT_a_time_relabelling":
        float(np.median(dists) / np.median(same_time)),
    "gyroradius": 1.0,
    "h": H, "T": TF}

# LL: the same quantity is exactly zero (1-D phase space)
r_e, c_l = 2.8179403262e-15, 2.99792458e8
omega = 2 * math.pi * c_l / 1.0e-6
EPS_PHYS = (2 * r_e * omega / (3 * c_l)) * 857.0
llc1 = {}
for pname, (al, ep, th0, T) in {"physical": (0.36, EPS_PHYS, 0.8, 8.0),
                                "generic": (1.0, 0.1, 0.3, 2.0)}.items():
    sysL = rc.LL(al, ep, th0)
    tt, thn, Q = sysL.run("euler", 0.05, T)
    s = sysL.t_of_theta(thn) - tt
    resid = np.abs(thn - sysL.theta_exact(tt + s))
    same = np.abs(thn - sysL.theta_exact(tt))
    llc1[pname] = {"median_distance_to_exact_orbit_as_curve": float(np.median(resid)),
                   "max_distance_to_exact_orbit_as_curve": float(np.max(resid)),
                   "median_same_time_state_error": float(np.median(same[len(tt) // 2:])),
                   "fraction_of_error_that_is_NOT_a_time_relabelling":
                       float(np.median(resid) / np.median(same[len(tt) // 2:]))}
OUT["C1_LL_off_orbit_distance"] = llc1

# ------------------------------------------------------------------- C2 -----
SETS = {"physical": dict(alpha=0.36, eps=EPS_PHYS, theta0=0.8, T=8.0),
        "generic": dict(alpha=1.0, eps=0.1, theta0=0.3, T=2.0)}
c2 = {}
for pname, Sp in SETS.items():
    sysL = rc.LL(Sp["alpha"], Sp["eps"], Sp["theta0"])
    for h in (0.05, 0.025):
        rows = []
        for sch in rc.LL_SCHEMES:
            tt, thn, Q = sysL.run(sch, h, Sp["T"])
            m = rc.metrics(tt, Q, sysL.Qref, sysL.dQref_dt, h)
            hh = len(tt) // 2
            e = Q[hh:] - sysL.Qref(tt[hh:])
            rows.append({"scheme": sch, "error_sign": int(np.sign(np.median(e))),
                         "dev0": m["dev0"], "A_ref": m["A_ref"],
                         "A_signed": m["A_signed"],
                         "A_signed_minus_A_ref": m["A_signed"] - m["A_ref"],
                         "A_ref_minus_2dev0": m["A_ref"] - 2 * m["dev0"],
                         "predicted_A_signed":
                             m["A_ref"] if np.median(e) < 0 else m["A_ref"] - 2 * m["dev0"]})
        A = np.array([r["A_signed"] for r in rows])
        lag = np.array([r["A_signed"] for r in rows if r["error_sign"] < 0])
        c2[f"{pname}_h{h}"] = {
            "rows": rows,
            "spread_all_schemes_max_over_min": float(np.max(np.abs(A)) / np.min(np.abs(A))),
            "spread_lagging_schemes_only_rel":
                (float((lag.max() - lag.min()) / lag.mean()) if len(lag) > 1 else None),
            "max_abs_err_of_sign_rule":
                float(np.max([abs(r["A_signed"] - r["predicted_A_signed"]) for r in rows]))}
OUT["C2_scheme_independence_is_an_identity"] = c2

with open(os.path.join(HERE, "rc4_audit.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)
print(json.dumps(OUT["C1_magnetic_off_orbit_distance"], indent=1))
print(json.dumps(OUT["C1_LL_off_orbit_distance"], indent=1))
for k, v in OUT["C2_scheme_independence_is_an_identity"].items():
    print("--", k, "spread_all=%.4g  spread_lagging=%s  sign_rule_err=%.3g"
          % (v["spread_all_schemes_max_over_min"], v["spread_lagging_schemes_only_rel"],
             v["max_abs_err_of_sign_rule"]))
    for r in v["rows"]:
        print(f"   {r['scheme']:11s} sgn={r['error_sign']:+d} dev0={r['dev0']:.6e} "
              f"A_ref={r['A_ref']:.10e} A_signed={r['A_signed']:.10e} "
              f"pred={r['predicted_A_signed']:.10e}")
print("wrote rc4_audit.json")
