"""Follow-up round.  Fixes two preregistration misses of round 1 and adds three tests.

F1  aliasing with the correct (cosine) phase              -> repairs P2-alias
F2  convention law with the correct normaliser            -> repairs P6a
F3  erosion of the protection: defect linear in (theta-theta*)   -> new P10
F4  neutral-channel exponent at a longer horizon          -> completes P5
F5  noise ensemble for the neutral channel                -> completes P5 (p = 1/2)
F6  certification gap: decay rate of the energy channel   -> tests P8

Predictions are written to prereg2.json BEFORE the measurement section runs.
"""
import json, math, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8"))
P, G = PRE["params"]["physical"], PRE["params"]["generic"]

def make(al, ep):
    def f(t):  return al*math.tanh(t) - ep*math.sinh(t)*math.cosh(t)
    def fp(t): ch = math.cosh(t); return al/(ch*ch) - ep*math.cosh(2*t)
    def g(t):  return 1.0 - math.tanh(t)
    return f, fp, g
def rk4(f, th, h):
    k1 = f(th); k2 = f(th+.5*h*k1); k3 = f(th+.5*h*k2); k4 = f(th+h*k3)
    return th + h*(k1+2*k2+2*k3+k4)/6.0
def R_rk4(z): return 1 + z + z*z/2 + z**3/6 + z**4/24
def env_exp(t, d):
    e = np.maximum.accumulate(np.abs(d)); s = (t > t[-1]/100.) & (e > 0)
    return (float(np.polyfit(np.log10(t[s]), np.log10(e[s]), 1)[0]) if s.sum() > 10
            else float("nan")), e

# ============================================================ PREDICTIONS ====
KAP, H = 1e-6, 0.05
pred = {"note": "PREREGISTERED round 2, written before the measurement section runs."}

rho_g = R_rk4(-G["Lambda"]*H); rho_p = R_rk4(-P["Lambda"]*H)
pred["F1_alias_cosine"] = {
    "generic_cos_2pi_over_h_equals_DC": KAP/(1-rho_g),
    "generic_cos_pi_over_h_equals_kappa_over_1_plus_rho": KAP/(1+rho_g),
    "statement": "cos at omega = 2pi/h aliases exactly onto DC (e^{i w h}=1); at omega = pi/h it "
                 "alternates and the plateau is kappa/(1+rho).  Round 1 used sin, which vanishes "
                 "identically at those sample points -- a degenerate test, not a refutation."}
pred["F2_convention_normaliser"] = {
    "law": "artefact(delta) = delta * |d(gamma/gamma_0)/dt| = delta * sinh(theta)*|f(theta)|/cosh(theta_0)",
    "round1_error": "round 1 predicted delta*|dln gamma/dt| = delta*tanh(theta)*|f|, i.e. it "
                    "normalised by gamma(t) instead of gamma_0; the miss factor must equal "
                    "gamma(t_med)/gamma_0 exactly",
    "tolerance_rel": 0.03}
er = {}
for cfrac in (0.0, 0.5, 0.9, 0.99, 1.05):
    Le = G["Lambda"]*(1-cfrac)
    r = R_rk4(-Le*H)
    er[f"c_over_Lambda={cfrac}"] = {"Lambda_eff": Le, "rho": float(r),
        "plateau_dtheta": (float(KAP/(1-r)) if cfrac < 1 else None),
        "expected": "bounded, p=0" if cfrac < 1 else "unstable fixed point, exponential blow-up"}
pred["F3_protection_erosion"] = {"grid": er,
    "statement": "a defect linear in (theta-theta*) with coefficient c erodes the contraction to "
                 "Lambda-c; the plateau scales as 1/(Lambda-c) and diverges at c=Lambda.  The "
                 "protection is proportional to the fidelity of the learned dissipation."}
H4 = 0.5; rho4 = R_rk4(-P["Lambda"]*H4); dth4 = KAP/(1-rho4)
pred["F4_neutral_long"] = {"h": H4, "T_end": 1e6, "rho": float(rho4),
    "plateau_dtheta": float(dth4),
    "dpsi_slope": float(P["sech2_star"]*dth4), "expected_exponent": 1.0, "tolerance_abs": 0.05}
pred["F5_noise_neutral_ensemble"] = {"n_seeds": 8, "expected_median_exponent": 0.5,
    "tolerance_abs": 0.2}
# --- F7: eps_rad sweep.  Breaks the dissipation / relativistic-kinematics confound. ---
SWEEP = [(0.0, 1e-6, 300.0), (0.1, 1e-6, 2e4), (0.5, 1e-6, 2e4), (0.9, 1e-6, 2e4),
         (0.99, 1e-7, 2e4), (0.999, 1e-8, 2e4)]
sw = {}
for epr, kap, T in SWEEP:
    L = 2.0*(1.0 - epr); q0 = kap/H
    if epr == 0.0:
        sw["eps_over_alpha=0"] = {"Lambda": 0.0, "kappa": kap, "T": T,
            "attractor": None, "predicted_exponent_theta": 1.0,
            "predicted_dtheta_slope_per_unit_time": q0,
            "predicted_dtheta_at_T": q0*T, "predicted_exponent_psi": 0.0,
            "note": "no attractor; f -> alpha, Green function -> 1, defect integrates freely"}
    else:
        r = R_rk4(-L*H)
        sw[f"eps_over_alpha={epr}"] = {"Lambda": L, "kappa": kap, "T": T,
            "theta_star": float(np.arccosh(np.sqrt(1.0/epr))),
            "rho": float(r), "predicted_plateau_dtheta": float(kap/(1-r)),
            "predicted_plateau_continuum_q0_over_Lambda": float(q0/L),
            "predicted_exponent_theta": 0.0, "Lambda_times_T": float(L*T)}
pred["F7_eps_sweep"] = {"grid": sw,
    "law": "plateau * Lambda / q0 = 1 across three decades of Lambda (eps/alpha = 0.1 -> 0.999); "
           "the eps=0 point is off the law entirely (linear growth, no plateau)",
    "law_tolerance_rel": 0.02,
    "crossover": "for eps/alpha=0.999 (Lambda=0.002) the envelope exponent measured on the early "
                 "window t in [1,1e2] (Lambda*T<1) must be ~1 and on the late window t in "
                 "[1e3,2e4] (Lambda*T>1) must be ~0 -- the crossover sits at Lambda*T ~ 1",
    "confound_statement": "eps=0 keeps the relativistic kinematics, the code, alpha, h, theta_0 "
                          "and T identical and removes only the contraction.  If p goes 1 -> 0 "
                          "along the sweep, the responsible factor is dissipation, not relativity.",
    "caveat": "at eps/alpha >= 0.9 the attractor sits at theta* <= 0.33 (sub-relativistic): that "
              "end of the sweep tests the dynamical law, not a physical regime."}

pred["F6_certification_gap"] = {
    "claim": "with no injected defect the energy-channel error of a consistent scheme decays like "
             "exp(-Lambda_min*t) with Lambda_min = min(Lambda, Lambda_h), while the rate error "
             "stays at its O(h^p) value; the understatement factor of energy-only certification "
             "therefore grows exponentially, unbounded in T",
    "fitted_decay_rate_expected_generic_euler_h0.3": min(G["Lambda"], 2.588429),
    "tolerance_rel": 0.05}
json.dump(pred, open(os.path.join(HERE, "prereg2.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("prereg2.json frozen")

# =========================================================== MEASUREMENTS ====
RES = {}

# --- F1 ----------------------------------------------------------------------
def f1():
    al, ep, ths = G["alpha"], G["eps"], G["theta_star"]
    f, fp, g = make(al, ep); out = {}
    T, h = 3e3, H; n = int(T/h)
    for lbl, w in (("cos_2pi_over_h", 2*math.pi/h), ("cos_pi_over_h", math.pi/h)):
        thu = thp = 0.3
        for k in range(n):
            thu = rk4(f, thu, h); thp = rk4(f, thp, h) + KAP*math.cos(w*k*h)
        out[lbl] = {"dtheta_final": float(thp - thu)}
    # amplitude over the last 200 steps for the alternating case
    return out
RES["F1"] = f1(); print("F1", RES["F1"])

# --- F2 ----------------------------------------------------------------------
def f2():
    out = {}
    for nm, S in (("physical", P), ("generic", G)):
        f, fp, g = make(S["alpha"], S["eps"])
        th0 = 0.8 if nm == "physical" else 0.3
        T = 8.0 if nm == "physical" else 2.0
        for h in (0.05, 0.025):
            hf = h/2000.; nf = int(T*1.5/hf)+1
            tf = np.arange(nf)*hf; gf = np.empty(nf); thv = np.empty(nf); th = th0
            gf[0] = math.cosh(th); thv[0] = th
            for i in range(1, nf):
                th = rk4(f, th, hf); gf[i] = math.cosh(th); thv[i] = th
            g0 = gf[0]; n = int(T/h)+1; tn = np.arange(n)*h
            gn = np.empty(n); th = th0; gn[0] = math.cosh(th)
            for i in range(1, n): th = rk4(f, th, h); gn[i] = math.cosh(th)
            half = n//2; tmed = float(np.median(tn[half:]))
            thm = float(np.interp(tmed, tf, thv))
            dev0 = float(np.median(np.abs(gn - np.interp(tn, tf, gf))[half:])/g0)
            devh = float(np.median(np.abs(gn - np.interp(tn+h/2, tf, gf))[half:])/g0)
            art = devh - dev0
            predicted = (h/2)*math.sinh(thm)*abs(f(thm))/g0
            signal = float(abs(gf[int(T/hf)] - g0)/g0)
            out[f"{nm}_h{h}"] = {"theta_med": thm, "artefact": art, "predicted": predicted,
                                 "ratio": art/predicted, "signal": signal,
                                 "R_conv_signal_over_artefact": signal/art,
                                 "R_conv_predicted": signal/predicted}
    return out
RES["F2"] = f2(); print("F2 done")

# --- F3 ----------------------------------------------------------------------
def f3():
    al, ep, ths, L = G["alpha"], G["eps"], G["theta_star"], G["Lambda"]
    f, fp, g = make(al, ep); out = {}; h = H; T = 3e3; n = int(T/h)
    for cf in (0.0, 0.5, 0.9, 0.99, 1.05):
        c = L*cf; thu = thp = 0.3; blew = False
        for k in range(n):
            thu = rk4(f, thu, h); thp = rk4(f, thp, h) + KAP + c*h*(thp - ths)
            if not math.isfinite(thp) or abs(thp-thu) > 1e3: blew = True; break
        out[f"c_over_Lambda={cf}"] = {"dtheta_final": (None if blew else float(thp-thu)),
                                      "diverged": blew, "steps_survived": k+1}
    return out
RES["F3"] = f3(); print("F3", RES["F3"])

# --- F4 ----------------------------------------------------------------------
def f4():
    al, ep, ths = P["alpha"], P["eps"], P["theta_star"]
    f, fp, g = make(al, ep); h = H4; T = 1e6; n = int(T/h)
    idx = np.unique(np.round(np.logspace(0, math.log10(n), 3000)).astype(int))
    thu = thp = 0.8; psu = psp = 0.0; j = 0
    ts = np.empty(len(idx)); dps = np.empty(len(idx)); dth = np.empty(len(idx))
    for k in range(n):
        psu += h*g(thu); psp += h*g(thp)
        thu = rk4(f, thu, h); thp = rk4(f, thp, h) + KAP
        if j < len(idx) and k+1 == idx[j]:
            ts[j] = (k+1)*h; dps[j] = psp-psu; dth[j] = thp-thu; j += 1
    p, e = env_exp(ts, dps)
    m = ts > ts[-1]/10
    return {"exponent_psi": p, "dtheta_plateau": float(dth[-1]),
            "dpsi_slope": float(np.polyfit(ts[m], dps[m], 1)[0]),
            "dpsi_final": float(dps[-1]),
            "exponent_energy": env_exp(ts, np.cosh(0.8+0*ts)*0 + dth)[0]}
RES["F4"] = f4(); print("F4", RES["F4"])

# --- F5 ----------------------------------------------------------------------
def f5():
    al, ep = G["alpha"], G["eps"]; f, fp, g = make(al, ep)
    h = H; T = 1e4; n = int(T/h); ps = []
    idx = np.unique(np.round(np.logspace(0, math.log10(n), 2000)).astype(int))
    for seed in range(8):
        rng = np.random.default_rng(1000+seed); nz = rng.standard_normal(n)*KAP
        thu = thp = 0.3; psu = psp = 0.0; j = 0
        ts = np.empty(len(idx)); dps = np.empty(len(idx))
        for k in range(n):
            psu += h*g(thu); psp += h*g(thp)
            thu = rk4(f, thu, h); thp = rk4(f, thp, h) + nz[k]
            if j < len(idx) and k+1 == idx[j]: ts[j] = (k+1)*h; dps[j] = psp-psu; j += 1
        ps.append(env_exp(ts, dps)[0])
    return {"exponents": [float(x) for x in ps], "median": float(np.median(ps)),
            "min": float(np.min(ps)), "max": float(np.max(ps))}
RES["F5"] = f5(); print("F5", RES["F5"])

# --- F6 ----------------------------------------------------------------------
def f6():
    al, ep, ths, L = G["alpha"], G["eps"], G["theta_star"], G["Lambda"]
    f, fp, g = make(al, ep); h = 0.3; T = 40.0; n = int(T/h)
    hf = h/4000.; nf = int(T/hf)+1; th = 0.3
    tf = np.arange(nf)*hf; gf = np.empty(nf); gf[0] = math.cosh(th)
    for i in range(1, nf): th = rk4(f, th, hf); gf[i] = math.cosh(th)
    th = 0.3; tn = np.arange(n+1)*h; gn = np.empty(n+1); gn[0] = math.cosh(th)
    for i in range(1, n+1): th = th + h*f(th); gn[i] = math.cosh(th)   # explicit Euler
    dev = np.abs(gn - np.interp(tn, tf, gf))/gf[0]
    m = (tn > 5) & (tn < 25) & (dev > 1e-14)
    rate = float(-np.polyfit(tn[m], np.log(dev[m]), 1)[0])
    return {"fitted_energy_decay_rate": rate, "Lambda": L, "Lambda_h_euler_h0.3": 2.588429,
            "dev_at_t5": float(dev[int(5/h)]), "dev_at_t25": float(dev[int(25/h)]),
            "rate_error_constant": abs(2.588429-L)/L}
RES["F6"] = f6(); print("F6", RES["F6"])

# --- F7 ------------------------------------------------------------------
def f7():
    out = {}
    for epr, kap, T in SWEEP:
        al = 1.0; ep = epr*al
        f, fp, g = make(al, ep)
        ths = float(np.arccosh(np.sqrt(al/ep))) if ep > 0 else None
        h = H; n = int(T/h)
        idx = np.unique(np.round(np.logspace(0, math.log10(n), 3000)).astype(int))
        thu = thp = 0.3; psu = psp = 0.0; j = 0
        ts = np.empty(len(idx)); dth = np.empty(len(idx)); dps = np.empty(len(idx))
        for k in range(n):
            psu += h*g(thu); psp += h*g(thp)
            thu = rk4(f, thu, h); thp = rk4(f, thp, h) + kap
            if j < len(idx) and k+1 == idx[j]:
                ts[j] = (k+1)*h; dth[j] = thp-thu; dps[j] = psp-psu; j += 1
        pT, eT = env_exp(ts, dth); pP, _ = env_exp(ts, dps)
        rec = {"Lambda": 2.0*(al-ep), "theta_star": ths, "kappa": kap, "T": T,
               "exponent_theta": pT, "exponent_psi": pP,
               "dtheta_final": float(dth[-1]),
               "dtheta_slope_per_unit_time": float(np.polyfit(ts[ts > ts[-1]/10],
                                                              dth[ts > ts[-1]/10], 1)[0]),
               "law_plateau_times_Lambda_over_q0": (float(abs(dth[-1])*2.0*(al-ep)/(kap/h))
                                                    if ep > 0 else None)}
        if epr == 0.999:
            for lo, hi, lbl in ((1., 1e2, "early_LambdaT_lt_1"), (1e3, 2e4, "late_LambdaT_gt_1")):
                m = (ts >= lo) & (ts <= hi)
                e = np.maximum.accumulate(np.abs(dth))[m]
                rec[f"window_{lbl}_exponent"] = float(np.polyfit(np.log10(ts[m]),
                                                                 np.log10(e), 1)[0])
        out[f"eps_over_alpha={epr}"] = rec
    return out
RES["F7"] = f7(); print("F7 done")

json.dump(RES, open(os.path.join(HERE, "results2.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("wrote results2.json")
