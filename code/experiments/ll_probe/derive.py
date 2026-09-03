"""LL rapidity system: analytic derivations -> PREREGISTERED predictions.

Run BEFORE run_experiments.py.  Everything here is closed form (symbolic or a
direct evaluation of a closed-form expression); nothing here uses a numerical
integration of the system under test.  Output: prereg.json  (frozen).

System:   dtheta/dt = f(theta) = alpha*tanh(theta) - eps*sinh(theta)*cosh(theta)
Quadrature (neutral) channel: dpsi/dt = g(theta) = 1 - tanh(theta)
Energy channel: gamma = cosh(theta)
"""
import json, os
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
out = {"note": "PREREGISTERED. Derived analytically before any integration of the system.",
       "derivations": {}, "params": {}, "predictions": {}}

# ---------------------------------------------------------------- symbolic ---
th, a, e = sp.symbols('theta alpha epsilon', positive=True)
f = a*sp.tanh(th) - e*sp.sinh(th)*sp.cosh(th)
thstar = sp.acosh(sp.sqrt(a/e))
lam = sp.simplify(-sp.diff(f, th).subs(th, thstar))
out["derivations"]["fixed_point_residual"] = str(sp.simplify(f.subs(th, thstar)))
out["derivations"]["Lambda_symbolic"] = str(lam)
out["derivations"]["Lambda_minus_2(alpha-eps)"] = str(sp.simplify(lam - 2*(a - e)))
out["derivations"]["fprime_general"] = str(sp.simplify(sp.diff(f, th).rewrite(sp.exp).simplify()))
out["derivations"]["sech2_at_thetastar"] = str(sp.simplify((1/sp.cosh(thstar)**2)))  # = eps/alpha
out["derivations"]["comment_factor_2"] = (
    "Lambda = eps*cosh(2 th*) - alpha*sech^2(th*).  cosh(2th*)=2cosh^2(th*)-1=2a/e-1, "
    "sech^2(th*)=e/a.  => Lambda = (2a-e) - e = 2(a-e).  The factor 2 comes from the "
    "double-angle identity, NOT from an energy-vs-rapidity conversion.")

# --------------------------------------------------------------- parameters --
r_e, c_l, lam_um = 2.8179403262e-15, 2.99792458e8, 1.0e-6
omega = 2*np.pi*c_l/lam_um
a0 = 857.0
eps_phys = (2*r_e*omega/(3*c_l))*a0
P = {"name": "physical", "alpha": 0.36, "eps": float(eps_phys)}
G = {"name": "generic",  "alpha": 1.0,  "eps": 0.1}
for S in (P, G):
    al, ep = S["alpha"], S["eps"]
    S["theta_star"] = float(np.arccosh(np.sqrt(al/ep)))
    S["gamma_star"] = float(np.sqrt(al/ep))
    S["Lambda"] = float(2*(al - ep))
    S["sech2_star"] = float(ep/al)          # neutral-channel coupling |dg/dtheta|
    S["sinh_star"] = float(np.sqrt(al/ep - 1.0))
out["params"] = {"physical": P, "generic": G,
                 "laser": {"lambda_m": lam_um, "omega_rad_s": float(omega), "a0": a0,
                           "eps_rad_formula": "2*r_e*omega/(3c)*a0", "r_e_m": r_e}}
out["derivations"]["preprint_crosscheck"] = {
    "claimed_theta_star": 5.93, "derived_theta_star": P["theta_star"],
    "claimed_gamma_L": 188.0, "derived_gamma_L": P["gamma_star"],
    "claimed_Lambda": 0.72, "derived_Lambda": P["Lambda"]}

# ----------------------------------------------- stability functions R(z) ----
def R_euler(z):  return 1 + z
def R_ieuler(z): return 1.0/(1 - z)
def R_trap(z):   return (1 + z/2)/(1 - z/2)
def R_rk4(z):    return 1 + z + z**2/2 + z**3/6 + z**4/24
SCHEMES = {"euler": R_euler, "ieuler": R_ieuler, "trapezoid": R_trap, "rk4": R_rk4}

# --------------------------------------------------------------- P7: rate ----
rate = {}
for S, hs in ((G, [0.3, 0.05]), (P, [0.3, 0.05])):
    L = S["Lambda"]
    for h in hs:
        key = f"{S['name']}_h{h}"
        rate[key] = {"Lambda_true": L, "Lambda_h": {}}
        for nm, R in SCHEMES.items():
            rho = R(-L*h)
            rate[key]["Lambda_h"][nm] = {"rho": float(rho),
                                         "Lambda_h": float(-np.log(abs(rho))/h),
                                         "rel_bias": float(-np.log(abs(rho))/h/L - 1)}
out["predictions"]["P7_rate_bias"] = rate
out["predictions"]["P7b_attractor_exact"] = {
    "claim": "the fixed point of any consistent one-step scheme applied to an autonomous ODE "
             "satisfies f(theta)=0 exactly, hence theta_inf^num = theta* to round-off",
    "tolerance": 1e-12}

# ------------------------------------------------- P1-P4: T3 defect response --
KAPPA = 1e-6         # per-step additive defect in theta
H_T3  = 0.05
t3 = {}
for S in (G, P):
    L = S["Lambda"]; h = H_T3
    rho = R_rk4(-L*h)
    dc = KAPPA/(1 - rho)
    entry = {"scheme": "rk4", "h": h, "kappa_per_step": KAPPA, "rho": float(rho),
             "Lambda": L,
             "P1_DC_plateau_dtheta_discrete": float(dc),
             "P1_DC_plateau_dtheta_continuum_q0_over_Lambda": float((KAPPA/h)/L),
             "P1_DC_plateau_dev_gamma": float(S["sinh_star"]*dc/np.cosh(0.3 if S is G else 0.8)),
             "P2_sinusoid_plateau": {}, "P2_ratio_to_DC": {},
             "P3_noise_rms_plateau": float(KAPPA/np.sqrt(1 - rho**2)),
             "P4_coherent_over_incoherent": float(np.sqrt((1+rho)/(1-rho))),
             "P4_max_exponent_over_all_omega": 0.0}
        # omega grid: 0, Lambda, 10*Lambda, Nyquist alias 2pi/h, and pi/h
    for lbl, w in (("0", 0.0), ("Lambda", L), ("10Lambda", 10*L),
                   ("alias_2pi_over_h", 2*np.pi/h), ("nyquist_pi_over_h", np.pi/h)):
        amp = KAPPA/abs(1 - rho*np.exp(1j*w*h))
        entry["P2_sinusoid_plateau"][lbl] = float(amp)
        entry["P2_ratio_to_DC"][lbl] = float(amp/dc)
    t3[S["name"]] = entry
out["predictions"]["P1_P4_T3_collapse"] = t3
out["predictions"]["P4_statement"] = (
    "TRICHOTOMY COLLAPSES: envelope exponent p = 0 for DC, for every sinusoid frequency "
    "(including aliases), and for white noise.  Resonant enhancement over DC is <= 1 for "
    "ALL omega (|1-rho e^{i w h}| >= 1-rho, minimised at w=0).  Contrast: in the conservative "
    "rotation model the same protocol gives p = 1 at w = omega_h.")

# ------------------------------------------------------ P5: neutral channel --
neutral = {}
for S in (G, P):
    L = S["Lambda"]; rho = R_rk4(-L*H_T3); dc = KAPPA/(1-rho)
    slope = S["sech2_star"]*dc          # |d psi_error / dt|
    neutral[S["name"]] = {"dpsi_dt_slope": float(slope),
                          "coupling_sech2_star_equals_eps_over_alpha": S["sech2_star"],
                          "abs_dpsi_at_T1e5": float(slope*1e5),
                          "predicted_exponent_p": 1.0}
out["predictions"]["P5_neutral_channel"] = neutral
out["predictions"]["P5_statement"] = (
    "Same run, two channels, two exponents: energy p=0 (bounded, plateau q0/Lambda), "
    "phase/quadrature p=1 (linear, slope (eps/alpha)*q0/Lambda).  The attractor protects the "
    "contracting direction and gives nothing to the neutral one.")

# ----------------------------------------------------- P6: T2 convention -----
out["predictions"]["P6_convention"] = {
    "law": "reported relative energy error at readout offset delta = "
           "| dev_true +- delta * dlnGamma/dt |, affine in delta",
    "P6a_slope_equals_dlnGamma_dt_tolerance_rel": 0.02,
    "P6b_half_step_artifact_scales_as_h": "halving h halves the artifact, factor 2.00 +- 0.02",
    "P6c_scheme_independence_tolerance_rel": 0.02,
    "P6d_collapse_at_delta_star": "for rk4 (true error far below the convention floor) there is "
                                  "delta* in (0,h) where the reported error drops by >= 100x",
    "P6d_min_collapse_factor": 100.0}

# ----------------------------------------------------- P9: BEA divisors ------
bea = {}
for S in (G, P):
    L = S["Lambda"]
    for h in (0.3, 0.05):
        rho = float(np.exp(-L*h))
        bea[f"{S['name']}_h{h}"] = {
            "rho": rho, "min_over_omega_abs(1-rho e^{i w h})": 1 - rho,
            "conservative_analogue_min": 0.0}
out["predictions"]["P9_BEA_divisor"] = bea
out["predictions"]["P9_statement"] = (
    "The resonant small divisor 1-e^{i m omega_h h} of the conservative case becomes "
    "1-rho e^{i m omega h} with rho=e^{-Lambda h}<1; its modulus is bounded below by "
    "1-rho = Lambda*h + O(h^2) > 0.  Dissipation regularises the homological equations: "
    "the T4 obstruction disappears in the contracting direction and survives only at m=0 "
    "(the secular/DC mode), which is exactly the bounded response q0/Lambda.")

# ---------------------------------------------------------- P8: T7 analogue --
out["predictions"]["P8_certification_gap"] = {
    "claim": "energy-only certification understates the state error by a factor that grows "
             "like exp(Lambda*T) (energy error decays exponentially to the attractor while the "
             "rate error stays O(h^p)); in the magnetic system the same factor was the fixed 1.07e6",
    "test": "ratio (|Lambda_h-Lambda|/Lambda) / (dev_gamma(T)) vs T, fitted slope in ln-space = Lambda",
    "tolerance_rel": 0.05}

with open(os.path.join(HERE, "prereg.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)

print("Lambda symbolic          :", out["derivations"]["Lambda_symbolic"])
print("Lambda - 2(alpha-eps)    :", out["derivations"]["Lambda_minus_2(alpha-eps)"])
print("physical: theta*=%.4f gamma*=%.2f Lambda=%.6f sech2*=%.4e"
      % (P["theta_star"], P["gamma_star"], P["Lambda"], P["sech2_star"]))
print("generic : theta*=%.4f gamma*=%.4f Lambda=%.6f sech2*=%.4e"
      % (G["theta_star"], G["gamma_star"], G["Lambda"], G["sech2_star"]))
print("eps_phys =", eps_phys)
for k, v in rate.items():
    print(k, {n: round(d["Lambda_h"], 6) for n, d in v["Lambda_h"].items()})
print("T3 generic:", json.dumps(t3["generic"], indent=1)[:700])
print("wrote prereg.json")
