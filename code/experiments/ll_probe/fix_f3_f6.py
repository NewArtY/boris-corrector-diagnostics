"""Repair of two badly designed tests from followup.py.

F3 (round 2) probed protection erosion with a defect c*h*(theta_p - theta*), which is
O(1) far from the attractor: it was not a small perturbation and the run left the basin.
F3b replaces it with (a) a linear-feedback probe c*(theta_p - theta_u), which is zero at
t=0 and stays small, and (b) the honest physical version: a relative gain error m on the
radiation-reaction coefficient.

F6 (round 2) fitted the energy-error decay over a window whose tail sat on the
interpolation/round-off floor of the reference grid.  F6b fits inside a clean dynamic range
and tests both signs of the bias (explicit Euler: Lambda_h > Lambda; implicit: Lambda_h < Lambda).

Predictions in prereg3.json are written before the measurement section runs.
"""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8"))["params"]["generic"]
AL, EP, THS, LAM = G["alpha"], G["eps"], G["theta_star"], G["Lambda"]

def make(al, ep):
    def f(t):  return al*math.tanh(t) - ep*math.sinh(t)*math.cosh(t)
    def fp(t): ch = math.cosh(t); return al/(ch*ch) - ep*math.cosh(2*t)
    return f, fp
def rk4(f, th, h):
    k1 = f(th); k2 = f(th+.5*h*k1); k3 = f(th+.5*h*k2); k4 = f(th+h*k3)
    return th + h*(k1+2*k2+2*k3+k4)/6.
def R_rk4(z): return 1 + z + z*z/2 + z**3/6 + z**4/24

KAP, H = 1e-6, 0.05
pred = {"note": "PREREGISTERED round 3."}

g3 = {}
for cf in (0.0, 0.5, 0.9, 0.99):
    Le = LAM*(1-cf); r = R_rk4(-Le*H)
    g3[f"c_over_Lambda={cf}"] = {"Lambda_eff": Le, "plateau_dtheta": float(KAP/(1-r)),
                                 "continuum_q0_over_Lambda_eff": float((KAP/H)/Le),
                                 "expected_exponent": 0.0}
g3["c_over_Lambda=1.05"] = {"Lambda_eff": float(LAM*(-0.05)),
                            "expected": "unstable: exponential growth of dtheta, exponent >> 1"}
pred["F3b_linear_feedback"] = {"grid": g3,
    "statement": "a defect proportional to the state error erodes the contraction to Lambda-c; "
                 "plateau = kappa/(1-R(-(Lambda-c)h)) ~ q0/(Lambda-c), divergent at c=Lambda."}
gain = {}
for m in (0.01, 0.1, 0.5):
    ep2 = EP*(1+m)
    gain[f"m={m}"] = {"theta_star_new": float(np.arccosh(np.sqrt(AL/ep2))),
                      "d_theta_star": float(np.arccosh(np.sqrt(AL/ep2)) - THS),
                      "gamma_star_rel_shift": float(np.sqrt(AL/ep2)/np.sqrt(AL/EP) - 1),
                      "Lambda_new": float(2*(AL-ep2)),
                      "Lambda_rel_shift": float(2*(AL-ep2)/LAM - 1)}
pred["F3c_drag_gain_error"] = {"grid": gain,
    "statement": "a relative error m on the radiation-reaction coefficient shifts the attractor "
                 "energy by (1+m)^{-1/2}-1 and the rate by -2*eps*m/Lambda: both bounded, both "
                 "first order in m, no secular growth."}
f6 = {}
for nm, Lh in (("euler", -math.log(abs(1-LAM*0.3))/0.3),
               ("ieuler", math.log(1+LAM*0.3)/0.3)):
    f6[nm] = {"Lambda_h": Lh, "predicted_dev_decay_rate": min(LAM, Lh)}
pred["F6b_energy_decay_rate"] = {"grid": f6, "h": 0.3, "tolerance_rel": 0.05,
    "statement": "the energy-channel error decays at min(Lambda, Lambda_h); the rate error stays "
                 "at its O(h^p) value, so energy-only certification understates the defect by a "
                 "factor growing like exp(min(Lambda,Lambda_h) t) -- unbounded in T."}
json.dump(pred, open(os.path.join(HERE, "prereg3.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("prereg3.json frozen")

RES = {}
f, fp = make(AL, EP)

# --- F3b ---------------------------------------------------------------------
o = {}
for cf in (0.0, 0.5, 0.9, 0.99, 1.05):
    c = LAM*cf; h = H; T = 4e3; n = int(T/h)
    idx = np.unique(np.round(np.logspace(0, math.log10(n), 2000)).astype(int))
    thu = thp = 0.3; j = 0; ts = np.empty(len(idx)); dt = np.empty(len(idx)); blew = False
    for k in range(n):
        d = thp - thu
        thu = rk4(f, thu, h); thp = rk4(f, thp, h) + KAP + c*h*d
        if not math.isfinite(thp) or abs(thp-thu) > 1.0: blew = True; break
        if j < len(idx) and k+1 == idx[j]: ts[j] = (k+1)*h; dt[j] = thp-thu; j += 1
    if blew:
        o[f"c_over_Lambda={cf}"] = {"diverged": True, "steps_survived": k+1,
                                    "t_survived": (k+1)*h}
    else:
        e = np.maximum.accumulate(np.abs(dt)); s = ts > ts[-1]/100
        o[f"c_over_Lambda={cf}"] = {"diverged": False, "dtheta_final": float(dt[-1]),
            "exponent": float(np.polyfit(np.log10(ts[s]), np.log10(e[s]), 1)[0])}
RES["F3b"] = o; print("F3b", o)

# --- F3c ---------------------------------------------------------------------
o = {}
for m in (0.01, 0.1, 0.5):
    f2, fp2 = make(AL, EP*(1+m)); h = H
    th = 0.3
    for _ in range(200000):
        nx = rk4(f2, th, h)
        if abs(nx-th) < 1e-16*max(1., abs(th)): th = nx; break
        th = nx
    d = 1e-7; rho = (rk4(f2, th+d, h) - rk4(f2, th-d, h))/(2*d)
    o[f"m={m}"] = {"theta_star_measured": float(th), "d_theta_star": float(th-THS),
                   "gamma_star_rel_shift": float(math.cosh(th)/math.cosh(THS)-1),
                   "Lambda_h_measured": float(-math.log(abs(rho))/h)}
RES["F3c"] = o; print("F3c", o)

# --- F6b ---------------------------------------------------------------------
o = {}
h = 0.3; T = 30.0
hf = h/4000.; nf = int(T/hf)+1
gf = np.empty(nf); th = 0.3; gf[0] = math.cosh(th)
for i in range(1, nf): th = rk4(f, th, hf); gf[i] = math.cosh(th)
for nm in ("euler", "ieuler"):
    n = int(T/h); th = 0.3; gn = np.empty(n+1); gn[0] = math.cosh(th)
    for i in range(1, n+1):
        if nm == "euler": th = th + h*f(th)
        else:
            x = th + h*f(th)
            for _ in range(60):
                dd = (x - th - h*f(x))/(1 - h*fp(x)); x -= dd
                if abs(dd) < 1e-16*max(1., abs(x)): break
            th = x
        gn[i] = math.cosh(th)
    tn = np.arange(n+1)*h
    dev = np.abs(gn - gf[(np.arange(n+1)*4000)])/gf[0]
    m = (dev > 1e-10) & (dev < 1e-2) & (tn > 0)
    o[nm] = {"fitted_decay_rate": float(-np.polyfit(tn[m], np.log(dev[m]), 1)[0]),
             "n_points": int(m.sum()), "t_window": [float(tn[m][0]), float(tn[m][-1])]}
RES["F6b"] = o; print("F6b", o)

# Merge rather than overwrite.  results3.json also carries
# "F6b_refit_clean_window", a refit over a later window that was computed
# inline during the campaign and never written into any script; Section 6 of
# the paper quotes two of its numbers (1.7954 and 1.4286).  Overwriting the
# file therefore silently deleted numbers the manuscript cites, which is what
# happened when the reproduction driver was first run end to end (W6.2).  Keys
# this script does write are replaced; keys it does not write are kept.
_out = os.path.join(HERE, "results3.json")
if os.path.exists(_out):
    _prev = json.load(open(_out, encoding="utf-8"))
    _prev.update(RES)
    RES = _prev
json.dump(RES, open(_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("wrote results3.json")
