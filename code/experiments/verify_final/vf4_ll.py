"""vf4: LL rapidity system, all from scratch.

  theta' = f(theta) = alpha tanh(theta) - eps sinh(theta) cosh(theta)
  y = sinh^2 theta  =>  y' = Lam y (1 - y/y*),  Lam = 2(alpha-eps), y* = (alpha-eps)/eps
  closed-form logistic reference and exact inverse t(y).

Checks:
  (3.2a) Euler resynchronisation s(t) = -(h/2) ln[f(theta)/f(theta0)]
  (3.2b) dev0/A_ref ~ h^{p-1}: measured orders for Euler/trapezoid/midpoint/RK4
  (3.3a) defect injection on the generic set (alpha=1, eps=0.1, h=0.05, RK4):
         DC / sin(Lam t) / sin(10 Lam t) / cos alias 2pi/h / cos Nyquist pi/h /
         white noise; envelope exponent + plateau vs kappa/|1-rho e^{i w h}|
  (3.3b) rho<0 caveat: explicit Euler, Lam h > 1 -> worst frequency is Nyquist,
         NOT omega=0 (condition 0<rho<1 unstated in 12_SECOND_SYSTEM)
  (3.3c) fixed point exact / Lambda_h biased +12.66% (physical set, Euler h=0.3)
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

def make_sys(alpha, eps):
    Lam = 2 * (alpha - eps)
    ystar = (alpha - eps) / eps
    def f(th):
        return alpha * np.tanh(th) - eps * np.sinh(th) * np.cosh(th)
    def y_of_t(t, th0):
        y0 = np.sinh(th0) ** 2
        return ystar / (1.0 + (ystar / y0 - 1.0) * np.exp(-Lam * t))
    def th_of_t(t, th0):
        return np.arcsinh(np.sqrt(y_of_t(t, th0)))
    def t_of_th(th, th0):
        y0 = np.sinh(th0) ** 2; y = np.sinh(th) ** 2
        return (1.0 / Lam) * np.log((y * (ystar - y0)) / (y0 * (ystar - y)))
    return f, Lam, ystar, th_of_t, t_of_th

# ---------------------------------------------------------------- schemes
def run_scheme(scheme, f, th0, h, T, kick=None):
    n = int(round(T / h))
    th = th0; t = 0.0
    ths = np.zeros(n + 1); ths[0] = th0
    for i in range(1, n + 1):
        if scheme == "euler":
            th = th + h * f(th)
        elif scheme == "rk4":
            k1 = f(th); k2 = f(th + h / 2 * k1)
            k3 = f(th + h / 2 * k2); k4 = f(th + h * k3)
            th = th + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif scheme == "trap":
            x = th + h * f(th)
            for _ in range(60):
                xn = th + h / 2 * (f(th) + f(x))
                if abs(xn - x) < 1e-16: x = xn; break
                x = xn
            th = x
        elif scheme == "mid":
            x = th + h * f(th)
            for _ in range(60):
                xn = th + h * f(0.5 * (th + x))
                if abs(xn - x) < 1e-16: x = xn; break
                x = xn
            th = x
        if kick is not None:
            th = th + kick(i - 1, t)
        t += h
        ths[i] = th
    return np.arange(n + 1) * h, ths

# ============================================== (3.2a) s(t) closed form, Euler
alpha, eps, th0, T = 0.36, 1.011582e-5, 0.8, 8.0
f, Lam, ystar, th_of_t, t_of_th = make_sys(alpha, eps)
res = {}
for h in (0.1, 0.05, 0.025, 0.0125):
    ts, ths = run_scheme("euler", f, th0, h, T)
    s_meas = t_of_th(ths[1:], th0) - ts[1:]
    s_form = -(h / 2) * np.log(f(ths[1:]) / f(th0))
    hh = len(ts) // 2
    res[f"h={h}"] = {"ratio_median_2nd_half": float(np.median(s_meas[hh:] / s_form[hh:])),
                     "ratio_final": float(s_meas[-1] / s_form[-1])}
out["s_formula_euler"] = res

# ============================================== (3.2b) dev0/A_ref orders
def dev0_over_Aref(scheme, h):
    ts, ths = run_scheme(scheme, f, th0, h, T)
    Q = np.cosh(ths) / np.cosh(th0)
    Qref = np.cosh(th_of_t(ts, th0)) / np.cosh(th0)
    hh = len(ts) // 2
    dev0 = np.median(np.abs(Q - Qref)[hh:])
    Aref = np.median(np.abs(np.cosh(th_of_t(ts[hh:] + h / 2, th0)) -
                            np.cosh(th_of_t(ts[hh:], th0))) / np.cosh(th0))
    return dev0 / Aref

orders = {}
hs = np.array([0.1, 0.05, 0.025, 0.0125])
for scheme in ("euler", "trap", "mid", "rk4"):
    vals = np.array([dev0_over_Aref(scheme, h) for h in hs])
    ords = list(np.round(np.log2(vals[:-1] / vals[1:]), 3))
    orders[scheme] = {"dev0_over_Aref": [float(v) for v in vals],
                      "orders": [float(o) for o in ords]}
out["dev0_Aref_orders"] = orders

# ============================================== (3.3a) defect injection, generic
alpha_g, eps_g = 1.0, 0.1
fg, LamG, ystarG, th_of_tG, _ = make_sys(alpha_g, eps_g)
h = 0.05
x = -LamG * h
rho = 1 + x + x**2/2 + x**3/6 + x**4/24          # RK4 stability function
kap = 1e-6
th0g, Tg = 0.3, 1e5

def envelope_exponent(t, dev):
    env = np.maximum.accumulate(dev)
    sel = (t > t[-1] / 100) & (env > 0)
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
    return float(p[0]), env

def defect_run(kick, label, pred_denom):
    ts, ths = run_scheme("rk4", fg, th0g, h, Tg, kick=kick)
    dth = np.abs(ths - th_of_tG(ts, th0g))
    # subsample for the envelope (campaign takes ~4000 samples)
    stride = max(1, len(ts) // 4000)
    tt = ts[1::stride]; dd = dth[1::stride]
    p, env = envelope_exponent(tt, dd)
    hh2 = len(dd) // 2
    plateau = float(np.max(dd[hh2:]))
    pred = kap / pred_denom if pred_denom else None
    return {"exponent": p, "plateau_max_2nd_half": plateau,
            "pred": (float(pred) if pred else None),
            "plateau/pred": (float(plateau / pred) if pred else None)}

rng = np.random.default_rng(2026)
wA = 2 * np.pi / h; wN = np.pi / h
den = lambda w: abs(1 - rho * np.exp(1j * w * h))
cases = {
    "DC": (lambda i, t: kap, den(0)),
    "sin_wLam": (lambda i, t: kap * np.sin(LamG * t), den(LamG)),
    "sin_10Lam": (lambda i, t: kap * np.sin(10 * LamG * t), den(10 * LamG)),
    "cos_alias_2pi_h": (lambda i, t: kap * np.cos(wA * t), den(wA)),
    "cos_nyquist_pi_h": (lambda i, t: kap * np.cos(wN * t), den(wN)),
}
g33 = {}
for lbl, (kick, dn) in cases.items():
    g33[lbl] = defect_run(kick, lbl, dn)
noise = rng.normal(size=int(round(Tg / h)) + 2)
g33["white_noise"] = defect_run(lambda i, t: kap * noise[i], "noise", None)
g33["white_noise"]["pred_rms"] = float(kap / np.sqrt(1 - rho ** 2))
g33["rho_rk4"] = float(rho)
out["generic_defects"] = g33

# ============================================== (3.3b) rho<0 caveat, Euler h big
h2 = 0.7                       # Lam h = 1.26 -> rho = -0.26
rho2 = 1 - LamG * h2
c33 = {}
for lbl, kick, dn in (("DC", lambda i, t: kap, abs(1 - rho2)),
                      ("nyquist", lambda i, t: kap * np.cos(np.pi / h2 * t), abs(1 + rho2))):
    ts, ths = run_scheme("euler", fg, th0g, h2, 2e4, kick=kick)
    dth = np.abs(ths - th_of_tG(ts, th0g))
    hh2 = len(dth) // 2
    c33[lbl] = {"plateau": float(np.max(dth[hh2:])), "pred_kap_over_denom": float(kap / dn)}
c33["rho"] = float(rho2)
out["rho_negative_caveat"] = c33

# ============================================== (3.3c) fixed point / Lambda_h
res = {}
th_star = np.arccosh(np.sqrt(alpha / eps))
for scheme, pred in (("euler", -np.log(1 - Lam * 0.3) / 0.3),
                     ("rk4", None)):
    ts, ths = run_scheme(scheme, f, th0, 0.3, 200.0)
    res[scheme] = {"fixed_point_gap": float(abs(ths[-1] - th_star))}
    d = np.abs(ths - th_star)
    m = (d > 1e-11) & (d < 1e-3) & (ts > 20)
    if m.sum() > 5:
        sl = np.polyfit(ts[m], np.log(d[m]), 1)[0]
        res[scheme]["Lambda_h_measured"] = float(-sl)
    if pred:
        res[scheme]["Lambda_h_predicted"] = float(pred)
        res[scheme]["bias_vs_Lam_percent"] = float((pred / Lam - 1) * 100)
res["Lambda_true"] = float(Lam)
res["theta_star"] = float(th_star)
out["fixed_point_and_rate"] = res

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf4_ll.json"), "w"), indent=1)
