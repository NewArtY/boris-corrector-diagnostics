"""LL rapidity system: experiments.  Run AFTER derive.py (prereg.json frozen).

E1  attractor exactness + measured contraction rate Lambda_h   -> tests P7, P7b
E2  read-out convention artefact                                -> tests P6 (T2)
E3  defect response: envelopes and exponents, both channels     -> tests P1-P5 (T3, T7)
E4  contrast control: conservative rotation model, same harness -> reproduces T3 p=1

Envelope methodology copied from horizon/long_runs.py: running maximum, power-law
fit of log10(envelope) vs log10(t) over t > t_end/100.
"""
import json, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE = json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8"))
PARAMS = {k: PRE["params"][k] for k in ("physical", "generic")}
RES = {"note": "measured; compare against prereg.json"}

# ------------------------------------------------------------------ system --
def make(al, ep):
    def f(t):   return al*math.tanh(t) - ep*math.sinh(t)*math.cosh(t)
    def fp(t):  ch = math.cosh(t); return al/(ch*ch) - ep*math.cosh(2*t)
    def g(t):   return 1.0 - math.tanh(t)          # neutral quadrature dpsi/dt
    return f, fp, g

def step_euler(f, fp, th, h):  return th + h*f(th)
def step_rk4(f, fp, th, h):
    k1 = f(th); k2 = f(th + .5*h*k1); k3 = f(th + .5*h*k2); k4 = f(th + h*k3)
    return th + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
def _newton(F, dF, x0):
    x = x0
    for _ in range(60):
        d = F(x)/dF(x); x -= d
        if abs(d) < 1e-16*max(1.0, abs(x)): break
    return x
def step_ieuler(f, fp, th, h):
    return _newton(lambda x: x - th - h*f(x), lambda x: 1 - h*fp(x), th + h*f(th))
def step_trap(f, fp, th, h):
    c = th + .5*h*f(th)
    return _newton(lambda x: x - c - .5*h*f(x), lambda x: 1 - .5*h*fp(x), th + h*f(th))
STEPS = {"euler": step_euler, "ieuler": step_ieuler, "trapezoid": step_trap, "rk4": step_rk4}

def envelope_exponent(t, dev):
    env = np.maximum.accumulate(np.abs(dev))
    sel = (t > t[-1]/100.0) & (env > 0)
    if sel.sum() < 10: return float("nan"), env
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
    return float(p[0]), env

# =============================================================== E1: rate ====
def e1():
    out = {}
    for pname, S in PARAMS.items():
        al, ep, ths, L = S["alpha"], S["eps"], S["theta_star"], S["Lambda"]
        f, fp, g = make(al, ep)
        th0 = 0.8 if pname == "physical" else 0.3
        for h in (0.3, 0.05):
            for sname, st in STEPS.items():
                # (a) fixed point: iterate to convergence from th0
                th = th0
                for _ in range(50000):
                    nxt = st(f, fp, th, h)
                    if abs(nxt - th) < 1e-16*max(1.0, abs(th)): th = nxt; break
                    th = nxt
                fp_err = abs(th - ths)
                # (b) multiplier of the map at the numerical fixed point (central FD)
                d = 1e-7
                rho = (st(f, fp, th + d, h) - st(f, fp, th - d, h))/(2*d)
                Lh = -math.log(abs(rho))/h
                # (c) independent: exponential fit of |theta(t)-theta*| in the linear regime
                th = th0; ts = []; ds = []
                for n in range(1, 400001):
                    th = st(f, fp, th, h); dd = abs(th - ths)
                    if dd < 1e-2 and dd > 1e-11: ts.append(n*h); ds.append(dd)
                    if dd < 1e-11: break
                Lfit = float(-np.polyfit(ts, np.log(ds), 1)[0]) if len(ts) > 5 else float("nan")
                out[f"{pname}_h{h}_{sname}"] = {
                    "theta_star": ths, "fixed_point_abs_err": float(fp_err),
                    "rho_measured": float(rho), "Lambda_h_measured": float(Lh),
                    "Lambda_h_from_fit": Lfit, "Lambda_true": L,
                    "rel_bias": float(Lh/L - 1)}
    return out

# ========================================================= E2: convention ====
def e2():
    out = {}
    for pname, S in PARAMS.items():
        al, ep, ths = S["alpha"], S["eps"], S["theta_star"]
        f, fp, g = make(al, ep)
        th0 = 0.8 if pname == "physical" else 0.3
        T = 8.0 if pname == "physical" else 2.0
        for h in (0.05, 0.025):
            # dense reference on a fine grid (rk4, h/2000) for interpolation
            hf = h/2000.0; nf = int(round(T*1.5/hf)) + 1
            tf = np.arange(nf)*hf; gf = np.empty(nf); th = th0
            gf[0] = math.cosh(th)
            for i in range(1, nf):
                th = step_rk4(f, fp, th, hf); gf[i] = math.cosh(th)
            g0 = gf[0]
            for sname in ("euler", "trapezoid", "rk4"):
                st = STEPS[sname]; n = int(round(T/h)) + 1
                tn = np.arange(n)*h; gn = np.empty(n); th = th0; gn[0] = math.cosh(th)
                for i in range(1, n):
                    th = st(f, fp, th, h); gn[i] = math.cosh(th)
                half = n//2
                rec = {}
                for dl in np.linspace(-h, h, 41):
                    gr = np.interp(tn + dl, tf, gf)
                    rec[f"{dl:+.5f}"] = float(np.median(np.abs(gn - gr)[half:])/g0)
                gr0 = np.interp(tn, tf, gf)
                dev0 = float(np.median(np.abs(gn - gr0)[half:])/g0)
                grh = np.interp(tn + h/2, tf, gf)
                devh = float(np.median(np.abs(gn - grh)[half:])/g0)
                # d ln gamma / dt at the median evaluation time
                tmed = float(np.median(tn[half:]))
                thm = float(np.interp(tmed, tf, np.arccosh(np.maximum(gf, 1.0))))
                dlng = abs(math.tanh(thm)*f(thm))
                vals = np.array(list(rec.values())); dls = np.array([float(k) for k in rec])
                signal = float(abs(gf[int(round(T/hf))] - g0)/g0)
                out[f"{pname}_h{h}_{sname}"] = {
                    "dev_at_offset_0": dev0, "dev_at_offset_half_h": devh,
                    "artefact_h_over_2": devh - dev0,
                    "predicted_artefact": float((h/2)*dlng),
                    "dlnGamma_dt_at_tmed": float(dlng), "t_med": tmed, "theta_at_tmed": thm,
                    "min_dev_over_offsets": float(vals.min()),
                    "argmin_offset": float(dls[vals.argmin()]),
                    "collapse_factor_vs_offset0": float(dev0/max(vals.min(), 1e-300)),
                    "signal": signal,
                    "R_signal_over_dev0": float(signal/max(dev0, 1e-300)),
                    "sweep": rec}
    return out

# ============================================================ E3: defect =====
def e3(T_end=1e5, h=0.05, kappa=1e-6, n_samp=4000):
    out = {}
    nst = int(round(T_end/h))
    idx = np.unique(np.round(np.logspace(0, math.log10(nst), n_samp)).astype(int))
    for pname, S in PARAMS.items():
        al, ep, ths, L = S["alpha"], S["eps"], S["theta_star"], S["Lambda"]
        f, fp, g = make(al, ep)
        th0 = 0.8 if pname == "physical" else 0.3
        modes = [("DC", lambda n, t: kappa),
                 ("sin_Lambda", lambda n, t, w=L: kappa*math.sin(w*t)),
                 ("sin_10Lambda", lambda n, t, w=10*L: kappa*math.sin(w*t)),
                 ("sin_alias_2pi_h", lambda n, t, w=2*math.pi/h: kappa*math.sin(w*t)),
                 ("sin_nyquist", lambda n, t, w=math.pi/h: kappa*math.sin(w*t)),
                 ("noise", None)]
        if pname == "physical":
            modes = [m for m in modes if m[0] in ("DC", "sin_Lambda", "noise")]
        for mname, kf in modes:
            rng = np.random.default_rng(20260831)
            noise = rng.standard_normal(nst)*kappa if mname == "noise" else None
            thu = th0; thp = th0; psu = 0.0; psp = 0.0
            g0 = math.cosh(th0)
            ts = np.empty(len(idx)); dth = np.empty(len(idx))
            dgm = np.empty(len(idx)); dps = np.empty(len(idx))
            j = 0; t0 = time.time()
            for n in range(nst):
                t = n*h
                psu += h*g(thu); psp += h*g(thp)
                thu = step_rk4(f, fp, thu, h)
                thp = step_rk4(f, fp, thp, h) + (noise[n] if noise is not None else kf(n, t))
                if j < len(idx) and n + 1 == idx[j]:
                    ts[j] = (n+1)*h; dth[j] = thp - thu
                    dgm[j] = (math.cosh(thp) - math.cosh(thu))/g0
                    dps[j] = psp - psu; j += 1
            pE, envE = envelope_exponent(ts, dgm)
            pT, envT = envelope_exponent(ts, dth)
            pP, envP = envelope_exponent(ts, dps)
            out[f"{pname}_{mname}"] = {
                "h": h, "kappa": kappa, "T_end": T_end, "seconds": round(time.time()-t0, 1),
                "exponent_energy_channel": pE, "exponent_theta": pT,
                "exponent_neutral_channel_psi": pP,
                "plateau_dtheta_final": float(abs(dth[-1])),
                "plateau_dtheta_envmax": float(envT[-1]),
                "rms_dtheta_last_decade": float(np.sqrt(np.mean(dth[ts > ts[-1]/10]**2))),
                "dev_gamma_envmax": float(envE[-1]),
                "dpsi_final": float(dps[-1]),
                "dpsi_slope_last_decade": float(np.polyfit(ts[ts > ts[-1]/10],
                                                           dps[ts > ts[-1]/10], 1)[0])}
    return out

# ========================================== E4: conservative contrast ========
def e4(T_end=1e5, h=0.3, kappa=1e-6, n_samp=4000):
    """z_{n+1} = e^{-i th_h} z_n + kappa_n ; Omega = 1, th_h = 2 atan(h/2)."""
    out = {}
    thh = 2*math.atan(h/2.0); wh = thh/h
    nst = int(round(T_end/h))
    idx = np.unique(np.round(np.logspace(0, math.log10(nst), n_samp)).astype(int))
    rot = complex(math.cos(thh), -math.sin(thh))
    modes = {"DC_along_v": None, "resonant_wh": wh, "detuned_1.0": 1.0, "noise": "noise"}
    for mname, w in modes.items():
        rng = np.random.default_rng(20260831)
        z = complex(1.0, 0.0); e0 = abs(z)**2
        ts = np.empty(len(idx)); dv = np.empty(len(idx)); j = 0
        for n in range(nst):
            t = n*h
            z = rot*z
            if mname == "DC_along_v":       z = z*(1 + kappa)
            elif mname == "noise":          z = z + complex(rng.standard_normal()*kappa, 0.0)
            else:                           z = z + complex(kappa*math.sin(w*t), 0.0)
            if j < len(idx) and n + 1 == idx[j]:
                ts[j] = (n+1)*h; dv[j] = abs(abs(z)**2 - e0)/e0; j += 1
        p, env = envelope_exponent(ts, dv)
        out[mname] = {"exponent": p, "env_max": float(env[-1]), "omega_h": wh}
    return out

if __name__ == "__main__":
    t0 = time.time(); RES["E1_rate"] = e1();        print("E1 done", round(time.time()-t0))
    t0 = time.time(); RES["E2_convention"] = e2();  print("E2 done", round(time.time()-t0))
    t0 = time.time(); RES["E3_defect"] = e3();      print("E3 done", round(time.time()-t0))
    t0 = time.time(); RES["E4_contrast"] = e4();    print("E4 done", round(time.time()-t0))
    json.dump(RES, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("wrote results.json")
