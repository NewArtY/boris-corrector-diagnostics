"""Independent verification, T2a/T2b: sampling-shift floor and the '600x'.

Checks, all with my own integrator loops (no reuse of theory_check code):

 1. Floor statistics of the shipped Boris in B4 (exp decay, tau=1.2e5,
    h=0.3): dev(t) vs the adiabatic law. Claimed: median = (h/2tau)e^{-t/tau},
    envelope max = h/tau (factor 2 above the pure sampling shift!), plus the
    unstated oscillatory component that must supply that factor.
 2. Generality of R = signal/error ~= 2 t_med/h = 600:
      a. exponential decay, tau in {1.2e4, 1.2e5, 1.2e6, 240, 120};
         exact prediction R = (e^x - 1)/x * 2 t/h,  x = t_med/tau.
      b. power-law decay B = B0 (1+t/tau)^{-1} (induced E consistent with
         Faraday), tau in {1.2e5, 240}; prediction R = (1+x) * 2 t/h.
      c. a different scheme in the same convention: exact-rotation
         semi-implicit variant (rotation by exactly h*Omega instead of
         2 atan(h Omega/2)); prediction: R still ~ 600.
      d. a scheme with genuine energy drift: classical RK4 read at integer
         times; prediction: R nowhere near 600 (dominated by intrinsic
         dissipation) -> the hidden applicability condition.
      e. h = 0.15 -> R ~ 1200.
 3. Staggered variant, my own loop: avg-recentring error vs
    sin^2(theta_h(t)/2) e^{-t/tau} pointwise; rotation-recentring collapse
    to the sampling floor; identical state sequence for both readouts.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 2 * np.pi


# ----------------------------------------------------------------- fields
def field_exp(r, t, tau, beta=None):
    Bz = np.exp(-t / tau)
    fac = 0.5 * Bz / tau                    # E = (rho/2)(B/tau) phi_hat
    return np.array([-fac * r[1], fac * r[0], 0.0]), Bz


def field_pow(r, t, tau, beta=None):
    beta = 1.0 if beta is None else beta
    Bz = (1.0 + t / tau) ** (-beta)
    fac = 0.5 * beta * Bz / (tau + t)       # E = (rho/2) beta B/(tau+t)
    return np.array([-fac * r[1], fac * r[0], 0.0]), Bz


# ----------------------------------------------------------------- schemes
def kick_boris(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])
    return vp + k * E


def kick_exactrot(v, E, Bz, h, q=-1.0):
    """Half E kick, EXACT rotation by q*Bz*h about z, half E kick."""
    k = 0.5 * q * h
    vm = v + k * E
    ang = q * Bz * h            # signed rotation angle (q=-1: clockwise?)
    # Boris with t = tan(ang/2): reproduce the same rotation direction as
    # kick_boris but with the exact angle.
    tz = np.tan(0.5 * ang)
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])
    return vp + k * E


def run_shipped(kick, field, tau, h, t_final, beta=None):
    n = int(round(t_final / h))
    r = np.array([1.0, 0.0, 0.0]); v = np.array([0.0, 1.0, 0.0])
    t = 0.0
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1); sp2[0] = 1.0
    for i in range(1, n + 1):
        E, Bz = field(r, t, tau, beta)
        v = kick(v, E, Bz, h)
        r = r + v * h
        t += h
        ts[i] = t; sp2[i] = v @ v
    return ts, sp2


def rk4_lorentz(field, tau, h, t_final, beta=None, q=-1.0):
    n = int(round(t_final / h))
    y = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    t = 0.0
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1); sp2[0] = 1.0

    def f(t, y):
        r = y[:3]; v = y[3:]
        E, Bz = field(r, t, tau, beta)
        B = np.array([0.0, 0.0, Bz])
        return np.concatenate([v, q * (E + np.cross(v, B))])

    for i in range(1, n + 1):
        k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        ts[i] = t; sp2[i] = y[3:] @ y[3:]
    return ts, sp2


def adiabatic(ts, tau, law):
    if law == "exp":
        return np.exp(-ts / tau)
    return (1.0 + ts / tau) ** (-1.0)


def R_of(ts, sp2, tau, law):
    """Campaign diagnostic: signal(t_med) / median dev over 2nd half."""
    Ead = adiabatic(ts, tau, law)
    dev = np.abs(0.5 * sp2 - 0.5 * Ead) / 0.5
    half = len(ts) // 2
    med = float(np.median(dev[half:]))
    t_med = float(np.median(ts[half:]))
    signal = float(1.0 - adiabatic(np.array([t_med]), tau, law)[0])
    return signal / med, med, t_med


out = {}

# ---------------- 1. floor statistics, B4 paper config ----------------------
tau, h = 1.2e5, 0.3
ts, sp2 = run_shipped(kick_boris, field_exp, tau, h, 120.0)
Ead = np.exp(-ts / tau)
dev = np.abs(0.5 * sp2 - 0.5 * Ead) / 0.5
half = len(ts) // 2
# oscillation decomposition over the 2nd half
d2 = dev[half:]
out["floor_stats_t120"] = {
    "median_2nd_half": float(np.median(d2)),
    "pred_median_h_over_2tau_at_t90": float(h / (2 * tau) * np.exp(-90 / tau)),
    "max": float(np.max(dev)), "pred_max_h_over_tau": h / tau,
    "min_2nd_half": float(np.min(d2)),
    "osc_amplitude_(max-min)/2_2nd_half": float((np.max(d2) - np.min(d2)) / 2),
    "pred_osc_amp_h_over_2tau": h / (2 * tau),
}
# long horizon for the envelope claim (1e3 gyro)
ts_l, sp2_l = run_shipped(kick_boris, field_exp, tau, h, 1e3 * TWO_PI)
dev_l = np.abs(0.5 * sp2_l - 0.5 * np.exp(-ts_l / tau)) / 0.5
out["envelope_1e3gyro_max"] = {"measured": float(np.max(dev_l)),
                               "pred_h_over_tau": h / tau,
                               "campaign_long_runs": 2.499690216373196e-06}

# ---------------- 2. generality of the 600x ---------------------------------
rows = []
for law, fld, taus in (("exp", field_exp, (1.2e4, 1.2e5, 1.2e6, 240.0, 120.0)),
                       ("pow", field_pow, (1.2e5, 240.0))):
    for tv in taus:
        ts, sp2 = run_shipped(kick_boris, fld, tv, 0.3, 120.0)
        R, med, t_med = R_of(ts, sp2, tv, law)
        x = t_med / tv
        if law == "exp":
            R_exact = (np.expm1(x) / x) * 2 * t_med / 0.3
        else:
            R_exact = (1 + x) * 2 * t_med / 0.3
        rows.append({"scheme": "boris", "law": law, "tau": tv,
                     "x=t_med/tau": x, "R_measured": R,
                     "R_2t_over_h": 2 * t_med / 0.3, "R_exact_pred": R_exact})
# other schemes, exp tau=1.2e5
for name, runner in (("exactrot", lambda: run_shipped(kick_exactrot, field_exp, 1.2e5, 0.3, 120.0)),
                     ("rk4", lambda: rk4_lorentz(field_exp, 1.2e5, 0.3, 120.0))):
    ts, sp2 = runner()
    R, med, t_med = R_of(ts, sp2, 1.2e5, "exp")
    rows.append({"scheme": name, "law": "exp", "tau": 1.2e5,
                 "R_measured": R, "R_2t_over_h": 2 * t_med / 0.3,
                 "median_dev": med})
# h = 0.15
ts, sp2 = run_shipped(kick_boris, field_exp, 1.2e5, 0.15, 120.0)
R, med, t_med = R_of(ts, sp2, 1.2e5, "exp")
rows.append({"scheme": "boris", "law": "exp", "tau": 1.2e5, "h": 0.15,
             "R_measured": R, "R_2t_over_h": 2 * t_med / 0.15})
out["R_600x"] = rows

# ---------------- 3. staggered variant, my own loop -------------------------
def staggered(h, tau, t_final):
    n = int(round(t_final / h))
    r = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
    E, Bz = field_exp(r, 0.0, tau)
    w = kick_boris(v0, E, Bz, -0.5 * h)     # v_{-1/2}
    ts = np.zeros(n + 1)
    v_avg = np.zeros((n + 1, 3)); v_rot = np.zeros((n + 1, 3))
    v_avg[0] = v0; v_rot[0] = v0
    rs = np.zeros((n + 1, 3)); rs[0] = r
    t = 0.0
    for i in range(1, n + 1):
        E, Bz = field_exp(r, t, tau)
        w_new = kick_boris(w, E, Bz, h)
        r = r + w_new * h
        t += h
        # avg readout
        v_avg[i] = 0.5 * (w + w_new)
        # rotation readout: rotate w_new BACK by half the step rotation.
        # step rotation angle (signed, q=-1, about z): -theta_h where
        # theta_h = 2 atan(h Bz/2); w rotated clockwise; undo half => rotate
        # counterclockwise by theta_h/2? Determine sign empirically robust:
        th = np.arctan(0.5 * h * Bz)        # theta_h/2 magnitude
        c, s = np.cos(th), np.sin(th)
        cand1 = np.array([c * w_new[0] - s * w_new[1],
                          s * w_new[0] + c * w_new[1], w_new[2]])
        cand2 = np.array([c * w_new[0] + s * w_new[1],
                          -s * w_new[0] + c * w_new[1], w_new[2]])
        # pick the rotation that best matches the avg direction (the true
        # integer-time velocity); both preserve |w_new| exactly
        v_rot[i] = cand1 if np.dot(cand1, v_avg[i]) >= np.dot(cand2, v_avg[i]) else cand2
        w = w_new
        ts[i] = t; rs[i] = r
    return ts, v_avg, v_rot, rs

h = 0.3; tau = 1.2e5
ts, v_avg, v_rot, rs = staggered(h, tau, 120.0)
Ead = np.exp(-ts / tau)
dev_avg = np.abs(np.sum(v_avg ** 2, axis=1) - Ead)
dev_rot = np.abs(np.sum(v_rot ** 2, axis=1) - Ead)
half = len(ts) // 2
Om = np.exp(-ts / tau)
pred_avg = (np.sin(np.arctan(h * Om / 2)) ** 2) * np.exp(-ts / tau)
ratio = dev_avg[half:] / pred_avg[half:]
out["staggered_avg"] = {
    "median_measured": float(np.median(dev_avg[half:])),
    "median_predicted_sin2": float(np.median(pred_avg[half:])),
    "pointwise_ratio_mean": float(np.mean(ratio)),
    "pointwise_ratio_std": float(np.std(ratio)),
}
out["staggered_rot"] = {
    "median_measured": float(np.median(dev_rot[half:])),
    "sampling_floor_pred": float(h / (2 * tau) * np.exp(-90 / tau)),
    "collapse_factor": float(np.median(dev_avg[half:]) / np.median(dev_rot[half:])),
}
out["staggered_ratio_of_conventions"] = {
    "measured": float(np.median(dev_avg[half:]) / (h / (2 * tau) * np.exp(-90 / tau))),
    "pred_sin2_x_2tau_over_h": float(np.median(pred_avg[half:]) * 2 * tau / h)}

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt2_floor_600x.json"), "w"), indent=1)
