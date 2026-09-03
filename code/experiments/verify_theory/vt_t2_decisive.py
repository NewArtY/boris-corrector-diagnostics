"""VT-T2 decisive tests.

(1) GUIDING-CENTRE TEST.  My mechanism says the shipped-Boris energy floor is
    caused by the O(h) displacement of the DISCRETE guiding centre off the
    field axis (|c| = h|v|/2), which modulates rho and hence the betatron
    work at omega_h.  Prediction: choose r0 so that the DISCRETE guiding
    centre sits exactly on the axis, and the floor must collapse by O(h),
    while the 'h/2 sampling shift of a decaying energy' mechanism predicts
    NO change at all (it does not depend on the orbit).

(2) READOUT STRUCTURE.  For the staggered variant with rotation-recentring the
    readout speed is |w_{n+1/2}| = the true speed at a half-integer time; that
    IS a pure sampling shift and must give a CONSTANT offset h/(2tau)
    (envelope == median).  For the shipped variant the error oscillates
    0..h/tau (envelope == 2*median).  Same number, different objects.

(3) GENERALITY OF R = 2 t_med / h ('600x'): schemes x decay laws x tau x h.

(4) MAP IDENTITY and what the initial half-step-back really costs
    (convergence order in position).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
TWO_PI = 2 * np.pi


# ------------------------------------------------------------------ fields
def mk_field(law, tau, beta=1.0):
    """Return f(t) = Bz/B0 and the consistent induced-E prefactor
    fac = -0.5 dBz/dt, so that E = fac*(zhat x r)."""
    if law == "exp":
        return (lambda t: np.exp(-t / tau),
                lambda t: 0.5 * np.exp(-t / tau) / tau)
    if law == "pow":
        return (lambda t: (1.0 + t / tau) ** (-beta),
                lambda t: 0.5 * beta / tau * (1.0 + t / tau) ** (-beta - 1.0))
    if law == "lin":                      # B = 1 - t/tau
        return (lambda t: 1.0 - t / tau, lambda t: 0.5 / tau + 0.0 * t)
    if law == "cos":                      # B = 1 - a(1-cos(t/tau)) : non-monotone
        a = 0.5
        return (lambda t: 1.0 - a * (1.0 - np.cos(t / tau)),
                lambda t: 0.5 * a * np.sin(t / tau) / tau)
    if law == "gauss":
        return (lambda t: np.exp(-(t / tau) ** 2),
                lambda t: 0.5 * (2 * t / tau ** 2) * np.exp(-(t / tau) ** 2))
    raise ValueError(law)


# ----------------------------------------------------------------- schemes
def kick_boris(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    return np.array([vm[0] + vpy * sz + k * E[0],
                     vm[1] - vpx * sz + k * E[1], vm[2]])


def kick_exactrot(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = np.tan(0.5 * q * Bz * h)
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    return np.array([vm[0] + vpy * sz + k * E[0],
                     vm[1] - vpx * sz + k * E[1], vm[2]])


def run_generic(kick, fB, ffac, h, t_final, r0, v0, drift="new"):
    """drift='new': r += v_{n+1} h (shipped).  'mid': r += (v_n+v_{n+1})/2 h."""
    n = int(round(t_final / h))
    r = np.array(r0, float); v = np.array(v0, float)
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1); rho = np.zeros(n + 1)
    sp2[0] = v @ v; rho[0] = np.hypot(r[0], r[1])
    t = 0.0
    for i in range(1, n + 1):
        Bz = fB(t); fac = ffac(t)
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        vo = v
        v = kick(v, E, Bz, h)
        r = r + (v if drift == "new" else 0.5 * (vo + v)) * h
        t += h
        ts[i] = t; sp2[i] = v @ v; rho[i] = np.hypot(r[0], r[1])
    return ts, sp2, rho


def rk4(fB, ffac, h, t_final, r0, v0, q=-1.0):
    n = int(round(t_final / h))
    y = np.concatenate([np.array(r0, float), np.array(v0, float)])
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1); sp2[0] = y[3:] @ y[3:]
    t = 0.0

    def f(t, y):
        r = y[:3]; v = y[3:]
        fac = ffac(t); Bz = fB(t)
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        return np.concatenate([v, q * (E + np.cross(v, np.array([0, 0, Bz])))])
    for i in range(1, n + 1):
        k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h
        ts[i] = t; sp2[i] = y[3:] @ y[3:]
    return ts, sp2, None


out = {}
TAU, H, TF = 1.2e5, 0.3, 120.0
fB, ffac = mk_field("exp", TAU)

# ==================================================================== (1)
# discrete guiding centre of the shipped map: r_n - c = M v_n,
# M = h R (R - I)^{-1}, R = rotation by -theta_h about z.
th = 2 * np.arctan(H / 2)
c_, s_ = np.cos(th), np.sin(th)
R = np.array([[c_, s_], [-s_, c_]])                 # clockwise (q=-1)
M = H * R @ np.linalg.inv(R - np.eye(2))
v0 = np.array([0.0, 1.0])
c_off = M @ v0                                      # r_0 - c  if GC at r0 - Mv0
# to place the discrete GC at the ORIGIN we need r0 = M v0
r0_gc = np.array([M[0, 0] * v0[0] + M[0, 1] * v0[1],
                  M[1, 0] * v0[0] + M[1, 1] * v0[1], 0.0])
rows = []
for lbl, r0 in (("standard r0=(1,0,0)", np.array([1.0, 0.0, 0.0])),
                ("GC-on-axis r0 = M v0", r0_gc)):
    ts, sp2, rho = run_generic(kick_boris, fB, ffac, H, TF, r0, (0.0, 1.0, 0.0))
    E0 = sp2[0]
    dev = np.abs(sp2 / E0 - fB(ts))
    hh = len(ts) // 2
    rows.append({"case": lbl, "r0": list(np.round(r0, 6)),
                 "rho_min_2nd_half": float(np.min(rho[hh:])),
                 "rho_max_2nd_half": float(np.max(rho[hh:])),
                 "GC_offset_est_(rmax-rmin)/2": float((np.max(rho[hh:]) - np.min(rho[hh:])) / 2),
                 "median_dev": float(np.median(dev[hh:])),
                 "max_dev": float(np.max(dev)),
                 "h_over_2tau": H / (2 * TAU)})
out["1_guiding_centre_test"] = rows
out["1_guiding_centre_test_note"] = (
    "sampling-shift mechanism predicts both rows identical at h/2tau; "
    "guiding-centre mechanism predicts the 2nd row collapses")

# ==================================================================== (2)
def staggered(h, tau, t_final, fB, ffac):
    n = int(round(t_final / h))
    r = np.array([1.0, 0.0, 0.0]); v0 = np.array([0.0, 1.0, 0.0])
    E = np.array([-ffac(0.0) * r[1], ffac(0.0) * r[0], 0.0])
    w = kick_boris(v0, E, fB(0.0), -0.5 * h)
    ts = np.zeros(n + 1); e_avg = np.zeros(n + 1); e_rot = np.zeros(n + 1)
    e_avg[0] = e_rot[0] = 1.0
    t = 0.0
    for i in range(1, n + 1):
        Bz = fB(t); fac = ffac(t)
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        wn = kick_boris(w, E, Bz, h)
        r = r + wn * h
        t += h
        e_avg[i] = np.sum((0.5 * (w + wn)) ** 2)
        e_rot[i] = np.sum(wn ** 2)          # rotation-recentring preserves |w|
        w = wn
        ts[i] = t
    return ts, e_avg, e_rot


ts, e_avg, e_rot = staggered(H, TAU, TF, fB, ffac)
hh = len(ts) // 2
d_avg = e_avg - fB(ts); d_rot = e_rot - fB(ts)
Om = fB(ts)
out["2_readout_structure"] = {
    "staggered_avg": {"median_abs": float(np.median(np.abs(d_avg[hh:]))),
                      "min": float(np.min(d_avg[hh:])), "max": float(np.max(d_avg[hh:])),
                      "pred_sin2_theta_h_half": float(np.median(
                          (np.sin(np.arctan(H * Om / 2)) ** 2 * Om)[hh:]))},
    "staggered_rot": {"median_abs": float(np.median(np.abs(d_rot[hh:]))),
                      "min": float(np.min(d_rot[hh:])), "max": float(np.max(d_rot[hh:])),
                      "envelope_over_median": float(np.max(np.abs(d_rot)) /
                                                    np.median(np.abs(d_rot[hh:]))),
                      "h_over_2tau": H / (2 * TAU)},
    "shipped": {"envelope_over_median": float(np.max(np.abs(
        run_generic(kick_boris, fB, ffac, H, TF, (1.0, 0, 0), (0, 1.0, 0))[1] - fB(ts))) /
        np.median(np.abs((run_generic(kick_boris, fB, ffac, H, TF, (1.0, 0, 0), (0, 1.0, 0))[1] - fB(ts))[hh:])))},
}

# ==================================================================== (3)
rows = []
def R_of(ts, sp2, fB):
    E0 = sp2[0]
    dev = np.abs(sp2 / E0 - fB(ts))
    hh = len(ts) // 2
    med = float(np.median(dev[hh:])); t_med = float(np.median(ts[hh:]))
    sig = float(1.0 - fB(np.array([t_med]))[0]) if np.ndim(fB(np.array([0.0]))) else float(1.0 - fB(t_med))
    return sig / med, med, t_med

for law, taus, beta in (("exp", (1.2e3, 1.2e4, 1.2e5, 1.2e6, 1.2e7, 480.0, 240.0, 120.0), 1.0),
                        ("pow", (1.2e5, 1.2e4, 240.0), 1.0),
                        ("pow", (1.2e5,), 3.0),
                        ("lin", (1.2e5, 1.2e4), 1.0),
                        ("cos", (1.2e5, 1.2e4), 1.0),
                        ("gauss", (1.2e5, 3000.0), 1.0)):
    for tv in taus:
        fb, ff = mk_field(law, tv, beta)
        ts, sp2, _ = run_generic(kick_boris, fb, ff, 0.3, TF, (1.0, 0, 0), (0, 1.0, 0))
        Rm, med, t_med = R_of(ts, sp2, fb)
        rows.append({"scheme": "boris_shipped", "law": law, "beta": beta, "tau": tv,
                     "h": 0.3, "R": Rm, "2t_med/h": 2 * t_med / 0.3,
                     "R/(2t/h)": Rm / (2 * t_med / 0.3), "median_dev": med})
# schemes
for name, fn in (("boris_shipped", lambda h: run_generic(kick_boris, fB, ffac, h, TF, (1.0, 0, 0), (0, 1.0, 0))),
                 ("boris_midpoint_drift", lambda h: run_generic(kick_boris, fB, ffac, h, TF, (1.0, 0, 0), (0, 1.0, 0), drift="mid")),
                 ("exactrot_shipped", lambda h: run_generic(kick_exactrot, fB, ffac, h, TF, (1.0, 0, 0), (0, 1.0, 0))),
                 ("rk4", lambda h: rk4(fB, ffac, h, TF, (1.0, 0, 0), (0, 1.0, 0)))):
    for h in (0.6, 0.3, 0.15, 0.05):
        ts, sp2, _ = fn(h)
        Rm, med, t_med = R_of(ts, sp2, fB)
        rows.append({"scheme": name, "law": "exp", "tau": TAU, "h": h,
                     "R": Rm, "2t_med/h": 2 * t_med / h,
                     "R/(2t/h)": Rm / (2 * t_med / h), "median_dev": med})
# horizon dependence at fixed everything
for tf in (60.0, 120.0, 600.0, 6000.0):
    ts, sp2, _ = run_generic(kick_boris, fB, ffac, 0.3, tf, (1.0, 0, 0), (0, 1.0, 0))
    Rm, med, t_med = R_of(ts, sp2, fB)
    rows.append({"scheme": "boris_shipped", "law": "exp", "tau": TAU, "h": 0.3,
                 "t_final": tf, "R": Rm, "2t_med/h": 2 * t_med / 0.3,
                 "R/(2t/h)": Rm / (2 * t_med / 0.3), "median_dev": med})
out["3_R_generality"] = rows

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt_t2_decisive.json"), "w"), indent=1)
