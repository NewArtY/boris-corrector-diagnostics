"""VT-T2a: what IS the energy 'floor' of the shipped Boris in B4?

Claim under test (11_THEORY T2a): the energy error of the shipped variant is
a pure h/2 sampling shift of a genuinely decaying energy, giving
  speed error   h/(4 tau)      (6.25e-7)
  energy median h/(2 tau)      (1.249e-6)
  energy envelope h/tau        (2.5e-6)

A pure sampling shift is a CONSTANT offset: it predicts
  dev(t) = (h/2)|dE/dt|/E0 = (h/2tau) e^{-t/tau}  for ALL t,
hence median == envelope == h/(2tau).  The claimed envelope h/tau is a
factor 2 ABOVE that.  Both cannot follow from the same mechanism.

This script (my own loop, no reuse of theory_check) resolves it:
  A. signed dev(t), its mean / oscillation amplitude / spectrum;
  B. direct test of the shift hypothesis: compare E_num(t) with
     E_phys(t - h/2) pointwise;
  C. scaling of mean and oscillation amplitude in h and tau separately;
  D. what the h/150 'reference' actually measures in the speed channel.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 2 * np.pi


def field_exp(r, t, tau):
    Bz = np.exp(-t / tau)
    fac = 0.5 * Bz / tau
    return np.array([-fac * r[1], fac * r[0], 0.0]), Bz


def kick(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    return np.array([vm[0] + vpy * sz + k * E[0],
                     vm[1] - vpx * sz + k * E[1], vm[2]])


def run(h, tau, t_final, r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0)):
    n = int(round(t_final / h))
    r = np.array(r0, float); v = np.array(v0, float)
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1); rr = np.zeros((n + 1, 3))
    sp2[0] = v @ v; rr[0] = r
    t = 0.0
    for i in range(1, n + 1):
        E, Bz = field_exp(r, t, tau)
        v = kick(v, E, Bz, h)
        r = r + v * h
        t += h
        ts[i] = t; sp2[i] = v @ v; rr[i] = r
    return ts, sp2, rr


out = {}
TAU, H, TF = 1.2e5, 0.3, 120.0

# ---------- A. signed deviation, mean and oscillation ----------------------
ts, sp2, rr = run(H, TAU, TF)
Eph = np.exp(-ts / TAU)                     # 2*E_phys (E0 = 0.5)
signed = (sp2 - Eph)                        # = 2*(E_num - E_phys) = dev with sign
dev = np.abs(signed)
half = len(ts) // 2
s2 = signed[half:]
out["A_signed"] = {
    "mean_signed_2nd_half": float(np.mean(s2)),
    "median_abs_2nd_half": float(np.median(dev[half:])),
    "max_abs": float(np.max(dev)),
    "min_signed": float(np.min(signed)), "max_signed": float(np.max(signed)),
    "h_over_4tau": H / (4 * TAU), "h_over_2tau": H / (2 * TAU),
    "h_over_tau": H / TAU,
    "note": "signed dev in units of E0; sign tells whether E_num > E_phys",
}

# oscillation: fit signed = c0 + a*cos(w t + ph) by FFT over 2nd half
seg = signed[half:] - np.mean(signed[half:])
F = np.fft.rfft(seg * np.hanning(len(seg)))
frq = np.fft.rfftfreq(len(seg), d=H) * TWO_PI      # angular
k = int(np.argmax(np.abs(F[1:])) + 1)
om_h = 2 * np.arctan(H / 2) / H
out["A_spectrum"] = {
    "dominant_angular_freq": float(frq[k]),
    "omega_h": float(om_h), "Omega_phys": 1.0,
    "2*omega_h_aliased": float(abs(((2 * om_h) + np.pi / H) % (TWO_PI / H) - np.pi / H)),
    "osc_amp_(max-min)/2": float((np.max(s2) - np.min(s2)) / 2),
    "mean_level": float(np.mean(s2)),
}

# ---------- B. direct test of the h/2 sampling-shift hypothesis -------------
# hypothesis: sp2(t) == exp(-(t - h/2)/tau)  =>  residual should vanish
resid_shift = sp2 - np.exp(-(ts - 0.5 * H) / TAU)
out["B_shift_hypothesis"] = {
    "median_abs_residual_after_h/2_shift": float(np.median(np.abs(resid_shift[half:]))),
    "median_abs_dev_no_shift": float(np.median(dev[half:])),
    "improvement_factor": float(np.median(dev[half:]) /
                                np.median(np.abs(resid_shift[half:]))),
    "verdict": "shift explains the error only if improvement >> 1",
}
# best-fit shift delta: minimise median |sp2 - exp(-(t-delta)/tau)|
ds = np.linspace(-2 * H, 2 * H, 4001)
score = [np.median(np.abs(sp2[half:] - np.exp(-(ts[half:] - d) / TAU))) for d in ds]
out["B_shift_hypothesis"]["best_fit_delta"] = float(ds[int(np.argmin(score))])
out["B_shift_hypothesis"]["best_fit_delta_over_h"] = float(ds[int(np.argmin(score))] / H)
out["B_shift_hypothesis"]["best_fit_residual"] = float(np.min(score))

# ---------- C. scaling of mean level and oscillation amplitude -------------
rows = []
for h in (0.6, 0.3, 0.15, 0.075, 0.0375):
    for tau in (1.2e4, 1.2e5, 1.2e6):
        t_, s_, _ = run(h, tau, TF)
        sg = s_ - np.exp(-t_ / tau)
        hh = len(t_) // 2
        seg = sg[hh:]
        rows.append({"h": h, "tau": tau,
                     "mean": float(np.mean(seg)),
                     "median_abs": float(np.median(np.abs(seg))),
                     "amp": float((np.max(seg) - np.min(seg)) / 2),
                     "max_abs": float(np.max(np.abs(sg))),
                     "mean/(h/4tau)": float(np.mean(seg) / (h / (4 * tau))),
                     "median/(h/2tau)": float(np.median(np.abs(seg)) / (h / (2 * tau))),
                     "amp/(h/2tau)": float((np.max(seg) - np.min(seg)) / 2 / (h / (2 * tau))),
                     "maxabs/(h/tau)": float(np.max(np.abs(sg)) / (h / tau))})
out["C_scaling"] = rows

# ---------- D. the speed channel against the h/150 reference ---------------
n_fine = int(round(TF / (H / 150)))
tf, spf, rf = run(H / 150, TAU, TF)
sp_w = np.sqrt(sp2); sp_r = np.sqrt(spf[::150])
d_signed = (sp_w - sp_r) / sp_r
out["D_speed_channel"] = {
    "median_abs": float(np.median(np.abs(d_signed[half:]))),
    "mean_signed": float(np.mean(d_signed[half:])),
    "osc_amp": float((np.max(d_signed[half:]) - np.min(d_signed[half:])) / 2),
    "h_over_4tau": H / (4 * TAU),
    "h_over_8tau": H / (8 * TAU),
    "ref_own_floor_hfine_over_4tau": (H / 150) / (4 * TAU),
}

# ---------- E. does the envelope really saturate?  long run ----------------
for ng in (1e2, 1e3, 1e4):
    t_, s_, _ = run(H, TAU, ng * TWO_PI)
    d_ = np.abs(s_ - np.exp(-t_ / TAU))
    out.setdefault("E_envelope", {})[f"{ng:.0e}gyro_max"] = float(np.max(d_))
out["E_envelope"]["h_over_tau"] = H / TAU

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt_t2a_mechanism.json"), "w"), indent=1)
