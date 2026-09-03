"""VT-T3: independent re-derivation and stress test of the 'trichotomy'.

Model (identical to 11_THEORY Lemma 3.1, re-derived here):
    z_{n+1} = e^{-i th_n} z_n + k_n,  th_n = 2 atan(h Om(t_n)/2)
    =>  z_N = e^{-i Phi_N} (z_0 + S_N),  Phi_N = sum_{k<N} th_k,
        S_N = sum_{k<N} e^{i Phi_{k+1}} k_k
    dev_N = |2 Re(conj(z0) S_N) + |S_N|^2| / |z0|^2.

Tests:
  A. verify Lemma 3.1 against brute-force iteration (machine precision)
  B. the claimed three regimes, with MY OWN envelope pipeline
  C. EXHAUSTIVENESS: is 'p in {0, 1/2, 1}' a trichotomy?  Counterexamples:
       C1 non-stationary incoherent amplitude sigma_n ~ n^a  -> p = a + 1/2
       C2 drive chirped to FOLLOW omega_h(t)                 -> p = 1, no stall
       C3 correlated (1/f-like) noise                        -> intermediate p
  D. resonance PROFILE: scan omega.  Is the dangerous set a measure-zero line
     or a band of width ~ alpha*T set by the chirp?  Includes the third
     v_caveat2 run (detune 2e-4) that theory_check never modelled.
  E. Fresnel stall time: where does the resonant run actually break?
  F. how many free parameters: refit kappa on each anchor separately.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
OM_H0 = 2 * np.arctan(H / 2) / H


def phases(n_gyr):
    N = int(round(n_gyr * TWO_PI / H))
    t_n = np.arange(N) * H
    th = 2 * np.arctan(H * np.exp(-t_n / TAU_Q) / 2)
    Phi = np.concatenate([[0.0], np.cumsum(th)])[:-1]
    return N, t_n, th, Phi


def envelope_exponent(ts, dev, n_samples=4000, gyros=(1e2, 1e3, 1e4)):
    stride = max(1, len(ts) // n_samples)
    idx = np.arange(stride - 1, len(ts), stride)
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / 100.0) & (env > 0)
    expo = float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])
    emax = {f"{g:.0e}": float(env[tw <= g * TWO_PI][-1]) for g in gyros
            if np.any(tw <= g * TWO_PI)}
    return expo, emax, tw, env


Z0 = 1j
out = {"omega_h0": OM_H0}

# ================================================== A. Lemma 3.1 exactness
N, t_n, th, Phi = phases(10.0)
rng = np.random.default_rng(1)
kk = 1e-3 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
z = Z0
brute = np.empty(N, complex)
for n in range(N):
    z = np.exp(-1j * th[n]) * z + kk[n]
    brute[n] = z
S = np.cumsum(np.exp(1j * (Phi + th)) * kk)         # Phi_{k+1} = Phi_k + th_k
closed = np.exp(-1j * (Phi + th)) * (Z0 + S)
out["A_lemma31_max_abs_error"] = float(np.max(np.abs(brute - closed)))

# ================================================== B/E. resonant run
def sin_drive(omega, kappa, n_gyr=1e4, chirped=False):
    N, t_n, th, Phi = phases(n_gyr)
    ph = (np.concatenate([[0.0], np.cumsum(th)])[:-1] if chirped
          else omega * (t_n + 0.5 * H))
    if chirped:                     # drive exactly at the running omega_h
        ph = Phi + 0.5 * th
    kick = kappa * np.sin(ph if chirped else omega * (t_n + 0.5 * H))
    S = np.cumsum(np.exp(1j * (Phi + th)) * kick)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    return t_n + H, dev


# calibrate kappa once on emax(1e2 gyro) of the resonant v_caveat2 run
TARGET_E2 = 7.180920873978147e-04
ts, dev = sin_drive(OM_H0, 1e-9)
i2 = int(round(1e2 * TWO_PI / H)) - 1
KAPPA = 1e-9 * TARGET_E2 / np.max(dev[:i2])
out["kappa_calibrated"] = float(KAPPA)

# independent cross-check: what velocity kick does v_caveat2's SinNet make?
a_mid = 0.5                       # Bz/2 at t~0
AMP = 3.494e-07
g = 0.5 * AMP                     # d(dL_d)/dq in both slots, x-component
den = 1.0 + (a_mid * H) ** 2
dD = np.array([g * H / den, g * a_mid * H * H / den])
dp1 = np.array([g + dD[0] / H, dD[1] / H])
dA = a_mid * np.array([-dD[1], dD[0]])
dv = dp1 + dA
out["kappa_from_dLd_amplitude"] = {
    "AMP_in_v_caveat2": AMP,
    "derived_per_step_velocity_kick_amplitude": float(np.hypot(*dv)),
    "conversion_factor_kick/AMP": float(np.hypot(*dv) / AMP),
    "kick_direction_deg_from_x": float(np.degrees(np.arctan2(dv[1], dv[0]))),
    "ratio_calibrated_kappa_over_derived": float(KAPPA / np.hypot(*dv)),
    "note": ("11_THEORY compares 3.472e-7 with 3.494e-7 as if the conversion "
             "were 1; the derived conversion is this factor"),
}

rows = []
for lbl, om in (("resonant omega_h", OM_H0),
                ("detune 2e-4 (v_caveat2 run 2, never modelled)", OM_H0 * (1 + 2e-4)),
                ("physical omega=1 (detune 0.74%)", 1.0),
                ("off-resonant 0.37", 0.37)):
    ts, dev = sin_drive(om, KAPPA)
    e, em, tw, env = envelope_exponent(ts, dev)
    rows.append({"case": lbl, "omega": om, "exponent": e, "emax": em})
out["B_regimes"] = rows
out["B_measured_v_caveat2"] = {
    "resonant": {"exponent": 0.5107, "emax": {"1e+02": 7.1809e-4, "1e+03": 7.1908e-3, "1e+04": 1.75326e-2}},
    "detune2e-4": {"exponent": 0.1643, "emax": {"1e+02": 7.1618e-4, "1e+03": 5.1752e-3, "1e+04": 5.2363e-3}},
    "omega=1": {"exponent": -1.8e-16, "emax": {"1e+04": 1.56191e-4}},
}

# E. where does the resonant envelope actually break?
ts, dev = sin_drive(OM_H0, KAPPA, n_gyr=3e4)
e, em, tw, env = envelope_exponent(ts, dev, gyros=(1e2, 1e3, 2e3, 4419, 1e4, 3e4))
alpha = 1.0 / ((1 + (H / 2) ** 2) * TAU_Q)
Tstar = np.sqrt(2 * np.pi / alpha)
# empirical break: last time the envelope is within 5% of the linear extrapolation
lin = env[np.argmax(tw > 1e2 * TWO_PI)] * tw / tw[np.argmax(tw > 1e2 * TWO_PI)]
ok = env > 0.9 * lin
t_break = float(tw[np.max(np.where(ok)[0])] / TWO_PI)
out["E_fresnel"] = {"alpha": float(alpha), "T_star_time": float(Tstar),
                    "T_star_gyro": float(Tstar / TWO_PI),
                    "empirical_break_gyro_(10%_below_linear)": t_break,
                    "emax": em,
                    "plateau_formula_(kappa/2h)sqrt(2pi/alpha)*2":
                        float(2 * (KAPPA / (2 * H)) * Tstar)}

# ================================================== C. exhaustiveness
crows = []
# C1: incoherent with a power-law amplitude envelope sigma_n ~ (n+1)^a
for a in (-0.25, 0.0, 0.25, 0.4):
    N, t_n, th, Phi = phases(1e4)
    rg = np.random.default_rng(7)
    sig = KAPPA * ((np.arange(N) + 1.0) / N) ** a
    kk = sig * (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
    S = np.cumsum(np.exp(1j * (Phi + th)) * kk)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    e, em, _, _ = envelope_exponent(t_n + H, dev)
    crows.append({"case": f"incoherent sigma_n ~ n^{a}", "a": a,
                  "predicted_p_a_plus_half": a + 0.5, "exponent": e})
# C2: drive chirped to follow omega_h(t) -> stays resonant forever
ts, dev = sin_drive(None, KAPPA, n_gyr=3e4, chirped=True)
e, em, _, _ = envelope_exponent(ts, dev, gyros=(1e2, 1e3, 1e4, 3e4))
crows.append({"case": "chirp-matched drive omega(t)=omega_h(t)",
              "exponent": e, "emax": em,
              "note": "no Fresnel stall: T3.5's 'chirp always stalls' is "
                      "conditional on a FIXED drive frequency"})
# C3: correlated noise with spectral weight piled near omega_h (AR(1) in the
#     co-rotating frame) -> intermediate exponent
N, t_n, th, Phi = phases(1e4)
rg = np.random.default_rng(11)
for rho in (0.9, 0.99, 0.999):
    w = rg.standard_normal(N) + 1j * rg.standard_normal(N)
    x = np.empty(N, complex); acc = 0
    for i in range(N):
        acc = rho * acc + w[i]; x[i] = acc
    x *= KAPPA * np.sqrt(1 - rho ** 2) / np.sqrt(2)
    kk = np.real(x * np.exp(-1j * (Phi + th)))       # co-rotating AR(1)
    S = np.cumsum(np.exp(1j * (Phi + th)) * kk)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    e, _, _, _ = envelope_exponent(t_n + H, dev)
    crows.append({"case": f"co-rotating AR(1) rho={rho}", "exponent": e})
out["C_exhaustiveness"] = crows

# ================================================== D. resonance profile
prof = []
for rel in np.concatenate([-np.logspace(-2, -5.5, 18), [0.0], np.logspace(-5.5, -2, 18)]):
    om = OM_H0 * (1 + rel)
    ts, dev = sin_drive(om, KAPPA)
    e, em, _, _ = envelope_exponent(ts, dev)
    prof.append({"rel_detune": float(rel), "emax_1e4": em["1e+04"], "exponent": e})
out["D_resonance_profile"] = prof
T_obs = 1e4 * TWO_PI
out["D_widths"] = {
    "naive_1/T_relative_width": float(np.pi / T_obs / OM_H0),
    "chirp_band_alpha*T_relative_width": float(alpha * T_obs / OM_H0),
    "note": ("the dangerous band is set by the chirp sweep alpha*T, which "
             "GROWS with the horizon; 'measure zero' is a statement about a "
             "fixed autonomous frequency, not about this configuration"),
}

# ================================================== F. free parameters
# refit kappa on each anchor separately: how consistent is one kappa?
anchors = {"emax(1e2)": (1e2, 7.1809e-4), "emax(1e3)": (1e3, 7.1908e-3),
           "emax(1e4)": (1e4, 1.75326e-2)}
fits = {}
ts, dev = sin_drive(OM_H0, 1e-9)
for k, (g, targ) in anchors.items():
    i = int(round(g * TWO_PI / H)) - 1
    fits[k] = float(1e-9 * targ / np.max(dev[:i]))
out["F_kappa_per_anchor"] = fits
out["F_note"] = ("a single-parameter model must give the SAME kappa from every "
                 "anchor; the spread is the model's real accuracy")

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt_t3_trichotomy.json"), "w"), indent=1)
