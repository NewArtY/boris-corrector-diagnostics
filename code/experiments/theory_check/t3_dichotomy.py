"""t3: the secular dichotomy on the exactly-solvable kicked-rotation model.

Model: planar velocity z in C,  z_{n+1} = e^{-i th_h(t_n)} z_n + kick_n,
th_h(t) = 2 atan(h Om(t)/2), Om(t) = exp(-t/tau_q), tau_q = 1.2e8 (the
quasistatic B4 of v_caveat2).  Solvable:  z_n = e^{-i Phi_n}(z_0 + S_n),
S_n = sum_{k<n} e^{+i Phi_k} kick_k.  Energy error = |z_n|^2 - 1.

Claims under test against experiments/verify_f0/v_caveat2.json (nonlinear
variational integrator + symplectic in-Lagrangian defect, same amplitude):

  D1 resonance condition is the NUMERICAL frequency: drive at omega_h(0)
     grows linearly until the chirp-decoherence time, envelope-fit exponent
     ~0.51 over [1e2,1e4] gyro; emax(1e4)/emax(1e3) ~ 2.4.
     (measured there: 0.511, 7.19e-3 -> 1.75e-2, ratio 2.44)
  D2 drive at the PHYSICAL frequency omega=1 (0.74% detuned): bounded,
     plateau = 2*rate/|domega|-level, reached before 1e2 gyro; exponent ~0.
     (measured: plateau 1.56e-4, exponent 0)
  D3 far off-resonance omega=0.37: bounded at ~1e-6 level. (measured 2.7e-6)
  D4 aligned kick dv||v: exponent 1 exactly, slope 2*eps per step.
     (measured: 1.013, slope 4.37e-7 per step at eps=2.2e-7)
  D5 iid random kicks with the measured self-extinguishing amplitude decay
     (~1/t): bounded => exponent ~0.  With constant amplitude: exponent ~0.5.

Kick amplitude for D1-D3 calibrated once to the measured emax(1e2 gyro) of
the resonant run; everything else (shape, break, exponents) is prediction.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
N_GYR = 1e4
N = int(round(N_GYR * TWO_PI / H))


def envelope_exponent(ts, dev, n_samples=4000):
    """Exact methodology of horizon/long_runs.py / v_caveat2.py."""
    stride = max(1, len(ts) // n_samples)
    idx = np.arange(stride - 1, len(ts), stride)
    # windowed running max
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / 100.0) & (env > 0)
    expo = float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])
    emax = {f"{g:.0e}": float(env[tw <= g * TWO_PI][-1]) for g in (1e2, 1e3, 1e4)}
    return expo, emax


t_n = np.arange(N) * H
Om = np.exp(-t_n / TAU_Q)
th = 2 * np.arctan(H * Om / 2)
Phi = np.concatenate([[0.0], np.cumsum(th)])[:-1]   # Phi_k before step k
ts_out = t_n + H

om_h0 = 2 * np.arctan(H / 2) / H

out = {"omega_h0": om_h0, "runs": []}


Z0 = 1j                                               # v0 = (0,1,0) of all runs


def sin_drive(omega, kappa):
    kick = kappa * np.sin(omega * (t_n + 0.5 * H))   # real kick along x
    S = np.cumsum(np.exp(1j * Phi) * kick)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)          # |dE|/E0 (|z0| = 1)
    return dev


# --- calibrate kappa on emax(1e2) of the resonant v_caveat2 run -------------
# two-step: probe in the linear regime, then scale linearly
target_e2 = 7.180920873978147e-04
kappa_probe = 1e-9
dev = sin_drive(om_h0, kappa_probe)
i2 = int(round(1e2 * TWO_PI / H)) - 1
kappa = kappa_probe * target_e2 / np.max(dev[:i2])
out["kappa_calibrated"] = float(kappa)
out["kappa_per_step_rms"] = float(kappa / np.sqrt(2))

for label, omega, ref in (
        ("D1 resonant omega_h", om_h0,
         {"exponent": 0.511, "emax": {"1e+02": 7.18e-4, "1e+03": 7.19e-3, "1e+04": 1.75e-2}}),
        ("D2 physical omega=1", 1.0,
         {"exponent": 0.0, "emax": {"1e+04": 1.56e-4}}),
        ("D3 off-resonant omega=0.37", 0.37,
         {"exponent": 0.0, "emax": {"1e+04": 2.7e-6}})):
    dev = sin_drive(omega, kappa)
    expo, emax = envelope_exponent(ts_out, dev)
    out["runs"].append({"label": label, "exponent_model": expo, "emax_model": emax,
                        "v_caveat2_measured": ref})

# analytic cap for D2: linear growth rate r = kappa/2 per step (RWA);
# plateau ~ 2*(kappa/2)/|delta_omega| / h ... report both
domega = 1.0 - om_h0
out["D2_analytic_cap"] = {"2*rate/domega": float(2 * (kappa / 2) / (domega * H) * 2),
                          "note": "order-of-magnitude; exact value phase-dependent"}
# Fresnel decoherence time of the resonant run: chirp rate
# alpha = |d omega_h/dt| = Om/( (1+(h Om/2)^2) tau_q )
alpha = 1.0 / ((1 + (H / 2) ** 2) * TAU_Q)
out["D1_fresnel_stall_gyro"] = float(np.sqrt(2 * np.pi / alpha) / TWO_PI)

# --- D4: aligned multiplicative kick ---------------------------------------
eps = 2.2e-7
n4 = int(round(1e4 * TWO_PI / H))
tt = (np.arange(n4) + 1) * H
dev4 = (1 + eps) ** (2 * (np.arange(n4) + 1)) - 1.0
expo4, emax4 = envelope_exponent(tt, dev4)
out["D4_aligned"] = {"exponent_model": expo4, "emax_model": emax4,
                     "slope_per_step_model": float(dev4[100] / 101),
                     "v_caveat_measured": {"exponent": 1.013,
                                           "emax": {"1e+02": 9.156e-4, "1e+03": 9.240e-3,
                                                    "1e+04": 9.647e-2},
                                           "slope_per_step": 4.37e-7}}

# --- D5: random kicks -------------------------------------------------------
rng = np.random.default_rng(0)
res5 = {}
for mode in ("const", "decay"):
    amp = np.full(N, kappa)
    if mode == "decay":
        # measured gradient decay of the frozen net (v_caveat.json): ~1/t
        amp = kappa * np.minimum(1.0, (1000.0 / np.maximum(t_n, 1.0)) ** 1.1)
    kicks = amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)
    S = np.cumsum(np.exp(1j * Phi) * kicks)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    e, em = envelope_exponent(ts_out, dev)
    res5[mode] = {"exponent": e, "emax": em}
out["D5_random"] = res5

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "t3_dichotomy.json"), "w"), indent=1)
