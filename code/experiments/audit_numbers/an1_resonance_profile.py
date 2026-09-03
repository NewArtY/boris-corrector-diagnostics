"""AN1: fine rescan of the resonance profile of Sec. 4.5.

Reuses the exact model and calibration of
`verify_theory/vt_t3_trichotomy.py` (sections B and D) and refines the
frequency grid, so that the peak height, the peak location and the FWHM
can be read off directly instead of off an 18-decade-log grid.

Question under audit: the manuscript prints a peak of 3.84e-2 at a
relative detuning of -2.5e-4 and a width of 4.6e-4 at 1e4 gyro-orbits.
The shipped scan (37 points) gives 3.27e-2 at -8.7e-5 and a chirp band
alpha*T/omega_h = 5.16e-4.

Outcome (W5.1, W5.3).  All four numbers of the manuscript are confirmed:
the coarse scan has no node between -2.25e-4 and -3.62e-4, and the true
peak sits at -2.476e-4, right inside that hole.  The one correction the
audit forced was terminological: 4.6e-4 is the FULL width at half
maximum (`fwhm_rel` below), not the half-width, which is half of it, and
the manuscript now says "full width at half maximum".  The comparison
with the Fourier-limited 5e-5 survives, because that too is an FWHM:
for a sinc response of duration T it is 0.886*pi/T = 4.46e-5.

This is the only source in the bundle for those four numbers; without it
Sections 4.5 and 8 are not reproducible.  It takes about a minute.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
OM_H0 = 2 * np.arctan(H / 2) / H
Z0 = 1j


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


N_MAX, T_N, TH, PHI = phases(3e4)
EXP_FACTOR = np.exp(1j * (PHI + TH))


def sin_drive(omega, kappa, n_gyr=1e4):
    N = int(round(n_gyr * TWO_PI / H))
    t_n, th, Phi = T_N[:N], TH[:N], PHI[:N]
    kick = kappa * np.sin(omega * (t_n + 0.5 * H))
    S = np.cumsum(EXP_FACTOR[:N] * kick)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    return t_n + H, dev


# calibration, identical to vt_t3_trichotomy.py
TARGET_E2 = 7.180920873978147e-04
ts, dev = sin_drive(OM_H0, 1e-9)
i2 = int(round(1e2 * TWO_PI / H)) - 1
KAPPA = 1e-9 * TARGET_E2 / np.max(dev[:i2])

out = {"omega_h0": OM_H0, "kappa_calibrated": float(KAPPA)}

alpha = 1.0 / ((1 + (H / 2) ** 2) * TAU_Q)

for horizon in (1e3, 3e3, 1e4, 3e4):
    # linear grid dense enough to resolve a band of width alpha*T/omega
    band = alpha * horizon * TWO_PI / OM_H0
    grid = np.linspace(-6.0 * band, 3.0 * band, 901)
    rows = []
    for rel in grid:
        om = OM_H0 * (1 + rel)
        ts, dv = sin_drive(om, KAPPA, n_gyr=horizon)
        rows.append((float(rel), float(np.max(dv))))
    rel = np.array([r[0] for r in rows])
    em = np.array([r[1] for r in rows])
    j = int(np.argmax(em))
    half = em[j] / 2.0
    above = np.where(em >= half)[0]
    fwhm = float(rel[above[-1]] - rel[above[0]])
    # value exactly on omega_h(0)
    _, dv0 = sin_drive(OM_H0, KAPPA, n_gyr=horizon)
    out[f"T={horizon:.0e}gyr"] = {
        "peak_emax": float(em[j]),
        "peak_rel_detune": float(rel[j]),
        "emax_on_omega_h0": float(np.max(dv0)),
        "fwhm_rel": fwhm,
        "half_width_rel": fwhm / 2.0,
        "pi_over_T_omega": float(np.pi / (horizon * TWO_PI) / OM_H0),
        "alpha_T_over_omega": float(alpha * horizon * TWO_PI / OM_H0),
        "grid_step": float(grid[1] - grid[0]),
    }
    print(horizon, out[f"T={horizon:.0e}gyr"])

json.dump(out, open(os.path.join(HERE, "an1_resonance_profile.json"), "w"),
          indent=1)
