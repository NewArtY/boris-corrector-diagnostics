"""t1: quantitative predictions for the shipped Boris scheme in B4.

Predictions under test (zero fit parameters, tau = 1.2e5, h = 0.3, Omega0 = 1):

  P1 (phase drift):     theta(t) = int_0^t [Omega(s) - omega_h(Omega(s))] ds,
                        omega_h(Om) = (2/h) atan(h Om / 2).
                        At t = 120: 50.9 deg;  median over [60,120]: 38.2 deg.
                        (article measured: 38.11 deg median, 50.8 deg end)
  P2 (speed error):     d|v|/|v| = (h/4) / tau = 6.25e-7  (half-step sampling
                        offset of |v| ~ exp(-t/2tau)).  measured: 6.23e-7
  P3 (energy error):    dE/E0 = (h/2tau) exp(-t/tau) -> median 1.249e-6.
                        measured: 1.246e-6
  P4 (the '600x'):      signal/error = (1-e^{-t_med/tau}) / (h/(2tau) e^{-t_med/tau})
                        ~= 2 t_med / h = 600.  measured: 601.95
  P5 (position error):  |dr(t)| ~= 2 r_L sin(theta(t)/2)  -> rms over run.
                        measured: 0.417 r_L

Reference = same Boris at h/150 (article methodology, offset floor 150x lower).
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 1.2e5
H = 0.3
T_FINAL = 120.0
TWO_PI = 2 * np.pi


def boris_run(h, n_steps, tau=TAU):
    """Shipped Boris variant (r += v_{n+1} h), scalar, B4 field."""
    rx, ry, rz = 1.0, 0.0, 0.0
    vx, vy, vz = 0.0, 1.0, 0.0
    t = 0.0
    k = -0.5 * h
    inv_tau = 1.0 / tau
    rs = np.empty((n_steps + 1, 3)); vs = np.empty((n_steps + 1, 3))
    rs[0] = (rx, ry, rz); vs[0] = (vx, vy, vz)
    for i in range(1, n_steps + 1):
        Bz = np.exp(-t * inv_tau)
        fac = 0.5 * Bz * inv_tau
        Ex = -fac * ry; Ey = fac * rx
        kEx = k * Ex; kEy = k * Ey
        vmx = vx + kEx; vmy = vy + kEy
        tz = k * Bz
        sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vmx + vmy * tz; vpy = vmy - vmx * tz
        vx = vmx + vpy * sz + kEx; vy = vmy - vpx * sz + kEy
        rx += vx * h; ry += vy * h; rz += vz * h
        t += h
        rs[i] = (rx, ry, rz); vs[i] = (vx, vy, vz)
    return rs, vs


n_work = int(round(T_FINAL / H))            # 400 steps
n_fine = n_work * 150
rs_w, vs_w = boris_run(H, n_work)
rs_f, vs_f = boris_run(H / 150, n_fine)
rs_r = rs_f[::150]; vs_r = vs_f[::150]      # matched integer times
ts = np.arange(n_work + 1) * H

# measured channels
sp_w = np.linalg.norm(vs_w, axis=1); sp_r = np.linalg.norm(vs_r, axis=1)
dspeed = np.abs(sp_w - sp_r) / sp_r
cosang = np.sum(vs_w * vs_r, axis=1) / (sp_w * sp_r)
theta = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
E0 = 0.5
dE = np.abs(0.5 * sp_w**2 - 0.5 * sp_r**2) / E0
pos_err = np.linalg.norm(rs_w - rs_r, axis=1)
half = len(ts) // 2

# predictions
om = np.exp(-ts / TAU)
drift_rate = om - (2 / H) * np.arctan(H * om / 2)
theta_pred = np.degrees(np.concatenate([[0], np.cumsum(0.5 * (drift_rate[1:] + drift_rate[:-1]) * H)]))
dE_pred = (H / (2 * TAU)) * np.exp(-ts / TAU)
dsp_pred = (H / (4 * TAU)) * np.ones_like(ts)
pos_pred = 2.0 * np.abs(np.sin(np.radians(theta_pred) / 2))
signal = 1 - np.exp(-90.0 / TAU)

out = {
    "theta_end_deg": {"measured": float(theta[-1]), "predicted": float(theta_pred[-1])},
    "theta_median_2nd_half_deg": {"measured": float(np.median(theta[half:])),
                                  "predicted": float(np.median(theta_pred[half:])),
                                  "article": 38.11},
    "speed_err_median": {"measured": float(np.median(dspeed[half:])),
                         "predicted": float(np.median(dsp_pred[half:])),
                         "article": 6.23e-7},
    "energy_err_median": {"measured": float(np.median(dE[half:])),
                          "predicted": float(np.median(dE_pred[half:])),
                          "article": 1.246e-6},
    "signal_over_error": {"measured": float(signal / np.median(dE[half:])),
                          "predicted_180_over_h": 180.0 / H,
                          "article": 601.95},
    "pos_err_rms": {"measured": float(np.sqrt(np.mean(pos_err**2))),
                    "predicted_phase_only": float(np.sqrt(np.mean(pos_pred**2))),
                    "article": 0.417},
    "theta_curve_ratio_meas_over_pred_at_t":
        {f"{tq:.0f}": float(theta[int(tq/H)] / max(theta_pred[int(tq/H)], 1e-30))
         for tq in (30, 60, 90, 120)},
}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "t1_boris_channels.json"), "w"), indent=1)
