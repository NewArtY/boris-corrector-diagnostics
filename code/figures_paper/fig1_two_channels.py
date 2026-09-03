"""Figure 1 -- the two error channels of one Boris run.  Cited in Section 2.

WHAT IT SHOWS
    (a) The direction channel.  Angle theta(t) between the Boris velocity and
        the reference velocity, in degrees, against gyroperiods.  Overlaid:
        the closed form

            theta(t) = \\int_0^t [ Omega(s) - (2/h) atan(h Omega(s)/2) ] ds,

        with no fitted parameter.
    (b) The two channels of the same run on one logarithmic axis: relative
        velocity error ||dv||/||v_ref|| (direction-dominated), relative speed
        error d|v|/||v_ref|| and the relative energy error |dE|/E_0 -- the
        quantity a practitioner actually monitors.  Almost six decades
        separate what is wrong from what is measured.

INPUT
    ../../code/experiments/verify_final/vf1_magnetic.json
        Committed output of the final verifier.  Its T1_T7 block is the
        source of the numbers quoted in plan/11_THEORY.md (I1.4).  Every
        scalar drawn or annotated here is asserted equal to that file
        (rtol 1e-9) before anything is plotted; the script aborts otherwise.

    The per-step time series is not committed -- only the summary is -- so the
    script reproduces the run.  The Boris recursion and the reference below
    are a verbatim copy of ``boris_run`` and ``reference`` from
    ``experiments/verify_final/vf1_magnetic.py``, called exactly as its T1/T7
    block calls them; both are deterministic, the whole run takes under a
    second, and the assertions are what prove the copy faithful.

    Setup: Boris, q = -1, m = 1, B = B0 exp(-t/tau) zhat,
    E = (Bz/2 tau)(-y, x, 0), h = 0.3, tau = 1.2e5, r0 = (1,0,0),
    v0 = (0,1,0), t_final = 120 (19.1 gyroperiods).

    Reference: DOP853 on the same ODE at rtol 1e-13, atol 1e-15 -- an
    integrator of a different family, not the scheme under test at a finer
    step.  Its own energy departs from the analytic |v|^2 = exp(-t/tau) by
    less than 1e-9, six decades below the error being measured; the script
    checks that too.

OUTPUT
    fig1_two_channels.pdf          -- the figure
    fig1_two_channels_values.json  -- every number drawn or annotated
"""
#
# ---------------------------------------------------------------------
# BUNDLE COPY.  This file is byte-for-byte the manuscript's figure script
# except for the one line that locates the data: in the manuscript tree
# the scripts sit in article/figures/ and reach the experiments through
# ../../code/bundle/code/experiments/, here they sit next to them and
# reach them through ../experiments/.  repro.py --check-sync verifies
# that this is still the only difference.
# ---------------------------------------------------------------------


import json
import os

import numpy as np
from scipy.integrate import solve_ivp

import paper_style as ps
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, os.pardir, "experiments",
                       "verify_final", "vf1_magnetic.json")

TAU, H, TF, B0 = 1.2e5, 0.3, 120.0, 1.0
Q = -1.0
TWO_PI = 2.0 * np.pi


def Bz(t):
    return B0 * np.exp(-t / TAU)


def fac(t):
    return 0.5 * Bz(t) / TAU          # E = fac * (zhat x r)


# ---------------------------------------------------------------------------
# verbatim from experiments/verify_final/vf1_magnetic.py
# ---------------------------------------------------------------------------
def boris_run(r0, v0, h=H, tf=TF):
    n = int(round(tf / h))
    r = np.array(r0, float)
    v = np.array(v0, float)
    rs = np.zeros((n + 1, 3))
    vs = np.zeros((n + 1, 3))
    ts = np.zeros(n + 1)
    rs[0], vs[0] = r, v
    t = 0.0
    for i in range(1, n + 1):
        f = fac(t)
        E = np.array([-f * r[1], f * r[0], 0.0])
        k = 0.5 * Q * h
        vm = v + k * E
        tz = k * Bz(t)
        sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vm[0] + vm[1] * tz
        vpy = vm[1] - vm[0] * tz
        vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])
        v = vp + k * E
        r = r + v * h
        t += h
        rs[i], vs[i], ts[i] = r, v, t
    return ts, rs, vs


def reference(r0, v0, t_eval, rtol=1e-12, atol=1e-14):
    def f(t, y):
        r, v = y[:3], y[3:]
        fc = fac(t)
        E = np.array([-fc * r[1], fc * r[0], 0.0])
        Bv = np.array([0.0, 0.0, Bz(t)])
        return np.concatenate([v, Q * (E + np.cross(v, Bv))])
    sol = solve_ivp(f, (0.0, t_eval[-1]), np.concatenate([r0, v0]),
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[:3].T, sol.y[3:].T


def block_median(x, n_per_block):
    """Median of |x| inside consecutive windows of one gyroperiod.

    The magnitude and energy channels oscillate at the gyrofrequency between
    zero and twice their mean, so the raw series is a hairy band on a log
    axis.  The per-gyroperiod median is the same statistic the committed
    summary reports (median over the second half), so the plateaux in panel
    (b) can be read against the numbers quoted in the text.
    """
    n = (len(x) // n_per_block) * n_per_block
    return np.median(np.abs(x[:n]).reshape(-1, n_per_block), axis=1)


# ---------------------------------------------------------------------------
# reproduce the T1/T7 run of vf1_magnetic.py
# ---------------------------------------------------------------------------
r0 = np.array([1.0, 0.0, 0.0])
v0 = np.array([0.0, 1.0, 0.0])
ts, rs, vs = boris_run(r0, v0)
rr, vr = reference(r0, v0, ts, rtol=1e-13, atol=1e-15)
gyro = ts / TWO_PI

sp = np.linalg.norm(vs, axis=1)
spr = np.linalg.norm(vr, axis=1)
cth = np.clip(np.sum(vs * vr, axis=1) / (sp * spr), -1.0, 1.0)
theta = np.degrees(np.arccos(cth))
half = len(ts) // 2

d_speed = np.abs(sp - spr) / spr                       # magnitude channel
d_energy = np.abs(sp ** 2 - spr ** 2) / (spr[0] ** 2)  # energy diagnostic
d_vel = np.linalg.norm(vs - vr, axis=1) / spr          # both channels
pos_err = np.linalg.norm(rs - rr, axis=1)

# closed form, zero fitted parameters (vf1's quadrature)
tt = np.linspace(0.0, TF, 20001)
Om = Bz(tt)
drift_rate = Om - (2.0 / H) * np.arctan(H * Om / 2.0)
cum = np.concatenate([[0.0], np.cumsum(0.5 * (drift_rate[1:] + drift_rate[:-1])
                                       * np.diff(tt))])
theta_pred = np.degrees(np.interp(ts, tt, cum))

# polar identity, T1.1:  ||dv||^2 = (d|v|)^2 + 2 |v||v_ref| (1 - cos theta)
dv = vs - vr
lhs = np.sum(dv * dv, axis=1)
rhs = (sp - spr) ** 2 + 2.0 * sp * spr * (1.0 - cth)
identity_residual = float(np.max(np.abs(lhs - rhs)))
ref_energy_drift = float(np.max(np.abs(spr ** 2 - np.exp(-ts / TAU))))

theta_end = float(theta[-1])
theta_end_pred = float(theta_pred[-1])
theta_med = float(np.median(theta[half:]))
theta_med_pred = float(np.median(theta_pred[half:]))
speed_med = float(np.median(d_speed[half:]))
energy_med = float(np.median(d_energy[half:]))
vel_end = float(d_vel[-1])
pos_rms = float(np.sqrt(np.mean(pos_err ** 2)))
dev_end_pct = abs(theta_end / theta_end_pred - 1.0) * 100.0
dev_med_pct = abs(theta_med / theta_med_pred - 1.0) * 100.0

# closed forms for the magnitude channel (T1 predictions, no fit)
speed_closed = H / (4.0 * TAU)                       # 6.25e-7
energy_closed = float(np.median((H / (2.0 * TAU)) * np.exp(-ts[half:] / TAU)))

# ---------------------------------------------------------------------------
# self-check against the committed summary -- nothing is drawn before this
# ---------------------------------------------------------------------------
ref = json.load(open(SUMMARY, encoding="utf-8"))
t17 = ref["T1_T7"]
checks = [
    ("theta_median_2nd_half", theta_med, t17["theta_median_2nd_half_deg"]),
    ("theta_median_closed_form", theta_med_pred,
     t17["theta_pred_median_2nd_half_deg"]),
    ("theta_final", theta_end, t17["theta_final_deg"]),
    ("theta_final_closed_form", theta_end_pred, t17["theta_pred_final_deg"]),
    ("ratio_meas_over_pred_final", theta_end / theta_end_pred,
     t17["ratio_meas_over_pred_final"]),
    ("polar_identity_residual", identity_residual,
     t17["polar_identity_max_resid"]),
    # vf1 reports pos_err_rms in its P1.1 block, where the reference runs at
    # the looser rtol 1e-12; the two agree to eleven digits, not to fifteen.
    ("pos_err_rms", pos_rms,
     ref["P1_1_r0_shift"]["base r0=(1,0,0)"]["pos_err_rms"]),
]
for name, got, want in checks:
    assert np.isclose(got, want, rtol=1e-9, atol=0.0), \
        f"{name}: reproduced {got!r} != committed {want!r}"
assert identity_residual < 6e-16, identity_residual
assert ref_energy_drift < 1e-9, ref_energy_drift
# the magnitude channel sits on its closed form h/(4 tau) to better than 1 %
assert abs(speed_med / speed_closed - 1.0) < 0.01, (speed_med, speed_closed)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
ps.use_style()
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(ps.TEXTWIDTH_IN, 2.42),
    gridspec_kw=dict(wspace=0.36, left=0.085, right=0.995,
                     bottom=0.175, top=0.94))

# ---- (a) direction channel ------------------------------------------------
axA.plot(gyro, theta_pred, color=ps.VERMILLION, lw=2.4, alpha=0.55,
         solid_capstyle="round", zorder=1, label="closed form, no fit")
axA.plot(gyro[::12], theta[::12], color=ps.BLACK, lw=0.0, marker="o",
         ms=2.8, mfc="none", mew=0.85, zorder=3, label="measured")
axA.plot(gyro, theta, color=ps.BLACK, lw=0.9, ls="-", zorder=2)

g_half = gyro[half]
axA.plot([g_half, gyro[-1]], [theta_med, theta_med], color=ps.BLACK,
         lw=0.7, ls=(0, (1, 1.6)), zorder=4)
axA.annotate("median over the\n"
             f"second half, ${theta_med:.2f}^{{\\circ}}$",
             xy=(18.2, theta_med), xytext=(18.9, 20.0),
             fontsize=7.0, ha="right", va="center", color=ps.BLACK,
             arrowprops=dict(arrowstyle="-", lw=0.6, color=ps.BLACK,
                             shrinkA=2, shrinkB=2))
axA.annotate(f"${theta_end:.2f}^{{\\circ}}$",
             xy=(gyro[-1], theta_end), xytext=(-31.0, 7.0),
             textcoords="offset points", fontsize=7.4, ha="left",
             va="center", color=ps.BLACK)
axA.text(0.04, 0.985, "closed form vs. measured:\n"
         f"{max(dev_end_pct, dev_med_pct):.3f} % over the run",
         transform=axA.transAxes, fontsize=7.0, va="top", ha="left",
         color=ps.VERMILLION)

axA.set_xlabel("gyroperiods")
axA.set_ylabel(r"direction error $\theta$  (deg)")
axA.set_xlim(0.0, gyro[-1] * 1.02)
axA.set_ylim(0.0, 62.0)
axA.set_yticks([0, 10, 20, 30, 40, 50])
axA.legend(loc="upper left", bbox_to_anchor=(-0.012, 0.830))
ps.panel_label(axA, "(a)")

# ---- (b) the two channels side by side ------------------------------------
n_per = int(round(TWO_PI / H))                       # 21 samples per gyroperiod
med_gyro = np.arange(len(ts) // n_per) + 1.0
med_vel = block_median(d_vel, n_per)
med_speed = block_median(d_speed, n_per)
med_energy = block_median(d_energy, n_per)
gap = float(med_vel[-1] / med_energy[-1])

axB.plot(med_gyro, med_vel, color=ps.BLACK, ls="-", lw=1.3, marker="o",
         ms=2.6, mfc="none", mew=0.8, markevery=2,
         label=r"$\|\Delta\mathbf{v}\|\,/\,\|\mathbf{v}\|$  (both channels)")
axB.plot(med_gyro, med_speed, color=ps.BLUE, ls=(0, (4.5, 1.8)), lw=1.3,
         marker="s", ms=2.6, mfc="none", mew=0.8, markevery=2,
         label=r"$\Delta|\mathbf{v}|\,/\,\|\mathbf{v}\|$  (magnitude only)")
axB.plot(med_gyro, med_energy, color=ps.VERMILLION, ls=(0, (1.2, 1.4)),
         lw=1.5, marker="^", ms=3.0, mfc="none", mew=0.8, markevery=(1, 2),
         label=r"$|\Delta E|/E_0$  (energy diagnostic)")

axB.set_yscale("log")
axB.set_xlim(0.0, gyro[-1] * 1.02)
axB.set_ylim(7e-8, 4.0)
axB.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
axB.yaxis.set_minor_locator(LogLocator(base=10.0,
                                       subs=tuple(np.arange(2, 10) * 0.1),
                                       numticks=60))
axB.yaxis.set_minor_formatter(NullFormatter())
axB.set_xlabel("gyroperiods")
axB.set_ylabel("relative error")

x_gap = med_gyro[-1]
axB.annotate("", xy=(x_gap, med_vel[-1]), xytext=(x_gap, med_energy[-1]),
             arrowprops=dict(arrowstyle="<->", lw=0.8, color=ps.GREY,
                             shrinkA=0, shrinkB=0))
axB.text(x_gap - 0.55, 1.2e-3, f"${gap / 1e5:.1f}\\times10^{{5}}$",
         fontsize=7.2, rotation=90, ha="center", va="center", color=ps.GREY)
axB.text(0.5, 2.4e-7, f"${speed_med * 1e7:.2f}\\times10^{{-7}}$",
         fontsize=7.0, ha="left", va="center", color=ps.BLUE)
axB.legend(loc="center left", bbox_to_anchor=(-0.018, 0.40))
ps.panel_label(axB, "(b)")

ps.save_pdf(fig, "fig1_two_channels")

# ---------------------------------------------------------------------------
values = {
    "source": "reproduced from experiments/verify_final/vf1_magnetic.py "
              "(T1/T7 block); asserted equal to vf1_magnetic.json at rtol 1e-9",
    "setup": {"h": H, "tau": TAU, "t_final": TF,
              "gyroperiods": float(gyro[-1]),
              "reference": "DOP853, rtol 1e-13, atol 1e-15"},
    "panel_a": {
        "theta_end_deg_measured": theta_end,
        "theta_end_deg_closed_form": theta_end_pred,
        "theta_median_2nd_half_deg_measured": theta_med,
        "theta_median_2nd_half_deg_closed_form": theta_med_pred,
        "deviation_end_percent": dev_end_pct,
        "deviation_median_percent": dev_med_pct,
    },
    "panel_b": {
        "smoothing": "median over one gyroperiod (21 steps)",
        "rel_velocity_error_end": vel_end,
        "two_sin_half_theta_end": float(2.0 * np.sin(np.radians(theta_end) / 2)),
        "speed_err_median_2nd_half": speed_med,
        "speed_err_closed_form_h_over_4tau": speed_closed,
        "energy_err_median_2nd_half": energy_med,
        "energy_err_closed_form_median": energy_closed,
        "speed_last_gyroperiod_median": float(med_speed[-1]),
        "energy_last_gyroperiod_median": float(med_energy[-1]),
        "velocity_last_gyroperiod_median": float(med_vel[-1]),
        "gap_velocity_over_energy_annotated": gap,
    },
    "polar_identity_max_residual": identity_residual,
    "reference_energy_drift_from_analytic": ref_energy_drift,
    "pos_err_rms_over_larmor_radius": pos_rms,
}
with open(os.path.join(HERE, "fig1_two_channels_values.json"), "w",
          encoding="utf-8") as fh:
    json.dump(values, fh, indent=1)
print(json.dumps(values, indent=1))
