"""
fig9_spectral_analysis.py
=========================
Figure 7 -- frequency-domain test of the two error channels.

The Article claims a specific mechanism: the classical scheme's dominant error
is phase error concentrated near the gyration frequency, which integrates into
position drift, while a learned correction that is not energy-constrained
deposits residual power at LOW frequency, i.e. exactly where a slow physical
energy signal lives. This script tests that claim directly, in the same
decaying configuration B4 and at the same working step as Fig. 4, comparing
the three schemes that appear there.

Output: output_figures/Figure7_spectral_analysis.{png,pdf}
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import COLORS, FIGURE_DIR, SEED, set_global_seed, get_logger
from figures.plot_style import save_fig, add_panel_label
from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DT_WORK, DT_FINE, T_FINAL, TAU_MAIN
from diagnostics.eval_corrector import load_corrector, integrate_corrected

torch.set_default_dtype(torch.float64)
set_global_seed(SEED)
logger = get_logger("figure7")

C_RAW = "#D55E00"


def main():
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    n_fine = int(round(T_FINAL / DT_FINE))
    rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)

    n_work = int(round(T_FINAL / DT_WORK))
    model = load_corrector()
    runs = {
        "Boris": (integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field), COLORS["boris"]),
        "unconstrained correction":
            (integrate_corrected(field, r0, v0, DT_WORK, n_work, model, project=False), C_RAW),
        "constrained hybrid":
            (integrate_corrected(field, r0, v0, DT_WORK, n_work, model, project=True),
             COLORS["boris_corrector"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    peak_info = {}

    for lab, ((rs, vs, ts), col) in runs.items():
        v_ref = np.vstack([np.interp(ts, ts_r, vs_r[:, j]) for j in range(3)]).T
        err = np.linalg.norm(vs - v_ref, axis=1)

        axes[0].semilogy(ts, np.maximum(err, 1e-18), color=col, lw=1.3, label=lab)

        e = err.copy()
        w = np.hanning(len(e))
        f = np.fft.rfftfreq(len(e), d=DT_WORK) * 2.0 * np.pi   # in units of Omega_c
        psd = np.abs(np.fft.rfft(e * w)) ** 2
        psd /= len(e) ** 2   # absolute, comparable between schemes
        m = f > 0
        axes[1].loglog(f[m], np.maximum(psd[m], 1e-20), color=col, lw=1.2, label=lab)

        low = m & (f < 0.2)
        peak_info[lab] = {"low_freq_power_fraction": float(psd[low].sum() / psd[m].sum())}
        logger.info("%-26s low-frequency power fraction = %.3f",
                    lab, peak_info[lab]["low_freq_power_fraction"])

    ax = axes[0]
    ax.set_xlabel("time  $\\Omega_c t$")
    ax.set_ylabel("velocity error  $|\\mathbf{v}-\\mathbf{v}_{\\rm ref}|$")
    ax.set_title("Velocity error against converged reference", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.axvline(1.0, color="0.55", ls=":", lw=1.0)
    ax.text(1.05, 3e-18, "$\\Omega_c$", fontsize=9, color="0.35")
    ax.axvspan(f[1], 0.2, color=COLORS["physical"], alpha=0.10)
    ax.text(0.03, 0.06, "band of the\nphysical signal", transform=ax.transAxes,
            fontsize=8, color=COLORS["physical"], va="bottom")
    ax.set_xlabel("frequency  $f/\\Omega_c$")
    ax.set_ylabel("PSD of velocity error  (absolute)")
    ax.set_title("Where each scheme puts its error", fontsize=11)
    ax.set_ylim(1e-18, 1e-1)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    for a, l in zip(axes, "ab"):
        add_panel_label(a, l)
    fig.tight_layout()
    save_fig(fig, "Figure7_spectral_analysis")

    with open(os.path.join(FIGURE_DIR, "figure7_numbers.json"), "w") as f:
        json.dump(peak_info, f, indent=2)
    print(json.dumps(peak_info, indent=2))


if __name__ == "__main__":
    main()
