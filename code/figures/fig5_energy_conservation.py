"""
fig5_energy_conservation.py
==============================
Figure 3 (per required renaming): energy-conservation comparison across all
five integrators in the uniform field, over long-term integration (tens of
gyroperiods), following Eq. (13) of the Methods.

Output: Figure3_energy_conservation.{png,pdf}
Panels: (a) relative energy drift dE/E0 vs time, (b) RMS energy error bar
comparison, (c) |v| conservation vs time (log-scale deviation).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import COLORS, LABELS, SEED, set_global_seed, T_C
from figures.plot_style import save_fig, add_panel_label
from fields import UniformField
from diagnostics.integrator_runner import integrate, ALL_INTEGRATORS
from diagnostics.energy_drift import relative_energy_drift, rms_energy_error, mean_abs_energy_drift

set_global_seed(SEED)


def main():
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    dt = 0.05
    n_gyro = 20
    n_steps = int(n_gyro * (2 * np.pi) / dt)

    field = UniformField(B0=1.0)

    fig = plt.figure(figsize=(14, 5.2))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    rms_values = {}
    mean_abs_values = {}

    for name in ALL_INTEGRATORS:
        rs, vs, ts = integrate(name, field, r0, v0, 0.0, dt, n_steps)
        drift = relative_energy_drift(vs)
        speed = np.linalg.norm(vs, axis=-1)

        ax_a.plot(ts / (2 * np.pi), np.abs(drift) + 1e-16, color=COLORS[name], label=LABELS.get(name, name),
                  linewidth=1.5, alpha=0.9)
        ax_c.plot(ts / (2 * np.pi), np.abs(speed - speed[0]) + 1e-16, color=COLORS[name],
                  label=LABELS.get(name, name), linewidth=1.5, alpha=0.9)

        rms_values[name] = rms_energy_error(vs)
        mean_abs_values[name] = mean_abs_energy_drift(vs)

    ax_a.set_xlabel("time (gyroperiods)")
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$|\delta E / E_0|$")
    ax_a.set_title("Relative energy drift")
    ax_a.legend(frameon=False, fontsize=8.5, loc="upper left")
    add_panel_label(ax_a, "a")

    names_sorted = sorted(ALL_INTEGRATORS, key=lambda n: rms_values[n])
    bar_colors = [COLORS[n] for n in names_sorted]
    ax_b.bar([LABELS.get(n, n) for n in names_sorted], [rms_values[n] for n in names_sorted],
              color=bar_colors)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("RMS energy error")
    ax_b.set_title("RMS energy error by integrator")
    ax_b.tick_params(axis="x", rotation=30)
    for tick in ax_b.get_xticklabels():
        tick.set_ha("right")
    add_panel_label(ax_b, "b")

    ax_c.set_xlabel("time (gyroperiods)")
    ax_c.set_ylabel(r"$||v| - |v_0||$")
    ax_c.set_yscale("log")
    ax_c.set_title("Speed-conservation deviation")
    add_panel_label(ax_c, "c")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "Figure3_energy_conservation")
    plt.close(fig)

    return rms_values, mean_abs_values


if __name__ == "__main__":
    main()
