"""
fig7_generalization.py
========================
Figure 5: generalization to unseen field configurations.

All neural integrators (PINN-symplectic, HNN, SympNet, Boris+Corrector) are
trained on trajectories from the uniform / dipole / stochastic field
hierarchy (see fields/__init__.py). This figure stress-tests them, together
with the plain Boris baseline, on four field configurations that were never
seen during training:

    B1 radial_field.py    - quadratic radial gradient
    B2 wave_field.py      - weak spatiotemporal travelling wave
    B3 tilted_field.py    - static field tilted off the training axis
    B4 decaying_field.py  - time-decaying field (adiabatic energy change)

Panels (2x2), one per field, show the relative energy deviation
|dE/E0|(t) for all five integrators on a log-scale y-axis.

Output: output_figures/Figure5_generalization.{png,pdf}
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import COLORS, LABELS, SEED, set_global_seed, get_logger
from figures.plot_style import save_fig, add_panel_label
from fields import RadialField, WaveField, TiltedField, DecayingField
from diagnostics.integrator_runner import integrate, ALL_INTEGRATORS
from diagnostics.energy_drift import relative_energy_drift

set_global_seed(SEED)
logger = get_logger("figure5")

# Modest but sufficient run length for a clear generalization comparison.
DT = 0.05
N_GYRO = 15
N_STEPS = int(N_GYRO * (2 * np.pi) / DT)
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
EPS = 1e-16  # floor for log-scale plotting of |dE/E0|

FIELD_SPECS = [
    ("B1", "radial", RadialField()),
    ("B2", "wave", WaveField()),
    ("B3", "tilted", TiltedField()),
    ("B4", "decaying", DecayingField(tau=40.0)),
]


def run_all(field):
    """Return {integrator_name: (ts_in_gyroperiods, |dE/E0|)}."""
    out = {}
    for name in ALL_INTEGRATORS:
        rs, vs, ts = integrate(name, field, R0, V0, 0.0, DT, N_STEPS)
        drift = relative_energy_drift(vs)
        out[name] = (ts / (2 * np.pi), np.abs(drift) + EPS)
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    panel_labels = ["a", "b", "c", "d"]

    for ax, label, (tag, key, field) in zip(axes.flat, panel_labels, FIELD_SPECS):
        results = run_all(field)
        for name in ALL_INTEGRATORS:
            t_gyro, abs_drift = results[name]
            # Clip absurd divergences for plot readability but keep log-scale.
            abs_drift = np.clip(abs_drift, EPS, 1e12)
            ax.plot(t_gyro, abs_drift, color=COLORS[name], label=LABELS.get(name, name),
                     linewidth=1.3, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("time (gyroperiods)")
        ax.set_ylabel(r"$|\delta E / E_0|$")
        ax.set_title(f"{tag}: {field.description.split(':')[-1].strip()}", fontsize=10)
        add_panel_label(ax, label)
        logger.info("Panel %s (%s) done", label, tag)

    # Single shared legend below all panels to avoid covering any data.
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    save_fig(fig, "Figure5_generalization")
    plt.close(fig)


if __name__ == "__main__":
    main()
