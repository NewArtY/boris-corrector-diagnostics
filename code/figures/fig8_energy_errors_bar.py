"""
fig8_energy_errors_bar.py
============================
Figure 6: grouped bar comparison of the time-averaged relative energy
deviation <|dE/E0|> for all five integrators across the four unseen field
configurations B1-B4 (see fig7_generalization.py for field definitions).

Output: output_figures/Figure6_energy_errors_bar.{png,pdf}
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
from diagnostics.energy_drift import mean_abs_energy_drift

set_global_seed(SEED)
logger = get_logger("figure6")

DT = 0.05
N_GYRO = 15
N_STEPS = int(N_GYRO * (2 * np.pi) / DT)
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
EPS = 1e-16

FIELD_SPECS = [
    ("B1", RadialField()),
    ("B2", WaveField()),
    ("B3", TiltedField()),
    ("B4", DecayingField(tau=40.0)),
]


def compute_matrix():
    """Return dict[integrator][field_tag] = mean |dE/E0|."""
    values = {name: {} for name in ALL_INTEGRATORS}
    for tag, field in FIELD_SPECS:
        for name in ALL_INTEGRATORS:
            rs, vs, ts = integrate(name, field, R0, V0, 0.0, DT, N_STEPS)
            val = mean_abs_energy_drift(vs)
            val = float(np.clip(val, EPS, 1e12))
            values[name][tag] = val
        logger.info("Field %s done", tag)
    return values


def main():
    values = compute_matrix()
    field_tags = [tag for tag, _ in FIELD_SPECS]
    n_fields = len(field_tags)
    n_int = len(ALL_INTEGRATORS)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    x = np.arange(n_fields)
    width = 0.8 / n_int

    for i, name in enumerate(ALL_INTEGRATORS):
        offsets = x + (i - (n_int - 1) / 2.0) * width
        heights = [values[name][tag] for tag in field_tags]
        ax.bar(offsets, heights, width=width * 0.95, color=COLORS[name],
               label=LABELS.get(name, name))

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(field_tags)
    ax.set_xlabel("field configuration")
    ax.set_ylabel(r"$\langle|\delta E/E_0|\rangle$  (time-averaged)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9.5,
              borderaxespad=0.0)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_fig(fig, "Figure6_energy_errors_bar")
    plt.close(fig)

    return values


if __name__ == "__main__":
    main()
