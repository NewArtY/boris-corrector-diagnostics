"""
fig2_phase_portraits.py
=========================
Figure 2: combined phase portraits (Vx, Vy) and configuration-space tracks
(x, y) for all five integrators in the uniform field, as described in
Section 3 ("комбинированные диаграммы...фазовый портрет скоростей...и
треки частиц").

Output: Figure2_phase_portraits.{png,pdf}
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import COLORS, LABELS, SEED, set_global_seed
from figures.plot_style import save_fig, add_panel_label
from fields import UniformField, DipoleField
from diagnostics.integrator_runner import integrate, ALL_INTEGRATORS

set_global_seed(SEED)

# Draw order: put the most divergent / least accurate models first so
# accurate ones (Boris, Boris+Corrector) are drawn on top and remain visible.
DRAW_ORDER = ["hnn", "sympnet", "pinn_symplectic", "boris_corrector", "boris"]
LINESTYLES = {
    "boris": "-", "boris_corrector": "--", "pinn_symplectic": "-.",
    "hnn": "-", "sympnet": ":",
}


def main():
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    dt = 0.05          # moderate step: reveals integrator differences without HNN blowup dominating scale
    n_steps = 150

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))

    configs = [
        ("Uniform field", UniformField(B0=1.0), axes[0, 0], axes[0, 1], "a", "b"),
        ("Dipole field", DipoleField(B0=1.0, r0=5.0), axes[1, 0], axes[1, 1], "c", "d"),
    ]

    for title, field, ax_v, ax_r, lab_v, lab_r in configs:
        traj = {}
        for name in ALL_INTEGRATORS:
            rs, vs, ts = integrate(name, field, r0, v0, 0.0, dt, n_steps)
            traj[name] = (rs, vs)

        for name in DRAW_ORDER:
            rs, vs = traj[name]
            ax_v.plot(vs[:, 0], vs[:, 1], color=COLORS[name], label=LABELS.get(name, name),
                      alpha=0.9, linewidth=1.6, linestyle=LINESTYLES[name])
            ax_r.plot(rs[:, 0], rs[:, 1], color=COLORS[name], label=LABELS.get(name, name),
                      alpha=0.9, linewidth=1.6, linestyle=LINESTYLES[name])

        # Zoom the axes to the well-behaved integrators (Boris, Boris+Corrector,
        # PINN, SympNet) so the plot isn't dominated by an outlier (HNN) scale.
        well_behaved = ["boris", "boris_corrector", "pinn_symplectic", "sympnet"]
        v_all = np.concatenate([traj[n][1] for n in well_behaved], axis=0)
        r_all = np.concatenate([traj[n][0] for n in well_behaved], axis=0)
        pad_v = 0.3 * (np.ptp(v_all[:, 0]) + np.ptp(v_all[:, 1]) + 1e-3)
        pad_r = 0.3 * (np.ptp(r_all[:, 0]) + np.ptp(r_all[:, 1]) + 1e-3)
        ax_v.set_xlim(v_all[:, 0].min() - pad_v, v_all[:, 0].max() + pad_v)
        ax_v.set_ylim(v_all[:, 1].min() - pad_v, v_all[:, 1].max() + pad_v)
        ax_r.set_xlim(r_all[:, 0].min() - pad_r, r_all[:, 0].max() + pad_r)
        ax_r.set_ylim(r_all[:, 1].min() - pad_r, r_all[:, 1].max() + pad_r)

        ax_v.set_xlabel("$V_x$")
        ax_v.set_ylabel("$V_y$")
        ax_v.set_title(f"{title}: velocity phase portrait")
        ax_v.set_aspect("equal")
        add_panel_label(ax_v, lab_v)

        ax_r.set_xlabel("x")
        ax_r.set_ylabel("y")
        ax_r.set_title(f"{title}: configuration-space track")
        ax_r.set_aspect("equal")
        add_panel_label(ax_r, lab_r)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # restore natural (non-reversed draw) legend order
    order = [labels.index(LABELS[n]) for n in ALL_INTEGRATORS]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Figure 2. Phase portraits and configuration-space tracks "
                 f"(dt={dt} in normalized gyro-units). Axes zoomed to symplectic/hybrid methods;\n"
                 "HNN and SympNet traces shown at true scale, may extend beyond zoomed view.",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, "Figure2_phase_portraits")
    plt.close(fig)


if __name__ == "__main__":
    main()
