"""
fig1_field_hierarchy.py
=========================
Figure 1: hierarchy of magnetic field configurations used as a test
sequence from the analytically solvable uniform case to non-stationary /
stochastic fields (Section 2.3, "иерархия тестов").

Output: Figure1_field_hierarchy.{png,pdf}
Panels (a)-(f): uniform, dipole, stochastic, radial (B1), wave (B2),
tilted (B3), and an inset for decaying (B4) field magnitude vs time.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.plot_style import save_fig, add_panel_label
from fields import (UniformField, DipoleField, StochasticField,
                     RadialField, WaveField, TiltedField, DecayingField)

np.random.seed(42)


def field_magnitude_grid(field, extent=8.0, n=60, t=0.0):
    xs = np.linspace(-extent, extent, n)
    ys = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(xs, ys)
    Bmag = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            r = np.array([X[i, j], Y[i, j], 0.0])
            Bmag[i, j] = np.linalg.norm(field.B(r, t))
    return X, Y, Bmag


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.2))

    fields_static = [
        ("Uniform", UniformField(B0=1.0), axes[0, 0], "a"),
        ("Dipole", DipoleField(B0=1.0, r0=5.0), axes[0, 1], "b"),
        ("Radial (B1)", RadialField(B0=1.0, alpha=0.3, L=5.0), axes[0, 2], "c"),
        ("Tilted (B3)", TiltedField(B0=1.0, theta_deg=30.0), axes[0, 3], "d"),
    ]

    for title, field, ax, label in fields_static:
        X, Y, Bmag = field_magnitude_grid(field)
        im = ax.pcolormesh(X, Y, Bmag, shading="auto", cmap="viridis")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("|B|", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        add_panel_label(ax, label)

    # (e) stochastic field: |B| vs time at fixed point
    ax = axes[1, 0]
    field = StochasticField(B0=1.0, eps=0.15, omega=0.5, seed=42)
    ts = np.linspace(0, 400, 2000)
    Bz = np.array([field.B(np.zeros(3), t)[2] for t in ts])
    ax.plot(ts, Bz, color="#CC79A7")
    ax.set_title("Stochastic modulation")
    ax.set_xlabel("t")
    ax.set_ylabel("$B_z(t)$")
    add_panel_label(ax, "e")

    # (f) wave field B2: |B| vs x at two times
    ax = axes[1, 1]
    field = WaveField(B0=1.0, delta=0.05, k=0.4, omega_w=0.3)
    xs = np.linspace(-10, 10, 300)
    for t, c in zip([0, 5, 10], ["#0072B2", "#D55E00", "#009E73"]):
        Bz = np.array([field.B(np.array([x, 0, 0]), t)[2] for x in xs])
        ax.plot(xs, Bz, label=f"t={t}", color=c)
    ax.set_title("Weak spatiotemporal wave (B2)")
    ax.set_xlabel("x")
    ax.set_ylabel("$B_z$")
    ax.legend(frameon=False)
    add_panel_label(ax, "f")

    # (g) decaying field B4: |B(t)|
    ax = axes[1, 2]
    field = DecayingField(B0=1.0, tau=40.0)
    ts = np.linspace(0, 200, 500)
    Bz = field.Bz_of_t(ts)
    ax.plot(ts, Bz, color="#E69F00")
    ax.set_title(r"Time-decaying field (B4): $B_0 e^{-t/\tau}$")
    ax.set_xlabel("t")
    ax.set_ylabel("$B_z(t)$")
    add_panel_label(ax, "g")

    # (h) summary text panel: complexity hierarchy
    ax = axes[1, 3]
    ax.axis("off")
    text = (
        "Field complexity hierarchy\n\n"
        "1. Uniform - analytic reference\n"
        "2. Radial (B1) - quadratic gradient\n"
        "3. Wave (B2) - weak spatiotemporal\n"
        "4. Tilted (B3) - mixed x/z axis\n"
        "5. Dipole - spatial inhomogeneity\n"
        "6. Stochastic - OOD generalization\n"
        "7. Decaying (B4) - non-stationary,\n"
        "   physical energy change vs.\n"
        "   numerical drift separation"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#f2f2f2", edgecolor="#999999"))
    add_panel_label(ax, "h")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, "Figure1_field_hierarchy")
    plt.close(fig)


if __name__ == "__main__":
    main()
