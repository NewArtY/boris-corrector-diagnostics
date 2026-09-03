"""
plot_style.py
==============
Shared matplotlib style configuration for all publication figures:
colorblind-friendly palette, serif-neutral fonts, consistent panel
labeling helpers, and dual PNG(300dpi)+PDF saving.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import FIGURE_DIR, COLORS, LABELS

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # serif-neutral, widely available, clean
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "lines.linewidth": 1.8,
    "axes.linewidth": 1.0,
})


def save_fig(fig, basename):
    """Save a figure as both 300dpi PNG and vector PDF into output_figures/."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    png_path = os.path.join(FIGURE_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"Saved {png_path} and {pdf_path}")
    return png_path, pdf_path


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add a bold panel label (a, b, c, ...) at the top-left of an axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold",
             va="bottom", ha="right")
