"""Shared matplotlib style for the manuscript figures (paper/figures/fig*.py).

Why a second style module, next to ``code/figures/plot_style.py``
----------------------------------------------------------------
``code/figures/plot_style.py`` styles the *repository* figures: 11 pt type,
1.8 pt lines, in-axes titles, and ``save_fig`` writing a 300 dpi PNG **and** a
PDF into ``code/output_figures/``.  Those defaults are right for standalone
plots that are looked at on their own.  They are wrong for the manuscript:

  * the figure must be included at ``width=\\linewidth`` with **no scaling**,
    so its physical width has to equal the text width of the class actually in
    use (``elsarticle`` ``preprint,12pt``: ``\\textwidth = 390 pt = 5.397 in``,
    measured, not assumed);
  * at 1:1 the type size inside the artwork must be *smaller* than the body
    text, not equal to it -- 8-9 pt against a 12 pt body;
  * the manuscript wants vector PDF only, written next to the script
    (``paper/figures/README.md``), not a PNG in ``output_figures/``;
  * Elsevier rejects Type 3 fonts, which is what matplotlib's PDF backend
    emits by default; ``pdf.fonttype = 42`` embeds TrueType instead;
  * captions carry the titles, so axes must have none.

What is carried over unchanged is the part that should be shared: the
Okabe-Ito colourblind-safe palette of ``code/common.py`` and the panel-label
convention.  Figures 3 and 4 (wave W2.2) import this module, so all four
manuscript figures read as one set.

Encoding rule
-------------
No series is distinguished by colour alone.  Every series carries a distinct
dash pattern *and*, where markers are used, a distinct marker, so the figures
survive both colourblind viewing and greyscale printing.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- geometry -------------------------------------------------------------
# elsarticle [preprint,12pt]: \textwidth = 390 pt (TeX pt) = 390/72.27 in.
TEXTWIDTH_IN = 390.0 / 72.27          # 5.397 in
FIGDIR = os.path.dirname(os.path.abspath(__file__))

# --- palette (Okabe-Ito, identical to code/common.py COLORS) --------------
BLACK = "#000000"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREY = "#7F7F7F"

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "axes.titlesize": 9.0,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "grid.color": "#000000",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#BBBBBB",
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.32,
    "legend.handlelength": 2.6,
    "legend.handletextpad": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # embed TrueType; Elsevier rejects Type 3
    "ps.fonttype": 42,
    "pdf.compression": 6,
}


def use_style():
    plt.rcParams.update(RC)


def panel_label(ax, text, x=-0.155, y=1.02):
    """Panel tag (a), (b) ... outside the axes, top left."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")


def save_pdf(fig, basename):
    """Vector PDF next to the script, per paper/figures/README.md."""
    path = os.path.join(FIGDIR, f"{basename}.pdf")
    fig.savefig(path)
    print(f"wrote {path}")
    return path
