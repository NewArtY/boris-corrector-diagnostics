"""
phase_space.py
================
Phase-space diagnostics: phase portraits (Vx, Vy) and configuration-space
tracks (x, y), plus Poincare sections for long-term stability visualization,
as described in Section 3 ("комбинированные диаграммы, одновременно
отображающие фазовый портрет скоростей ... и треки частиц").
"""

import numpy as np


def velocity_phase_portrait(vs):
    """Return (Vx, Vy) arrays for a phase portrait."""
    return vs[:, 0], vs[:, 1]


def configuration_track(rs):
    """Return (x, y) arrays for the configuration-space track."""
    return rs[:, 0], rs[:, 1]


def poincare_section(rs, vs, ts, period, plane_coord=2, plane_value=0.0, tol=None):
    """Extract a Poincare section by sampling stroboscopically once per
    `period` (e.g., once per gyroperiod) rather than a true crossing-based
    section -- robust for CPU-only demonstration runs.

    Returns (r_section, v_section): points sampled at t = 0, period,
    2*period, ...
    """
    t_max = ts[-1]
    sample_times = np.arange(0, t_max, period)
    idx = np.searchsorted(ts, sample_times)
    idx = idx[idx < len(ts)]
    return rs[idx], vs[idx]


def gyroradius(v_perp, omega_c):
    """Instantaneous gyroradius from perpendicular speed and gyrofrequency."""
    return v_perp / omega_c
