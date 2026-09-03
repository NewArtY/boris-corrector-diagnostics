"""
wave_field.py
=============
Field configuration B2: weak spatiotemporal wave perturbation superimposed
on a uniform background field.

    B_z(x, t) = B0 * [1 + delta * sin(k x - omega_w t)]

This is a small-amplitude travelling-wave modulation (delta << 1),
representative of low-frequency MHD-like wave activity in a background
magnetic field. It probes the ability of the integrators (classical and
neural) to track weak, explicitly space- AND time-dependent perturbations
without accumulating secular phase or energy drift.
"""

import numpy as np


class WaveField:
    """Weak spatiotemporal magnetic wave B2."""

    name = "wave"
    description = "B2: B_z = B0 (1 + delta sin(k x - omega_w t)), weak spatiotemporal wave"

    def __init__(self, B0: float = 1.0, delta: float = 0.05, k: float = 0.4,
                 omega_w: float = 0.3, E0: np.ndarray = None):
        self.B0 = B0
        self.delta = delta
        self.k = k
        self.omega_w = omega_w
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r).astype(float)
        Bz = self.B0 * (1.0 + self.delta * np.sin(self.k * r[:, 0] - self.omega_w * t))
        out = np.zeros_like(r)
        out[:, 2] = Bz
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
