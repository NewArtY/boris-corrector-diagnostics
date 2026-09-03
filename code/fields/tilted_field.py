"""
tilted_field.py
================
Field configuration B3: mixed x/z (tilted) uniform magnetic field.

    B = B0 * (sin(theta) x_hat + cos(theta) z_hat)

A static but non-axis-aligned field that produces coupled gyration in all
three velocity components. This tests whether integrators (especially
neural ones trained mostly on z-aligned fields) generalize correctly when
the field direction is rotated away from the training-time symmetry axis,
without introducing any spatial or temporal inhomogeneity.
"""

import numpy as np


class TiltedField:
    """Static uniform field tilted by angle theta from z toward x, B3."""

    name = "tilted"
    description = "B3: static field mixed between x and z axes (tilt angle theta)"

    def __init__(self, B0: float = 1.0, theta_deg: float = 30.0, E0: np.ndarray = None):
        self.B0 = B0
        self.theta = np.deg2rad(theta_deg)
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)
        self.direction = np.array([np.sin(self.theta), 0.0, np.cos(self.theta)])

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.B0 * self.direction, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
