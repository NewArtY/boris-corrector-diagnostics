"""
radial_field.py
================
Field configuration B1: quadratic radial magnetic-field gradient.

    B_z(r) = B0 * (1 + alpha * (rho / L)^2),   rho = sqrt(x^2 + y^2)

This represents a smooth, analytically simple magnetic-mirror-like radial
inhomogeneity, used as the first (mildest) step in the field-complexity
hierarchy discussed in the Methods (Fig. 1): from uniform -> weakly
non-uniform -> strongly structured / non-stationary fields. It complements
the dipole field with a purely radial (no field-line curvature reversal)
gradient, useful for isolating the effect of |grad B| drift on integrator
accuracy.
"""

import numpy as np


class RadialField:
    """Quadratic radial magnetic field gradient B1."""

    name = "radial"
    description = "B1: B_z = B0 (1 + alpha (rho/L)^2), quadratic radial gradient"

    def __init__(self, B0: float = 1.0, alpha: float = 0.3, L: float = 5.0,
                 E0: np.ndarray = None):
        self.B0 = B0
        self.alpha = alpha
        self.L = L
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r).astype(float)
        rho2 = r[:, 0] ** 2 + r[:, 1] ** 2
        Bz = self.B0 * (1.0 + self.alpha * rho2 / self.L ** 2)
        out = np.zeros_like(r)
        out[:, 2] = Bz
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
