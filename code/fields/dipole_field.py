"""
dipole_field.py
================
Dipole magnetic field configuration (Eq. 14 in the Methods section):

    B(r) = B0 * (r0 / |r|)^3 * [3 (m_hat . r_hat) r_hat - m_hat]

where r0 is the characteristic dipole scale length and m_hat is the dipole
moment direction (taken along z). This models magnetospheric / laboratory
plasma configurations with spatial inhomogeneity and field-line curvature,
used to test sensitivity of integrators to magnetic gradients.
"""

import numpy as np


class DipoleField:
    """Magnetic dipole field, characteristic scale r0."""

    name = "dipole"
    description = "Dipole field B ~ (r0/|r|)^3 (magnetospheric-like inhomogeneity)"

    def __init__(self, B0: float = 1.0, r0: float = 5.0, m_hat: np.ndarray = None,
                 E0: np.ndarray = None, soften: float = 1e-3):
        self.B0 = B0
        self.r0 = r0
        self.m_hat = np.array([0.0, 0.0, 1.0]) if m_hat is None else np.asarray(m_hat, dtype=float)
        self.m_hat = self.m_hat / np.linalg.norm(self.m_hat)
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)
        self.soften = soften  # regularizes 1/r singularity near origin

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r).astype(float)
        # offset the dipole center so the particle (started near x=r0) is
        # in a smooth, non-singular region of the field
        rr = r.copy()
        rr[:, 0] += self.r0  # shift dipole origin behind the orbit
        norm = np.linalg.norm(rr, axis=1, keepdims=True) + self.soften
        r_hat = rr / norm
        dot = np.sum(r_hat * self.m_hat, axis=1, keepdims=True)
        B = self.B0 * (self.r0 / norm) ** 3 * (3 * dot * r_hat - self.m_hat)
        return B if r.shape[0] > 1 else B[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
