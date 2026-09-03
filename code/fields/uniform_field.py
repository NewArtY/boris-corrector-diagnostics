"""
uniform_field.py
=================
Uniform (homogeneous) magnetic field configuration.

This is the reference / analytically solvable test case described in the
Methods section: "Однородное магнитное поле использовалось как эталонный
случай, при котором движение заряженной частицы представляет собой
идеальную циклотронную орбиту." It is used to validate energy invariance
and symplectic conservation of every integrator against the exact analytic
cyclotron solution.

B(r, t) = B0 * z_hat            (constant everywhere)
E(r, t) = E0  (small, optional uniform electric field; default zero)
"""

import numpy as np


class UniformField:
    """Constant magnetic field along z, optional constant weak E field."""

    name = "uniform"
    description = "Homogeneous magnetic field B = B0 z_hat (analytic cyclotron orbit)"

    def __init__(self, B0: float = 1.0, E0: np.ndarray = None):
        self.B0 = B0
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.zeros_like(r, dtype=float)
        out[:, 2] = self.B0
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
