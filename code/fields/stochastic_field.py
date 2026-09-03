"""
stochastic_field.py
====================
Stochastically modulated magnetic field (Eq. 15 in the Methods section):

    B(t) = B0 * [1 + eps * sin(omega t + phi(t))] * z_hat

where eps is the modulation amplitude, omega the modulation frequency, and
phi(t) a slowly-varying random phase with correlation length of a few
gyroperiods. Used as an out-of-distribution generalization test for the
neural integrators (Section 2.3: "тест на обобщающую способность
нейросетей за пределами обучающего распределения").
"""

import numpy as np


class StochasticField:
    """Magnetic field with stochastic phase modulation about a mean B0."""

    name = "stochastic"
    description = "B0*(1+eps*sin(omega t + phi(t))) z_hat, phi(t) correlated random walk"

    def __init__(self, B0: float = 1.0, eps: float = 0.15, omega: float = 0.5,
                 corr_gyroperiods: float = 3.0, T_c: float = 2 * np.pi,
                 E0: np.ndarray = None, seed: int = 42):
        self.B0 = B0
        self.eps = eps
        self.omega = omega
        self.T_c = T_c
        self.corr_time = corr_gyroperiods * T_c
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)
        self.rng = np.random.default_rng(seed)
        # Pre-generate a smooth random phase process phi(t) via an
        # Ornstein-Uhlenbeck process sampled finely, then interpolated.
        self._t_grid = np.linspace(0.0, 2000 * T_c, 200000)
        dt = self._t_grid[1] - self._t_grid[0]
        theta = dt / self.corr_time
        noise = self.rng.normal(0, 1, size=self._t_grid.size)
        phi = np.zeros_like(self._t_grid)
        for i in range(1, phi.size):
            phi[i] = phi[i - 1] * (1 - theta) + noise[i] * np.sqrt(2 * theta)
        self._phi_grid = phi

    def _phi(self, t):
        return np.interp(t, self._t_grid, self._phi_grid)

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        phase = self.omega * t + self._phi(t)
        Bz = self.B0 * (1.0 + self.eps * np.sin(phase))
        out = np.zeros_like(r, dtype=float)
        out[:, 2] = Bz
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.tile(self.E0, (r.shape[0], 1))
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
