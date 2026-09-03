"""
decaying_field.py
==================
Field configuration B4: time-decaying magnetic field.

    B(t) = B0 * exp(-t / tau) * z_hat

This is the KEY test case of the study. As |B(t)| decays, Faraday's law
(curl E = -dB/dt) induces an azimuthal electric field

    E_phi(rho, t) = -(rho/2) * dB/dt = (rho / (2 tau)) * B0 * exp(-t/tau)

which does real work on the gyrating particle. To leading order this
induced field conserves the first adiabatic invariant
(magnetic moment mu = m v_perp^2 / (2 |B|)), so the perpendicular kinetic
energy decreases smoothly in time together with |B(t)| -- a genuine,
physically expected change of kinetic energy that is NOT a numerical
artifact. This physical energy change is exactly the signal that Figure 4
contrasts against the numerical energy-drift error of each integrator.

The scientific question addressed by Figure 4 (fig6_decaying_field_case.py)
is whether the *numerical* energy-drift error of a given integrator can be
made small enough, relative to this *physical* energy change, that the two
effects become cleanly separable. The Boris+Corrector hybrid is designed to
push the discretization error 2-3 orders of magnitude below the physical
signal, which is exactly what B4 is built to demonstrate.
"""

import numpy as np


class DecayingField:
    """Time-decaying magnetic field B(t) = B0 exp(-t/tau) z_hat, B4."""

    name = "decaying"
    description = "B4: B(t) = B0 exp(-t/tau) z_hat, adiabatic energy change test case"

    def __init__(self, B0: float = 1.0, tau: float = 40.0, E0: np.ndarray = None,
                 induced_E: bool = True):
        self.B0 = B0
        self.tau = tau
        self.E0 = np.zeros(3) if E0 is None else np.asarray(E0, dtype=float)
        self.induced_E = induced_E

    def Bz_of_t(self, t):
        return self.B0 * np.exp(-np.asarray(t) / self.tau)

    def B(self, r: np.ndarray, t: float) -> np.ndarray:
        r = np.atleast_2d(r)
        out = np.zeros_like(r, dtype=float)
        out[:, 2] = self.Bz_of_t(t)
        return out if r.shape[0] > 1 else out[0]

    def E(self, r: np.ndarray, t: float) -> np.ndarray:
        """Azimuthal electric field induced by dB/dt via Faraday's law:
        E_phi = -(rho/2) dB/dt = (rho/(2 tau)) B0 exp(-t/tau), i.e.
        E = E_phi * phi_hat = (E_phi/rho) * (-y, x, 0)."""
        r = np.atleast_2d(r).astype(float)
        out = np.tile(self.E0, (r.shape[0], 1))
        if self.induced_E:
            dBdt = -self.Bz_of_t(t) / self.tau  # dB_z/dt
            factor = -0.5 * dBdt  # = E_phi / rho
            out = out.copy()
            out[:, 0] += -factor * r[:, 1]
            out[:, 1] += factor * r[:, 0]
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r: np.ndarray, t: float):
        return self.E(r, t), self.B(r, t)
