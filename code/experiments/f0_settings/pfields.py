"""
pfields.py -- perturbed B4 fields for the F0.1 adversarial elimination.
=======================================================================
Variant B needs a setting where a learned component is *necessary*: where the
right-hand side is not known everywhere, so refining the step cannot help.
This module builds the candidate settings.

Construction. Everything is generated from an axisymmetric vector potential
    dA_phi(rho, t) = (rho/2) a(t)
so that
    dB_z    = (1/rho) d(rho dA_phi)/drho = a(t)      (uniform in space)
    dE_phi  = -d(dA_phi)/dt = -(rho/2) a'(t)
Maxwell consistency is therefore automatic, which the shipped
stochastic_field/wave_field do NOT have (they leave E = 0 under a
time-varying B).

The total field seen by the *reference* is B_smooth + dB; the integrator under
test is told only B_smooth. That gap -- epistemic, not numerical -- is the only
thing that can justify a learned component at all, since experiments/classical
showed that where the field is known everywhere, vps4 beats the hybrid by 418x
in flops.

Sign conventions follow fields/decaying_field.py exactly:
    factor = -0.5 * dBz/dt,  E = factor * (-y, x, 0).
"""
import numpy as np


class PerturbedDecaying:
    """B_z(t) = B0 exp(-t/tau) + a(t), with the Faraday-induced azimuthal E.

    a(t) and its derivative are supplied as callables. a = None reproduces the
    unperturbed B4 field bit-for-bit (checked in test_consistency)."""

    name = "perturbed_decaying"

    def __init__(self, B0=1.0, tau=1.2e5, a=None, adot=None):
        self.B0, self.tau = float(B0), float(tau)
        self._a = a
        self._adot = adot

    # ---------------------------------------------------------------- field
    def Bz_of_t(self, t):
        base = self.B0 * np.exp(-np.asarray(t, dtype=float) / self.tau)
        return base if self._a is None else base + self._a(t)

    def dBz_dt(self, t):
        base = -(self.B0 / self.tau) * np.exp(-np.asarray(t, dtype=float) / self.tau)
        return base if self._adot is None else base + self._adot(t)

    def B(self, r, t):
        r = np.atleast_2d(r)
        out = np.zeros((r.shape[0], 3), dtype=float)
        out[:, 2] = self.Bz_of_t(t)
        return out if r.shape[0] > 1 else out[0]

    def E(self, r, t):
        r = np.atleast_2d(r).astype(float)
        factor = -0.5 * self.dBz_dt(t)          # = E_phi / rho
        out = np.zeros((r.shape[0], 3), dtype=float)
        out[:, 0] = -factor * r[:, 1]
        out[:, 1] = factor * r[:, 0]
        return out if r.shape[0] > 1 else out[0]

    def __call__(self, r, t):
        return self.E(r, t), self.B(r, t)


# ------------------------------------------------------------ perturbations
class QuasiPeriodic:
    """S2/P1: a(t) = sum_i A_i sin(w_i t + p_i), few incommensurate tones.

    Fully deterministic and described by 3*n_tones numbers, so a classical
    attacker who is allowed to *fit* those numbers recovers the field exactly.
    That is the point of the setting: it is a debugging ladder rung, not a
    candidate for the paper."""

    def __init__(self, amps, freqs, phases):
        self.amps = np.asarray(amps, float)
        self.freqs = np.asarray(freqs, float)
        self.phases = np.asarray(phases, float)

    def a(self, t):
        t = np.asarray(t, float)
        return np.sum(self.amps * np.sin(np.outer(np.atleast_1d(t), self.freqs)
                                         + self.phases), axis=-1).reshape(np.shape(t))

    def adot(self, t):
        t = np.asarray(t, float)
        return np.sum(self.amps * self.freqs
                      * np.cos(np.outer(np.atleast_1d(t), self.freqs)
                               + self.phases), axis=-1).reshape(np.shape(t))

    @property
    def n_params(self):
        return 3 * len(self.amps)


class Broadband:
    """S3/P2: stationary Gaussian process by random-phase spectral synthesis.

        a(t) = sum_k c_k cos(w_k t + p_k),   p_k ~ U[0, 2pi)

    The *statistics* (frequency grid w_k and amplitudes c_k, i.e. the PSD) are
    public. The *realization* (the phases p_k) is not. With K modes the
    realization needs K numbers to specify, so for K large no small parametric
    fit recovers it -- that is exactly the information asymmetry the setting is
    built on. Variance = sum(c_k^2)/2, and a'(t) stays analytic."""

    def __init__(self, w_lo=0.5, w_hi=5.0, n_modes=64, rms=1e-3,
                 slope=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.freqs = np.linspace(w_lo, w_hi, n_modes)
        shape = self.freqs ** (-slope) if slope else np.ones(n_modes)
        shape /= np.sqrt(np.sum(shape ** 2) / 2.0)     # unit rms
        self.amps = shape * float(rms)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
        self.rms = float(rms)
        self.seed = int(seed)

    def a(self, t):
        t = np.asarray(t, float)
        return np.sum(self.amps * np.cos(np.outer(np.atleast_1d(t), self.freqs)
                                         + self.phases), axis=-1).reshape(np.shape(t))

    def adot(self, t):
        t = np.asarray(t, float)
        return np.sum(-self.amps * self.freqs
                      * np.sin(np.outer(np.atleast_1d(t), self.freqs)
                               + self.phases), axis=-1).reshape(np.shape(t))

    @property
    def n_params(self):
        return len(self.freqs)          # phases; amps/freqs are public


class SplineField:
    """S1: the smooth field is known only on a coarse time grid.

    A classical attacker interpolates B_z(t) by a cubic spline through the same
    samples and takes dB_z/dt from the spline derivative, so E stays consistent
    with the interpolant. No learning involved."""

    def __init__(self, t_nodes, bz_nodes):
        from scipy.interpolate import CubicSpline
        self.spl = CubicSpline(np.asarray(t_nodes, float),
                               np.asarray(bz_nodes, float))
        self.dspl = self.spl.derivative()
        self.n_nodes = len(t_nodes)

    def Bz_of_t(self, t):
        return self.spl(t)

    def dBz_dt(self, t):
        return self.dspl(t)

    def B(self, r, t):
        r = np.atleast_2d(r)
        out = np.zeros((r.shape[0], 3), dtype=float)
        out[:, 2] = self.Bz_of_t(t)
        return out if r.shape[0] > 1 else out[0]

    def E(self, r, t):
        r = np.atleast_2d(r).astype(float)
        factor = -0.5 * self.dBz_dt(t)
        out = np.zeros((r.shape[0], 3), dtype=float)
        out[:, 0] = -factor * r[:, 1]
        out[:, 1] = factor * r[:, 0]
        return out if r.shape[0] > 1 else out[0]
