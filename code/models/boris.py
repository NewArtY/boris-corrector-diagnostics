"""
boris.py
========
Classical Boris pusher, implementing Eqs. (2)-(4) of the Methods section.

The Boris algorithm splits one Lorentz-force step into:
  1) a half electric-field acceleration kick,
  2) an exact rotation of the velocity about B using the (t, s) rotation
     vectors,
  3) a second half electric-field kick.

This is used both as (a) the reference / ground-truth trajectory generator
(at a very small time step, 0.001-0.01 gyroperiod), and (b) the baseline
integrator against which all neural schemes and the Boris+Corrector hybrid
are compared at larger, "production" time steps.

Properties (Methods 2.1): symplecticity (phase-volume conservation),
excellent energy stability at large steps, exact conservation of |v| in a
purely magnetic field.
"""

import numpy as np


def boris_step(r, v, t, dt, field, q=-1.0, m=1.0):
    """Advance (r, v) by one Boris step of size dt.

    Parameters
    ----------
    r, v : ndarray, shape (3,) or (N,3)
        Position and velocity.
    t : float
        Current time (for time-dependent fields).
    dt : float
        Time step.
    field : object with .E(r,t) and .B(r,t) methods
    q, m : float
        Charge and mass (normalized units by default: q=-1, m=1 => omega_c=|B|).

    Returns
    -------
    r_new, v_new : ndarray, same shape as input.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    E = field.E(r, t)
    B = field.B(r, t)

    qmdt2 = 0.5 * q * dt / m

    # (2) half electric-field kick
    v_minus = v + qmdt2 * E

    # (3) magnetic rotation via (t, s) vectors
    t_vec = qmdt2 * B
    t_mag2 = np.sum(t_vec * t_vec, axis=-1, keepdims=True)
    s_vec = 2.0 * t_vec / (1.0 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)

    # (4) second half electric-field kick
    v_new = v_plus + qmdt2 * E

    # position update (drift over full step with new velocity, standard leapfrog)
    r_new = r + v_new * dt

    return r_new, v_new


def integrate_boris(r0, v0, t0, dt, n_steps, field, q=-1.0, m=1.0):
    """Integrate n_steps of the Boris algorithm starting from (r0, v0, t0).

    Returns arrays of shape (n_steps+1, 3) for positions/velocities and
    (n_steps+1,) for times.
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)

    rs[0], vs[0], ts[0] = r0, v0, t0
    r, v, t = r0.copy(), v0.copy(), t0
    for i in range(1, n_steps + 1):
        r, v = boris_step(r, v, t, dt, field, q=q, m=m)
        t = t + dt
        rs[i], vs[i], ts[i] = r, v, t

    return rs, vs, ts


class BorisIntegrator:
    """Thin object-oriented wrapper around the functional Boris pusher,
    exposing a uniform `.step()` / `.integrate()` interface shared with the
    neural integrators for direct benchmarking."""

    name = "boris"

    def __init__(self, field, q=-1.0, m=1.0):
        self.field = field
        self.q = q
        self.m = m

    def step(self, r, v, t, dt):
        return boris_step(r, v, t, dt, self.field, q=self.q, m=self.m)

    def integrate(self, r0, v0, t0, dt, n_steps):
        return integrate_boris(r0, v0, t0, dt, n_steps, self.field, q=self.q, m=self.m)
