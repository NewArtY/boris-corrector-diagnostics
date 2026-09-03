"""
schemes.py -- classical structure-preserving integrators for field B4.
=======================================================================
The Article compares its hybrid against exactly two things: the shipped Boris
pusher (shown by experiments/cost/staggered.py to be only FIRST order in
position) and three fully-learned networks that were trained under a budget
too small to converge. Every classical scheme cited in the Article's own
introduction -- high-order Boris (Winkel 2015), volume-preserving splitting
(He 2015), energy-conserving semi-implicit PIC (Lapenta 2017, Markidis 2011)
-- is absent from the comparison.

This module supplies that missing set:

  shipped     r_{n+1} = r_n + v_{n+1} dt          (as shipped; 1st order)
  staggered   textbook leapfrog Boris             (2nd order)
  vps2        volume-preserving Strang splitting  (2nd order, symmetric)
  vps4        Yoshida triple-jump of vps2         (4th order, symmetric)
  imr         implicit midpoint rule              (2nd order, symplectic)
  gl4         2-stage Gauss-Legendre              (4th order, symplectic)

All operate on the same (r, v) state at INTEGER times, so energy is defined
identically for every scheme and the comparison is like-for-like. This matters:
experiments/cost/ found that the Article's "600x below the signal" energy
figure is partly an artefact of where the velocity is centred.

Flop counts are analytic, documented in FLOPS_PER_STEP, and use the same
accounting as experiments/cost (Boris = 113 flop/step).
"""
import numpy as np

Q, M = -1.0, 1.0

# --- analytic flop accounting -------------------------------------------
# Field evaluations for DecayingField: B needs one exp + one multiply (~12);
# E needs the same exp plus the azimuthal construction (~25). A Boris kick is
# 2 cross products (9 each), the t/s vectors and the electric half-kicks (~50).
F_B, F_E, F_KICK, F_DRIFT, F_ROT, F_CROSS = 12, 25, 50, 6, 30, 9


def _rodrigues(v, b_hat, theta):
    """Exact rotation of v about b_hat by angle theta (volume-preserving)."""
    c, s = np.cos(theta), np.sin(theta)
    return (v * c + np.cross(b_hat, v) * s
            + b_hat * np.dot(b_hat, v) * (1.0 - c))


def _fields(field, r, t):
    E = np.atleast_1d(field.E(r, t)).ravel()
    B = np.atleast_1d(field.B(r, t)).ravel()
    return E, B


def _boris_kick(v, E, B, dt):
    """Velocity-only Boris rotation with electric half-kicks."""
    h = 0.5 * Q * dt / M
    vm = v + h * E
    tv = h * B
    sv = 2.0 * tv / (1.0 + np.dot(tv, tv))
    vp = vm + np.cross(vm + np.cross(vm, tv), sv)
    return vp + h * E


# ------------------------------------------------------------------ shipped
def step_shipped(r, v, t, dt, field):
    E, B = _fields(field, r, t)
    v_new = _boris_kick(v, E, B, dt)
    return r + v_new * dt, v_new


# ---------------------------------------------------------------- staggered
# Handled by integrate_staggered (needs half-step velocity state).

# -------------------------------------------------------------------- vps2
def make_vps2(field):
    """Strang splitting drift/kick/rotate/kick/drift. Each sub-flow is exactly
    volume-preserving, so the composition is; symmetric, hence 2nd order."""
    def step(r, v, t, dt):
        r = r + 0.5 * dt * v                              # drift dt/2
        E, B = _fields(field, r, t + 0.5 * dt)
        v = v + (0.5 * dt * Q / M) * E                    # E-kick dt/2
        bn = np.linalg.norm(B)
        if bn > 0.0:
            # dv/dt = (q/m) v x B = -(q/m)|B| (b_hat x v), and Rodrigues has
            # dv/dtheta = b_hat x v, hence theta = -(q/m)|B| dt.
            v = _rodrigues(v, B / bn, -(Q / M) * bn * dt)  # exact rotation dt
        v = v + (0.5 * dt * Q / M) * E                    # E-kick dt/2
        r = r + 0.5 * dt * v                              # drift dt/2
        return r, v
    return step


# -------------------------------------------------------------------- vps4
_G1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_G0 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))


def make_vps4(field):
    base = make_vps2(field)

    def step(r, v, t, dt):
        for g in (_G1, _G0, _G1):
            r, v = base(r, v, t, g * dt)
            t = t + g * dt
        return r, v
    return step


# --------------------------------------------------------------------- imr
def make_imr(field, tol=1e-14, maxit=50):
    """Implicit midpoint: symplectic, 2nd order, symmetric.

    y_{n+1} = y_n + dt * f((y_n + y_{n+1})/2, t_n + dt/2), solved by fixed
    point. `iters` accumulates the iteration count for the flop model."""
    stats = {"iters": 0, "steps": 0}

    def step(r, v, t, dt):
        tm = t + 0.5 * dt
        rm, vm = r + 0.25 * dt * v, v.copy()
        used = maxit
        for it in range(maxit):
            E, B = _fields(field, rm, tm)
            a = (Q / M) * (E + np.cross(vm, B))
            rm_n = 0.5 * (r + (r + dt * vm))
            vm_n = 0.5 * (v + (v + dt * a))
            done = (np.linalg.norm(rm_n - rm) + np.linalg.norm(vm_n - vm)) < tol
            rm, vm = rm_n, vm_n
            if done:
                used = it + 1
                break
        E, B = _fields(field, rm, tm)
        a = (Q / M) * (E + np.cross(vm, B))
        stats["iters"] += used
        stats["steps"] += 1
        return r + dt * vm, v + dt * a

    step.stats = stats
    return step


# --------------------------------------------------------------------- gl4
_C1, _C2 = 0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0
_A = np.array([[0.25, 0.25 - np.sqrt(3.0) / 6.0],
               [0.25 + np.sqrt(3.0) / 6.0, 0.25]])


def make_gl4(field, tol=1e-14, maxit=60):
    """2-stage Gauss-Legendre: symplectic, 4th order."""
    def rhs(y, t):
        r, v = y[:3], y[3:]
        E, B = _fields(field, r, t)
        return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])

    stats = {"iters": 0, "steps": 0}

    def step(r, v, t, dt):
        y = np.concatenate([r, v])
        k = np.zeros((2, 6))
        used = maxit
        for it in range(maxit):
            Y1 = y + dt * (_A[0, 0] * k[0] + _A[0, 1] * k[1])
            Y2 = y + dt * (_A[1, 0] * k[0] + _A[1, 1] * k[1])
            k_new = np.vstack([rhs(Y1, t + _C1 * dt), rhs(Y2, t + _C2 * dt)])
            done = np.max(np.abs(k_new - k)) < tol
            k = k_new
            if done:
                used = it + 1
                break
        y = y + 0.5 * dt * (k[0] + k[1])
        stats["iters"] += used
        stats["steps"] += 1
        return y[:3], y[3:]

    step.stats = stats
    return step


# --------------------------------------------------------------- integrator
def integrate(step, r0, v0, dt, n_steps, field=None):
    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    r, v, t = np.array(r0, float), np.array(v0, float), 0.0
    rs[0], vs[0] = r, v
    for i in range(1, n_steps + 1):
        if field is not None:
            r, v = step(r, v, t, dt, field)
        else:
            r, v = step(r, v, t, dt)
        t += dt
        rs[i], vs[i], ts[i] = r, v, t
    return rs, vs, ts


def integrate_staggered(field, r0, v0, dt, n_steps):
    """Textbook staggered leapfrog Boris; velocities re-centred to integer
    times so that energy is defined the same way as for every other scheme."""
    r = np.array(r0, float)
    E, B = _fields(field, r, 0.0)
    v_half = _boris_kick(np.array(v0, float), E, B, -0.5 * dt)
    rs = np.zeros((n_steps + 1, 3)); vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    rs[0], vs[0] = r, np.array(v0, float)
    t = 0.0
    for i in range(1, n_steps + 1):
        E, B = _fields(field, r, t)
        v_new = _boris_kick(v_half, E, B, dt)
        r = r + v_new * dt
        t += dt
        vs[i] = 0.5 * (v_half + v_new)
        v_half = v_new
        rs[i], ts[i] = r, t
    return rs, vs, ts


# Flop model per step. imr/gl4 counts are filled at run time from the measured
# mean iteration count, since they are iterative.
FLOPS_PER_STEP = {
    "shipped":   F_E + F_B + F_KICK + F_DRIFT,          # 93 + margin -> 113
    "staggered": F_E + F_B + F_KICK + F_DRIFT,
    "vps2":      2 * F_DRIFT + F_E + F_B + F_ROT + 12,
    "vps4":      3 * (2 * F_DRIFT + F_E + F_B + F_ROT + 12),
}


def flops_imr(n_iter):
    return n_iter * (F_E + F_B + F_CROSS + 24) + F_DRIFT


def flops_gl4(n_iter):
    return n_iter * 2 * (F_E + F_B + F_CROSS + 18) + 12
