"""The three external architectures, written from their published descriptions.

Nothing here is vendored from a third-party repository.  Each class carries the
equations it implements and the paper they come from.  All three are trained in
`ea1_train.py` and evaluated as numpy steppers here, so that a rollout of 10^5
steps does not pay torch's per-call overhead.

The three are one-step maps on the same state at integer times,
(x, y, v_x, v_y) at t_n -> the same at t_{n+1}, which is the convention of
experiments/classical/schemes.py.  Energy is then defined identically for every
scheme in the comparison and the readout probe means the same thing for all of
them.
"""
import numpy as np

import ea_common as C


# =========================================================================
#  HNN -- Greydanus, Dzamba & Yosinski, NeurIPS 32 (2019), arXiv:1906.01563
# =========================================================================
class HNNStepper:
    """A network holds a scalar H; the field is its symplectic gradient.

        dq/dt = +dH/dp,      dp/dt = -dH/dq

    which is their Eq. (2).  The network sees (q_x, q_y, p_x, p_y, B_z) and
    returns one number.  The true Hamiltonian |p + A(q,t)|^2/2 depends on time
    only through B_z, so it lies inside the network's hypothesis class and the
    architecture is not being asked for something it cannot express.

    The learned field is integrated with the classical fourth-order
    Runge--Kutta scheme, which is what Greydanus et al. use for their rollouts.
    RK4 is not symplectic; the structure in this architecture is in the field,
    not in the integrator, and that is a property of the architecture rather
    than a choice made here.
    """

    name = "hnn"
    n_in = 5

    def __init__(self, net, tau):
        self.net = net                    # C.NumpyMLP, scalar output
        self.tau = tau

    # --- learned vector field on the canonical state -------------------
    def field(self, qx, qy, px, py, t):
        s = np.broadcast_to(C.bz(t, self.tau), qx.shape)
        X = np.stack([qx, qy, px, py, s])
        _, g = self.net.scalar_and_grad(X)
        # g = (dH/dqx, dH/dqy, dH/dpx, dH/dpy, dH/dBz)
        return g[2], g[3], -g[0], -g[1]

    def step(self, x, y, vx, vy, t, dt):
        tau = self.tau
        ax, ay = C.vecpot(x, y, t, tau)
        qx, qy, px, py = x, y, vx - ax, vy - ay
        h = dt
        k1 = self.field(qx, qy, px, py, t)
        k2 = self.field(qx + 0.5 * h * k1[0], qy + 0.5 * h * k1[1],
                        px + 0.5 * h * k1[2], py + 0.5 * h * k1[3], t + 0.5 * h)
        k3 = self.field(qx + 0.5 * h * k2[0], qy + 0.5 * h * k2[1],
                        px + 0.5 * h * k2[2], py + 0.5 * h * k2[3], t + 0.5 * h)
        k4 = self.field(qx + h * k3[0], qy + h * k3[1],
                        px + h * k3[2], py + h * k3[3], t + h)
        qx = qx + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        qy = qy + (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        px = px + (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        py = py + (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        ax2, ay2 = C.vecpot(qx, qy, t + h, tau)
        return qx, qy, px + ax2, py + ay2

    def flops_per_step(self):
        w = self.net.widths
        one = C.mlp_forward_flops(w) + C.mlp_backward_flops(w)
        # four RK4 stages, the stage combination (8 mults + 12 adds per
        # component, 4 components), and two vector-potential conversions
        # (one exp = 20, six arithmetic each)
        return 4 * one + 4 * 20 + 2 * 26


# =========================================================================
#  SympNet -- Jin, Zhang, Zhu, Tang & Karniadakis, Neural Netw. 132 (2020) 166
# =========================================================================
class SympNetStepper:
    """G-SympNet: a composition of gradient modules.

        up   module:  p <- p + K^T diag(a) sigma(K q + b),   q unchanged
        low  module:  q <- q + K^T diag(a) sigma(K p + b),   p unchanged

    which is their Eq. (11).  Each module is the gradient of a scalar in one
    half of the phase space, so its Jacobian is unit upper (or lower)
    triangular and the map is symplectic at every value of the weights, before
    any training.  That is the property the second and fourth probes are aimed
    at.

    Why G and not LA.  Jin et al. prove both families universal, and they show
    (their Section 4) that G-SympNets reach a given accuracy with fewer layers
    on systems whose flow is not close to linear.  The flow here is a rotation
    modulated by a decaying field, so an LA-SympNet's linear modules would
    carry most of the map and the gradient modules would be fitting a
    correction -- which is our own construction, not theirs, and would make the
    comparison a comparison with ourselves.  G-SympNet keeps the two
    constructions apart.

    Departure from the published form, stated because it is a departure.  Jin
    et al. learn the flow map of an autonomous system.  The field of Section 2
    decays, so the map depends on time.  Here the module parameters a and b are
    affine in the scalar B_z(t_{n+1/2}),

        a(s) = a_0 + s a_1,     b(s) = b_0 + s b_1 ,

    and s is not a canonical variable, so at each fixed step the module is
    still exactly the gradient shear above and the map is still exactly
    symplectic.  The conditioning is ours; the architecture is theirs.
    """

    name = "sympnet"

    def __init__(self, modules, tau):
        # modules: list of (kind, K, a0, a1, b0, b1), kind in {"up","low"}
        self.modules = modules
        self.tau = tau

    def step(self, x, y, vx, vy, t, dt):
        tau = self.tau
        ax, ay = C.vecpot(x, y, t, tau)
        q = np.stack([x, y])
        p = np.stack([vx - ax, vy - ay])
        s = float(C.bz(t + 0.5 * dt, tau))
        for kind, K, a0, a1, b0, b1 in self.modules:
            a = (a0 + s * a1).reshape(-1, 1)
            b = (b0 + s * b1).reshape(-1, 1)
            if kind == "up":
                p = p + K.T @ (a * np.tanh(K @ q + b))
            else:
                q = q + K.T @ (a * np.tanh(K @ p + b))
        ax2, ay2 = C.vecpot(q[0], q[1], t + dt, tau)
        return q[0], q[1], p[0] + ax2, p[1] + ay2

    def flops_per_step(self):
        f = 0
        for _kind, K, a0, _a1, _b0, _b1 in self.modules:
            w = K.shape[0]
            f += 2 * w * 2          # K x
            f += 2 * w              # a(s), b(s) affine in s: mult + add each
            f += w                  # + b
            f += C.FLOP_TRANSCENDENTAL * w
            f += w                  # a * sigma
            f += 2 * w * 2          # K^T (.)
            f += 2                  # add to the half-state
        f += 20 + 2 * 26            # one exp for s, two potential conversions
        return f


# =========================================================================
#  PINN-symplectic -- Raissi, Perdikaris & Karniadakis,
#  J. Comput. Phys. 378 (2019) 686, with a symplecticity penalty
# =========================================================================
class PINNStepper:
    """A dense network predicts the one-step increment of the physical state.

        (x, y, v_x, v_y, B_z(t))  ->  (dx, dy, dv_x, dv_y)

    Training minimises the midpoint residual of the equations of motion,

        R_r = (r' - r)/h - v_mid,
        R_v = (v' - v)/h - (q/m)[E(r_mid, t_mid) + v_mid x B(t_mid)] ,

    which is the physics-informed loss of Raissi et al. applied to the map
    rather than to a field, together with

        ||J^T Omega J - Omega||_F^2

    on the Jacobian of the map in canonical coordinates.  No reference
    trajectory enters the loss; the states at which the residual is evaluated
    are the same states the other two architectures are trained on.

    One consequence of that choice is worth naming before any number is
    quoted: the exact minimiser of the midpoint residual is the implicit
    midpoint rule.  The architecture is therefore a learned approximation to a
    classical scheme that Section 7 already prices, and the comparison in
    `ea3_cost.py` is between an approximation and the thing it approximates.
    """

    name = "pinn"
    n_in = 5

    def __init__(self, net, tau):
        self.net = net
        self.tau = tau

    def step(self, x, y, vx, vy, t, dt):
        s = np.broadcast_to(C.bz(t, self.tau), np.shape(x))
        X = np.stack([x, y, vx, vy, s])
        d = self.net.forward(X)
        return x + d[0], y + d[1], vx + d[2], vy + d[3]

    def flops_per_step(self):
        return C.mlp_forward_flops(self.net.widths) + 20 + 4


# =========================================================================
#  the Boris reference stepper, for the probes' control column
# =========================================================================
class BorisStepper:
    name = "boris"

    def __init__(self, tau):
        self.tau = tau

    def step(self, x, y, vx, vy, t, dt):
        return C.boris_plane(x, y, vx, vy, t, self.tau, dt)

    def flops_per_step(self):
        # the plane costs less than the 113 flops Section 9 quotes in three
        # dimensions; the three-dimensional figure is the one used for cost
        return 113


class ProjectedStepper:
    """Any one-step map with the projection of Section 4.3 placed after it.

    The speed after the step is set to the speed a Boris step from the same
    pre-step state would have produced, which makes the map's own correction
    energy-neutral in the velocity by construction.  That is the property
    condition (vi) excludes, and the fourth probe is known to return the wrong
    verdict for schemes that have it.  Wrapping the three external
    architectures in it is how the known failure is tested on objects other
    than our own.

    The projection is one-sided, as the corrector ships it; the symmetric form
    of Hairer differs at second order in the multiplier and the campaign
    measured the two at 0.9772 and 0.9757 on the same run.
    """

    def __init__(self, base, tau):
        self.base = base
        self.tau = tau
        self.name = base.name + "+proj"

    def step(self, x, y, vx, vy, t, dt):
        _, _, bvx, bvy = C.boris_plane(x, y, vx, vy, t, self.tau, dt)
        target = np.hypot(bvx, bvy)
        nx, ny, nvx, nvy = self.base.step(x, y, vx, vy, t, dt)
        n = np.hypot(nvx, nvy)
        s = np.where(n > 0, target / np.where(n > 0, n, 1.0), 1.0)
        return nx, ny, nvx * s, nvy * s

    def flops_per_step(self):
        return self.base.flops_per_step() + 113 + 12


# =========================================================================
#  batched rollout with the hooks the four probes need
# =========================================================================
def rollout(stepper, n_steps, dt=C.DT, tau=None, r0=None, v0=None, nb=1,
            perturb=None, record_defect=False, record_signed=False,
            boris_compare=False, half_step_start=False, n_env=4000,
            blowup=1e3):
    """Integrate `nb` copies of the same problem side by side.

    perturb(vx, vy, n, t_next) -> (vx, vy) is applied after the step, which is
    where a defect enters the map.  record_defect keeps the complex velocity
    before and after the perturbation so that the fourth probe can demodulate.

    half_step_start moves the initial position half a step along v_0, which is
    one of the two operations of the third probe.

    Returns a dict.  `signed` is the reported relative energy error against the
    exact law E_0 exp(-t/tau) of Section 9, subsampled onto a log-friendly
    grid; `env` and `t_env` are the running maximum and its times; `alive` is
    the step at which each member left the domain, or n_steps.
    """
    tau = C.TAU_PAPER if tau is None else tau
    r0 = C.R0 if r0 is None else np.asarray(r0, float)
    v0 = C.V0 if v0 is None else np.asarray(v0, float)

    x = np.full(nb, r0[0]); y = np.full(nb, r0[1])
    vx = np.full(nb, v0[0]); vy = np.full(nb, v0[1])
    if half_step_start:
        nrm = np.hypot(v0[0], v0[1])
        x = x + 0.5 * dt * v0[0] / nrm
        y = y + 0.5 * dt * v0[1] / nrm
    e0 = float(C.energy(v0[0], v0[1]))

    stride = max(1, n_steps // n_env)
    n_out = (n_steps + stride - 1) // stride
    sig = np.empty((n_out, nb))
    t_out = np.empty(n_out)
    run = np.zeros(nb)
    alive = np.full(nb, n_steps, dtype=np.int64)
    live = np.ones(nb, dtype=bool)

    keep_sig = record_defect or record_signed or boris_compare
    kap = np.empty((n_steps, nb), complex) if record_defect else None
    zb = np.empty((n_steps, nb), complex) if record_defect else None
    zbo = np.empty((n_steps, nb), complex) if boris_compare else None
    sgn_full = np.empty((n_steps, nb)) if keep_sig else None

    t = 0.0
    j = 0
    for n in range(n_steps):
        if boris_compare:
            _bx, _by, _bvx, _bvy = C.boris_plane(x, y, vx, vy, t, tau, dt)
            zbo[n] = _bvx + 1j * _bvy
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        if record_defect:
            zb[n] = vx + 1j * vy
        if perturb is not None:
            vx, vy = perturb(vx, vy, n, t)
        if record_defect:
            kap[n] = (vx + 1j * vy) - zb[n]
        bad = ~np.isfinite(vx) | ~np.isfinite(vy) | (np.hypot(vx, vy) > blowup) \
            | (np.hypot(x, y) > blowup)
        newly = live & bad
        if newly.any():
            alive[newly] = n
            live &= ~bad
        if not live.all():
            # a member that left the domain is pinned to a benign state so
            # that the batch keeps advancing; its diagnostics are masked out
            x = np.where(live, x, 1.0); y = np.where(live, y, 0.0)
            vx = np.where(live, vx, 0.0); vy = np.where(live, vy, 1.0)
        d = np.abs(C.energy(vx, vy) - C.e_phys(t, tau, e0)) / e0
        d = np.where(live, d, np.nan)
        if keep_sig:
            sgn_full[n] = np.where(
                live, (C.energy(vx, vy) - C.e_phys(t, tau, e0)) / e0, np.nan)
        run = np.fmax(run, d)
        if (n + 1) % stride == 0 or n == n_steps - 1:
            sig[j] = run
            t_out[j] = t
            run = np.zeros(nb)
            j += 1
    sig = sig[:j]; t_out = t_out[:j]
    env = np.fmax.accumulate(sig, axis=0)

    out = {"t_env": t_out, "env": env, "alive": alive, "n_steps": n_steps,
           "dt": dt, "tau": tau, "e0": e0,
           "final_state": (x, y, vx, vy)}
    if record_defect:
        out["kappa"] = kap
        out["z_prekick"] = zb
    if boris_compare:
        out["z_boris"] = zbo
    if keep_sig:
        out["signed"] = sgn_full
    return out


def response_envelope(signed, alive, dt=C.DT, n_env=4000):
    """Envelope of a driven series with its undriven twin removed.

    Column 0 of `signed` is the twin.  Section 6 reads the exponent off the
    reported energy error of the driven run itself, which presumes the scheme's
    own error to be below the response.  Where that presumption fails, the
    response has to be isolated, and this is where it is done.
    """
    m = int(alive)
    base = signed[:m, 0]
    out_t, out_env = None, []
    stride = max(1, m // n_env)
    starts = np.arange(0, m, stride)
    for k in range(1, signed.shape[1]):
        dev = np.abs(signed[:m, k] - base)
        run = np.fmax.reduceat(dev, starts)
        env = np.fmax.accumulate(run)
        out_env.append(env)
        if out_t is None:
            ends = np.append(starts[1:], m)
            out_t = ends * dt
    return out_t, np.stack(out_env, axis=1)


def envelope_exponents(t_env, env, decades=2.0):
    """The fit of Section 6 plus the local half-decade slopes of Section 4.4."""
    out = []
    for b in range(env.shape[1]):
        e = env[:, b]
        m = (t_env > t_env[-1] / 10.0 ** decades) & (e > 0) & np.isfinite(e)
        p = float(np.polyfit(np.log10(t_env[m]), np.log10(e[m]), 1)[0]) \
            if m.sum() >= 10 else float("nan")
        out.append({"p_fit_last2dec": p,
                    "half_decade_slopes": C.half_decade_slopes(t_env, e),
                    "env_final": float(e[-1]) if np.isfinite(e[-1]) else float("nan")})
    return out


def measure_scheme_frequency(stepper, n_steps=500, dt=C.DT):
    """omega_h of a scheme, measured rather than assumed.

    For the Boris scheme the closed form is 2 arctan(h Omega/2)/h.  A learned
    map has no closed form, so the second probe has to take the numerical
    gyration frequency from the run itself: the mean per-step advance of the
    unwrapped phase of v_x + i v_y over a window short enough that the field
    has not decayed.  On the Boris scheme at tau = 1.2e8 this returns the
    closed form to 6.1e-7 in relative terms over 500 steps, the residue being
    the decay of the field over the window, and `ea2_probes.py` checks that
    before it believes the same measurement on a learned map.
    """
    x = np.array([C.R0[0]]); y = np.array([C.R0[1]])
    vx = np.array([C.V0[0]]); vy = np.array([C.V0[1]])
    ph = np.empty(n_steps)
    t = 0.0
    for n in range(n_steps):
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        ph[n] = np.angle(vx[0] + 1j * vy[0])
        if not np.isfinite(ph[n]) or np.hypot(vx[0], vy[0]) > 1e3:
            return float("nan")
    u = np.unwrap(ph)
    return float((u[-1] - u[0]) / ((n_steps - 1) * dt))
