"""Shared machinery for the spectrum study of Wave 11.

WHAT THIS DIRECTORY ASKS
------------------------
Wave 10 measured, on our own reproduction of the SympMat architecture of
Drimalas et al. 2025, that the trained matrix is symplectic to 7e-16 and yet
carries eigenvalues off the unit circle, up to |lambda| = 1.0098.  The question
this directory answers is whether that is a property of SympMat or a property
of trained structure-preserving maps in general.

The three architectures of Wave 8 are the material: a Hamiltonian neural
network, a G-SympNet and a PINN with a symplecticity penalty, four checkpoints
each, in experiments/external_arch/ckpt.  Nothing here retrains them.  The
checkpoints are loaded exactly as `ea1_train.load_stepper` writes them and
every number in this directory is a function of those frozen weights.

THE THREE OBJECTS THAT MUST NOT BE CONFUSED
-------------------------------------------
SympMat is a *linear* map, so three different quantities coincide there and
one number tells the whole story.  For a nonlinear map they are three
different things, and the whole point of this directory is that they separate:

  (1) rho(J(z))    the spectral radius of the one-step Jacobian at one point.
                   For a linear map this is the growth rate of the state.  For
                   a nonlinear one it is the growth rate of a perturbation
                   held at one frozen point, which no trajectory experiences.
  (2) lambda_max   the leading finite-time Lyapunov exponent of the product of
                   Jacobians along the orbit.  This *is* the growth rate of a
                   perturbation, and for a linear map it equals log rho.
  (3) the growth of the state itself along the run, which is what the error
                   at a long horizon is made of.

A positive (2) with a bounded (3) is not a contradiction: a volume-preserving
map can stretch phase space indefinitely while every orbit stays in a bounded
region.

THE ONE THEOREM THIS DIRECTORY LEANS ON
---------------------------------------
If J is real symplectic, J^T Omega J = Omega, then Omega J Omega^{-1} = J^{-T},
so J is similar to J^{-T} and the spectrum is invariant under lambda -> 1/lambda
as well as under conjugation.  Eigenvalues therefore come in quadruples
{lambda, 1/lambda, conj(lambda), 1/conj(lambda)}, which collapse to reciprocal
pairs on the real axis and to conjugate pairs on the unit circle.

Two consequences, both of which are measured rather than assumed here:

  rho(J) >= 1 for every symplectic J, with equality if and only if the whole
  spectrum lies on the unit circle.  A symplectic map can be neutral; it can
  never be asymptotically stable.

  The same argument applies to any product of symplectic matrices, so the
  Lyapunov exponents of a symplectic map come in +/- pairs and sum to zero.

That is the general statement that survives.  "A trained symplectic
architecture leaves the unit circle" is not implied by it: leaving the circle
is a training outcome, not a structural one, and the measurement below finds
maps that are symplectic and neutral, maps that are not symplectic and neutral,
and maps that are not symplectic and contracting.

SEED LEDGER
-----------
The grid of linearisation points is deterministic.  One random cloud of points
is drawn as a check that the grid is not special, and it is the only draw in
the directory.  Its seeds come from a block nothing else in the bundle
touches -- the highest seed anywhere else is 11,704,999, in
experiments/sympmat/sm_common.py:

    13_000_000 + 1_000 * arch_index + rep

arch_index  0 hnn, 1 sympnet, 2 pinn, 3 boris
rep         0..N_REP-1

`sp_seed()` is the only place a seed is formed, the generator is built once
outside every loop, and the seed is written into the JSON beside the numbers it
produced.

COORDINATES
-----------
Jacobians are taken in the canonical coordinates of experiments/external_arch,
w = (x, y, p_x, p_y) with p = v - A(q,t), which is the convention of
`ea3_cost.symplecticity` and the convention in which SympMat lives.  The
finite-difference stencil is the one that file uses, central differences at
fd = 1e-6, so that the symplecticity residuals here are comparable leaf for
leaf with Section 5.2 of the Wave 9 report; `sp2_spectra.py` asserts the
agreement on the checkpoints that report used.

THE FROZEN FIELD
----------------
The primary setting is tau = infinity: B_z = B_0 exactly, E = 0 exactly, so
the one-step map is autonomous and its spectrum is a time-independent object
about which "(rho)^n after n steps" is a well-posed prediction.  In that field
the exact solution is a closed form,

    r(t) = (cos t, sin t),      v(t) = (-sin t, cos t),

so no reference integrator enters any error reported here.  `C.bz(t, inf)`
returns B_0 and `C.efield(...)` returns exactly zero, so the frozen field
needs no new code path: it is the same functions at a different tau.  The
decaying field tau = 1.2e5 of Wave 9 is measured too, and the difference
between the two is reported rather than assumed away.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, EXP, os.path.join(EXP, "external_arch"),
           os.path.join(EXP, "sympmat")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ea_common as C            # noqa: E402
import ea_arch as A              # noqa: E402
import ea1_train as T            # noqa: E402

from ea_common import check_or_write  # noqa: E402,F401  (re-exported)

CKPT = os.path.join(EXP, "external_arch", "ckpt")
SYMPMAT_JSON = os.path.join(EXP, "sympmat", "sm4_gyrocentre.json")

# --------------------------------------------------------------- declared --
# Everything below is fixed before the first run and is not tuned afterwards.
ARCHS = ("hnn", "sympnet", "pinn")
N_REP = 4                        # the four checkpoints Wave 8 committed
DT = C.DT                        # 0.3, Omega h = 0.3
FD = 1e-6                        # the stencil of ea3_cost.symplecticity
TAU_FROZEN = np.inf              # B_z = B_0, E = 0, exactly
TAU_DECAY = C.TAU_PAPER          # 1.2e5, the field of Wave 9

RHO_GRID = (0.7, 1.0, 1.3)       # gyroradii at which the map is linearised
N_PHASE = 8                      # phases per radius; 24 grid points in all
N_CLOUD = 32                     # random points, one generator, drawn once
CLOUD_RADIUS = 1.5               # the cloud fills the ball of this radius

N_LYAP = 4000                    # steps of the QR run, 191 gyro-orbits
N_LYAP_SHORT = 1000              # the same run a quarter as long, for the
                                 # convergence test described in `lyapunov`
N_LONG = 100_000                 # steps of the long run, 4775 gyro-orbits
LONG_CHECKPOINTS = (100, 400, 1000, 4000, 10_000, 40_000, 100_000)
BLOWUP = 1e3                     # the threshold of ea_arch.rollout
N_FREQ = 2000                    # steps over which omega_h is measured

OMEGA4 = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                   [-1., 0., 0., 0.], [0., -1., 0., 0.]])

ARCH_SLOT = {"hnn": 0, "sympnet": 1, "pinn": 2, "boris": 3}
SEED_BLOCK = 13_000_000


def sp_seed(arch, rep=0):
    """The one place a seed is formed.  See the ledger in the module docstring."""
    a = ARCH_SLOT[arch]
    assert 0 <= a < 8 and 0 <= rep < 1000
    return SEED_BLOCK + 1_000 * a + rep


def load(arch, rep, tau):
    """A frozen Wave 8 checkpoint as a numpy stepper.  Nothing is retrained."""
    return T.load_stepper(os.path.join(CKPT, "%s_r%d.npz" % (arch, rep)), tau)


# ------------------------------------------------------- the frozen field --
def exact_state(t):
    """The closed-form solution at tau = infinity, r0 = (1,0), v0 = (0,1).

    dv/dt = (q/m) v x B with q = -1, m = 1, B = B_0 z gives dv/dt = (-v_y, v_x),
    a counter-clockwise rotation at unit rate, so r(t) = (cos t, sin t) and
    v(t) = (-sin t, cos t): the unit circle, traversed once per 2 pi.
    """
    t = np.asarray(t, float)
    return np.stack([np.cos(t), np.sin(t), -np.sin(t), np.cos(t)])


def to_canon(x, y, vx, vy, t, tau):
    ax, ay = C.vecpot(x, y, t, tau)
    return np.stack([x, y, vx - ax, vy - ay])


def from_canon(w, t, tau):
    ax, ay = C.vecpot(w[0], w[1], t, tau)
    return w[0], w[1], w[2] + ax, w[3] + ay


# ------------------------------------------------------------- Jacobians ---
def jacobian(stepper, w, t, tau, dt=DT, fd=FD):
    """d(w_{n+1})/d(w_n) in canonical coordinates, and the image of w.

    Nine states go through the stepper in one batched call -- the eight legs of
    the central-difference stencil and the point itself -- so a Jacobian costs
    one call rather than nine.  The steppers of ea_arch are written on numpy
    arrays and are exactly as happy with nine columns as with one.
    """
    w = np.asarray(w, float).reshape(4)
    W = np.repeat(w.reshape(4, 1), 9, axis=1)
    for k in range(4):
        W[k, 2 * k] += fd
        W[k, 2 * k + 1] -= fd
    px, py, pvx, pvy = from_canon(W, t, tau)
    nx, ny, nvx, nvy = stepper.step(px, py, pvx, pvy, t, dt)
    out = to_canon(nx, ny, nvx, nvy, t + dt, tau)
    J = (out[:, 0:8:2] - out[:, 1:8:2]) / (2.0 * fd)
    return J, out[:, 8].copy()


def spec(J):
    """Everything this study reads off one Jacobian.

    `reciprocity` is max_i |a_i a_{3-i} - 1| over the sorted moduli: zero for a
    symplectic matrix by the pairing lemma of the module docstring, and a
    measurement of how far from symplectic the map is that is independent of
    the residual below.
    """
    ev = np.linalg.eigvals(J)
    a = np.sort(np.abs(ev))
    order = np.argsort(np.abs(ev))
    ang = np.angle(ev[order])
    R = J.T @ OMEGA4 @ J - OMEGA4
    return {
        "abs": [float(v) for v in a],
        "arg": [float(v) for v in ang],
        "max_abs": float(a[-1]),
        "max_abs_minus_1": float(a[-1] - 1.0),
        "min_abs": float(a[0]),
        "reciprocity": float(max(abs(a[0] * a[3] - 1.0), abs(a[1] * a[2] - 1.0))),
        "symplectic_defect": float(np.sqrt(np.sum(R ** 2))),
        "det_minus_one": float(abs(np.linalg.det(J) - 1.0)),
        "rotation_arg": float(_rotation_arg(ev)),
    }


def _rotation_arg(ev):
    """The argument of the eigenvalue pair that carries the gyration.

    The exact one-step map at tau = infinity has spectrum {1, 1, e^{+ih},
    e^{-ih}}; the pair with the largest |arg| is the gyration and its argument
    is the per-step phase advance of the scheme.  Returned as a positive angle.
    """
    ang = np.abs(np.angle(ev))
    return float(np.max(ang))


def grid_points(rho_grid=RHO_GRID, n_phase=N_PHASE):
    """Canonical states on circles of several gyroradii, at several phases.

    Point (rho, phi) is the state a particle of gyroradius rho would have at
    phase phi on the exact orbit of the frozen field, so rho = 1 lies on the
    trajectory the run actually follows and the other two bracket it.
    """
    pts = []
    for rho in rho_grid:
        for phi in np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False):
            x, y = rho * np.cos(phi), rho * np.sin(phi)
            vx, vy = -rho * np.sin(phi), rho * np.cos(phi)
            pts.append((float(rho), float(phi),
                        [x, y, vx + 0.5 * y, vy - 0.5 * x]))
    return pts


def cloud_points(rng, n=N_CLOUD, radius=CLOUD_RADIUS):
    """A random cloud in the ball of the given radius, in canonical coordinates.

    The grid above is aligned with the exact orbit by construction; if the grid
    and the cloud disagree about the spread of rho(J) then the grid is special
    and the grid statistics are not to be believed.  The generator is built by
    the caller, once, outside every loop.
    """
    z = rng.normal(size=(n, 4))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    r = radius * rng.random(size=(n, 1)) ** 0.25
    return [list(map(float, v)) for v in z * r]


def orbit_points(stepper, tau, n_pts=8, stride=None, dt=DT):
    """Canonical states visited by the run itself, evenly spaced along it."""
    stride = int(round(2.0 * np.pi / dt)) // n_pts if stride is None else stride
    stride = max(1, stride)
    x = np.array([1.0]); y = np.array([0.0])
    vx = np.array([0.0]); vy = np.array([1.0])
    t = 0.0
    pts = []
    for i in range(n_pts * stride):
        if i % stride == 0:
            pts.append((t, [float(v) for v in
                            to_canon(x[0], y[0], vx[0], vy[0], t, tau)]))
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        if not np.isfinite(vx[0]):
            break
    return pts


# -------------------------------------------------------------- Lyapunov ---
def lyapunov(stepper, w0, tau, n=N_LYAP, dt=DT):
    """The finite-time Lyapunov spectrum of the map along its own orbit.

    Y is carried through the linearised map and re-orthonormalised by a QR
    factorisation at every step, with the sign of each diagonal entry of R
    absorbed into Q so that the diagonal is positive and its logarithm is the
    per-step stretch of that direction.  The exponents are per step, not per
    unit time, so that they compare directly with log rho(J).

    For a symplectic map the four exponents must come in +/- pairs and sum to
    zero; `sum` is printed for exactly that check and is not a fitted quantity.

    THE FLOOR OF THIS ESTIMATOR, AND WHY IT IS NOT ZERO
    ---------------------------------------------------
    A finite-time exponent divides an accumulated logarithm by the number of
    steps, so any *bounded* transient in ||J^n|| appears as an exponent of
    order log(transient)/n and decays like 1/n rather than converging to zero.
    The Boris scheme in the frozen field is the cleanest possible case of this:
    its one-step matrix is constant, diagonalisable, with all four eigenvalues
    on the unit circle, so its true exponents are exactly zero, and ||M^n||
    merely wanders between 1.12 and 2.01 for ever.  This function nonetheless
    returns 1.2e-4 per step for it at n = 4000, which is log(1.6)/4000.

    That number is the resolution of the instrument, and `sp2_spectra.py`
    measures it rather than assuming it.  The way to tell a real exponent from
    the floor is to run twice at different n: a real exponent is constant in n,
    a transient has n * lambda constant instead.  Both lengths are computed for
    every cell and both are written into the JSON.
    """
    w = np.asarray(w0, float).copy()
    Y = np.eye(4)
    acc = np.zeros(4)
    t = 0.0
    n_done = 0
    for _ in range(n):
        J, w = jacobian(stepper, w, t, tau, dt)
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite(w)):
            break
        Y = J @ Y
        Q, R = np.linalg.qr(Y)
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        Q = Q * d
        R = d[:, None] * R
        dg = np.abs(np.diag(R))
        if np.any(dg <= 0) or not np.all(np.isfinite(dg)):
            break
        acc += np.log(dg)
        Y = Q
        t += dt
        n_done += 1
    if n_done == 0:
        return {"n_steps": 0}
    e = np.sort(acc / n_done)[::-1]
    return {"n_steps": n_done,
            "exponents_per_step": [float(v) for v in e],
            "lambda_max_per_step": float(e[0]),
            "sum_per_step": float(np.sum(e)),
            "pairing_residual": float(max(abs(e[0] + e[3]), abs(e[1] + e[2])))}


# -------------------------------------------------------------- long runs --
def long_run(stepper, tau, n=N_LONG, dt=DT, checkpoints=LONG_CHECKPOINTS,
             blowup=BLOWUP):
    """One long run in the frozen field, scored against the closed form.

    Three channels, because the three ask different questions:

      amplitude  |w| in canonical coordinates.  On the exact solution it is
                 exactly sqrt(5)/2 for every t, so its running maximum is the
                 channel in which an eigenvalue off the unit circle would show
                 itself, and the only one in which "growth by (rho)^n" is a
                 statement about the run.
      energy     the relative error of |v|^2/2 against its exact value, the
                 channel Section 7 of the manuscript scores.
      position   |r_n - r(t_n)|, which saturates near the diameter of the orbit
                 once the phase has decorrelated and is reported so that the
                 saturation is on the record rather than inferred.

    `blowup_step` is the first step at which the amplitude passes `blowup`, or
    None.  A run that blows up is stopped there.
    """
    x = np.array([1.0]); y = np.array([0.0])
    vx = np.array([0.0]); vy = np.array([1.0])
    t = 0.0
    a0 = float(np.linalg.norm(to_canon(x[0], y[0], vx[0], vy[0], 0.0, tau)))
    amp_max = a0
    rows = {}
    blow = None
    # a log-spaced trace of the running maximum, for the growth-rate fit
    trace_n, trace_a = [], []
    nxt = 1
    for step in range(1, n + 1):
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        w = to_canon(x[0], y[0], vx[0], vy[0], t, tau)
        a = float(np.linalg.norm(w))
        if not np.isfinite(a) or a > blowup:
            blow = step
            break
        amp_max = max(amp_max, a)
        if step >= nxt:
            trace_n.append(step)
            trace_a.append(amp_max)
            nxt = max(step + 1, int(step * 1.05))
        if step in checkpoints:
            ex = exact_state(t)
            e = 0.5 * (vx[0] ** 2 + vy[0] ** 2)
            rows[str(step)] = {
                "amplitude": a,
                "amplitude_running_max": amp_max,
                "energy_rel_err": float(abs(e - 0.5) / 0.5),
                "position_err": float(np.hypot(x[0] - ex[0], y[0] - ex[1])),
            }
    out = {"n_steps_declared": n, "blowup_step": blow,
           "amplitude_0": a0, "amplitude_running_max": amp_max,
           "checkpoints": rows}
    out["growth_per_step_measured"] = _growth_rate(trace_n, trace_a)
    return out


def _growth_rate(ns, amps):
    """log-amplitude growth per step, fitted over the last decade of the run.

    A bounded run returns a rate consistent with zero; an exponentially growing
    one returns log(rho).  The fit is on the running maximum so that the
    gyration itself, which modulates |w| by a bounded factor within every
    orbit, cannot masquerade as growth.
    """
    ns = np.asarray(ns, float)
    amps = np.asarray(amps, float)
    m = (ns >= ns[-1] / 10.0) & (amps > 0) & np.isfinite(amps)
    if m.sum() < 10:
        return float("nan")
    return float(np.polyfit(ns[m], np.log(amps[m]), 1)[0])


def measured_frequency(stepper, tau, n=N_FREQ, dt=DT):
    """The per-step phase advance of the scheme, measured on its own run.

    This is `ea_arch.measure_scheme_frequency` with the window declared here
    and the answer returned per step rather than per unit time, so that it
    compares directly with the argument of the gyration eigenvalue.  On the
    Boris scheme in the frozen field it must return 2 arctan(Omega h / 2)
    exactly, and `sp1_calibration.py` checks that before this function is
    believed anywhere else.
    """
    x = np.array([1.0]); y = np.array([0.0])
    vx = np.array([0.0]); vy = np.array([1.0])
    t = 0.0
    ph = np.empty(n)
    for i in range(n):
        x, y, vx, vy = stepper.step(x, y, vx, vy, t, dt)
        t += dt
        ph[i] = np.angle(vx[0] + 1j * vy[0])
        if not np.isfinite(ph[i]) or np.hypot(vx[0], vy[0]) > BLOWUP:
            return float("nan")
    u = np.unwrap(ph)
    return float((u[-1] - u[0]) / (n - 1))


# ------------------------------------------- the analytically known maps ---
def canonicalise_matrix(M, b=1.0):
    """A one-step matrix written in (x, y, v_x, v_y) rewritten in (x, y, p_x, p_y).

    In the frozen field the change of variables is the constant linear map
    w = T z with T = [[I, 0], [G, I]] and G = [[0, b/2], [-b/2, 0]], because
    p = v - A and A = (b/2)(-y, x).  The one-step matrix therefore becomes
    T M T^{-1}, a similarity, so the spectrum is the same in both frames and
    only the matrices differ.  In the *decaying* field T is taken at t and at
    t + h and the map is no longer a similarity; the difference is of order
    h/tau = 2.5e-6 at tau = 1.2e5 and is measured rather than assumed in
    sp2_spectra.py.
    """
    G = np.array([[0.0, 0.5 * b], [-0.5 * b, 0.0]])
    T = np.eye(4)
    T[2:, :2] = G
    Ti = np.eye(4)
    Ti[2:, :2] = -G
    return T @ M @ Ti


def boris_matrix(dt=DT, b=1.0):
    """The Boris one-step map in the frozen field, in closed form.

    From `ea_common.boris_plane` with E = 0 and t_z = q b dt / (2m):

        v_{n+1} = R v_n,   R = [[1 - t s,  s], [-s, 1 - t s]],  s = 2t/(1+t^2),
        r_{n+1} = r_n + h v_{n+1},

    and 1 - t s = (1 - t^2)/(1 + t^2), s = -2 t /(1 + t^2) in the entries above
    with t = t_z < 0 for q = -1, so R is the rotation by

        theta = 2 arctan(|q| b h / 2 m),

    which is the standard Boris angle.  The state matrix is block triangular,

        M = [[I, h R], [0, R]],

    so its spectrum is {1, 1, e^{+i theta}, e^{-i theta}}: on the unit circle
    exactly, for every step size, although the map is not symplectic.
    """
    tz = 0.5 * C.Q * dt * b / C.M
    sz = 2.0 * tz / (1.0 + tz * tz)
    R = np.array([[1.0 - tz * sz, sz], [-sz, 1.0 - tz * sz]])
    M = np.zeros((4, 4))
    M[:2, :2] = np.eye(2)
    M[:2, 2:] = dt * R
    M[2:, 2:] = R
    return M


def boris_angle(dt=DT, b=1.0):
    return float(2.0 * np.arctan(abs(C.Q) * b * dt / (2.0 * C.M)))


def exact_matrix(dt=DT, b=1.0):
    """The exact flow map over one step in the frozen field.

    v(t+h) = R(bh) v(t) with R the counter-clockwise rotation, and
    r(t+h) = r(t) + S(bh) v(t)/b with S = [[sin, -(1-cos)], [1-cos, sin]].
    Spectrum {1, 1, e^{+ibh}, e^{-ibh}}.
    """
    th = b * dt
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    S = np.array([[s, -(1.0 - c)], [1.0 - c, s]]) / b
    M = np.zeros((4, 4))
    M[:2, :2] = np.eye(2)
    M[:2, 2:] = S
    M[2:, 2:] = R
    return M


def rk4_matrix(dt=DT, b=1.0):
    """RK4 applied to the exact field of the frozen problem.

    The control for the HNN: it is what that architecture would be if the
    learned Hamiltonian were exact, since the structure there sits in the field
    and the step is taken by RK4.  The field is linear, dz/dt = F z with

        F = [[0, I], [0, bW]],   W = [[0, -1], [1, 0]],

    so RK4 is the matrix polynomial I + hF + (hF)^2/2 + (hF)^3/6 + (hF)^4/24 and
    its eigenvalues are R(0), R(0), R(+ibh), R(-ibh) with R the RK4 stability
    function.  |R(iy)|^2 = 1 - y^6/72 + y^8/576 < 1 for 0 < y < 2 sqrt(2), so
    RK4 on an exact Hamiltonian field is *contracting*, by 5.0e-6 per step at
    y = 0.3.  Any architecture that wraps RK4 around a learned field inherits
    that, and it is the opposite sign from the SympMat observation.
    """
    F = np.zeros((4, 4))
    F[0, 2] = F[1, 3] = 1.0
    F[2, 3] = -b
    F[3, 2] = b
    Z = dt * F
    M = np.eye(4)
    P = np.eye(4)
    for k in (1, 2, 3, 4):
        P = P @ Z / k
        M = M + P
    return M


def rk4_stability_modulus(y):
    """|R(iy)| for the classical RK4 stability function, in closed form."""
    return float(np.sqrt(1.0 - y ** 6 / 72.0 + y ** 8 / 576.0))
