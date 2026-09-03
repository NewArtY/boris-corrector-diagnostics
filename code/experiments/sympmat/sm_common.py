"""Shared machinery for the W10 reproduction of Drimalas et al., Phys. Plasmas
32, 103901 (2025), DOI 10.1063/5.0283551.

WHAT THIS DIRECTORY IS FOR
--------------------------
Their abstract claims that SympMat "outperforms the traditional Boris particle
pusher down to the sub-gyroperiod scale in the case of charged particles in
uniform magnetic fields".  The measurement behind that claim is their Fig. 9
(Sec. III.D): the L1 error against the *analytical* solution as a function of
step size, for a particle carried to a common final time omega_0 t_f = 8 in a
fixed background field, at B = 0.5 B_0 and B = 2.5 B_0, with a Boris pusher run
through the same protocol.  Their crossing is at omega_g dt ~ 0.1, "about 0.015
of the gyroperiod".

Gate G0 of plan/prereg/W10_sympmat.md: reproduce that crossing to within a
factor of 2 in dt.  Nothing else in the wave runs unless G0 passes.

WHAT IS IMPLEMENTED HERE
------------------------
No third-party code is vendored.  SympMat is written from the published
description in Secs. III.B and III.C:

  G-reflector    G = I - beta u u^T J,  beta scalar, u in R^{2n}, both trained,
                 u normalised.  Symplectic at any parameters because u^T J u = 0
                 identically for an antisymmetric J -- normalisation is only a
                 gauge fixing, not a condition for symplecticity.
  SympMat        the composition of 4n = 8 such reflectors (n = 2 here), which
                 by their Ref. 50 can represent any real symplectic matrix.
  parametric     beta_i and u_i of each reflector are the outputs of shallow
                 tanh MLPs of the physical parameter mu = B/B_0; hidden width 10
                 for beta and 20 for u, their Table I.

CANONICAL COORDINATES
---------------------
Their state is canonical, p = m r' + (q/c) A with A = (B/2)(-y, x), and their
transfer matrix M(t) of Eq. (5) acts on (x, y, p_x, p_y).  L1 is therefore an
average over four canonical components, and the Boris pusher -- which lives in
mechanical variables -- is converted into the same coordinates before the error
is taken.  `analytic_M` is derived here from their Hamiltonian Eq. (3) rather
than copied off the page, because the PDF's Eq. (5) does not survive text
extraction; `sm0_analytic.py` checks the derivation against a stiff-tolerance
ODE integration and against symplecticity.

NORMALISATION
-------------
Theirs, Sec. III.C: time in units of omega_0^{-1} = mc/(q B_0), positions in
r_{g,0} = m c u_0 / (q B_0), momenta in m u_0, fields in B_0 and
E_0 = (u_0/c) B_0.  In those units the whole problem depends on one number,
b = B/B_0, and the normalised transfer matrix is `analytic_M(b, tau)`.

THE BORIS VARIANTS
------------------
The paper says "a Boris pusher" and never says which.  The words leapfrog,
staggered, half-step and synchronised do not occur anywhere in it, although it
cites Chin and Cator (their Ref. 30) for the anatomy of Boris solvers.  Chin
and Cator, J. Comput. Phys. 466, 111422 (2022), arXiv:2109.01901v2, distinguish

  BLF   the conventional leapfrog Boris solver, their Eqs. (3.8)-(3.9) with the
        Boris angle: positions live at half-integer steps, velocities at integer
        steps.  In their operator form (5.2) a start from (r_0, v_0) is a half
        drift followed by a rotation, so the state after n steps is
        (r_{n-1/2}, v_n).  Error in a constant field: the gyroradius
        R_g = r_g sqrt(1 + theta^2/4), their Eq. (3.11); not on-orbit.
  B2B   the symmetric second-order Boris solver, their Eq. (4.3) with the Boris
        angle: half drift, full rotation, half drift, both variables at integer
        steps.  In a constant field its positions lie exactly on the true
        gyrocircle -- correct centre, correct radius -- and the only error is
        the phase deficit theta - 2 arctan(theta/2) per step.
  B1A   their Eq. (3.2) with the Boris angle: rotate, then drift.  y_c = +r_g/2.
  B1B   their Eq. (3.3) with the Boris angle: drift, then rotate.  y_c = -r_g/2.
        Parker and Birdsall's solver is B1B, not BLF (their Sec. III).

Their Sec. V: "positions at integer time step r_n do not exist for BLF.  Any
attempt to define an integer time-step position r_n for BLF is an ad hoc
alteration of the algorithm".  Their own such definition is Eq. (5.3),
r_n = r_{n-1/2} + (dt/2) v_n, which is the midpoint of the two stored positions;
this directory calls it the centred readout and the raw stored one the
as-stored readout.  In a uniform field the rotation does not depend on position,
so BLF and B2B share their velocity sequence exactly and differ *only* by that
readout -- BLF centred and B2B are the same numbers.  That is Chin and Cator's
Sec. V stated as an identity, and it is why predictions P1 and P2 of the
preregistration are two faces of one measurement here.

DECLARED BEFORE THE RUN
-----------------------
Fixed on 02.09.2026 before any training, per the discipline of the campaign:

  step-size ladder   omega_0 dt = 8/N for N in 4, 8, 16, ..., 1024: nine values
                     from 2.0 down to 0.0078125.  Nine, because their Fig. 3
                     shows nine parametric models; dyadic, because every horizon
                     used in the wave (8, 80, 800, 1000, 8000) must be an exact
                     integer number of steps at every ladder rung, which their
                     "common final time" requires.
  seeds              THREE independent seeds per (parametric model, dt).  Each
                     seed redraws the training set and the initialisation.  The
                     spread reported is min/median/max over those three.
  evaluation set     625 particles -- their ensemble size in Sec. III.D --
                     sampled once, uniformly in the training box [-1, 1]^4 of
                     canonical phase space, from a single fixed seed, and shared
                     by every scheme, every step size and every horizon.
  metric             their L1, Eq. (11), averaged over N particles and d = 4
                     canonical components.  No metric of ours is introduced.
  field values       b = 0.5 and b = 2.5, both reported, both endpoints of the
                     parametric training range.

SEED LEDGER
-----------
Three seed accidents were found during this campaign: a seed reused between
experiments, a seed frozen inside `build_model()`, and a generator rebuilt
inside a loop.  Seeds here come from a block that nothing else in the bundle
touches -- the highest seed anywhere else is 9,304,000 -- they are disjoint
between role, step-size index and repetition by construction, and each is
written into the output JSON beside the number it produced.

    11_000_000 + 100_000 * role + 1_000 * dt_index + rep

role 0 parametric weight init, 1 parametric training data, 2 parametric batch
order, 3 parametric augmentation angles, 4 standard weight init, 5 standard
training data, 6 standard batch order, 7 evaluation ensemble.

`seed_of()` is the only place a seed is formed.  Nothing here calls the global
numpy or torch random state.

FLOP MODEL
----------
One flop per arithmetic operation, twenty per transcendental: the model of
Section 9 of the manuscript and of experiments/classical/schemes.py, reused
unchanged so that the numbers in this directory sit on the same scale as the
rest of the paper.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
for _p in (EXP, os.path.join(EXP, "external_arch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ea_common import check_or_write, FLOP_TRANSCENDENTAL   # noqa: E402

#: SM_CKPT and SM_OUT exist only so that the analysis scripts can be smoke-tested
#: against synthetic checkpoints in a scratch directory while the real training
#: is still running.  Unset, everything reads and writes beside this file.
CKPT_DIR = os.environ.get("SM_CKPT", os.path.join(HERE, "ckpt"))
OUT_DIR = os.environ.get("SM_OUT", HERE)


def outpath(name):
    return os.path.join(OUT_DIR, name)

TWO_PI = 2.0 * np.pi

# ------------------------------------------------------------------ declared
DT_LADDER = tuple(8.0 / (4 * 2 ** k) for k in range(9))     # 2.0 ... 0.0078125
B_EVAL = (0.5, 2.5)
B_TRAIN = tuple(np.linspace(0.5, 2.5, 40))
N_ENSEMBLE = 625
N_SEEDS = 3
N_PAIRS_PER_B = 100
TF_MAIN = 8.0

# ------------------------------------------------------------------- seeds --
_ROLE = {"pinit": 0, "pdata": 1, "pbatch": 2, "paug": 3,
         "sinit": 4, "sdata": 5, "sbatch": 6, "ensemble": 7}


def seed_of(role, dt_index=0, rep=0):
    """The one place a seed is formed.  See the ledger in the module docstring."""
    r = _ROLE[role]
    assert 0 <= dt_index < 100 and 0 <= rep < 1000
    return 11_000_000 + 100_000 * r + 1_000 * dt_index + rep


# ------------------------------------------------------------------- J, M ---
J4 = np.array([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0, 0.0],
               [0.0, -1.0, 0.0, 0.0]])


def analytic_M(b, tau):
    """Their Eq. (5) in the normalisation of Sec. III.C, state (x, y, p_x, p_y).

    Derived from their Hamiltonian Eq. (3),
        H = [ (p_x + (m w /2) y)^2 + (p_y - (m w /2) x)^2 ] / 2m,   w = qB/mc,
    whose flow in normalised variables depends only on b = B/B_0 through the
    gyrophase theta = b tau.  `sm0_analytic.py` checks it three ways.
    """
    b = float(b)
    th = b * np.asarray(tau, dtype=float)
    c, s = np.cos(th), np.sin(th)
    o = np.ones_like(c)
    M = np.array([
        [(o + c) / 2, s / 2, s / b, (o - c) / b],
        [-s / 2, (o + c) / 2, -(o - c) / b, s / b],
        [-b * s / 4, -b * (o - c) / 4, (o + c) / 2, s / 2],
        [b * (o - c) / 4, -b * s / 4, -s / 2, (o + c) / 2]])
    return M if M.ndim == 2 else np.moveaxis(M, 2, 0)


def sympl_defect(M):
    """max |M^T J M - J|, zero for a symplectic matrix."""
    return float(np.max(np.abs(M.T @ J4 @ M - J4)))


def can_to_mech(b):
    """(x, y, p_x, p_y) -> (x, y, v_x, v_y);  v = p + (m w/2)(y, -x), normalised."""
    A = np.eye(4)
    A[2, 1] = b / 2.0
    A[3, 0] = -b / 2.0
    return A


def mech_to_can(b):
    A = np.eye(4)
    A[2, 1] = -b / 2.0
    A[3, 0] = b / 2.0
    return A


# --------------------------------------------------------- Boris variants ---
def _drift(h):
    D = np.eye(4)
    D[0, 2] = h
    D[1, 3] = h
    return D


def boris_angle(theta):
    """Chin and Cator Eq. (3.10), tan(theta_B/2) = theta/2; returned as the pair
    (cos, sin) of their Eqs. (3.13)-(3.14) so that no arctan is taken."""
    q = 0.5 * theta
    d = 1.0 + q * q
    return (1.0 - q * q) / d, theta / d


def _rot_boris(theta):
    """v <- (cos v_x + sin v_y, -sin v_x + cos v_y) at the Boris angle: the sense
    of the exact gyration for q > 0 and B along +z, which is what the Boris kick
    with t = (theta/2) z-hat produces.  The exact rotation is the same matrix at
    theta itself, and the difference between the two angles,
    theta - 2 arctan(theta/2), is the whole of B2B's error in a uniform field."""
    c, s = boris_angle(theta)
    R = np.eye(4)
    R[2, 2] = c
    R[2, 3] = s
    R[3, 2] = -s
    R[3, 3] = c
    return R


def boris_step_matrices(b, h):
    """One-step propagators in mechanical variables for the Chin-Cator family.

    Returns a dict name -> (first, repeat, readout) where the state after n >= 1
    steps is  readout @ repeat^(n-1) @ first @ z0.  Everything in a uniform
    field is linear, so the whole trajectory is a matrix power and the
    autoregressive rollout is exact rather than sampled.

    BLF_stored   the leapfrog algorithm read where it stores, (r_{n-1/2}, v_n).
    BLF_centred  the same run with Chin and Cator's Eq. (5.3) readout
                 r_n = r_{n-1/2} + (h/2) v_n; identical to B2B in a uniform
                 field, which is their Sec. V.
    B2B          half drift, Boris rotation, half drift.
    B1A          rotate then drift.   B1B  drift then rotate.
    """
    th = b * h
    RB = _rot_boris(th)
    Dh, Dh2 = _drift(h), _drift(0.5 * h)
    I = np.eye(4)
    lf_first = RB @ Dh2                      # e^{h V_B} e^{(h/2) T}
    lf_rep = RB @ Dh                         # e^{h V_B} e^{h T}
    return {
        "BLF_stored":  (lf_first, lf_rep, I),
        "BLF_centred": (lf_first, lf_rep, Dh2),
        "B2B":         (lf_first, lf_rep, Dh2),
        "B1A":         (Dh @ RB, Dh @ RB, I),
        "B1B":         (RB @ Dh, RB @ Dh, I),
    }


def rollout_matrix(first, repeat, readout, n):
    """readout @ repeat^(n-1) @ first, accumulated one step at a time."""
    A = np.array(first, dtype=float)
    for _ in range(int(n) - 1):
        A = repeat @ A
    return readout @ A


def boris_total_map(b, h, n, variant):
    """The n-step Boris map in canonical coordinates."""
    first, rep, out = boris_step_matrices(b, h)[variant]
    return mech_to_can(b) @ rollout_matrix(first, rep, out, n) @ can_to_mech(b)


# ---------------------------------------------------------------- metrics ---
def l1_error(Zp, Zt):
    """Their Eq. (11): mean over particles and over the d = 4 components."""
    return float(np.mean(np.abs(np.asarray(Zp) - np.asarray(Zt))))


def ensemble(seed=None):
    """625 canonical states drawn once, uniformly in the training box."""
    rng = np.random.default_rng(seed_of("ensemble") if seed is None else seed)
    return rng.uniform(-1.0, 1.0, size=(N_ENSEMBLE, 4))


def guiding_centre(Z, b):
    """(X_c, Y_c) = (x + v_y/b, y - v_x/b), the two invariants of the exact flow,
    from canonical states Z."""
    V = Z @ can_to_mech(b).T
    return np.stack([V[:, 0] + V[:, 3] / b, V[:, 1] - V[:, 2] / b], axis=1)


# ---------------------------------------------------------------- crossing --
def crossings(x, y_a, y_b):
    """Every step size at which log y_a - log y_b changes sign, linear in
    (log x, log y).  y_a is the scheme whose error grows with the step and y_b
    the one whose error falls with it, so a single crossing is the expected
    picture and anything else is a result rather than a failure."""
    x, y_a, y_b = np.asarray(x, float), np.asarray(y_a, float), np.asarray(y_b, float)
    o = np.argsort(x)
    lx, d = np.log(x[o]), np.log(y_a[o]) - np.log(y_b[o])
    out = []
    for i in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
        t = -d[i] / (d[i + 1] - d[i])
        out.append(float(np.exp(lx[i] + t * (lx[i + 1] - lx[i]))))
    return out


def loglog_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)),
                            np.log(np.asarray(y, float)), 1)[0])


def accumulate(first, repeat, snapshots):
    """{n: repeat^(n-1) @ first} for the n in `snapshots`, one step at a time."""
    snaps, A, n = {}, np.array(first, dtype=float), 1
    want = sorted(int(s) for s in snapshots)
    if 1 in want:
        snaps[1] = A.copy()
    for n in range(2, want[-1] + 1):
        A = repeat @ A
        if n in want:
            snaps[n] = A.copy()
    return snaps


# ------------------------------------------------------------------ flops ---
def flops_matvec(n=4):
    return 2 * n * n - n


def flops_boris_step_uniform():
    """One B2B / BLF step in the plane with cos and sin of the Boris angle
    precomputed (a uniform field): two half drifts and one 2x2 rotation."""
    return 2 * (2 * 2) + (4 + 2)


def flops_sympmat_step_uniform():
    """One SympMat step once the eight reflectors have been collapsed into a
    single 4x4 matrix, which the paper states is what one does for a fixed
    symplectic matrix (Sec. III.B)."""
    return flops_matvec(4)


def flops_parametric_build(nref=8, wb=10, wu=20, nin=1, dim=4):
    """Cost of rebuilding the SympMat matrix when B changes: the 2 x 8 shallow
    MLPs, the eight reflectors and the seven 4x4 products."""
    def mlp(win, wh, wout):
        f = 2 * win * wh + wh + FLOP_TRANSCENDENTAL * wh
        return f + 2 * wh * wout + wout
    per_ref = mlp(nin, wb, 1) + mlp(nin, wu, dim)
    per_ref += dim + (dim - 1) + FLOP_TRANSCENDENTAL + dim      # normalise u
    per_ref += dim * dim + dim * dim + dim * dim                # I - beta u u^T J
    tot = nref * per_ref
    tot += (nref - 1) * (2 * dim ** 3 - dim ** 2)               # matrix products
    return tot
