"""Shared machinery for the three external architectures of Section 7.

WHAT THIS DIRECTORY IS FOR
--------------------------
The four-probe protocol of Section 6 was applied to one learned scheme, our
own, and to a family of classical ones.  A protocol validated on a single
object is a protocol shaped by that object.  This directory runs the same four
probes on three learned architectures taken from the literature -- a
Hamiltonian neural network, a SympNet and a physics-informed network with a
symplecticity penalty -- on the same field, at the same step size, under the
same readout, and scores them against the same classical schemes on flops.

WHAT IS IMPLEMENTED HERE AND WHAT IS NOT
----------------------------------------
No third-party code is vendored.  Each architecture is written from its
published description, in the smallest form that is faithful to it:

  HNN     Greydanus, Dzamba & Yosinski, NeurIPS 32 (2019).  A network holds a
          scalar H and the vector field is its symplectic gradient.  Trained
          on the field-matching loss of their Eq. (3), integrated with RK4 as
          in their experiments.
  SympNet Jin, Zhang, Zhu, Tang & Karniadakis, Neural Netw. 132 (2020) 166.
          G-SympNet: a composition of gradient modules
          p <- p + K^T diag(a) sigma(K q + b), q <- q + K^T diag(a) sigma(K p + b),
          each of which is a shear in one half of the phase space and
          therefore symplectic at any weights.  G rather than LA is argued in
          `ea1_train.py`.
  PINN    Raissi, Perdikaris & Karniadakis, J. Comput. Phys. 378 (2019) 686,
          with the symplecticity penalty that the label "PINN-symplectic"
          implies.  The network predicts the one-step increment and is trained
          on the midpoint residual of the equations of motion plus
          ||J^T Omega J - Omega||_F^2.  No reference trajectory enters its loss.

CANONICAL COORDINATES
---------------------
The motion of Section 2 is planar: B is along z, E lies in the (x,y) plane,
v_z(0) = 0, so r_z and v_z stay at zero identically.  The phase space is
four-dimensional.  With the vector potential

    A(r,t) = (1/2) B_z(t) (-y, x),        B_z(t) = B_0 exp(-t/tau),

and q = -1, m = 1, the motion is canonical in

    q = (x, y),      p = v - A(q,t),      H(q,p,t) = |p + A(q,t)|^2 / 2,

with E = -dA/dt reproducing the induced field of Section 2 exactly.  H depends
on time only through the scalar B_z, so B_z is what the networks are given as
their time input; the true Hamiltonian is then exactly representable in their
input variables, and no architecture is handicapped by the parameterisation.

Working in the plane rather than in three dimensions makes the external
architectures cheaper than they would otherwise be, by about a third of a
forward pass.  The handicap runs in their favour and the conclusion of
`ea3_cost.py` is unchanged by it.

SEED LEDGER
-----------
Three seed accidents were found during this campaign: a seed reused between
experiments, a seed frozen inside `build_model()` so that an ensemble was one
model five times, and a generator rebuilt inside a loop so that four points
were correlated at 0.55 to 0.98.  Seeds here are drawn from one block that no
other script in the bundle touches (the highest seed anywhere else is 500000),
they are disjoint between architecture, role and repetition by construction,
and each one is written into the output JSON beside the number it produced.

    9_000_000 + 100_000 * arch_index + 1_000 * role_index + rep

arch_index  0 hnn, 1 sympnet, 2 pinn
role_index  0 weight initialisation, 1 data shuffling, 2 collocation sampling,
            3 probe perturbations, 4 capacity/budget controls
rep         0..N-1, the repetition index

`seed_of()` is the only place a seed is ever formed.  Nothing in this
directory calls the global `numpy.random` functions or `torch.manual_seed`;
every draw goes through a generator built once, outside its loop, from a seed
that `seed_of()` returned.

FLOP MODEL
----------
One flop per arithmetic operation, twenty per transcendental, which is the
model of Section 9 and of experiments/classical/schemes.py.  It reproduces the
manuscript's 113,958 flops for one forward pass of the learned corrector
exactly; `ea3_cost.py` asserts that as a calibration.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, os.path.join(EXP, "classical"), os.path.join(EXP, "f0_settings"),
           os.path.join(EXP, "p_law_check"), os.path.join(EXP, "probe4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------- physics --
Q, M = -1.0, 1.0
B0 = 1.0
DT = 0.3                      # Omega h = 0.3
T_FINAL = 120.0               # 19.1 gyro-orbits, the window of Section 7
TAU_PAPER = 1.2e5             # decaying
TAU_QUASI = 1.2e8             # quasistatic
R0 = np.array([1.0, 0.0])
V0 = np.array([0.0, 1.0])

# training-set spread, copied from training/train_corrector_b4.py so that the
# external architectures see exactly the states the corrector saw
TAU_TRAIN = (0.8e5, 1.0e5, 1.5e5, 2.0e5, 3.0e5)
N_TRAJ_PER_TAU = 3
RHO_RANGE = (0.7, 1.3)

# training budget, also from train_corrector_b4.py: 400 epochs over ~5400
# training samples at batch 512 is 4400 Adam steps
ADAM_STEPS = 4400
BATCH = 512
LR = 1e-3


def seed_of(arch_index, role_index, rep=0):
    """The one place a seed is formed.  See the ledger in the module docstring."""
    assert 0 <= arch_index < 8 and 0 <= role_index < 8 and 0 <= rep < 1000
    return 9_000_000 + 100_000 * arch_index + 1_000 * role_index + rep


ARCH_INDEX = {"hnn": 0, "sympnet": 1, "pinn": 2}
ROLE = {"init": 0, "shuffle": 1, "collocation": 2, "probe": 3, "control": 4}


# ------------------------------------------------------------------ field --
def bz(t, tau):
    return B0 * np.exp(-np.asarray(t, dtype=float) / tau)


def vecpot(x, y, t, tau):
    """A = (1/2) B_z(t) (-y, x)."""
    h = 0.5 * bz(t, tau)
    return -h * y, h * x


def efield(x, y, t, tau):
    """E = -dA/dt, the induced field of Section 2."""
    h = 0.5 * bz(t, tau) / tau
    return -h * y, h * x


def accel(x, y, vx, vy, t, tau):
    """dv/dt = (q/m)(E + v x B) in the plane."""
    ex, ey = efield(x, y, t, tau)
    b = bz(t, tau)
    return (Q / M) * (ex + vy * b), (Q / M) * (ey - vx * b)


def to_canonical(x, y, vx, vy, t, tau):
    ax, ay = vecpot(x, y, t, tau)
    return x, y, vx - ax, vy - ay


def to_physical(x, y, px, py, t, tau):
    ax, ay = vecpot(x, y, t, tau)
    return x, y, px + ax, py + ay


def energy(vx, vy):
    return 0.5 * (vx * vx + vy * vy)


def e_phys(t, tau, e0):
    """The exact energy law E(t) = E_0 exp(-t/tau) of Section 9, which follows
    from the conservation of the magnetic moment."""
    return e0 * np.exp(-np.asarray(t, dtype=float) / tau)


# -------------------------------------------------------------- reference --
def dop853(tau, t_eval, r0=None, v0=None, rtol=1e-12, atol=1e-14):
    """The reference of Section 7, restricted to the plane."""
    from scipy.integrate import solve_ivp
    r0 = R0 if r0 is None else np.asarray(r0, dtype=float)
    v0 = V0 if v0 is None else np.asarray(v0, dtype=float)

    def rhs(t, s):
        ax, ay = accel(s[0], s[1], s[2], s[3], t, tau)
        return [s[2], s[3], ax, ay]

    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), [r0[0], r0[1], v0[0], v0[1]],
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
    assert sol.success, sol.message
    return sol.y[:2].T, sol.y[2:].T


# -------------------------------------------------------------- boris step --
def boris_plane(x, y, vx, vy, t, tau, dt):
    """The Boris step of Section 2 in the plane, arithmetic identical to
    experiments/symproj/symproj.py:boris_kick followed by a drift."""
    k = 0.5 * Q * dt / M
    ex, ey = efield(x, y, t, tau)
    b = bz(t, tau)
    vmx, vmy = vx + k * ex, vy + k * ey
    tz = k * b
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx, vpy = vmx + vmy * tz, vmy - vmx * tz
    vlx, vly = vmx + vpy * sz, vmy - vpx * sz
    nvx, nvy = vlx + k * ex, vly + k * ey
    return x + nvx * dt, y + nvy * dt, nvx, nvy


# ------------------------------------------------------------- flop model --
# 1 flop per arithmetic operation, 20 per transcendental (Section 9).
FLOP_TRANSCENDENTAL = 20


def mlp_forward_flops(widths, act="tanh", standardise=True):
    """Flops of one forward pass of a dense MLP with the given layer widths.

    Calibration: widths = [13,128,128,128,128,6] with standardisation returns
    113,958, which is the figure Section 9 prints for the learned corrector.
    """
    f = 0
    for i in range(len(widths) - 1):
        f += 2 * widths[i] * widths[i + 1] + widths[i + 1]
        if i < len(widths) - 2 and act == "tanh":
            f += FLOP_TRANSCENDENTAL * widths[i + 1]
    if standardise:
        f += 2 * widths[0] + widths[-1]
    return f


def mlp_backward_flops(widths, act="tanh"):
    """Flops of one reverse-mode gradient of a scalar-output MLP.

    Each linear layer costs its transpose product again, 2 n_in n_out; each
    hidden unit costs its activation derivative (1 - s^2 is one multiply and
    one subtract) and the elementwise product with the incoming adjoint.
    """
    f = 0
    for i in range(len(widths) - 1):
        f += 2 * widths[i] * widths[i + 1]
        if i < len(widths) - 2 and act == "tanh":
            f += 3 * widths[i + 1]
    return f


# ------------------------------- numpy evaluation of a torch-trained MLP ----
class NumpyMLP:
    """Weights lifted out of torch into plain numpy, evaluated on a batch.

    State is (n_in, nb): one column per member of the batch.  The rollouts
    below carry the whole probe sweep as one batch so that a frequency scan
    costs one matmul per step rather than one run per frequency.
    """

    def __init__(self, Ws, bs, x_mean=None, x_std=None, y_scale=None):
        self.Ws = [np.ascontiguousarray(W, dtype=float) for W in Ws]
        self.bs = [np.ascontiguousarray(b, dtype=float).reshape(-1, 1) for b in bs]
        n_in = self.Ws[0].shape[1]
        n_out = self.Ws[-1].shape[0]
        self.x_mean = np.zeros((n_in, 1)) if x_mean is None else \
            np.asarray(x_mean, dtype=float).reshape(-1, 1)
        self.x_std = np.ones((n_in, 1)) if x_std is None else \
            np.asarray(x_std, dtype=float).reshape(-1, 1)
        self.y_scale = np.ones((n_out, 1)) if y_scale is None else \
            np.asarray(y_scale, dtype=float).reshape(-1, 1)
        self.widths = [n_in] + [W.shape[0] for W in self.Ws]

    def forward(self, X):
        z = (X - self.x_mean) / self.x_std
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            z = np.tanh(W @ z + b)
        return (self.Ws[-1] @ z + self.bs[-1]) * self.y_scale

    def scalar_and_grad(self, X):
        """Value and d(value)/dX of a scalar-output MLP, reverse mode."""
        z = (X - self.x_mean) / self.x_std
        acts = []
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            z = np.tanh(W @ z + b)
            acts.append(z)
        out = (self.Ws[-1] @ z + self.bs[-1]) * self.y_scale
        g = self.Ws[-1].T @ (self.y_scale * np.ones_like(out))
        for i in range(len(self.Ws) - 2, -1, -1):
            g = g * (1.0 - acts[i] ** 2)
            g = self.Ws[i].T @ g
        return out[0], g / self.x_std


def lift_torch_mlp(seq, x_mean=None, x_std=None, y_scale=None):
    import torch
    Ws, bs = [], []
    for layer in seq:
        if isinstance(layer, torch.nn.Linear):
            Ws.append(layer.weight.detach().double().numpy().copy())
            bs.append(layer.bias.detach().double().numpy().copy())
    return NumpyMLP(Ws, bs, x_mean, x_std, y_scale)


# ------------------------------------------------------------ diagnostics --
def import_pc():
    """The estimators of Section 6 as the campaign wrote them.

    `estimate_aH` is the two-parameter estimate; `envelope_exponent_from_series`
    is the envelope fit; `loglog_slope` and `sub_index` are the local-slope
    machinery.  They are imported rather than copied so that a change to the
    estimator changes these results too.
    """
    import pc_defect as PC
    return PC


def import_pb4():
    """The channel-selection rule of the fourth probe, Table 2 of Section 6.

    `channel_search` is the whole of lines 1 to 13 of that table, including the
    extrapolation gate G0, the carriage gate G1 and the cancellation gate G2.
    It is imported rather than reimplemented, so that this directory decides
    the fourth probe by exactly the rule the manuscript prints, and a change to
    the rule changes these verdicts.
    """
    import pb4_channel as PB
    return PB


def half_decade_slopes(t, env, n=6):
    """The local half-decade slopes that Section 4.4 requires beside every
    exponent this paper prints."""
    good = (t > 0) & (env > 0) & np.isfinite(env)
    t, env = t[good], env[good]
    if t.size < 20:
        return []
    hi = np.log10(t[-1])
    lo = max(np.log10(t[0]), hi - 3.0)
    edges = np.linspace(lo, hi, n + 1)
    out = []
    for i in range(n):
        m = (np.log10(t) >= edges[i]) & (np.log10(t) <= edges[i + 1])
        if m.sum() < 6:
            out.append(float("nan"))
            continue
        out.append(float(np.polyfit(np.log10(t[m]), np.log10(env[m]), 1)[0]))
    return out


# ----------------------------------------------------------- JSON gatekeeping
#: leaves whose key path contains one of these is checked for presence only.
#: Elapsed time is a property of the machine and the interpreter, not of the
#: scheme, which is the whole reason Section 7 scores flops instead; a gate
#: that failed on a 5% timing wobble would fail on every rerun and would
#: therefore stop being read.
NOT_REPRODUCIBLE = ("wall_s", "_wall_by", "seconds", "train_seconds")


def check_or_write(path, payload, rtol=1e-9, atol=0.0, force=False,
                   ignore=NOT_REPRODUCIBLE):
    """Write `payload` if the file is absent, otherwise compare and fail loudly.

    Every script in this bundle that produces numbers the manuscript prints
    exits non-zero when a rerun no longer reproduces the committed file
    (experiments/recover_numbers/rn1_f6b_refit.py is the model).  This is that
    check, generalised over a nested dictionary.  Pass --force to overwrite
    deliberately.
    """
    import json
    if force or not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote %s" % os.path.basename(path))
        return 0
    stored = json.load(open(path, encoding="utf-8"))
    bad = []
    _diff(stored, payload, "", bad, rtol, atol, ignore)
    if bad:
        print("\nMISMATCH against the committed %s:" % os.path.basename(path))
        for b in bad[:60]:
            print("   " + b)
        if len(bad) > 60:
            print("   ... %d more" % (len(bad) - 60))
        return 1
    print("%s reproduces (%d leaves compared)" % (os.path.basename(path),
                                                  _count_leaves(stored)))
    return 0


def _count_leaves(o):
    if isinstance(o, dict):
        return sum(_count_leaves(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_count_leaves(v) for v in o)
    return 1


def _diff(a, b, path, bad, rtol, atol, ignore=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                bad.append("%s.%s: new key" % (path, k))
            elif k not in b:
                bad.append("%s.%s: missing key" % (path, k))
            else:
                _diff(a[k], b[k], "%s.%s" % (path, k), bad, rtol, atol, ignore)
        return
    if any(tok in path for tok in ignore):
        return
    # json.dump turns a tuple into a list, so a freshly computed tuple and the
    # list read back from the committed file are the same value.  Comparing
    # them as different types made every loss_history entry report a mismatch
    # whose two sides printed identically.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            bad.append("%s: length %d vs %d" % (path, len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, "%s[%d]" % (path, i), bad, rtol, atol, ignore)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            bad.append("%s: %r vs %r" % (path, b, a))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return
        if not np.isclose(fb, fa, rtol=rtol, atol=atol, equal_nan=True):
            bad.append("%s: %r vs stored %r" % (path, b, a))
        return
    if a != b:
        bad.append("%s: %r vs stored %r" % (path, b, a))
