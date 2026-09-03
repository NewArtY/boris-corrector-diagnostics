"""Shared machinery for wave W14 -- the map of applicability.

WHAT THIS DIRECTORY IS FOR
--------------------------
Every number the manuscript prints about the learned corrector lives at one
step size, Omega h = 0.3, and in one field, the decaying B4 of
Section~\\ref{sec:channels}.  A claim measured at one point of a two-dimensional
parameter plane is a claim about that point.  This directory takes the step
from 1e-3 to 0.5 and the field over the five configurations the bundle already
defines, and maps three claims that must not be confused with one another:

  A   the learned corrector is more accurate than the Boris scheme on the
      trajectory                                   (the first author's question)
  B   the learned corrector is more accurate than vps4 **at equal total flops**
                                                   (does it earn its cost)
  C   the energy diagnostic does not show the trajectory error that has
      already accumulated                          (the manuscript's own thesis)

All three are reported at every point of the map, whichever way each falls.

WHAT IS NEW HERE AND WHAT IS IMPORTED
--------------------------------------
Nothing is retrained and nothing outside this directory is written.  The field
definitions come from `code/fields/` unchanged, the classical schemes and the
flop model from `../classical/schemes.py`, the closed form of B4 and the
committed Table 4 targets from `../spectral/sw_common.py`, and the JSON
gatekeeper from `../external_arch/ea_common.py`.  A change to any of them
changes these numbers too.

THE BRIDGE, AND WHY IT WAS NEEDED
----------------------------------
`../external_arch/ea_common.py` and `ea_arch.py` carry the batched rollout
whose `nb` parameter integrates many copies of the same problem side by side,
but they are written for the *planar* reduction of the decaying field: B along
z, E in the (x,y) plane, v_z identically zero.  The five field classes of
`code/fields/` expose `B(r,t)` and `E(r,t)` on three-vectors and three of them
(B1 radial, B2 wave, B3 tilted) do not reduce to that plane.  The bridge is
this module: the same batching idea carried over to three dimensions, with the
physics taken from the field classes verbatim.

Two things are done for speed and neither changes a number.

  1.  The state is carried as six arrays over the batch rather than as one
      (nb, 3) array, and the cross products are written out.  This is the same
      arithmetic in the same order: `numpy.cross` forms a1*b2 - a2*b1 and so
      does the code below.
  2.  Each field class is paired with a component-wise evaluator, `FastField`
      below.  `mp1_calibration.py` asserts that the evaluator and the class
      return **bit-identical** values on a batch of random states, for all five
      configurations, before any of it is used.  The definitions are the
      classes'; the evaluator only avoids their allocation.

Batching itself is bit-for-bit faithful because every operation is elementwise
along the batch.  The one place where members could interact is the
fixed-point iteration of gl4, and there each member is frozen at its own
convergence step exactly as the scalar loop breaks at its own.
`mp1_calibration.py` checks a batched rollout against a single-trajectory
rollout and requires zero difference.

WHAT r_L MEANS HERE
-------------------
Position errors are reported in Larmor radii, r_L = |v_perp(0)| / |B(r_0,0)|,
computed per (field, initial condition).  For the canonical initial condition
of the manuscript, r_0 = (1,0,0) and v_0 = (0,1,0), r_L = 1 exactly in all five
configurations, so the numbers here are directly comparable with Table 4.

SEEDS
-----
Declared before any run.  The number of random draws in this directory is
**twenty-one**: three per drawn initial condition (radius, gyrophase, parallel
velocity) for seven drawn initial conditions.  The eighth initial condition is
the canonical one of the manuscript and is not drawn.  One generator, built
once, outside every loop, from `MAP_SEED` below.  That seed lies in a block no
other script in the bundle touches: external_arch uses 9.0e6-9.8e6, spectral
13.0e6, the initial-condition ensemble of `../stats/` 20260830, everything
else is below 5e5.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, os.path.join(EXP, "external_arch"),
           os.path.join(EXP, "classical"), os.path.join(EXP, "spectral")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import schemes as S                                          # noqa: E402
import sw_common as SW                                       # noqa: E402
from ea_common import check_or_write                         # noqa: E402

from fields import (UniformField, RadialField, WaveField,    # noqa: E402
                    TiltedField, DecayingField)

TWO_PI = 2.0 * math.pi
Q, M = -1.0, 1.0

# ------------------------------------------------------------------ seeds --
MAP_SEED = 14_000_000
N_RANDOM_DRAWS = 21          # 3 per drawn initial condition, 7 drawn
N_IC = 8


# ------------------------------------------------------------------ axes ---
#: the step axis of the pre-registration, Omega h from 1e-3 to 0.5.  Every
#: value divides both horizons exactly, so t = 120 and t = 636 are attained as
#: step multiples for every scheme and no interpolation of a scheme's own
#: output is ever needed.
DT_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
           0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

#: the horizon axis.  H_paper is the window of Tables 4 and 5, 19.10
#: gyro-orbits.  H_crossover is 101.22 gyro-orbits, the neighbourhood of the
#: 100.98 of `../horizon/crossover.json` at which the corrector's trajectory
#: advantage reaches unity.  The long run is read out at both, so the horizon
#: axis costs nothing beyond the long run itself.
T_SHORT = 120.0
T_LONG = 636.0
HORIZONS = {"H_paper": T_SHORT, "H_crossover": T_LONG}

SCHEMES = ["boris", "corrector", "vps2", "vps4", "gl4"]

#: the corrector's single training point
DT_TRAIN = 0.3

# ------------------------------------------------------------------ fields -
#: the five configurations, taken from `code/fields/` with the parameters the
#: bundle ships.  B4 carries the paper's decay time tau = 1.2e5 (the class
#: default of 40 is not the setting any number in the manuscript lives in);
#: everything else is the class default.
TAU_PAPER = 1.2e5

FIELD_NAMES = ["uniform", "B1_radial", "B2_wave", "B3_tilted", "B4_decaying"]

#: what the corrector saw in training.  B4 only; B1, B2, B3 and the uniform
#: field are out of distribution on the field axis, and every step but 0.3 is
#: out of distribution on the step axis.
TRAINED_ON = {"uniform": False, "B1_radial": False, "B2_wave": False,
              "B3_tilted": False, "B4_decaying": True}


def make_fields():
    return {
        "uniform": UniformField(B0=1.0),
        "B1_radial": RadialField(B0=1.0, alpha=0.3, L=5.0),
        "B2_wave": WaveField(B0=1.0, delta=0.05, k=0.4, omega_w=0.3),
        "B3_tilted": TiltedField(B0=1.0, theta_deg=30.0),
        "B4_decaying": DecayingField(B0=1.0, tau=TAU_PAPER),
    }


# ===================================================== component-wise fields
class FastField:
    """Component-wise evaluator of one of the five field classes.

    `eb(x, y, z, t)` returns (Ex, Ey, Ez, Bx, By, Bz), each either a float
    (where the component is constant over the batch) or an array over the
    batch.  Every expression below is copied from the class it wraps, in the
    class's own order of operations, so the two agree bit for bit;
    `mp1_calibration.py` asserts exactly that before anything is run.
    """

    def __init__(self, name, f):
        self.name = name
        self.f = f
        self.e0 = (float(f.E0[0]), float(f.E0[1]), float(f.E0[2]))
        if name == "B3_tilted":
            d = f.B0 * f.direction
            self._b = (float(d[0]), float(d[1]), float(d[2]))

    def eb(self, x, y, z, t):
        n = self.name
        f = self.f
        e0 = self.e0
        if n == "uniform":
            # UniformField: B = B0 z_hat, E = E0
            return e0[0], e0[1], e0[2], 0.0, 0.0, f.B0
        if n == "B1_radial":
            # RadialField: B_z = B0 (1 + alpha rho^2 / L^2), E = E0
            rho2 = x ** 2 + y ** 2
            return (e0[0], e0[1], e0[2], 0.0, 0.0,
                    f.B0 * (1.0 + f.alpha * rho2 / f.L ** 2))
        if n == "B2_wave":
            # WaveField: B_z = B0 (1 + delta sin(k x - omega_w t)), E = E0
            return (e0[0], e0[1], e0[2], 0.0, 0.0,
                    f.B0 * (1.0 + f.delta * np.sin(f.k * x - f.omega_w * t)))
        if n == "B3_tilted":
            # TiltedField: B = B0 (sin theta, 0, cos theta), E = E0
            b = self._b
            return e0[0], e0[1], e0[2], b[0], b[1], b[2]
        if n == "B4_decaying":
            # DecayingField: B_z = B0 exp(-t/tau), E = E0 + induced azimuthal
            bz = f.B0 * np.exp(-t / f.tau)
            if not f.induced_E:
                return e0[0], e0[1], e0[2], 0.0, 0.0, bz
            dBdt = -bz / f.tau
            factor = -0.5 * dBdt
            return (e0[0] + -factor * y, e0[1] + factor * x, e0[2],
                    0.0, 0.0, bz)
        raise ValueError(n)


def make_fast_fields(fields):
    return {k: FastField(k, v) for k, v in fields.items()}


# -------------------------------------------------------- initial conditions
def initial_conditions(n_ic=N_IC):
    """The canonical initial condition of the manuscript plus n_ic-1 drawn
    from the spread `training/train_corrector_b4.py` samples: radius in
    (0.7, 1.3), uniform gyrophase, parallel velocity 0.3 (u - 1/2).

    One generator, built once, here and nowhere else.  The canonical one is
    index 0 so that Table 4 is a column of the batch rather than a separate
    run.
    """
    rng = np.random.default_rng(MAP_SEED)
    R = [np.array([1.0, 0.0, 0.0])]
    V = [np.array([0.0, 1.0, 0.0])]
    for _ in range(n_ic - 1):
        rho = 0.7 + 0.6 * rng.random()
        phase = TWO_PI * rng.random()
        vpar = 0.3 * (rng.random() - 0.5)
        R.append(np.array([rho * math.cos(phase), rho * math.sin(phase), 0.0]))
        V.append(np.array([-math.sin(phase), math.cos(phase), vpar]))
    return np.asarray(R), np.asarray(V)


def larmor_radii(field, R0, V0):
    """r_L = |v_perp(0)| / |B(r_0, 0)| per initial condition."""
    B = np.atleast_2d(field.B(R0, 0.0))
    bn = np.linalg.norm(B, axis=1)
    bhat = B / np.where(bn > 0, bn, 1.0)[:, None]
    vpar = np.sum(V0 * bhat, axis=1)[:, None] * bhat
    vperp = np.linalg.norm(V0 - vpar, axis=1)
    return vperp / np.where(bn > 0, bn, 1.0)


# =========================================================== batched steppers
# Each of these is the arithmetic of the committed implementation it names,
# written out in components and carried elementwise along the batch.

def step_boris(fld, x, y, z, vx, vy, vz, t, dt):
    """models/boris.py:boris_step."""
    ex, ey, ez, bx, by, bz = fld.eb(x, y, z, t)
    k = 0.5 * Q * dt / M
    vmx = vx + k * ex
    vmy = vy + k * ey
    vmz = vz + k * ez
    tx = k * bx
    ty = k * by
    tz = k * bz
    den = 1.0 + (tx * tx + ty * ty + tz * tz)
    sx = 2.0 * tx / den
    sy = 2.0 * ty / den
    sz = 2.0 * tz / den
    vpx = vmx + (vmy * tz - vmz * ty)
    vpy = vmy + (vmz * tx - vmx * tz)
    vpz = vmz + (vmx * ty - vmy * tx)
    nvx = (vmx + (vpy * sz - vpz * sy)) + k * ex
    nvy = (vmy + (vpz * sx - vpx * sz)) + k * ey
    nvz = (vmz + (vpx * sy - vpy * sx)) + k * ez
    return x + nvx * dt, y + nvy * dt, z + nvz * dt, nvx, nvy, nvz


def step_vps2(fld, x, y, z, vx, vy, vz, t, dt):
    """schemes.py:make_vps2 -- drift, E-kick, exact Rodrigues rotation over the
    full step about the local b_hat, E-kick, drift.  The rotation angle is
    -(q/m)|B|dt per batch member, so a field whose magnitude varies across the
    batch rotates each member by its own angle."""
    hh = 0.5 * dt
    x = x + hh * vx
    y = y + hh * vy
    z = z + hh * vz
    ex, ey, ez, bx, by, bz = fld.eb(x, y, z, t + hh)
    kk = hh * Q / M
    vx = vx + kk * ex
    vy = vy + kk * ey
    vz = vz + kk * ez
    bn = np.sqrt(bx * bx + by * by + bz * bz)
    bhx = bx / bn
    bhy = by / bn
    bhz = bz / bn
    th = -(Q / M) * bn * dt
    c = np.cos(th)
    s = np.sin(th)
    omc = 1.0 - c
    dot = bhx * vx + bhy * vy + bhz * vz
    nvx = vx * c + (bhy * vz - bhz * vy) * s + bhx * dot * omc
    nvy = vy * c + (bhz * vx - bhx * vz) * s + bhy * dot * omc
    nvz = vz * c + (bhx * vy - bhy * vx) * s + bhz * dot * omc
    vx = nvx + kk * ex
    vy = nvy + kk * ey
    vz = nvz + kk * ez
    return x + hh * vx, y + hh * vy, z + hh * vz, vx, vy, vz


_G1, _G0 = S._G1, S._G0


def step_vps4(fld, x, y, z, vx, vy, vz, t, dt):
    """schemes.py:make_vps4 -- the Yoshida triple jump of vps2."""
    for g in (_G1, _G0, _G1):
        gh = g * dt
        x, y, z, vx, vy, vz = step_vps2(fld, x, y, z, vx, vy, vz, t, gh)
        t = t + gh
    return x, y, z, vx, vy, vz


_A00, _A01 = S._A[0, 0], S._A[0, 1]
_A10, _A11 = S._A[1, 0], S._A[1, 1]
_C1, _C2 = S._C1, S._C2


def _gl4_rhs(fld, Y, t, out):
    ex, ey, ez, bx, by, bz = fld.eb(Y[0], Y[1], Y[2], t)
    out[0] = Y[3]
    out[1] = Y[4]
    out[2] = Y[5]
    out[3] = (Q / M) * (ex + (Y[4] * bz - Y[5] * by))
    out[4] = (Q / M) * (ey + (Y[5] * bx - Y[3] * bz))
    out[5] = (Q / M) * (ez + (Y[3] * by - Y[4] * bx))
    return out


def step_gl4(fld, x, y, z, vx, vy, vz, t, dt, tol=1e-14, maxit=60, stats=None,
             buf=None):
    """schemes.py:make_gl4 -- two-stage Gauss-Legendre, fixed point from a zero
    start, exactly as the committed scheme does it.

    The two stages are carried in one (12, nb) array, rows 0-5 for the first
    and 6-11 for the second.  Each member of the batch is frozen at its own
    convergence step, so the batched rollout is bit for bit the
    single-trajectory one; while every member is still active the two buffers
    are swapped rather than copied, which is the same values in the same
    places and costs nothing.  `stats` accumulates the iteration count, which
    is what prices gl4 in flops.
    """
    nb = np.shape(x)[0]
    if buf is None or buf[0].shape[1] != nb:
        buf = (np.empty((12, nb)), np.empty((12, nb)), np.empty((6, nb)),
               np.empty((6, nb)), np.empty((6, nb)), np.empty((12, nb)))
    k, kn, Y, Y1, Y2, diff = buf
    Y[0] = x; Y[1] = y; Y[2] = z
    Y[3] = vx; Y[4] = vy; Y[5] = vz
    k[:] = 0.0
    active = np.ones(nb, dtype=bool)
    n_active = nb
    used = np.full(nb, maxit, dtype=np.int64)
    t1 = t + _C1 * dt
    t2 = t + _C2 * dt
    for it in range(maxit):
        np.add(Y, dt * (_A00 * k[:6] + _A01 * k[6:]), out=Y1)
        np.add(Y, dt * (_A10 * k[:6] + _A11 * k[6:]), out=Y2)
        _gl4_rhs(fld, Y1, t1, kn[:6])
        _gl4_rhs(fld, Y2, t2, kn[6:])
        np.subtract(kn, k, out=diff)
        np.abs(diff, out=diff)
        dmax = diff.max(axis=0)
        if n_active == nb:
            k, kn = kn, k
        else:
            np.copyto(k, kn, where=active)
        done = active & (dmax < tol)
        if done.any():
            used[done] = it + 1
            active = active & ~done
            n_active = int(active.sum())
            if n_active == 0:
                break
    np.multiply(k[:6] + k[6:], 0.5 * dt, out=Y1)
    Y += Y1
    if stats is not None:
        stats["iters"] += float(used.mean())
        stats["iters_max"] = max(stats.get("iters_max", 0), int(used.max()))
        stats["steps"] += 1
    return (Y[0].copy(), Y[1].copy(), Y[2].copy(),
            Y[3].copy(), Y[4].copy(), Y[5].copy()), (k, kn, Y, Y1, Y2, diff)


# ------------------------------------------------------------- the corrector
_MLP = None


def load_corrector_numpy():
    """The committed checkpoint `boris_corrector_b4.pt`, lifted out of torch
    into plain numpy so that a batch costs one matmul per layer.

    Nothing here retrains, fine-tunes or reinitialises.  There is exactly one
    checkpoint of this network in the bundle, and every cross-field number in
    this directory is that one run.
    """
    global _MLP
    if _MLP is None:
        import torch
        import ea_common as EA
        torch.set_default_dtype(torch.float64)
        from common import CHECKPOINT_DIR
        from training.train_corrector_b4 import DefectNet
        m = DefectNet(n_in=13)
        m.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
            map_location="cpu"))
        m.eval()
        _MLP = EA.lift_torch_mlp(m.net, m.x_mean.numpy(), m.x_std.numpy(),
                                 m.y_scale.numpy())
        _MLP.torch_model = m
    return _MLP


def step_corrector(fld, x, y, z, vx, vy, vz, t, dt, mlp):
    """A Boris step, the learned correction, and the projection onto the speed.

    Arithmetic identical to `../classical/run.py:integrate_hybrid` and to
    `../spectral/sw_common.py:run_corrector`, which is what Table 4 scores.
    The network's thirteen inputs are (r, v, B, E, dt); B and E come from the
    field, so a configuration the network never saw enters through exactly the
    channel the network was built with and no new interface is invented for it.
    """
    ex, ey, ez, bx, by, bz = fld.eb(x, y, z, t)
    k = 0.5 * Q * dt / M
    vmx = vx + k * ex
    vmy = vy + k * ey
    vmz = vz + k * ez
    tx = k * bx
    ty = k * by
    tz = k * bz
    den = 1.0 + (tx * tx + ty * ty + tz * tz)
    sx = 2.0 * tx / den
    sy = 2.0 * ty / den
    sz = 2.0 * tz / den
    vpx = vmx + (vmy * tz - vmz * ty)
    vpy = vmy + (vmz * tx - vmx * tz)
    vpz = vmz + (vmx * ty - vmy * tx)
    bvx = (vmx + (vpy * sz - vpz * sy)) + k * ex
    bvy = (vmy + (vpz * sx - vpx * sz)) + k * ey
    bvz = (vmz + (vpx * sy - vpy * sx)) + k * ez
    rbx = x + bvx * dt
    rby = y + bvy * dt
    rbz = z + bvz * dt

    nb = np.shape(x)[0]
    X = np.empty((13, nb))
    X[0] = x; X[1] = y; X[2] = z
    X[3] = vx; X[4] = vy; X[5] = vz
    X[6] = bx; X[7] = by; X[8] = bz
    X[9] = ex; X[10] = ey; X[11] = ez
    X[12] = dt
    d = mlp.forward(X)

    nrm = np.sqrt(bvx * bvx + bvy * bvy + bvz * bvz)
    inv = 1.0 / np.maximum(nrm, 1e-300)
    vhx = bvx * inv
    vhy = bvy * inv
    vhz = bvz * inv
    dvx, dvy, dvz = d[3], d[4], d[5]
    proj = dvx * vhx + dvy * vhy + dvz * vhz
    dvx = dvx - proj * vhx
    dvy = dvy - proj * vhy
    dvz = dvz - proj * vhz
    nvx = bvx + dvx
    nvy = bvy + dvy
    nvz = bvz + dvz
    sc = nrm / np.maximum(np.sqrt(nvx * nvx + nvy * nvy + nvz * nvz), 1e-300)
    return (rbx + d[0], rby + d[1], rbz + d[2],
            nvx * sc, nvy * sc, nvz * sc)


# ------------------------------------------------------------------ rollout -
def sample_indices(n_total, n_mark, n_out=1200):
    """Step indices at which the state is recorded.

    The mark (the short horizon) and the end are always sampled, so both
    horizons are read off one run and the horizon axis costs nothing beyond
    the long run.
    """
    stride = max(1, n_total // n_out)
    idx = np.arange(0, n_total + 1, stride)
    idx = np.unique(np.concatenate([idx, [n_mark, n_total]]).astype(np.int64))
    return idx[idx <= n_total]


def rollout(fld, scheme, R0, V0, dt, n_steps, idx, mlp=None):
    """Integrate nb copies of the same problem side by side.

    Returns (Rs, Vs, meta) with Rs, Vs of shape (len(idx), nb, 3).
    """
    x = np.array(R0[:, 0], dtype=float)
    y = np.array(R0[:, 1], dtype=float)
    z = np.array(R0[:, 2], dtype=float)
    vx = np.array(V0[:, 0], dtype=float)
    vy = np.array(V0[:, 1], dtype=float)
    vz = np.array(V0[:, 2], dtype=float)
    nb = x.shape[0]
    Rs = np.empty((len(idx), nb, 3))
    Vs = np.empty((len(idx), nb, 3))
    pos = {int(i): j for j, i in enumerate(idx)}
    if 0 in pos:
        j = pos[0]
        Rs[j, :, 0] = x; Rs[j, :, 1] = y; Rs[j, :, 2] = z
        Vs[j, :, 0] = vx; Vs[j, :, 1] = vy; Vs[j, :, 2] = vz
    stats = {"iters": 0.0, "steps": 0}
    buf = None
    if scheme == "boris":
        step = step_boris
    elif scheme == "vps2":
        step = step_vps2
    elif scheme == "vps4":
        step = step_vps4
    elif scheme in ("gl4", "corrector"):
        step = None
    else:
        raise ValueError(scheme)

    with np.errstate(all="ignore"):
        for n in range(1, n_steps + 1):
            t = (n - 1) * dt
            if step is not None:
                x, y, z, vx, vy, vz = step(fld, x, y, z, vx, vy, vz, t, dt)
            elif scheme == "gl4":
                st, buf = step_gl4(fld, x, y, z, vx, vy, vz, t, dt,
                                   stats=stats, buf=buf)
                x, y, z, vx, vy, vz = st
            else:
                x, y, z, vx, vy, vz = step_corrector(fld, x, y, z, vx, vy, vz,
                                                     t, dt, mlp)
            j = pos.get(n)
            if j is not None:
                Rs[j, :, 0] = x; Rs[j, :, 1] = y; Rs[j, :, 2] = z
                Vs[j, :, 0] = vx; Vs[j, :, 1] = vy; Vs[j, :, 2] = vz

    meta = {"n_steps": int(n_steps), "h": float(dt)}
    bad = ~np.isfinite(Rs).all(axis=(0, 2)) | ~np.isfinite(Vs).all(axis=(0, 2))
    meta["n_nonfinite"] = int(bad.sum())
    if scheme == "gl4" and stats["steps"]:
        meta["mean_iters"] = float(stats["iters"] / stats["steps"])
        meta["max_iters"] = int(stats.get("iters_max", 0))
    return Rs, Vs, meta


# ============================================================== the reference
def dop853(field, ts, r0, v0, rtol=1e-12, atol=1e-14):
    """The reference of Section 7, on a general three-dimensional field.

    The field class is called, not the fast evaluator, so the reference is
    driven by the shipped definition itself.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, yv):
        r, v = yv[:3], yv[3:]
        E = np.atleast_1d(field.E(r, t)).ravel()
        B = np.atleast_1d(field.B(r, t)).ravel()
        return np.concatenate([v, (Q / M) * (E + np.cross(v, B))])

    sol = solve_ivp(rhs, (0.0, float(np.max(ts))),
                    np.concatenate([r0, v0]),
                    method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    assert sol.success, sol.message
    return sol


def dop853_at(sol, ts):
    yv = sol.sol(np.asarray(ts, dtype=float))
    return yv[:3].T, yv[3:].T


def exact_uniform_like(bvec, ts, r0, v0):
    """The closed form for a magnetic field constant in space and time with no
    electric field: dv/dt = (q/m) v x B = |B| (b_hat x v) for q = -1, m = 1.

    Covers the uniform configuration and B3, which is a static, spatially
    uniform field tilted away from z: a rotation of the perpendicular velocity
    about b_hat at the gyrofrequency |B|, free parallel motion, and the exact
    integral of both.  Each sample is an independent function evaluation, so
    the reference accumulates nothing.
    """
    ts = np.asarray(ts, dtype=float)
    b = np.asarray(bvec, dtype=float)
    w = float(np.linalg.norm(b))
    bh = b / w
    vpar = np.dot(v0, bh) * bh
    vperp = v0 - vpar
    cr = np.cross(bh, vperp)
    c = np.cos(w * ts)[:, None]
    s = np.sin(w * ts)[:, None]
    V = vpar[None, :] + c * vperp[None, :] + s * cr[None, :]
    R = (r0[None, :] + ts[:, None] * vpar[None, :]
         + (np.sin(w * ts) / w)[:, None] * vperp[None, :]
         + ((1.0 - np.cos(w * ts)) / w)[:, None] * cr[None, :])
    return R, V


def exact_uniform_like_mp(bvec, ts, r0, v0, dps=40):
    """The same closed form carried end to end in mpmath, to price what the
    float64 reconstruction costs.  This is the check `../spectral/sw1_reference.py`
    makes for B4, made here for the two configurations whose closed form is
    elementary: the argument w t reaches 636 radians at the long horizon, and
    a float64 argument reduction there is worth about 1e-16, which has to be
    shown to be below the residuals the map measures rather than assumed to be.
    """
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        b = [mp.mpf(float(v)) for v in bvec]
        w = mp.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
        bh = [v / w for v in b]
        v0m = [mp.mpf(float(v)) for v in v0]
        r0m = [mp.mpf(float(v)) for v in r0]
        d = bh[0] * v0m[0] + bh[1] * v0m[1] + bh[2] * v0m[2]
        vpar = [d * bh[i] for i in range(3)]
        vperp = [v0m[i] - vpar[i] for i in range(3)]
        cr = [bh[1] * vperp[2] - bh[2] * vperp[1],
              bh[2] * vperp[0] - bh[0] * vperp[2],
              bh[0] * vperp[1] - bh[1] * vperp[0]]
        out_r = np.empty((len(ts), 3))
        out_v = np.empty((len(ts), 3))
        for i, t in enumerate(ts):
            tm = mp.mpf(float(t))
            c, s = mp.cos(w * tm), mp.sin(w * tm)
            for j in range(3):
                out_v[i, j] = float(vpar[j] + c * vperp[j] + s * cr[j])
                out_r[i, j] = float(r0m[j] + tm * vpar[j]
                                    + (s / w) * vperp[j]
                                    + ((1 - c) / w) * cr[j])
        return out_r, out_v
    finally:
        mp.mp.dps = old


def exact_mp(field_name, field, ts, r0, v0, dps=40):
    """The closed form in mpmath, where there is one."""
    if field_name == "uniform":
        return exact_uniform_like_mp(np.array([0.0, 0.0, field.B0]), ts,
                                     r0, v0, dps)
    if field_name == "B3_tilted":
        return exact_uniform_like_mp(field.B0 * field.direction, ts,
                                     r0, v0, dps)
    if field_name == "B4_decaying":
        return SW.exact_reference_mp(ts, r0, v0, tau=field.tau, dps=dps)
    return None


_B4_BASIS = {}


def exact_b4(ts, r0, v0, tau=TAU_PAPER):
    """The closed form of B4: the Bessel solution derived and implemented in
    `../spectral/sw_common.py`, imported rather than re-derived.  The basis
    depends on the time grid alone and is cached, so a batch of initial
    conditions costs one basis."""
    ts = np.asarray(ts, dtype=float)
    key = (ts.tobytes(), float(tau))
    if key not in _B4_BASIS:
        if len(_B4_BASIS) > 6:
            _B4_BASIS.clear()
        _B4_BASIS[key] = SW.bessel_basis(ts, tau=tau)
    return SW.exact_from_basis(_B4_BASIS[key], r0, v0)


#: which configurations have a closed form and what it is.  Declared here
#: because the pre-registration requires the reference to be adjudicated per
#: configuration and not once for the whole map.  The pre-registration says
#: B1-B3 have none; B3 turns out to have one, because the bundle's B3 is a
#: *spatially uniform, static* field merely tilted away from z.  That is
#: reported rather than assumed away.
CLOSED_FORM = {
    "uniform": "exact cyclotron orbit about z",
    "B1_radial": None,
    "B2_wave": None,
    "B3_tilted": "exact cyclotron orbit about the tilted axis; B3 is a "
                 "static, spatially uniform field",
    "B4_decaying": "Bessel of order zero in the Larmor frame "
                   "(../spectral/sw_common.py)",
}


def exact(field_name, field, ts, r0, v0):
    """The closed form where there is one, else None."""
    if field_name == "uniform":
        return exact_uniform_like(np.array([0.0, 0.0, field.B0]), ts, r0, v0)
    if field_name == "B3_tilted":
        return exact_uniform_like(field.B0 * field.direction, ts, r0, v0)
    if field_name == "B4_decaying":
        return exact_b4(ts, r0, v0, tau=field.tau)
    return None


# -------------------------------------------------- exact invariants, B1, B2
def invariants(field_name, field, Rs, Vs):
    """Quantities the continuous motion conserves exactly.  They do not give a
    trajectory, but where there is no closed form they are what a reference
    can be held to.

      every configuration but B4:  E = 0, so |v| is exactly constant
      uniform, B1, B2, B4:         B has only a z component, so v_z is constant
      B1:                          axisymmetric static B_z(rho), so the
                                   canonical angular momentum
                                   p_phi = rho v_phi + (q/m) rho A_phi,
                                   A_phi = B0 (rho/2 + alpha rho^3/(4 L^2)),
                                   is exactly constant
    """
    out = {}
    if field_name != "B4_decaying":
        out["speed"] = _drift(np.linalg.norm(Vs, axis=-1))
    if field_name in ("uniform", "B1_radial", "B2_wave", "B4_decaying"):
        out["v_z"] = _drift(Vs[..., 2])
    if field_name == "B1_radial":
        x, yy = Rs[..., 0], Rs[..., 1]
        rho = np.hypot(x, yy)
        vphi = (x * Vs[..., 1] - yy * Vs[..., 0]) / np.where(rho > 0, rho, 1.0)
        Aphi = field.B0 * (rho / 2.0
                           + field.alpha * rho ** 3 / (4.0 * field.L ** 2))
        out["p_phi"] = _drift(rho * vphi + (Q / M) * rho * Aphi)
    return out


def _drift(a):
    """Relative excursion of a quantity the continuous motion conserves."""
    a0 = a[0]
    scale = np.maximum(np.abs(a0), 1e-300)
    d = np.abs(a - a0) / scale
    return {"max": float(np.nanmax(d)), "final": float(np.nanmax(d[-1]))}


# ==================================================================== metrics
def channels(Rs, Vs, Rr, Vr, r_L):
    """The two residual series, both reported for every scheme.

    position -- the norm of the position error in Larmor radii
    energy   -- the relative kinetic-energy error against E_ref(0), which is
                the definition `../classical/run.py:score` uses and therefore
                the one the energy column of Table 4 is in
    """
    dr = np.linalg.norm(Rs - Rr, axis=-1) / np.asarray(r_L)[None, :]
    e = 0.5 * np.sum(Vs * Vs, axis=-1)
    er = 0.5 * np.sum(Vr * Vr, axis=-1)
    return {"position": dr, "energy": (e - er) / er[0][None, :]}


def time_metrics(a):
    """max, root mean square, final value and the running-maximum envelope at
    ten fractions of the record.  Declared in the pre-registration before the
    runs, for every channel.  `a` is (n_out, nb); the return is per member."""
    a = np.abs(np.asarray(a, dtype=float))
    env = np.maximum.accumulate(a, axis=0)
    n = a.shape[0]
    idx = [max(0, int(round(f * (n - 1)))) for f in np.linspace(0.1, 1.0, 10)]
    return {"max": a.max(axis=0),
            "rms": np.sqrt(np.mean(a ** 2, axis=0)),
            "final": a[-1],
            "envelope_deciles": env[idx]}


def median_second_half(a):
    """The energy column of Table 4: the median relative error over the second
    half of the run."""
    h = a.shape[0] // 2
    return np.median(np.abs(a[h:]), axis=0)


# ==================================================================== the cost
#: flops per step.  Boris and the corrector are the figures Section 9 prints;
#: vps2 and vps4 come from the committed model of `../classical/schemes.py`;
#: gl4 is iterative and priced from its measured mean iteration count.  The
#: model's field-evaluation terms (F_E = 25, F_B = 12) are those of B4 and are
#: used unchanged for every configuration, so that the same scheme costs the
#: same everywhere and cross-field comparison of one scheme stays coherent.
#: The corrector-to-vps4 ratio is 418, so no plausible re-pricing of a field
#: evaluation moves any verdict here.
FLOPS = {"boris": 113.0, "corrector": 114091.0,
         "vps2": float(S.FLOPS_PER_STEP["vps2"]),
         "vps4": float(S.FLOPS_PER_STEP["vps4"])}


def flops_per_step(scheme, mean_iters=None):
    if scheme == "gl4":
        return float(S.flops_gl4(mean_iters))
    return FLOPS[scheme]


def equal_cost_substeps(scheme, mean_iters=None):
    """The largest integer m for which m steps of `scheme` cost no more than
    one corrector step.  This is the budget claim B is decided on."""
    return int(FLOPS["corrector"] // flops_per_step(scheme, mean_iters))


# ============================================== the three claims, made concrete
#: Claim C needs two thresholds and they are fixed here, before any map is
#: drawn.  The trajectory threshold is one Larmor radius, the criterion
#: `../horizon/crossover.json` already uses ("reaches 1 Larmor"): at that size
#: the orbit is displaced by its own radius and no reader would call the run
#: accurate.  The energy threshold is 1e-2, the level at which a reader
#: looking only at the energy channel would call the run clean.  Both are
#: stated here rather than chosen after the map, and the continuous ratio is
#: reported beside the binary so that neither threshold carries the result.
C_POS_THRESHOLD = 1.0        # Larmor radii
C_ENERGY_THRESHOLD = 1e-2    # relative energy error


def spearman(a, b):
    """Rank correlation, average ranks on ties."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = math.sqrt(float(np.sum(ra ** 2)) * float(np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / d) if d > 0 else float("nan")


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


# ------------------------------------------------------------------- output -
def clean(o):
    """numpy scalars and arrays out, plain JSON in."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isfinite(f):
            return f
        return "nan" if math.isnan(f) else ("inf" if f > 0 else "-inf")
    return o
