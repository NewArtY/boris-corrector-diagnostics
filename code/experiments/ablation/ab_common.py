"""Shared machinery for wave W17 -- ablations, the non-learned baseline, the gate.

WHAT THIS DIRECTORY IS FOR
--------------------------
Three questions, all of them from the first author's list of required
experiments, and all of them about the same object: the learned corrector of
Section~\\ref{sec:family}.

  1  ABLATIONS.  The corrector is a Boris step, a network, and a hard
     constraint on the output.  Each of those pieces is removed on its own and
     the same measurement is repeated, so that the reader can see which piece
     carries which number.  The pre-registration `plan/prereg/W17_ablation.md`
     predicts (P1) that the reproducible part of the advantage comes from the
     projection and not from the network.

  2  A NON-LEARNED CORRECTOR.  The one-step defect of the shipped Boris map is
     written out by hand from its Taylor expansion, with no training and no
     data, and put through the same measurements.  If it takes a significant
     share of the advantage, learning is not necessary here (P2).

  3  A FALLBACK GATE.  W14 found the mechanism by which the corrector fails
     outside its training point.  This directory turns that mechanism into a
     runtime test and demonstrates it firing on the map of W14 (P3).

WHAT IS IMPORTED AND WHAT IS NEW
--------------------------------
Nothing here retrains the committed corrector and nothing outside this
directory is written.  Imported unchanged:

  ../stats/seed_sweep_b4.py:evaluate     the statistic Section 7 prints
  ../refcheck/rc_common.py:closed_form   the independent reference
  ../map/map_common.py                   the batched 3-D rollout, the field
                                         bridge, the flop model, `clean`
  ../gtable/gt_common.py                 the four channels
  ../external_arch/ea_common.py          the JSON gatekeeper, the flop model
  training/train_corrector_b4.py         the architecture and every
                                         hyper-parameter
  models/boris.py                        the shipped Boris step

What is new is `integrate_variant` below -- one integrator with the pieces of
the corrector on switches -- the analytic defect `analytic_defect`, and the
gate `gate_signal`.  `ab1_calibration.py` asserts that with every switch in
its shipped position `integrate_variant` reproduces
`../stats/seed_sweep_b4.py:integrate_corrected` to the last bit, on both of
the two settings that function has, before any ablation is reported.

THE REFERENCE
-------------
The closed form of B4, imported from `../spectral/sw_common.py` through
`../refcheck/rc_common.py`.  Never a Boris run at h/150: W18 established that
the corrector was trained against that surface, so ruler and pupil are the same
object.  Every number in this directory is against the closed form, and the
h/150 ruler is carried alongside only where a committed number has to be
reproduced before it is replaced.

SEEDS
-----
**This directory draws nothing.**  The checkpoints are the committed file and
the twenty of W16 and I1.3, read and md5-checked; the initial conditions of the
map are `../map/map_common.py`'s, drawn there from `MAP_SEED`; the fields, the
steps and the horizons are fixed.  `ab5_loss_ablation.py` is the one exception
and it is not an exception to the ledger either: it retrains, and it retrains
at seeds that already exist -- the committed 42 and W16's 16_000_000 and
16_000_001 -- because a loss-term ablation is only interpretable against the
run it ablates.  No new seed is formed anywhere in this directory.  Should a
later extension need one, the block reserved for W17 -- declared here, before
any code was run, and free by `../refcheck/rc0_seed_audit.json` -- is

    17_000_000 .. 17_999_999

COST
----
In flops, on the model of Section 9: one flop per arithmetic operation, twenty
per transcendental (`ea_common.FLOP_TRANSCENDENTAL`).  Counted operation by
operation in `flops_analytic_defect` and `flops_gate` below, against the
committed 114,091 of one corrector step.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, EXP,
           os.path.join(EXP, "external_arch"),
           os.path.join(EXP, "classical"),
           os.path.join(EXP, "spectral"),
           os.path.join(EXP, "map"),
           os.path.join(EXP, "gtable"),
           os.path.join(EXP, "refcheck"),
           os.path.join(EXP, "seeds"),
           os.path.join(EXP, "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_common as MC                                        # noqa: E402
import rc_common as RC                                         # noqa: E402
from ea_common import (check_or_write, FLOP_TRANSCENDENTAL,     # noqa: E402
                       mlp_forward_flops)

clean = MC.clean

# ------------------------------------------------------------------ setup --
Q, M = MC.Q, MC.M                       # -1.0, 1.0
DT = RC.DT                              # 0.3
TAU = RC.TAU                            # 1.2e5
T_FINAL = RC.T_FINAL                    # 120.0
R0 = RC.R0                              # (1, 0, 0)
V0 = RC.V0                              # (0, 1, 0)
N_WORK = int(round(T_FINAL / DT))       # 400

BUNDLE_CHECKPOINTS = os.path.join(ROOT, "checkpoints")
COMMITTED_CORRECTOR = os.path.join(BUNDLE_CHECKPOINTS, "boris_corrector_b4.pt")
COMMITTED_MD5 = "0fe271bdb54de8a720f11eec85ee01f5"

CKPT = os.path.join(HERE, "ckpt")
os.makedirs(CKPT, exist_ok=True)

#: the four inputs whose training variance was zero, found in W14.  Their
#: standard deviation is `clamp_min(1e-12)` and therefore exactly 1e-12 in the
#: committed checkpoint; the assertion is in `ab4_gate.py`, not here.
DEAD_INPUTS = {6: "Bx", 7: "By", 11: "Ez", 12: "dt"}
INPUT_NAMES = ["rx", "ry", "rz", "vx", "vy", "vz",
               "Bx", "By", "Bz", "Ex", "Ey", "Ez", "dt"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_committed_untouched():
    got = md5(COMMITTED_CORRECTOR)
    if got != COMMITTED_MD5:
        raise SystemExit("the committed checkpoint changed: %s != %s"
                         % (got, COMMITTED_MD5))
    return got


def load_torch(path):
    import torch
    torch.set_default_dtype(torch.float64)
    from training.train_corrector_b4 import DefectNet
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def ensemble_members():
    """(tag, seed, source, path) -- `../seeds/sd3_measure.py:members`, verbatim
    order, so that this directory's ensemble is W16's and not a second one."""
    import sd_common as SD
    STATS = os.path.join(EXP, "stats")
    out = [("committed", SD.COMMITTED_SEED, "committed", SD.COMMITTED_CORRECTOR)]
    for s in SD.CORRECTOR_SEEDS:
        p = SD.seed_ckpt(s)
        if os.path.exists(p):
            out.append(("w16_s%d" % s, s, "W16", p))
    for s in (1, 7, 123, 2026):
        p = os.path.join(STATS, "checkpoints", "corrector_b4_seed%d.pt" % s)
        if os.path.exists(p):
            out.append(("i13_s%d" % s, s, "I1.3", p))
    return out


# ======================================================== the reference =====
_REF = {}


def closed_form_ref(n_steps=N_WORK, dt=DT):
    """(r, v, t) of the closed form on the working grid, the `ref` argument
    `../stats/seed_sweep_b4.py:evaluate` takes.  Cached: it does not depend on
    the scheme."""
    key = (int(n_steps), float(dt))
    if key not in _REF:
        ts = np.arange(n_steps + 1) * dt
        R, V = RC.closed_form(ts)
        _REF[key] = (R, V, ts)
    return _REF[key]


# ================================================== the corrector, on switches
class Variant:
    """One point of the ablation lattice.

    source     'net'       the network of the checkpoint       (shipped)
               'analytic'  the hand-written defect below       (no training)
               'none'      no correction at all                (plain Boris)
    zero_dr    drop the position half of the correction
    zero_dv    drop the velocity half of the correction
    ortho      subtract the component of dv along the Boris velocity
    rescale    restore |v| to the speed the Boris step would have produced
    gate       None, or (x_mean, x_std, threshold): fall back to plain Boris on
               any step whose standardised input exceeds the threshold

    'ortho' and 'rescale' together are the symmetric projection as shipped;
    `../stats/seed_sweep_b4.py:integrate_corrected(project=True)` is
    (net, ortho, rescale) and `project=False` is (net, no ortho, no rescale).
    """

    __slots__ = ("source", "zero_dr", "zero_dv", "ortho", "rescale", "gate",
                 "label", "analytic_kind")

    def __init__(self, source="net", zero_dr=False, zero_dv=False,
                 ortho=True, rescale=True, gate=None, label=None,
                 analytic_kind="rotation"):
        self.source = source
        self.zero_dr = zero_dr
        self.zero_dv = zero_dv
        self.ortho = ortho
        self.rescale = rescale
        self.gate = gate
        self.analytic_kind = analytic_kind
        self.label = label or self.describe()

    def describe(self):
        bits = [self.source]
        if self.source == "analytic":
            bits.append(self.analytic_kind)
        if self.zero_dr:
            bits.append("dr=0")
        if self.zero_dv:
            bits.append("dv=0")
        bits.append("ortho" if self.ortho else "no-ortho")
        bits.append("rescale" if self.rescale else "no-rescale")
        if self.gate is not None:
            bits.append("gated")
        return " ".join(bits)

    def asdict(self):
        return {"source": self.source, "zero_dr": bool(self.zero_dr),
                "zero_dv": bool(self.zero_dv), "ortho": bool(self.ortho),
                "rescale": bool(self.rescale),
                "gated": self.gate is not None,
                "analytic_kind": self.analytic_kind
                if self.source == "analytic" else None,
                "label": self.label}


# --------------------------------------------------------- the analytic defect
def analytic_defect(r, v, E, B, dt, kind="rotation"):
    """The one-step defect of the shipped Boris map, written out by hand.

    No training, no data, no fit.  Uses exactly the thirteen numbers the
    network is given -- (r, v, B, E, h) -- and nothing else.

    THE DERIVATION, IN FULL
    -----------------------
    The shipped map (models/boris.py:boris_step) is

        v_{n+1} = Rot(b_hat, 2 arctan(|B| h / 2)) (v_n - (h/2) E) - (h/2) E
        r_{n+1} = r_n + h v_{n+1}

    with q/m = -1, so that a = dv/dt = -(E + v x B) and the exact rotation over
    a step in a constant B is by the angle |B| h about b_hat = B/|B|.

    POSITION.  Expanding the exact flow and the map about (r_n, v_n):

        r_exact  = r + h v + (h^2/2) a + (h^3/6) a' + O(h^4)
        v_Boris  = v + h a + (h^2/2) a' + O(h^3)        (the map is 2nd order
                                                         in the velocity)
        r_Boris  = r + h v_Boris = r + h v + h^2 a + (h^3/2) a' + O(h^4)

    so the position defect is

        dr = r_exact - r_Boris = -(h^2/2) a - (h^3/3) a' + O(h^4)
           = (h^2/2) (E + v x B) + O(h^3) .

    The leading term is the whole of what the network's inputs can express: a'
    needs dE/dt, dB/dt and the spatial gradients, none of which is an input.
    It is kept and the rest is not, and that is stated rather than hidden.
    Adding it turns the first-order position update of the shipped map into a
    second-order one.

    VELOCITY.  In a constant B with no E the map is an exact rotation about
    b_hat, so its only error is in the angle:

        theta_exact = |B| h ,   theta_Boris = 2 arctan(|B| h / 2) ,
        dtheta      = |B| h - 2 arctan(|B| h / 2) = (|B| h)^3 / 12 + O(h^5) .

    The correction is the rotation that completes it,

        dv = Rot(b_hat, dtheta) v_Boris - v_Boris ,

    which is exactly norm-preserving and therefore exactly energy-neutral in
    the sense Section~\\ref{sec:conditions} uses -- the same property the
    learned correction has to be given by a projection.  `kind="linear"`
    replaces the rotation by its first-order form dtheta (b_hat x v_Boris) with
    dtheta taken from the series (|B|h)^3/12, which costs no transcendental at
    all; both are measured.

    WHAT IS LEFT OUT, AND WHY
    -------------------------
    The map evaluates E at the left end of the step, so the exact impulse
    h E(r_mid, t_mid) differs from h E(r_n, t_n) at O(h^2 dE/dt).  In the field
    of Section~\\ref{sec:channels} that is ~2e-7 per step against a velocity
    defect of ~2e-3, and it cannot be formed from the network's inputs in any
    case.  It is not corrected here, and `ab3_baseline.py` reports what it
    would be worth.

    `r`, `v`, `E`, `B` are (3,) or (3, nb); `dt` is a scalar.  Returns the
    position half, (3,) or (3, nb); the velocity half needs the Boris velocity
    and is `_rotation_completion` below.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    h = float(dt)

    # ---- the position half: -(h^2/2) a with a = -(E + v x B)
    vxB = np.stack([v[1] * B[2] - v[2] * B[1],
                    v[2] * B[0] - v[0] * B[2],
                    v[0] * B[1] - v[1] * B[0]])
    dr = (0.5 * h * h) * (E + vxB)
    if kind != "order3":
        return dr

    # ---- the next term, -(h^3/3) a', still from the same thirteen numbers.
    # a' = -(E' + a x B + v x B'), and for a field whose magnetic part is
    # B(t) = B_0 e^{-t/tau} z_hat with the induced E it implies,
    #
    #     E  = -(1/(2 tau)) (r x B) ,   B' = -B/tau ,
    #     E' = dE/dt + (v.grad)E = -E/tau - (v x B)/(2 tau) ,
    #
    # so   a' = E/tau + (3/(2 tau)) (v x B) - a x B .
    #
    # 1/tau is not an input, but it does not have to be: E = -(r x B)/(2 tau)
    # inverts to  1/tau = -2 E.(r x B) / |r x B|^2 , which is formed from the
    # inputs alone.  That inversion is why the network can fit this term at
    # all, and it is done here the same way so that the comparison stays an
    # equal-information one.  Where r x B vanishes the term is dropped.
    a = -(E + vxB)
    rxB = np.stack([r[1] * B[2] - r[2] * B[1],
                    r[2] * B[0] - r[0] * B[2],
                    r[0] * B[1] - r[1] * B[0]])
    den = rxB[0] ** 2 + rxB[1] ** 2 + rxB[2] ** 2
    num = E[0] * rxB[0] + E[1] * rxB[1] + E[2] * rxB[2]
    inv_tau = np.where(den > 1e-300, -2.0 * num / np.where(den > 0, den, 1.0),
                       0.0)
    axB = np.stack([a[1] * B[2] - a[2] * B[1],
                    a[2] * B[0] - a[0] * B[2],
                    a[0] * B[1] - a[1] * B[0]])
    adot = E * inv_tau + (1.5 * inv_tau) * vxB - axB
    return dr - (h ** 3 / 3.0) * adot


def _rotation_completion(v_boris, B, h, kind="rotation"):
    """dv = Rot(b_hat, dtheta) v_Boris - v_Boris, the angle the map is short by.

    Split out of `analytic_defect` because it needs the Boris velocity, which
    exists only after the Boris step.
    """
    if kind in ("trapezoid_rot", "order3"):
        kind = "rotation"
    B = np.asarray(B, dtype=float)
    w = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    safe = np.where(w > 0, w, 1.0)
    bx, by, bz = B[0] / safe, B[1] / safe, B[2] / safe
    x = w * h
    if kind == "linear":
        dth = (x ** 3) / 12.0
    else:
        dth = x - 2.0 * np.arctan(0.5 * x)
    vx, vy, vz = v_boris
    cx = by * vz - bz * vy
    cy = bz * vx - bx * vz
    cz = bx * vy - by * vx
    if kind == "linear":
        return np.stack([dth * cx, dth * cy, dth * cz])
    c = np.cos(dth)
    s = np.sin(dth)
    omc = 1.0 - c
    dot = bx * vx + by * vy + bz * vz
    nx = vx * c + cx * s + bx * dot * omc
    ny = vy * c + cy * s + by * dot * omc
    nz = vz * c + cz * s + bz * dot * omc
    return np.stack([nx - vx, ny - vy, nz - vz])


# ----------------------------------------------------------------- the gate --
def gate_signal(X, x_mean, x_std):
    """The largest standardised input, |(x - mean)/std| over the thirteen.

    This is the quantity the network's first layer sees.  W14 established the
    failure mechanism: four of the thirteen inputs had zero variance in
    training, `clamp_min(1e-12)` made their divisor 1e-12, and any state that
    moves one of them off its training value drives the first layer into
    saturation, at which point the correction degenerates to a single constant
    vector.  The gate is therefore not a heuristic laid over the network -- it
    is the network's own argument, read before the tanh flattens it.

    `X` is (13,) or (13, nb).  Returns a scalar or (nb,).
    """
    X = np.asarray(X, dtype=float)
    z = (X - np.asarray(x_mean).reshape(-1, *([1] * (X.ndim - 1)))) \
        / np.asarray(x_std).reshape(-1, *([1] * (X.ndim - 1)))
    return np.max(np.abs(z), axis=0)


def step_corrector_gated(fld, x, y, z, vx, vy, vz, t, dt, mlp, x_mean, x_std,
                         thr, counter=None):
    """`../map/map_common.py:step_corrector` with the fallback in front of it.

    Batched, component-wise, in the layout of the map's rollout.  The Boris
    part, the network call and the projection are that function's, character
    for character; the only addition is that the correction is zeroed on any
    member of the batch whose standardised input exceeds the threshold, which
    leaves that member on the plain Boris step for that step.

    The gate is evaluated *before* the network, so a gated step does not pay
    the network's 113,958 flops; it pays 40.  That is what makes the fallback
    free rather than an extra cost on top of the corrector.
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
    g = gate_signal(X, x_mean, x_std)
    fired = g > thr
    if counter is not None:
        counter["n_gated"] += int(fired.sum())
        counter["n_steps"] += nb
        counter["g_max"] = max(counter["g_max"], float(np.max(g)))
        counter["per_ic"] += fired
        counter["g_max_per_ic"] = np.maximum(counter["g_max_per_ic"], g)
        if counter["n_step_calls"] == 0:
            counter["fired_first_step"] = fired.copy()
        counter["n_step_calls"] += 1

    if bool(fired.all()):
        return rbx, rby, rbz, bvx, bvy, bvz

    d = mlp.forward(X)
    if bool(fired.any()):
        d = np.where(fired[None, :], 0.0, d)

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


def rollout_gated(fld, R0, V0, dt, n_steps, idx, mlp, x_mean, x_std, thr,
                  mark=None):
    """`../map/map_common.py:rollout` for the gated scheme.

    Written out rather than threaded through the committed `rollout` so that
    nothing in `../map/` has to change; the loop, the sampling and the
    non-finite accounting are that function's.
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
    counter = {"n_gated": 0, "n_steps": 0, "g_max": 0.0,
               "per_ic": np.zeros(nb, dtype=np.int64),
               "g_max_per_ic": np.zeros(nb),
               "fired_first_step": np.zeros(nb, dtype=bool),
               "n_step_calls": 0}
    with np.errstate(all="ignore"):
        for n in range(1, n_steps + 1):
            t = (n - 1) * dt
            x, y, z, vx, vy, vz = step_corrector_gated(
                fld, x, y, z, vx, vy, vz, t, dt, mlp, x_mean, x_std, thr,
                counter)
            if mark is not None and n == mark:
                at_mark = {"n_gated": counter["n_gated"],
                           "n_steps": counter["n_steps"],
                           "per_ic": counter["per_ic"].copy(),
                           "g_max": counter["g_max"],
                           "g_max_per_ic": counter["g_max_per_ic"].copy()}
            j = pos.get(n)
            if j is not None:
                Rs[j, :, 0] = x; Rs[j, :, 1] = y; Rs[j, :, 2] = z
                Vs[j, :, 0] = vx; Vs[j, :, 1] = vy; Vs[j, :, 2] = vz
    meta = {"n_steps": int(n_steps), "h": float(dt),
            "gated_fraction": counter["n_gated"] / max(counter["n_steps"], 1),
            "n_gated": counter["n_gated"],
            "n_state_steps": counter["n_steps"],
            "gate_signal_max": counter["g_max"],
            "gated_fraction_per_ic":
                (counter["per_ic"] / max(n_steps, 1)).tolist(),
            "gate_signal_max_per_ic": counter["g_max_per_ic"].tolist(),
            "fired_first_step_per_ic":
                counter["fired_first_step"].tolist()}
    if mark is not None:
        meta["at_mark"] = {
            "n_steps_of_the_run": int(mark),
            "gated_fraction": at_mark["n_gated"] / max(at_mark["n_steps"], 1),
            "gated_fraction_per_ic":
                (at_mark["per_ic"] / max(mark, 1)).tolist(),
            "gate_signal_max": at_mark["g_max"],
            "gate_signal_max_per_ic": at_mark["g_max_per_ic"].tolist()}
    bad = ~np.isfinite(Rs).all(axis=(0, 2)) | ~np.isfinite(Vs).all(axis=(0, 2))
    meta["n_nonfinite"] = int(bad.sum())
    return Rs, Vs, meta


# ------------------------------------------------------------- the integrator
def integrate_variant(field, r0, v0, dt, n_steps, model, var, mlp=None):
    """One trajectory under one point of the ablation lattice.

    The arithmetic of the Boris part, of the network call and of the projection
    is `../stats/seed_sweep_b4.py:integrate_corrected` line for line, so that
    with `Variant(source='net', ortho=True, rescale=True)` this function is
    that one; `ab1_calibration.py` asserts bit-identity on both of its
    settings before anything else in this directory runs.

    Returns (rs, vs, ts, diag) with diag carrying the gate statistics.
    """
    import torch
    from models.boris import boris_step

    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    r, v, t = np.array(r0, float), np.array(v0, float), 0.0
    rs[0], vs[0] = r, v
    n_gated = 0
    g_max = 0.0
    g_series = np.zeros(n_steps)
    # The constraint of Section~\ref{sec:conditions}, checked step by step:
    # | ||v_{n+1}|| - ||v_Boris(v_n)|| |.  The manuscript quotes 1.8e-15 on
    # this run; it is a property of the projection and not of the field, so it
    # is measured here rather than inferred from the energy channel, which also
    # carries the physical decay of the field.
    c_abs = 0.0
    c_rel = 0.0

    with torch.no_grad():
        for i in range(1, n_steps + 1):
            r_b, v_b = boris_step(r, v, t, dt, field)
            B = np.atleast_1d(field.B(r, t)).ravel()
            E = np.atleast_1d(field.E(r, t)).ravel()

            fired = False
            if var.gate is not None:
                xm, xs, thr = var.gate
                g = float(gate_signal(np.concatenate([r, v, B, E, [dt]]),
                                      xm, xs))
                g_series[i - 1] = g
                g_max = max(g_max, g)
                fired = g > thr

            if fired or var.source == "none":
                dr = np.zeros(3)
                dv = np.zeros(3)
                n_gated += int(fired)
            elif var.source == "net":
                x = torch.tensor(np.concatenate([r, v, B, E, [dt]]))[None, :]
                d = model(x).numpy()[0]
                dr, dv = d[:3].copy(), d[3:].copy()
            elif var.source == "analytic":
                k = var.analytic_kind
                if k.startswith("trapezoid"):
                    # r_{n+1} = r_n + (h/2)(v_n + v_{n+1}) written as a
                    # correction to the shipped r_n + h v_{n+1}
                    dr = (0.5 * dt) * (v - v_b)
                else:
                    dr = analytic_defect(r, v, E, B, dt, kind=k)
                if k == "trapezoid":
                    dv = np.zeros(3)
                else:
                    dv = _rotation_completion(v_b, B, dt, kind=k)
            else:
                raise ValueError(var.source)

            if var.zero_dr:
                dr = np.zeros(3)
            if var.zero_dv:
                dv = np.zeros(3)

            nb = np.linalg.norm(v_b)
            if var.ortho:
                vh = v_b / max(nb, 1e-300)
                dv = dv - np.dot(dv, vh) * vh
            v_new = v_b + dv
            if var.rescale:
                v_new = v_new * (nb / max(np.linalg.norm(v_new), 1e-300))

            dev = abs(float(np.linalg.norm(v_new)) - nb)
            c_abs = max(c_abs, dev)
            c_rel = max(c_rel, dev / max(nb, 1e-300))

            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t

    diag = {"n_steps": int(n_steps), "n_gated": int(n_gated),
            "gated_fraction": float(n_gated) / float(n_steps),
            "constraint_max_abs": float(c_abs),
            "constraint_max_rel": float(c_rel),
            "gate_signal_max": float(g_max) if var.gate is not None else None,
            "gate_signal_median": (float(np.median(g_series))
                                   if var.gate is not None else None)}
    return rs, vs, ts, diag


# ------------------------------------------------------- the volume property
def one_step_jacobian(field, r, v, t, dt, var, model=None, fd=1e-7):
    """d(r_{n+1}, v_{n+1}) / d(r_n, v_n) of one step of a variant, 6 x 6.

    Central differences at the bundle's own `fd = 1e-7`
    (`../f0_variational/validate.py`, `../spectrum/sp_common.py`).  The lab
    frame is used, not the canonical one: the question asked here is whether
    the map preserves phase-space volume, and |det J| is the same number in
    either frame because the two are related by a volume-preserving change of
    variables (p = v - qA/m at fixed r).  Symplecticity is not asked here --
    `../spectrum/sp2_spectra.py` already measures that, in the canonical frame
    where it is meaningful, and this directory does not restate it.

    The Boris map is volume preserving exactly (Qin et al. 2013), so the Boris
    row of this measurement is the resolution of the difference and is what
    every other row has to be read against.
    """
    w = np.concatenate([np.asarray(r, float), np.asarray(v, float)])
    J = np.empty((6, 6))
    for k in range(6):
        out = []
        for s in (+1.0, -1.0):
            wp = w.copy()
            wp[k] += s * fd
            rs, vs, _, _ = integrate_variant(field, wp[:3], wp[3:], dt, 1,
                                             model, var)
            out.append(np.concatenate([rs[1], vs[1]]))
        J[:, k] = (out[0] - out[1]) / (2.0 * fd)
    return J


def volume_defect(field, var, model=None, r=None, v=None, t=0.0, dt=None,
                  fd=1e-7):
    """|det J - 1| of one step, at the canonical initial condition."""
    r = R0 if r is None else r
    v = V0 if v is None else v
    dt = DT if dt is None else dt
    J = one_step_jacobian(field, r, v, t, dt, var, model, fd)
    return {"det_minus_one": float(abs(np.linalg.det(J) - 1.0)),
            "det": float(np.linalg.det(J)), "fd": fd}


# ============================================== the statistic Section 7 prints
def evaluate_variant(variants, model=None, ref=None, tau=TAU,
                     r0=R0, v0=V0, dt=DT, n_work=N_WORK):
    """`../stats/seed_sweep_b4.py:evaluate`, generalised over the lattice.

    The reference, the Boris baseline, the physical signal, the median over the
    second half and the two ratios are formed exactly as that function forms
    them; the only change is that the runs being scored are the variants of
    this directory instead of its fixed pair.  `ab1_calibration.py` checks the
    generalisation against the original on the two points they share.
    """
    from fields import DecayingField
    from models.boris import integrate_boris

    field = DecayingField(B0=1.0, tau=tau)
    r0 = np.asarray(r0, float)
    v0 = np.asarray(v0, float)
    if ref is None:
        ref = closed_form_ref(n_work, dt)
    rs_r, vs_r, ts_r = ref
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1)
    E0 = E_ref[0]

    runs = {"boris": integrate_boris(r0, v0, 0.0, dt, n_work, field)[:3]}
    diags = {}
    for name, var in variants.items():
        rr, vv, tt, dg = integrate_variant(field, r0, v0, dt, n_work,
                                           model, var)
        runs[name] = (rr, vv, tt)
        diags[name] = dg

    out = {}
    half = n_work // 2
    for k, (rs, vs, tt) in runs.items():
        Ei = np.interp(tt, ts_r, E_ref)
        Ee = 0.5 * np.sum(vs ** 2, axis=1)
        e_err = np.abs(Ee - Ei) / E0
        r_ref_i = np.vstack([np.interp(tt, ts_r, rs_r[:, j])
                             for j in range(3)]).T
        pos_err = np.linalg.norm(rs - r_ref_i, axis=1)
        speed_dev = np.abs(np.linalg.norm(vs, axis=1)
                           - np.linalg.norm(vs[0])) / np.linalg.norm(vs[0])
        out[k] = {"energy_err_median_2nd_half": float(np.median(e_err[half:])),
                  "energy_err_max": float(e_err.max()),
                  "pos_err_final": float(pos_err[-1]),
                  "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2))),
                  # the physical decay of the field, not a constraint residual
                  "speed_change_from_initial_max": float(speed_dev.max())}
        if k in diags:
            out[k]["gate"] = diags[k]
            out[k]["constraint_max_abs"] = diags[k]["constraint_max_abs"]
            out[k]["constraint_max_rel"] = diags[k]["constraint_max_rel"]

    ts_b = runs["boris"][2]
    phys = float(np.median(np.abs((np.interp(ts_b, ts_r, E_ref) - E0) / E0)[half:]))
    out["physical_signal_median"] = phys
    for k in runs:
        e = out[k]["energy_err_median_2nd_half"]
        out[k]["energy_separation"] = phys / e if e > 0 else float("inf")
        out[k]["traj_gain_over_boris"] = (out["boris"]["pos_err_rms"]
                                          / out[k]["pos_err_rms"])
    return out


# ==================================================================== the cost
#: the committed figures, imported and not restated.
FLOPS_BORIS = MC.FLOPS["boris"]                 # 113
FLOPS_CORRECTOR = MC.FLOPS["corrector"]         # 114091
FLOPS_NET_FORWARD = mlp_forward_flops([13, 128, 128, 128, 128, 6])   # 113958
FLOPS_VPS4 = MC.FLOPS["vps4"]


def flops_analytic_defect(kind="rotation"):
    """Counted operation by operation against the code above.

      v x B                       6 mul + 3 add                    =  9
      E + v x B                   3 add                            =  3
      c = h^2/2 ; c * (.)         2 mul + 3 mul                    =  5
      |B|                         3 mul + 2 add + 1 sqrt           = 25
      b_hat = B/|B|               1 div + 3 mul                    =  4
      x = |B| h                   1 mul                            =  1
      b_hat x v_Boris             6 mul + 3 add                    =  9
      dv = dtheta * (b x v)       3 mul                            =  3   [linear]
      dtheta (linear)             x^3 = 2 mul, /12 = 1 mul         =  3   [linear]
      dtheta (rotation)           x/2, arctan, *2, subtract        = 23   [rotation]
      Rot(b_hat, dtheta) v        cos + sin = 40 ; 1-c = 1 ;
                                  dot = 3 mul + 2 add = 5 ;
                                  3 mul + 3 mul + 1 mul + 3 mul
                                  + 6 add = 16                     = 62   [rotation]
      dv = rotated - v_Boris      3 sub                            =  3   [rotation]
    """
    if kind == "trapezoid":
        # dr = (h/2)(v_n - v_Boris): three subtractions and three multiplies
        # by a step constant formed once.
        return 6
    rot = 25 + 4 + 1 + 9 + 23 + 62 + 3      # |B|, b_hat, x, cross, angle, Rot
    if kind == "trapezoid_rot":
        return 6 + rot
    common = 9 + 3 + 5 + 25 + 4 + 1 + 9
    if kind == "linear":
        return common + 3 + 3
    #   r x B 9 ; E.(r x B) 5 ; |r x B|^2 5 ; -2 num/den 2 ; a = -(E+vxB) 6 ;
    #   a x B 9 ; E/tau 3 ; (3/2tau) 1 and times v x B 3 ; the sum 6 ;
    #   h^3/3 2 and times a' 3 ; the subtraction 3
    third = 9 + 5 + 5 + 2 + 6 + 9 + 3 + 4 + 6 + 5 + 3
    if kind == "order3":
        return common + 23 + 62 + 3 + third
    return common + 23 + 62 + 3


def flops_gate(n_in=13):
    """13 subtractions, 13 multiplications by a precomputed reciprocal, 13
    absolute values and 12 comparisons for the maximum, then one comparison
    against the threshold."""
    return 3 * n_in + (n_in - 1) + 1


def flops_projection():
    """The shipped symmetric projection, per step: |v_B| (25), the reciprocal
    and three multiplies for v_hat (4), the dot product (5), three multiplies
    and three subtractions for the orthogonal part (6), three additions for
    v_B + dv (3), |v_new| (25), one division and three multiplies (4)."""
    return 25 + 4 + 5 + 6 + 3 + 25 + 4


# ------------------------------------------------------------------ output --
def write(path, payload, force=False, rtol=1e-9, ignore=None):
    kw = {} if ignore is None else {"ignore": ignore}
    return check_or_write(path, json.loads(json.dumps(clean(payload))),
                          rtol=rtol, force=force, **kw)


def outpath(name):
    return os.path.join(HERE, name)


_DRAWS_DECLARED = 0


def assert_no_draws(n=0):
    assert n == _DRAWS_DECLARED, (
        "this directory declared %d random draws before the first run and has "
        "just made %d" % (_DRAWS_DECLARED, n))


def summarise(values):
    """W16's convention, `../refcheck/rc3_seeds.py:summarise` verbatim."""
    v = np.asarray(sorted(values), dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    q1, med, q3 = (float(x) for x in np.percentile(v, [25.0, 50.0, 75.0]))
    return {"n": int(v.size), "median": med, "q1": q1, "q3": q3,
            "min": float(v.min()), "max": float(v.max()), "iqr": q3 - q1,
            "sd": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "ratio_max_min": float(v.max() / v.min()) if v.min() > 0 else None}


def place(committed_value, values):
    v = np.asarray(values, dtype=float)
    n_below = int((v < committed_value).sum())
    return {"n": int(v.size), "value": float(committed_value),
            "n_below": n_below, "n_above": int((v > committed_value).sum()),
            "percentile_below": 100.0 * n_below / v.size}
