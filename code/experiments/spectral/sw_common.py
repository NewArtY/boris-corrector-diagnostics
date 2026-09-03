"""Shared machinery for wave W13 -- the spectral probe in the signal band.

WHAT THIS DIRECTORY IS FOR
--------------------------
The first author observes that in the band of the slow physical signal,
f/Omega_c < 0.2, the residual power of the classical scheme stands about six
orders of magnitude above the corrected ones, and that the corrector pushes
what is left out of that band into narrow harmonic lines at Omega_c and above.
The observation is his.  This directory measures it, adds vps4 to the
comparison, puts the comparison on a flop budget, and checks whether the
reference is the floor of the measurement rather than the schemes being
measured.

Nothing here retrains anything and nothing here writes outside this directory.
`../external_arch/ea_common.py` and `../classical/schemes.py` are imported, not
copied, so a change to the flop model or to a scheme changes these numbers too.

THE REFERENCE PROBLEM HAS A CLOSED FORM
---------------------------------------
This is the point on which the whole wave turns.  The motion of
Section~\\ref{sec:channels} is planar and linear, and it is exactly solvable.
With z = x + iy, w = v_x + i v_y, q = -1, m = 1 and B_z(t) = B_0 e^{-t/tau},

    E  = i (B_z/2 tau) z ,      v x B = -i B_z w ,
    w' = i B_z w - i (B_z/2 tau) z ,     z' = w .

Passing to the Larmor frame z = e^{i theta(t)} zeta with theta' = B_z/2 kills
the first-order term and leaves

    zeta'' + (B_z(t)^2 / 4) zeta = 0 ,

a real oscillator whose frequency decays exponentially.  Substituting
s = (B_0 tau / 2) e^{-t/tau} turns it into Bessel's equation of order zero,

    s^2 zeta_ss + s zeta_s + s^2 zeta = 0 ,

so zeta(t) = a J_0(s(t)) + c Y_0(s(t)) with

    theta(t) = (B_0 tau / 2) (1 - e^{-t/tau}) = s(0) - s(t) ,
    dzeta/dt = (s/tau) (a J_1(s) + c Y_1(s)) ,
    z = e^{i theta} zeta ,   w = e^{i theta} (dzeta/dt + i (s/tau) zeta) .

The equation is real, so a complex initial condition splits into two real
solves against the same basis.  The parallel motion is free: B has no
transverse component and E no z-component, so v_z is constant and
z_par(t) = z_par(0) + v_z t exactly.

The consequence for this wave is the whole reason it was worth deriving.  The
reference of Section~\\ref{sec:family} is DOP853 at rtol 1e-12, and W12
measured vps4 at equal cost reaching 6.21e-12 Larmor radii against it.  The
closed form says what the reference itself is worth: `sw1_reference.py`
measures the DOP853 position error at t = 120 as 8.7e-12 Larmor radii, so that
number was the reference and not the scheme.  Evaluating the Bessel basis in
mpmath removes the floor entirely -- each sample is an independent function
evaluation with no accumulation.

WHY THE BASIS IS COMPUTED IN mpmath AND USED IN float64
--------------------------------------------------------
s(0) = B_0 tau / 2 = 60000 for the paper's tau.  A double-precision J_0 at that
argument loses about four digits to argument reduction, which puts its error at
1e-12 -- exactly the size of the effect being measured.  mpmath carries the
argument at 40 digits and returns J_0, Y_0, J_1, Y_1 correct to far below
double precision.  The linear combination that follows is done in float64:
the coefficients are O(3e2) and the basis values O(1e-3), so rounding the basis
to float64 costs 1e-19 absolute and 3e-17 after the combination, which is two
orders below double-precision unit roundoff on an O(1) trajectory.

FREQUENCY CONVENTION AND WHAT A BAND COSTS
-------------------------------------------
Omega_c = 1 in these units, so f/Omega_c is the dimensionless ratio
omega/omega_c and the gyration line sits at 1 under either the ordinary or the
angular reading.  A record of N samples at spacing h resolves bins spaced
2 pi / (N h), so the number of independent bins strictly inside f/Omega_c < nu
is

    n_bins = nu * (N h) / (2 pi) = nu * (number of gyro-orbits) .

At nu = 0.2 that is one fifth of the gyro-orbits in the record.  The window of
Section~\\ref{sec:family} is 19.1 gyro-orbits and therefore carries 3.8 bins in
the band the claim is about.  This identity is why three horizons are run.

SEEDS
-----
`sw1_reference.py` and `sw2_spectra.py` draw nothing: one initial condition,
fixed schemes, committed checkpoints, deterministic throughout.  The number of
random draws in this directory is declared before the runs as **eight**, all of
them in `sw3_ensemble.py`, from one generator built once outside every loop
with the seed `SPECTRAL_SEED` below.  That seed lies in a block no other script
in the bundle touches (external_arch uses 9.0e6 to 9.8e6, everything else is
below 5e5).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, os.path.join(EXP, "external_arch"), os.path.join(EXP, "classical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ea_common as EA                                       # noqa: E402
import schemes as S                                          # noqa: E402

TWO_PI = 2.0 * math.pi

# ------------------------------------------------------------------ setup --
# Every constant below is imported or copied from the setup of
# \ref{sec:app_setups}; nothing here is a new choice.
Q, M = EA.Q, EA.M
B0 = EA.B0
TAU = EA.TAU_PAPER                # 1.2e5, the decaying case of the paper
DT = EA.DT                        # 0.3, Omega h = 0.3
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])

#: the three record lengths, in output samples at h = 0.3.  Declared before the
#: runs together with the reason for each.
HORIZONS = {
    # the window of Table 4 and Table 5, t = 120, 19.1 gyro-orbits.  This is the
    # setting the claim's own numbers (0.42 and 3.5e-3 Larmor radii) live in.
    # It carries 3.8 bins below f/Omega_c = 0.2.
    "H1_paper": 400,
    # the shortest record in which the band carries twenty bins: 100.3
    # gyro-orbits.  Also, to within the resolution of Section 7, the horizon at
    # which the corrector's trajectory advantage reaches unity.
    "H2_100orb": 2100,
    # 391.2 gyro-orbits, 78.2 bins in the band, a power of two for the FFT.
    "H3_long": 8192,
}

BAND = 0.2                        # f/Omega_c, the first author's number, verbatim

#: flops per step.  Boris and the corrector are the figures printed in
#: \ref{sec:app_setups}; vps2 and vps4 come from the committed flop model of
#: ../classical/schemes.py; gl4 is iterative and is priced from its measured
#: mean iteration count through S.flops_gl4.
FLOPS_BORIS = 113
FLOPS_CORRECTOR = 114091
FLOPS_VPS2 = S.FLOPS_PER_STEP["vps2"]      # 91
FLOPS_VPS4 = S.FLOPS_PER_STEP["vps4"]      # 273

SPECTRAL_SEED = 13_000_000
N_ENSEMBLE_DRAWS = 8


# ------------------------------------------------------- exact reference ---
def bessel_basis(ts, tau=TAU, dps=40):
    """J_0, Y_0, J_1, Y_1 of s(t) = (B_0 tau/2) e^{-t/tau}, at 40 digits.

    Returned in float64; see the module docstring for why that is enough and
    why a float64 *argument* would not be.  The basis depends on the time grid
    alone, so it is shared by every initial condition on that grid.

    The Larmor phase needs the same care as the Bessel argument and for the
    same reason.  theta(t) = s(0) - s(t) is a difference of two numbers of size
    s(0) = 6e4, so forming it in float64 from a float64 s(t) would carry an
    absolute error of 6e-12 radians -- which is 6e-12 Larmor radii of position
    error, the size of the whole effect this wave measures.  theta is therefore
    evaluated in mpmath as s(0)(1 - e^{-t/tau}) and reduced modulo 2 pi before
    it is rounded, so that what reaches float64 is an angle in [0, 2 pi) with
    an absolute error of 1e-16.
    """
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        s0 = mp.mpf(B0) * mp.mpf(tau) / 2
        two_pi = 2 * mp.pi
        n = len(ts)
        j0 = np.empty(n); y0 = np.empty(n); j1 = np.empty(n); y1 = np.empty(n)
        sv = np.empty(n); th = np.empty(n)
        for i, t in enumerate(ts):
            u = mp.e ** (-mp.mpf(float(t)) / mp.mpf(tau))
            s = s0 * u
            sv[i] = float(s)
            th[i] = float(mp.fmod(s0 * (1 - u), two_pi))
            j0[i] = float(mp.besselj(0, s)); y0[i] = float(mp.bessely(0, s))
            j1[i] = float(mp.besselj(1, s)); y1[i] = float(mp.bessely(1, s))
        return {"t": np.asarray(ts, dtype=float), "s": sv, "s0": float(s0),
                "theta_mod2pi": th, "j0": j0, "y0": y0, "j1": j1, "y1": y1,
                "tau": float(tau), "dps": dps}
    finally:
        mp.mp.dps = old


def exact_from_basis(basis, r0, v0):
    """The closed-form trajectory on the grid the basis was built on.

    r0, v0 are three-vectors; the third component is the free parallel motion.
    """
    tau = basis["tau"]
    s = basis["s"]
    j0, y0, j1, y1 = basis["j0"], basis["y0"], basis["j1"], basis["y1"]
    assert basis["t"][0] == 0.0, "the time grid must start at t = 0"
    A = np.array([[j0[0], y0[0]],
                  [(s[0] / tau) * j1[0], (s[0] / tau) * y1[0]]])
    z0 = complex(r0[0], r0[1])
    w0 = complex(v0[0], v0[1])
    # The field strength has to be the one the basis was built with.  This
    # function used the module-level B0 unconditionally, which is right for
    # every basis built here -- they use the same global -- but wrong for one
    # built by refcheck/rc_common.fast_basis, which takes b0 as an argument.
    # No committed number is affected: every committed call is at B0.
    b0 = basis.get("b0", B0)
    assert abs(b0 - B0) < 1e-15 or "b0" in basis, "basis built at a different B0"
    d0 = w0 - 1j * (b0 / 2.0) * z0     # zeta'(0) = w(0) - i (B_z(0)/2) z(0)
    cR = np.linalg.solve(A, np.array([z0.real, d0.real]))
    cI = np.linalg.solve(A, np.array([z0.imag, d0.imag]))
    zeta = (cR[0] * j0 + cR[1] * y0) + 1j * (cI[0] * j0 + cI[1] * y0)
    dzeta = (s / tau) * ((cR[0] * j1 + cR[1] * y1)
                         + 1j * (cI[0] * j1 + cI[1] * y1))
    ph = np.exp(1j * basis["theta_mod2pi"])
    z = ph * zeta
    w = ph * (dzeta + 1j * (s / tau) * zeta)
    n = len(s)
    out_r = np.empty((n, 3)); out_v = np.empty((n, 3))
    out_r[:, 0] = z.real; out_r[:, 1] = z.imag
    out_r[:, 2] = r0[2] + v0[2] * basis["t"]
    out_v[:, 0] = w.real; out_v[:, 1] = w.imag
    out_v[:, 2] = v0[2]
    return out_r, out_v


def exact_reference_mp(ts, r0=R0, v0=V0, tau=TAU, dps=40):
    """The same closed form carried end to end in mpmath.

    Used only to price the float64 reconstruction of `exact_from_basis` on a
    handful of samples: it is the check that rounding the basis, the
    coefficients and the phase to double precision costs less than the
    residuals being measured.  Too slow to use on a whole record.
    """
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        s0 = mp.mpf(B0) * mp.mpf(tau) / 2
        taum = mp.mpf(tau)
        j0s, y0s = mp.besselj(0, s0), mp.bessely(0, s0)
        j1s, y1s = mp.besselj(1, s0), mp.bessely(1, s0)
        A = mp.matrix([[j0s, y0s],
                       [(s0 / taum) * j1s, (s0 / taum) * y1s]])
        z0 = mp.mpc(r0[0], r0[1]); w0 = mp.mpc(v0[0], v0[1])
        d0 = w0 - mp.mpc(0, 1) * (mp.mpf(B0) / 2) * z0
        cR = mp.lu_solve(A, mp.matrix([z0.real, d0.real]))
        cI = mp.lu_solve(A, mp.matrix([z0.imag, d0.imag]))
        out_r = np.empty((len(ts), 3)); out_v = np.empty((len(ts), 3))
        for i, t in enumerate(ts):
            u = mp.e ** (-mp.mpf(float(t)) / taum)
            s = s0 * u
            j0, y0 = mp.besselj(0, s), mp.bessely(0, s)
            j1, y1 = mp.besselj(1, s), mp.bessely(1, s)
            zeta = mp.mpc(cR[0] * j0 + cR[1] * y0, cI[0] * j0 + cI[1] * y0)
            dzeta = (s / taum) * mp.mpc(cR[0] * j1 + cR[1] * y1,
                                        cI[0] * j1 + cI[1] * y1)
            ph = mp.e ** (mp.mpc(0, 1) * (s0 * (1 - u)))
            z = ph * zeta
            w = ph * (dzeta + mp.mpc(0, 1) * (s / taum) * zeta)
            out_r[i] = (float(z.real), float(z.imag),
                        r0[2] + v0[2] * float(t))
            out_v[i] = (float(w.real), float(w.imag), v0[2])
        return out_r, out_v
    finally:
        mp.mp.dps = old


def dop853_ref(ts, r0=R0, v0=V0, tau=TAU, rtol=1e-12, atol=1e-14):
    """The reference of Section 7, in three dimensions, for the adequacy check."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        b = B0 * math.exp(-t / tau)
        he = 0.5 * b / tau
        ex, ey = -he * y[1], he * y[0]
        return [y[3], y[4], y[5],
                (Q / M) * (ex + y[4] * b),
                (Q / M) * (ey - y[3] * b),
                0.0]

    sol = solve_ivp(rhs, (0.0, float(ts[-1])),
                    [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]],
                    method="DOP853", rtol=rtol, atol=atol, t_eval=ts)
    assert sol.success, sol.message
    return sol.y[:3].T, sol.y[3:].T


# ------------------------------------------------- scalar plane schemes ----
# Written as scalar Python because the sub-stepped equal-cost runs take up to
# 1e7 steps and a numpy three-vector per step costs more in interpreter
# overhead than the arithmetic.  `sw1_reference.py` checks every one of them
# against the committed numpy implementation of ../classical/schemes.py and
# against the four trajectory errors Table 4 prints.

def _rollout_boris(r0, v0, h, m, n_out, tau):
    x, y, vx, vy = r0[0], r0[1], v0[0], v0[1]
    z, vz = r0[2], v0[2]
    t = 0.0
    R = np.empty((n_out + 1, 3)); V = np.empty((n_out + 1, 3))
    R[0] = (x, y, z); V[0] = (vx, vy, vz)
    k = 0.5 * (Q / M) * h
    for i in range(1, n_out + 1):
        for _ in range(m):
            b = B0 * math.exp(-t / tau)
            he = 0.5 * b / tau
            ex, ey = -he * y, he * x
            vmx = vx + k * ex; vmy = vy + k * ey
            tz = k * b
            sz = 2.0 * tz / (1.0 + tz * tz)
            vpx = vmx + vmy * tz; vpy = vmy - vmx * tz
            vlx = vmx + vpy * sz; vly = vmy - vpx * sz
            vx = vlx + k * ex; vy = vly + k * ey
            x += vx * h; y += vy * h; z += vz * h
            t += h
        R[i] = (x, y, z); V[i] = (vx, vy, vz)
    return R, V


def _vps2_scalar(x, y, vx, vy, t, h, tau):
    hh = 0.5 * h
    x += hh * vx; y += hh * vy
    b = B0 * math.exp(-(t + hh) / tau)
    he = 0.5 * b / tau
    ex, ey = -he * y, he * x
    k = hh * (Q / M)
    vx += k * ex; vy += k * ey
    th = b * h                       # -(Q/M) |B| h with Q/M = -1
    c, s = math.cos(th), math.sin(th)
    vx, vy = vx * c - vy * s, vx * s + vy * c
    vx += k * ex; vy += k * ey
    x += hh * vx; y += hh * vy
    return x, y, vx, vy


def _rollout_vps2(r0, v0, h, m, n_out, tau):
    x, y, vx, vy = r0[0], r0[1], v0[0], v0[1]
    z, vz = r0[2], v0[2]
    t = 0.0
    R = np.empty((n_out + 1, 3)); V = np.empty((n_out + 1, 3))
    R[0] = (x, y, z); V[0] = (vx, vy, vz)
    for i in range(1, n_out + 1):
        for _ in range(m):
            x, y, vx, vy = _vps2_scalar(x, y, vx, vy, t, h, tau)
            z += vz * h
            t += h
        R[i] = (x, y, z); V[i] = (vx, vy, vz)
    return R, V


def _rollout_vps4(r0, v0, h, m, n_out, tau):
    g1 = S._G1; g0 = S._G0
    x, y, vx, vy = r0[0], r0[1], v0[0], v0[1]
    z, vz = r0[2], v0[2]
    t = 0.0
    R = np.empty((n_out + 1, 3)); V = np.empty((n_out + 1, 3))
    R[0] = (x, y, z); V[0] = (vx, vy, vz)
    for i in range(1, n_out + 1):
        for _ in range(m):
            tt = t
            for g in (g1, g0, g1):
                gh = g * h
                x, y, vx, vy = _vps2_scalar(x, y, vx, vy, tt, gh, tau)
                z += vz * gh
                tt += gh
            t += h
        R[i] = (x, y, z); V[i] = (vx, vy, vz)
    return R, V


_C1, _C2 = S._C1, S._C2
_A00, _A01 = S._A[0, 0], S._A[0, 1]
_A10, _A11 = S._A[1, 0], S._A[1, 1]


def _rollout_gl4(r0, v0, h, m, n_out, tau, tol=1e-14, maxit=60):
    """Two-stage Gauss--Legendre in the plane, arithmetic of S.make_gl4."""
    x, y, vx, vy = r0[0], r0[1], v0[0], v0[1]
    z, vz = r0[2], v0[2]
    t = 0.0
    R = np.empty((n_out + 1, 3)); V = np.empty((n_out + 1, 3))
    R[0] = (x, y, z); V[0] = (vx, vy, vz)
    iters = 0; nsteps = 0
    for i in range(1, n_out + 1):
        for _ in range(m):
            k0a = k0b = k0c = k0d = 0.0
            k1a = k1b = k1c = k1d = 0.0
            used = maxit
            t1 = t + _C1 * h; t2 = t + _C2 * h
            b1 = B0 * math.exp(-t1 / tau); e1 = 0.5 * b1 / tau
            b2 = B0 * math.exp(-t2 / tau); e2 = 0.5 * b2 / tau
            for it in range(maxit):
                d0 = h * (_A00 * k0a + _A01 * k1a)
                d1 = h * (_A00 * k0b + _A01 * k1b)
                d2 = h * (_A00 * k0c + _A01 * k1c)
                d3 = h * (_A00 * k0d + _A01 * k1d)
                X1x = x + d0; X1y = y + d1; X1vx = vx + d2; X1vy = vy + d3
                d0 = h * (_A10 * k0a + _A11 * k1a)
                d1 = h * (_A10 * k0b + _A11 * k1b)
                d2 = h * (_A10 * k0c + _A11 * k1c)
                d3 = h * (_A10 * k0d + _A11 * k1d)
                X2x = x + d0; X2y = y + d1; X2vx = vx + d2; X2vy = vy + d3
                n0a = X1vx; n0b = X1vy
                n0c = (Q / M) * (-e1 * X1y + X1vy * b1)
                n0d = (Q / M) * (e1 * X1x - X1vx * b1)
                n1a = X2vx; n1b = X2vy
                n1c = (Q / M) * (-e2 * X2y + X2vy * b2)
                n1d = (Q / M) * (e2 * X2x - X2vx * b2)
                dmax = max(abs(n0a - k0a), abs(n0b - k0b), abs(n0c - k0c),
                           abs(n0d - k0d), abs(n1a - k1a), abs(n1b - k1b),
                           abs(n1c - k1c), abs(n1d - k1d))
                k0a, k0b, k0c, k0d = n0a, n0b, n0c, n0d
                k1a, k1b, k1c, k1d = n1a, n1b, n1c, n1d
                if dmax < tol:
                    used = it + 1
                    break
            hh = 0.5 * h
            x += hh * (k0a + k1a); y += hh * (k0b + k1b)
            vx += hh * (k0c + k1c); vy += hh * (k0d + k1d)
            z += vz * h
            t += h
            iters += used; nsteps += 1
        R[i] = (x, y, z); V[i] = (vx, vy, vz)
    return R, V, (iters / nsteps if nsteps else float("nan"))


ROLLOUT = {"boris": _rollout_boris, "vps2": _rollout_vps2,
           "vps4": _rollout_vps4, "gl4": _rollout_gl4}


def run_classical(name, h, m, n_out, r0=R0, v0=V0, tau=TAU):
    """One classical run.  `m` sub-steps of size `h` between output samples."""
    if name == "gl4":
        R, V, ni = _rollout_gl4(r0, v0, h, m, n_out, tau)
        return R, V, {"mean_iters": float(ni),
                      "flops_per_step": float(S.flops_gl4(ni))}
    R, V = ROLLOUT[name](r0, v0, h, m, n_out, tau)
    fps = {"boris": FLOPS_BORIS, "vps2": FLOPS_VPS2, "vps4": FLOPS_VPS4}[name]
    return R, V, {"flops_per_step": float(fps)}


# ------------------------------------------------------------- corrector ---
_MODEL = None


def load_corrector():
    """The checkpoint as committed.  Nothing here retrains or fine-tunes."""
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_default_dtype(torch.float64)
        sys.path.insert(0, ROOT)
        from common import CHECKPOINT_DIR
        from training.train_corrector_b4 import DefectNet
        m = DefectNet(n_in=13)
        m.load_state_dict(torch.load(
            os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
            map_location="cpu"))
        m.eval()
        _MODEL = m
    return _MODEL


def run_corrector(n_out, h=DT, r0=R0, v0=V0, tau=TAU):
    """Boris step, learned correction, symmetric projection onto the speed.

    Arithmetic identical to experiments/classical/run.py:integrate_hybrid,
    which is what Table 4 scores; `sw1_reference.py` checks that the two agree.
    """
    import torch
    model = load_corrector()
    R = np.empty((n_out + 1, 3)); V = np.empty((n_out + 1, 3))
    r = np.array(r0, float); v = np.array(v0, float)
    R[0] = r; V[0] = v
    t = 0.0
    k = 0.5 * (Q / M) * h
    xin = np.empty(13)
    xin[12] = h
    with torch.no_grad():
        for i in range(1, n_out + 1):
            b = B0 * math.exp(-t / tau)
            he = 0.5 * b / tau
            Ev = np.array([-he * r[1], he * r[0], 0.0])
            Bv = np.array([0.0, 0.0, b])
            # Boris step, models/boris.py arithmetic
            vm = v + k * Ev
            tv = k * Bv
            sv = 2.0 * tv / (1.0 + float(tv @ tv))
            vp = vm + np.cross(vm, tv)
            vpl = vm + np.cross(vp, sv)
            v_b = vpl + k * Ev
            r_b = r + v_b * h
            xin[0:3] = r; xin[3:6] = v; xin[6:9] = Bv; xin[9:12] = Ev
            d = model(torch.from_numpy(xin)[None, :]).numpy()[0]
            dr, dv = d[:3], d[3:]
            nb = float(np.linalg.norm(v_b))
            vh = v_b / max(nb, 1e-300)
            dv = dv - float(dv @ vh) * vh
            v_new = v_b + dv
            v_new = v_new * (nb / max(float(np.linalg.norm(v_new)), 1e-300))
            r, v = r_b + dr, v_new
            t += h
            R[i] = r; V[i] = v
    return R, V, {"flops_per_step": float(FLOPS_CORRECTOR)}


# ------------------------------------------------------------ spectra ------
def hann(n):
    return 0.5 - 0.5 * np.cos(TWO_PI * np.arange(n) / n)


def blackmanharris(n):
    """Four-term Blackman--Harris, sidelobes at -92 dB against Hann's -31 dB.

    Carried as the robustness window.  A scheme whose residual is one strong
    line at Omega_c has almost nothing genuinely inside f/Omega_c < 0.2, and
    what the periodogram reports there can be the skirt of that line rather
    than the scheme.  If an in-band power falls when the window is changed for
    one with lower sidelobes, that power was leakage; if it does not move, it
    is the scheme.  Every in-band figure in this directory is reported under
    both windows for exactly this reason.
    """
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    k = TWO_PI * np.arange(n) / n
    return (a[0] - a[1] * np.cos(k) + a[2] * np.cos(2 * k)
            - a[3] * np.cos(3 * k))


WINDOWS = {"hann": hann, "blackmanharris": blackmanharris}


def psd(series, dt, window="hann"):
    """One-sided periodogram of a real vector series.

    `series` is (N, ncomp).  The returned S sums the per-component spectra and
    is normalised so that sum(S) is the *window-weighted* mean of |series|^2.
    For the position channel that makes the total power the square of the
    window-weighted root-mean-square position error -- not of the plain
    root-mean-square, which the window reweights whenever the residual grows
    over the record, and which is reported separately.  What the normalisation
    buys is that

        P_band(A)/P_band(B) = [P_tot(A)/P_tot(B)] * [c_A/c_B],   c = P_band/P_tot

    is an identity rather than a fit.  Parseval is checked on every series by
    `spectral_selfcheck`.
    """
    if series.ndim == 1:
        series = series[:, None]
    n = series.shape[0]
    w = WINDOWS[window](n)
    U = float(np.sum(w * w))
    Y = np.fft.rfft(series * w[:, None], axis=0)
    # For a real sequence y, sum_k |Y_k|^2 = n sum_n y_n^2 over the full
    # two-sided transform.  Folding onto the one-sided grid doubles every bin
    # except DC and, for even n, Nyquist.  Dividing by n U therefore makes
    # sum(S) = sum_n w_n^2 x_n^2 / sum_n w_n^2, the window-weighted mean square.
    fold = np.full(Y.shape[0], 2.0)
    fold[0] = 1.0
    if n % 2 == 0:
        fold[-1] = 1.0
    S = (np.abs(Y) ** 2).sum(axis=1) * fold / (n * U)
    nu = TWO_PI * np.arange(S.shape[0]) / (n * dt)
    return nu, S


def spectral_selfcheck(series, dt, rtol=1e-12, window="hann"):
    """Parseval: the summed one-sided spectrum is the weighted mean square."""
    nu, S = psd(series, dt, window)
    n = series.shape[0]
    w = WINDOWS[window](n)
    lhs = float(S.sum())
    rhs = float(np.sum((w ** 2)[:, None] * series ** 2) / np.sum(w ** 2))
    ok = abs(lhs - rhs) <= rtol * max(abs(lhs), abs(rhs))
    return ok, lhs, rhs


def band_powers(series, dt, edge=BAND, window="hann"):
    """In-band, out-of-band and total power, with and without the DC bin."""
    nu, S = psd(series, dt, window)
    inb = nu < edge
    tot = float(S.sum())
    pb = float(S[inb].sum())
    pb_nodc = float(S[inb][1:].sum()) if inb.sum() > 1 else 0.0
    return {"p_band": pb, "p_out": tot - pb, "p_total": tot,
            "p_band_nodc": pb_nodc, "n_bins_in_band": int(inb.sum()),
            "rms_windowed": float(np.sqrt(tot))}


def band_sweep(series, dt, edges, window="hann"):
    nu, S = psd(series, dt, window)
    tot = float(S.sum())
    out = []
    for e in edges:
        m = nu < e
        out.append(float(S[m].sum()))
    return out, tot


SWEEP_EDGES = [0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2,
               0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25,
               1.5, 2.0, 2.5, 3.0, 5.0, 10.5]


# ------------------------------------------------------------- channels ----
def channels(R, V, Rr, Vr):
    """The two residual series declared in the pre-registration.

    Both are reported for every scheme regardless of which is the more
    convenient.  Position: the error vector in Larmor radii, so that the total
    spectral power is the mean square position error.  Energy: the relative
    error of the kinetic energy against E_ref(0).
    """
    dr = R - Rr
    e = 0.5 * np.sum(V * V, axis=1)
    er = 0.5 * np.sum(Vr * Vr, axis=1)
    de = ((e - er) / er[0])[:, None]
    return {"position": dr, "energy": de}


def time_metrics(series):
    """max, rms, final drift and the running-max envelope at ten fractions.

    Declared in the pre-registration before the runs, for every channel.
    """
    a = np.linalg.norm(series, axis=1) if series.ndim > 1 else np.abs(series)
    env = np.maximum.accumulate(a)
    n = len(a)
    idx = [max(0, int(round(f * (n - 1)))) for f in np.linspace(0.1, 1.0, 10)]
    return {"max": float(a.max()), "rms": float(np.sqrt(np.mean(a ** 2))),
            "final": float(a[-1]),
            "envelope_deciles": [float(env[i]) for i in idx]}
