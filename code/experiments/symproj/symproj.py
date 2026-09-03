"""Symmetric projection (Hairer, BIT 40, 726-734, 2000) for the B4 hybrid.

One-sided projection as shipped:      step, then rescale |v| to |v_Boris|.
Symmetric projection (Hairer 2.2):    perturb OFF the manifold by mu*grad g,
                                      step, project back with the SAME mu,
                                      solve g(y_{n+1}) = 0 for mu.

With g(v) = |v|^2 - c, grad g = 2v, so the perturbation is a rescaling:
    a  = 1 + 2 mu          (applied to the input velocity)
    b  = 1 / (1 - 2 mu)    (applied to the output velocity)
Both reduce to (1 + 2 mu) to first order -- that is the symmetry.

The constraint target keeps the Article's semantics: the LEARNED CORRECTION
must be energy neutral, i.e. |v_{n+1}| = |v_Boris(v_n)| with v_Boris taken
from the UNperturbed state.

Base methods:
  'shipped'   r_{n+1} = r_n + v_Boris * dt   (integer-time v; 1st order, NOT symmetric)
  'staggered' v at half-integer times          (2nd order, time-symmetric)

Hairer's theorem needs a SYMMETRIC base method, so both are measured.
"""
import os
import sys
import json
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from training.train_corrector_b4 import DefectNet, DT_WORK, TAU_MAIN

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def load_forward():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    Ws, bs = [], []
    for layer in m.net:
        if isinstance(layer, torch.nn.Linear):
            Ws.append(layer.weight.detach().numpy().copy())
            bs.append(layer.bias.detach().numpy().copy())
    x_mean = m.x_mean.numpy().copy()
    x_std = m.x_std.numpy().copy()
    y_scale = m.y_scale.numpy().copy()
    nW = len(Ws)

    def fwd(x):
        z = (x - x_mean) / x_std
        for i in range(nW - 1):
            z = np.tanh(Ws[i] @ z + bs[i])
        return (Ws[-1] @ z + bs[-1]) * y_scale
    return fwd


def features(rx, ry, rz, vx, vy, vz, Bz, Ex, Ey, dt, buf):
    buf[0] = rx; buf[1] = ry; buf[2] = rz
    buf[3] = vx; buf[4] = vy; buf[5] = vz
    buf[6] = 0.0; buf[7] = 0.0; buf[8] = Bz
    buf[9] = Ex; buf[10] = Ey; buf[11] = 0.0
    buf[12] = dt
    return buf


def boris_kick(vx, vy, vz, Ex, Ey, Bz, k):
    """Full Boris velocity update (both E half-kicks + rotation about z)."""
    kEx = k * Ex
    kEy = k * Ey
    vmx = vx + kEx
    vmy = vy + kEy
    vmz = vz
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vmx + vmy * tz
    vpy = vmy - vmx * tz
    vplx = vmx + vpy * sz
    vply = vmy - vpx * sz
    return vplx + kEx, vply + kEy, vmz


# --------------------------------------------------------------------------
# main integrator
# --------------------------------------------------------------------------
def run(mode, tau, dt, n_steps, fwd=None, base="shipped", n_samples=2000,
        B0=1.0, r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0),
        mu_tol=1e-15, mu_maxit=6, freeze_net=True, collect_mu=False,
        keep_traj=False, zero_dr=False, zero_dv=False):
    """mode: 'boris' | 'raw' | 'proj' | 'sym'.

    freeze_net=True evaluates the network once per step (at the warm-started
    mu) and holds it fixed while solving for mu; the Boris part is re-evaluated
    exactly.  freeze_net=False re-evaluates the network at every iteration.
    """
    rx, ry, rz = map(float, r0)
    vx, vy, vz = map(float, v0)
    t = 0.0
    k = -0.5 * dt
    inv_tau = 1.0 / tau
    E0 = 0.5 * (vx * vx + vy * vy + vz * vz)

    # staggered base: back the velocity up half a step
    half = False
    if base == "staggered":
        half = True
        Bz0 = B0
        fac0 = 0.5 * Bz0 * inv_tau
        vx, vy, vz = boris_kick(vx, vy, vz, -fac0 * ry, fac0 * rx, Bz0, +0.25 * dt)

    traj = np.empty((n_steps, 3)) if keep_traj else None
    stride = max(1, n_steps // n_samples)
    ts, Es, envs, mus_inv = [], [], [], []
    mu_hist = []
    run_max = 0.0
    mu = 0.0
    nfev_net = 0
    nfev_boris = 0
    buf = np.empty(13)

    for i in range(1, n_steps + 1):
        Bz = B0 * np.exp(-t * inv_tau)
        fac = 0.5 * Bz * inv_tau
        Ex = -fac * ry
        Ey = fac * rx

        # unperturbed Boris -> constraint target
        vb0x, vb0y, vb0z = boris_kick(vx, vy, vz, Ex, Ey, Bz, k)
        nfev_boris += 1
        target = np.sqrt(vb0x * vb0x + vb0y * vb0y + vb0z * vb0z)

        if mode == "boris":
            vnx, vny, vnz = vb0x, vb0y, vb0z
            vdx, vdy, vdz = vb0x, vb0y, vb0z      # velocity used for drift
            drx = dry = drz = 0.0
        else:
            if mode == "sym":
                # ---- symmetric projection: solve for mu ----
                d_frozen = None
                if freeze_net:
                    a = 1.0 + 2.0 * mu
                    features(rx, ry, rz, a * vx, a * vy, a * vz, Bz, Ex, Ey, dt, buf)
                    d_frozen = fwd(buf).copy()
                    nfev_net += 1

                def residual(m_):
                    nonlocal nfev_net, nfev_boris
                    a_ = 1.0 + 2.0 * m_
                    b_ = 1.0 / (1.0 - 2.0 * m_)
                    ax, ay, az = a_ * vx, a_ * vy, a_ * vz
                    bx_, by_, bz_ = boris_kick(ax, ay, az, Ex, Ey, Bz, k)
                    nfev_boris += 1
                    if freeze_net:
                        dd = d_frozen
                    else:
                        features(rx, ry, rz, ax, ay, az, Bz, Ex, Ey, dt, buf)
                        dd = fwd(buf)
                        nfev_net += 1
                    nx_ = b_ * (bx_ + dd[3])
                    ny_ = b_ * (by_ + dd[4])
                    nz_ = b_ * (bz_ + dd[5])
                    return (np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) - target,
                            nx_, ny_, nz_, bx_, by_, bz_, dd)

                # secant, warm-started from the previous step
                m0 = mu
                f0, *_ = residual(m0)
                m1 = m0 - f0 / (4.0 * max(target, 1e-300))   # analytic slope
                f1, nx, ny, nz, bx_, by_, bz_, dd = residual(m1)
                it = 2
                while abs(f1) > mu_tol * max(target, 1.0) and it < mu_maxit:
                    denom = (f1 - f0)
                    if abs(denom) < 1e-300:
                        break
                    m2 = m1 - f1 * (m1 - m0) / denom
                    m0, f0 = m1, f1
                    m1 = m2
                    f1, nx, ny, nz, bx_, by_, bz_, dd = residual(m1)
                    it += 1
                mu = m1
                if collect_mu:
                    mu_hist.append(mu)
                vnx, vny, vnz = nx, ny, nz
                vdx, vdy, vdz = bx_, by_, bz_
                drx, dry, drz = dd[0], dd[1], dd[2]
                if zero_dr:
                    drx = dry = drz = 0.0
            else:
                features(rx, ry, rz, vx, vy, vz, Bz, Ex, Ey, dt, buf)
                d = fwd(buf)
                nfev_net += 1
                dvx, dvy, dvz = d[3], d[4], d[5]
                if zero_dv:
                    dvx = dvy = dvz = 0.0
                if mode == "proj":
                    nb = target
                    inb = 1.0 / max(nb, 1e-300)
                    hx, hy, hz = vb0x * inb, vb0y * inb, vb0z * inb
                    dot = dvx * hx + dvy * hy + dvz * hz
                    dvx -= dot * hx; dvy -= dot * hy; dvz -= dot * hz
                    nvx = vb0x + dvx; nvy = vb0y + dvy; nvz = vb0z + dvz
                    nn = np.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
                    sc = nb / max(nn, 1e-300)
                    vnx, vny, vnz = nvx * sc, nvy * sc, nvz * sc
                else:                                   # raw
                    vnx, vny, vnz = vb0x + dvx, vb0y + dvy, vb0z + dvz
                vdx, vdy, vdz = vb0x, vb0y, vb0z
                drx, dry, drz = d[0], d[1], d[2]
                if zero_dr:
                    drx = dry = drz = 0.0

        # position update
        if base == "staggered":
            rx += vnx * dt; ry += vny * dt; rz += vnz * dt
        else:
            rx += vdx * dt; ry += vdy * dt; rz += vdz * dt
        rx += drx; ry += dry; rz += drz

        v_prev = (vx, vy, vz)
        vx, vy, vz = vnx, vny, vnz
        t += dt
        if keep_traj:
            traj[i - 1, 0] = rx; traj[i - 1, 1] = ry; traj[i - 1, 2] = rz

        # ---- diagnostics ----
        if half:      # re-centre half-integer velocity to integer time
            cx = 0.5 * (v_prev[0] + vx); cy = 0.5 * (v_prev[1] + vy); cz = 0.5 * (v_prev[2] + vz)
        else:
            cx, cy, cz = vx, vy, vz
        Ecur = 0.5 * (cx * cx + cy * cy + cz * cz)
        Ephys = E0 * np.exp(-t * inv_tau)
        dev = abs(Ecur - Ephys) / E0
        if dev > run_max:
            run_max = dev
        if i % stride == 0 or i == n_steps:
            ts.append(t); Es.append(dev); envs.append(run_max)
            run_max = 0.0

    out = {"t": np.array(ts), "e_err": np.array(Es), "env": np.array(envs),
           "nfev_net": nfev_net, "nfev_boris": nfev_boris, "E0": E0}
    if collect_mu:
        out["mu"] = np.array(mu_hist)
    if keep_traj:
        out["traj"] = traj
    return out


def envelope_exponent(t, env, decades=2.0):
    """Power-law fit of the running-max envelope over the last `decades`."""
    env = np.maximum.accumulate(env)
    sel = (t > t[-1] / 10.0 ** decades) & (env > 0)
    if sel.sum() < 10:
        return float("nan")
    p = np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)
    return float(p[0])
