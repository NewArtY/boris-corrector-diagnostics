"""Fast validated re-implementation of the B4 experiment for long horizons.

Scalar Boris (B along z, E in xy) + numpy forward pass of DefectNet.
Validated against the reference implementation before use.
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from training.train_corrector_b4 import DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN

TWO_PI = 2.0 * np.pi


def load_net_numpy():
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"),
                                 map_location="cpu"))
    m.eval()
    Ws, bs = [], []
    for layer in m.net:
        if isinstance(layer, torch.nn.Linear):
            Ws.append(layer.weight.detach().numpy().copy())
            bs.append(layer.bias.detach().numpy().copy())
    return (Ws, bs, m.x_mean.numpy().copy(), m.x_std.numpy().copy(),
            m.y_scale.numpy().copy(), m)


def make_forward(Ws, bs, x_mean, x_std, y_scale):
    nW = len(Ws)
    def fwd(x):
        z = (x - x_mean) / x_std
        for i in range(nW - 1):
            z = np.tanh(Ws[i] @ z + bs[i])
        return (Ws[-1] @ z + bs[-1]) * y_scale
    return fwd


def run(mode, tau, dt, n_steps, n_samples=2000, fwd=None, B0=1.0,
        r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), keep_traj_every=0):
    """mode: 'boris' | 'raw' | 'proj'.  Returns thinned diagnostics."""
    rx, ry, rz = map(float, r0)
    vx, vy, vz = map(float, v0)
    t = 0.0
    k = -0.5 * dt                      # qmdt2 with q=-1, m=1
    E0 = 0.5 * (vx * vx + vy * vy + vz * vz)
    inv_tau = 1.0 / tau

    stride = max(1, n_steps // n_samples)
    ts, mus, Es, envs = [], [], [], []
    run_max = 0.0
    traj_t, traj_r = [], []
    kt = keep_traj_every

    x = np.empty(13)
    for i in range(1, n_steps + 1):
        Bz = B0 * np.exp(-t * inv_tau)
        fac = 0.5 * Bz * inv_tau       # = -0.5*dBdt
        Ex = -fac * ry
        Ey = fac * rx
        if mode != 'boris':
            x[0] = rx; x[1] = ry; x[2] = rz
            x[3] = vx; x[4] = vy; x[5] = vz
            x[6] = 0.0; x[7] = 0.0; x[8] = Bz
            x[9] = Ex; x[10] = Ey; x[11] = 0.0
            x[12] = dt
            d = fwd(x)
        # --- Boris ---
        kEx = k * Ex; kEy = k * Ey
        vmx = vx + kEx; vmy = vy + kEy; vmz = vz
        tz = k * Bz
        sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vmx + vmy * tz; vpy = vmy - vmx * tz
        vplx = vmx + vpy * sz; vply = vmy - vpx * sz
        vbx = vplx + kEx; vby = vply + kEy; vbz = vmz
        rbx = rx + vbx * dt; rby = ry + vby * dt; rbz = rz + vbz * dt
        # --- correction ---
        if mode == 'boris':
            rx, ry, rz = rbx, rby, rbz
            vx, vy, vz = vbx, vby, vbz
        else:
            dvx, dvy, dvz = d[3], d[4], d[5]
            if mode == 'proj':
                nb = np.sqrt(vbx * vbx + vby * vby + vbz * vbz)
                inb = 1.0 / max(nb, 1e-300)
                hx, hy, hz = vbx * inb, vby * inb, vbz * inb
                dot = dvx * hx + dvy * hy + dvz * hz
                dvx -= dot * hx; dvy -= dot * hy; dvz -= dot * hz
                nvx = vbx + dvx; nvy = vby + dvy; nvz = vbz + dvz
                nn = np.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
                sc = nb / max(nn, 1e-300)
                vx, vy, vz = nvx * sc, nvy * sc, nvz * sc
            else:
                vx, vy, vz = vbx + dvx, vby + dvy, vbz + dvz
            rx, ry, rz = rbx + d[0], rby + d[1], rbz + d[2]
        t += dt
        # --- diagnostics ---
        Ecur = 0.5 * (vx * vx + vy * vy + vz * vz)
        Bcur = B0 * np.exp(-t * inv_tau)
        Ephys = E0 * np.exp(-t * inv_tau)
        dev = abs(Ecur - Ephys) / E0          # numerical energy error, /E0
        if dev > run_max:
            run_max = dev
        if kt and (i % kt == 0):
            traj_t.append(t); traj_r.append((rx, ry, rz))
        if i % stride == 0 or i == n_steps:
            ts.append(t)
            Es.append(dev)
            mus.append(abs((Ecur / Bcur) / (E0 / B0) - 1.0))
            envs.append(run_max)
            run_max = 0.0
    out = {"t": np.array(ts), "e_err": np.array(Es), "mu_err": np.array(mus),
           "env": np.array(envs), "E0": E0}
    if kt:
        out["traj_t"] = np.array(traj_t)
        out["traj_r"] = np.array(traj_r)
    return out


def fine_reference(tau, dt_fine, n_fine, sample_every, B0=1.0,
                   r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0)):
    """Fine Boris reference; stores r,E only every `sample_every` fine steps."""
    rx, ry, rz = map(float, r0)
    vx, vy, vz = map(float, v0)
    t = 0.0
    k = -0.5 * dt_fine
    inv_tau = 1.0 / tau
    E0 = 0.5 * (vx * vx + vy * vy + vz * vz)
    ts, rs, Es = [], [], []
    for i in range(1, n_fine + 1):
        Bz = B0 * np.exp(-t * inv_tau)
        fac = 0.5 * Bz * inv_tau
        Ex = -fac * ry; Ey = fac * rx
        kEx = k * Ex; kEy = k * Ey
        vmx = vx + kEx; vmy = vy + kEy; vmz = vz
        tz = k * Bz
        sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vmx + vmy * tz; vpy = vmy - vmx * tz
        vplx = vmx + vpy * sz; vply = vmy - vpx * sz
        vx = vplx + kEx; vy = vply + kEy; vz = vmz
        rx += vx * dt_fine; ry += vy * dt_fine; rz += vz * dt_fine
        t += dt_fine
        if i % sample_every == 0:
            ts.append(t); rs.append((rx, ry, rz))
            Es.append(0.5 * (vx * vx + vy * vy + vz * vz))
    return np.array(ts), np.array(rs), np.array(Es), E0
