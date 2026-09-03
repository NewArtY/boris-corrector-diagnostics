"""
integrator_runner.py
======================
Common helper that loads all trained models and exposes a uniform
`integrate(name, field, r0, v0, dt, n_steps)` function so that diagnostics
and figure scripts can benchmark the Boris baseline and all four neural
integrators identically.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CHECKPOINT_DIR, SEED, set_global_seed
from models.boris import boris_step
from models.pinn_symplectic import build_model as build_pinn
from models.hnn import build_model as build_hnn
from models.sympnet import build_model as build_symp
from models.boris_corrector import build_model as build_bc

set_global_seed(SEED)

_MODEL_CACHE = {}


def _load(name):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if name == "pinn_symplectic":
        m = build_pinn()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "pinn_symplectic.pt"),
                                      map_location="cpu"))
    elif name == "hnn":
        m = build_hnn()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "hnn.pt"), map_location="cpu"))
    elif name == "sympnet":
        m = build_symp()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "sympnet.pt"), map_location="cpu"))
    elif name == "boris_corrector":
        m = build_bc()
        m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "boris_corrector.pt"),
                                      map_location="cpu"))
    else:
        raise ValueError(f"Unknown model: {name}")
    m.eval()
    _MODEL_CACHE[name] = m
    return m


def integrate(name, field, r0, v0, t0, dt, n_steps, q=-1.0, m=1.0):
    """Integrate n_steps starting from (r0, v0, t0) using integrator `name`.

    name in {"boris", "pinn_symplectic", "hnn", "sympnet", "boris_corrector"}
    Returns rs, vs, ts arrays of shape (n_steps+1, 3) / (n_steps+1,).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    rs = np.zeros((n_steps + 1, 3))
    vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    rs[0], vs[0], ts[0] = r0, v0, t0
    r, v, t = r0.copy(), v0.copy(), t0

    if name == "boris":
        for i in range(1, n_steps + 1):
            r, v = boris_step(r, v, t, dt, field, q=q, m=m)
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
        return rs, vs, ts

    net = _load(name)

    with torch.no_grad():
        for i in range(1, n_steps + 1):
            B = field.B(r, t)
            if name == "pinn_symplectic":
                r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
                v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                B_t = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
                dt_t = torch.tensor([[dt]], dtype=torch.float32)
                r_next, v_next = net(r_t, v_t, B_t, dt_t)
                r, v = r_next.squeeze(0).numpy(), v_next.squeeze(0).numpy()
            elif name == "hnn":
                r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
                v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                B_t = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
                r, v = net.step(r_t, v_t, B_t, dt, q=q, m=m)
                r, v = r.squeeze(0).numpy(), v.squeeze(0).numpy()
            elif name == "sympnet":
                r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
                v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                r_next, v_next = net(r_t, v_t, torch.tensor(dt, dtype=torch.float32))
                r, v = r_next.squeeze(0).numpy(), v_next.squeeze(0).numpy()
            elif name == "boris_corrector":
                r_b, v_b = boris_step(r, v, t, dt, field, q=q, m=m)
                r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(0)
                v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                B_t = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
                dt_t = torch.tensor([[dt]], dtype=torch.float32)
                dr, dv = net(r_t, v_t, B_t, dt_t)
                r = r_b + dr.squeeze(0).numpy()
                v = v_b + dv.squeeze(0).numpy()
            else:
                raise ValueError(f"Unknown model: {name}")
            t += dt
            rs[i], vs[i], ts[i] = r, v, t

    return rs, vs, ts


ALL_INTEGRATORS = ["boris", "pinn_symplectic", "hnn", "sympnet", "boris_corrector"]
