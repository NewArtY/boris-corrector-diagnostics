"""
varint.py -- variational (discrete-Lagrangian) integrator for the B4 field,
with an optional frozen random defect dL_d living INSIDE the Lagrangian.

Physics (normalised units of common.py: m=1, q=-1, B0=1, Omega_c=1):

    B(t)  = B0 exp(-t/tau) z_hat
    A(q,t)= (Bz(t)/2) (-q_y, q_x, 0)          symmetric gauge, curl A = B z_hat
    phi   = 0   =>   E = -dA/dt = (Bz/(2 tau)) (-q_y, q_x, 0)

    which is byte-identical to fields/decaying_field.py:
        fac = 0.5*Bz/tau ;  E = fac*(-q_y, q_x, 0)          [verified in tests]

    L(q, qdot, t) = 1/2 |qdot|^2 + q_c A.qdot ,  q_c = -1
                  = 1/2 |qdot|^2 - A.qdot

Midpoint discrete Lagrangian (Marsden-West):

    L_d(q_k, q_k1, t_k) = |D|^2/(2h) - A(qm, tm).D ,
    D = q_k1 - q_k,  qm = (q_k+q_k1)/2,  tm = t_k + h/2

Discrete Euler-Lagrange in position-momentum form:

    p_k    = -D1 L_d(q_k, q_k1, t_k)          -> solve for q_k1
    p_{k+1}=  D2 L_d(q_k, q_k1, t_k)          -> explicit

For the unperturbed L_d this is LINEAR in D (no Newton needed):

    p_k = M D + b ,  a = Bz(tm)/2
    M   = [[1/h,  a , 0], [ -a , 1/h, 0], [0, 0, 1/h]]
    b   = a (q_k_y, -q_k_x, 0)
    p_{k+1} = D/h + a (q_k_y, -q_k_x, 0)      (mixed terms cancel exactly)

The map (q_k,p_k) -> (q_k1,p_k1) is symplectic for ANY smooth L_d with
non-degenerate D1D2 L_d -- that is the structural claim Ф0.2 tests.

Velocity is recovered from the discrete momentum by  v = p + A(q,t).
"""

import numpy as np

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# Field helpers (single source of truth for A, B, E)
# --------------------------------------------------------------------------- #

def Bz_of(t, tau, B0=1.0):
    return B0 * np.exp(-t / tau)


def A_of(q, t, tau, B0=1.0):
    """Vector potential, symmetric gauge. q shape (...,3)."""
    a = 0.5 * Bz_of(t, tau, B0)
    out = np.zeros_like(q)
    out[..., 0] = -a * q[..., 1]
    out[..., 1] = a * q[..., 0]
    return out


def E_of(q, t, tau, B0=1.0):
    """Induced field, -dA/dt. Must equal fields/decaying_field.py."""
    fac = 0.5 * Bz_of(t, tau, B0) / tau
    out = np.zeros_like(q)
    out[..., 0] = -fac * q[..., 1]
    out[..., 1] = fac * q[..., 0]
    return out


# --------------------------------------------------------------------------- #
# Batched frozen random MLP  dL_d(q_k, q_k1, t) -> scalar
# --------------------------------------------------------------------------- #

class DeltaLNet:
    """N independent tanh-MLPs, 7 -> H -> H -> 1, frozen random weights.

    Inputs are scaled to O(1) before entering the net so that the output
    magnitude is controlled by `amp` alone and not by the input units.
    Analytic gradients w.r.t. q_k (D1) and q_k1 (D2) via one backward pass.
    """

    def __init__(self, n_ens, seeds, amps, hidden=32, t_scale=None, q_scale=1.0):
        assert len(seeds) == n_ens and len(amps) == n_ens
        self.n, self.h, self.q_scale = n_ens, hidden, q_scale
        self.t_scale = t_scale
        self.amps = np.asarray(amps, float)
        W1 = np.empty((n_ens, hidden, 7))
        b1 = np.empty((n_ens, hidden))
        W2 = np.empty((n_ens, hidden, hidden))
        b2 = np.empty((n_ens, hidden))
        W3 = np.empty((n_ens, 1, hidden))
        b3 = np.empty((n_ens, 1))
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            # Glorot-like scaling keeps pre-activations O(1)
            W1[i] = rng.normal(0, np.sqrt(2.0 / (7 + hidden)), (hidden, 7))
            b1[i] = rng.normal(0, 0.1, hidden)
            W2[i] = rng.normal(0, np.sqrt(2.0 / (2 * hidden)), (hidden, hidden))
            b2[i] = rng.normal(0, 0.1, hidden)
            W3[i] = rng.normal(0, np.sqrt(2.0 / (hidden + 1)), (1, hidden))
            b3[i] = rng.normal(0, 0.1, 1)
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = W1, b1, W2, b2, W3, b3

    def _features(self, qk, qk1, t):
        """(N,3),(N,3),scalar -> (N,7) scaled features."""
        x = np.empty((self.n, 7))
        x[:, 0:3] = qk / self.q_scale
        x[:, 3:6] = qk1 / self.q_scale
        x[:, 6] = t / self.t_scale
        return x

    def grads(self, qk, qk1, t):
        """Return (D1, D2) each (N,3): d(dL_d)/dq_k and d(dL_d)/dq_k1."""
        x = self._features(qk, qk1, t)
        z1 = np.einsum('nij,nj->ni', self.W1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.einsum('nij,nj->ni', self.W2, a1) + self.b2
        a2 = np.tanh(z2)
        # backward: d out / d x
        g2 = self.W3[:, 0, :] * (1.0 - a2 * a2)                    # (N,H)
        g1 = np.einsum('ni,nij->nj', g2, self.W2) * (1.0 - a1 * a1)  # (N,H)
        gx = np.einsum('ni,nij->nj', g1, self.W1)                   # (N,7)
        gx *= (self.amps[:, None] / self.q_scale)
        return gx[:, 0:3].copy(), gx[:, 3:6].copy()

    def value(self, qk, qk1, t):
        x = self._features(qk, qk1, t)
        a1 = np.tanh(np.einsum('nij,nj->ni', self.W1, x) + self.b1)
        a2 = np.tanh(np.einsum('nij,nj->ni', self.W2, a1) + self.b2)
        out = np.einsum('nij,nj->ni', self.W3, a2) + self.b3
        return self.amps * out[:, 0]


# --------------------------------------------------------------------------- #
# Core stepping
# --------------------------------------------------------------------------- #

def _solve_M(rhs, a, h):
    """Solve M D = rhs for D, with M = [[1/h,a,0],[-a,1/h,0],[0,0,1/h]]."""
    det = 1.0 / (h * h) + a * a
    D = np.empty_like(rhs)
    D[..., 0] = (rhs[..., 0] / h - a * rhs[..., 1]) / det
    D[..., 1] = (a * rhs[..., 0] + rhs[..., 1] / h) / det
    D[..., 2] = h * rhs[..., 2]
    return D


def var_step(q, p, t, h, tau, net=None, tol=1e-14, max_it=40, B0=1.0):
    """One variational step. q,p shape (...,3). Returns q1,p1,n_iter."""
    tm = t + 0.5 * h
    a = 0.5 * Bz_of(tm, tau, B0)
    b = np.zeros_like(q)
    b[..., 0] = a * q[..., 1]
    b[..., 1] = -a * q[..., 0]

    if net is None:
        D = _solve_M(p - b, a, h)
        n_it = 0
    else:
        # p = M D + b - D1 dL_d(q, q+D, t)   ->  fixed point on D
        D = _solve_M(p - b, a, h)
        n_it = 0
        for n_it in range(1, max_it + 1):
            d1, _ = net.grads(q, q + D, t)
            D_new = _solve_M(p - b + d1, a, h)
            err = np.max(np.abs(D_new - D))
            D = D_new
            if err < tol:
                break

    q1 = q + D
    p1 = D / h + b
    if net is not None:
        _, d2 = net.grads(q, q1, t)
        p1 = p1 + d2
    return q1, p1, n_it


def integrate(mode, tau, h, n_steps, net=None, dv_net=None, n_samples=4000,
              r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), n_ens=1, B0=1.0,
              tol=1e-14):
    """mode: 'base' | 'varnet' | 'additive'.

    Diagnostics follow experiments/horizon/fast.py exactly:
        dev = |E_cur - E0 exp(-t/tau)| / E0     (deviation from adiabatic law)
        env = running max of dev inside each sampling window
    Returns dict of arrays with leading ensemble axis where relevant.
    """
    q = np.tile(np.asarray(r0, float), (n_ens, 1))
    v = np.tile(np.asarray(v0, float), (n_ens, 1))
    t = 0.0
    p = v - A_of(q, t, tau, B0)
    E0 = 0.5 * np.sum(v[0] * v[0])

    stride = max(1, n_steps // n_samples)
    ts, Es, mus, envs = [], [], [], []
    run_max = np.zeros(n_ens)
    it_tot, it_max = 0, 0
    dv_scale_acc = np.zeros(n_ens)

    use_net = net if mode == 'varnet' else None

    for i in range(1, n_steps + 1):
        if mode == 'additive':
            q1, p1, nit = var_step(q, p, t, h, tau, None, B0=B0)
            v1 = p1 + A_of(q1, t + h, tau, B0)
            d1, d2 = dv_net.grads(q, q1, t)      # reuse net as a source of noise
            dv = d1 + d2
            dv_scale_acc += np.sqrt(np.sum(dv * dv, axis=-1))
            v1 = v1 + dv
            p1 = v1 - A_of(q1, t + h, tau, B0)
        else:
            q1, p1, nit = var_step(q, p, t, h, tau, use_net, tol=tol, B0=B0)
            it_tot += nit
            it_max = max(it_max, nit)
            v1 = p1 + A_of(q1, t + h, tau, B0)

        q, p = q1, p1
        t += h

        Ecur = 0.5 * np.sum(v1 * v1, axis=-1)
        Ephys = E0 * np.exp(-t / tau)
        Bcur = Bz_of(t, tau, B0)
        dev = np.abs(Ecur - Ephys) / E0
        run_max = np.maximum(run_max, dev)

        if i % stride == 0 or i == n_steps:
            ts.append(t)
            Es.append(dev.copy())
            mus.append(np.abs((Ecur / Bcur) / (E0 / B0) - 1.0))
            envs.append(run_max.copy())
            run_max = np.zeros(n_ens)

    out = {"t": np.array(ts),
           "e_err": np.array(Es).T,        # (n_ens, n_samples)
           "mu_err": np.array(mus).T,
           "env": np.array(envs).T,
           "E0": E0,
           "newton_iters_mean": it_tot / max(n_steps, 1),
           "newton_iters_max": it_max}
    if mode == 'additive':
        out["dv_rel_mean"] = dv_scale_acc / n_steps
    return out


def envelope_exponent(t, env):
    """Power-law fit of the accumulated envelope over the last two decades.
    Identical definition to experiments/horizon/long_runs.py."""
    e = np.maximum.accumulate(env)
    sel = (t > t[-1] / 100.0) & (e > 0)
    if sel.sum() <= 10:
        return float('nan'), e
    p = np.polyfit(np.log10(t[sel]), np.log10(e[sel]), 1)
    return float(p[0]), e
