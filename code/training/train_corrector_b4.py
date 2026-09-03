"""
train_corrector_b4.py
=====================
Retraining of the hybrid Boris + Corrector integrator in the regime that
matters for the central claim of the Article: a WEAK, slow physical energy
change in a time-decaying magnetic field, where the numerical energy error of
the plain Boris scheme at a usable time step is comparable to (or larger than)
the physical signal itself.

Why this regime
---------------
`training/scan_dt.py` shows that when the field decays fast enough to change
the particle energy by tens of percent, plain Boris is already 3-4 orders of
magnitude more accurate than the signal: there is nothing to disentangle. The
degeneracy the Article is about appears when the physical signal is WEAK --
slow heating/cooling, adiabatic invariant breaking, radiative losses -- i.e.
when

        |dE_phys / E0|   ~   |dE_num / E0| .

With Omega_c * dt = 0.3 the Boris energy error over the run is ~2e-3, so a
decay time tau chosen such that the physical energy change is also ~1e-3
places the two exactly on top of each other. That is the configuration used
here and in Figure 4.

What is learned
---------------
The network never represents the dynamics. It represents only the ONE-STEP
DISCRETISATION DEFECT of the Boris map at the working step:

        target = (r_ref, v_ref)_{n+1} - BorisStep(r_n, v_n; dt_work)

where the reference is obtained by integrating the same state with the same
Boris scheme at a 150x smaller step. The defect is a smooth, small, locally
determined function of the state, which is why it is learnable with a compact
network and why it extrapolates.

Constraints (Eq. 2 of the Article) are imposed through the loss:
  * smallness      -- ||delta|| stays a small fraction of ||v||
  * orthogonality  -- delta_v . v_hat ~ 0 (acts on phase, not on magnitude)
  * energy neutral -- the correction may not change |v| by itself

Everything runs in float64: at the accuracy targeted here (1e-6 relative
energy error) float32 round-off is itself the dominant noise source.

Reproducible: fixed seed 42, all parameters logged to checkpoints/.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CHECKPOINT_DIR, SEED, set_global_seed, get_logger
from fields import DecayingField
from models.boris import boris_step, integrate_boris

torch.set_default_dtype(torch.float64)
set_global_seed(SEED)
logger = get_logger("train_corrector_b4")

# --------------------------------------------------------------------------
# Regime parameters (see module docstring)
# --------------------------------------------------------------------------
DT_WORK = 0.3            # working step, Omega_c * dt = 0.3  (~21 steps/gyration)
DT_FINE = DT_WORK / 150  # reference step used to define the defect
T_FINAL = 120.0          # ~19 gyrations
TAU_MAIN = 1.2e5         # decay time -> physical |dE/E0| ~ 1e-3 (weak signal)

# Training-set spread: several decay times and initial conditions
TAU_TRAIN = [0.8e5, 1.0e5, 1.5e5, 2.0e5, 3.0e5]
N_TRAJ_PER_TAU = 3
HIDDEN = 128
N_LAYERS = 4
EPOCHS = 400
BATCH = 512
LR = 1e-3

# Loss weights
LAMBDA_SMALL = 1e-3
LAMBDA_ORTHO = 1e-3
LAMBDA_ENERGY = 1e-3


# --------------------------------------------------------------------------
# Dataset: one-step discretisation defect of the Boris map
# --------------------------------------------------------------------------
def build_dataset():
    """Sample states along fine reference trajectories and record, for each,
    the difference between the fine-resolved propagation over dt_work and a
    single coarse Boris step."""
    X, Y = [], []
    rng = np.random.default_rng(SEED)

    for tau in TAU_TRAIN:
        field = DecayingField(B0=1.0, tau=tau)
        for k in range(N_TRAJ_PER_TAU):
            # varied initial conditions (radius, phase, parallel velocity)
            rho = 0.7 + 0.6 * rng.random()
            phase = 2 * np.pi * rng.random()
            vpar = 0.3 * (rng.random() - 0.5)
            r0 = np.array([rho * np.cos(phase), rho * np.sin(phase), 0.0])
            v0 = np.array([-np.sin(phase), np.cos(phase), vpar])

            n_coarse = int(round(T_FINAL / DT_WORK))
            r, v, t = r0.copy(), v0.copy(), 0.0
            for _ in range(n_coarse):
                # coarse Boris step
                r_b, v_b = boris_step(r, v, t, DT_WORK, field)
                # fine reference propagation of the same state over dt_work
                rs_f, vs_f, _ = integrate_boris(r, v, t, DT_FINE, 150, field)
                r_ref, v_ref = rs_f[-1], vs_f[-1]

                B = np.atleast_1d(field.B(r, t)).ravel()
                E = np.atleast_1d(field.E(r, t)).ravel()
                X.append(np.concatenate([r, v, B, E, [DT_WORK]]))
                Y.append(np.concatenate([r_ref - r_b, v_ref - v_b]))

                # advance along the reference so the sampled states stay on
                # the true trajectory rather than on the coarse one
                r, v = r_ref, v_ref
                t += DT_WORK

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    logger.info("dataset: X=%s  Y=%s  |Y| mean=%.3e", X.shape, Y.shape,
                np.abs(Y).mean())
    return X, Y


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
class DefectNet(nn.Module):
    """MLP predicting the standardised one-step Boris defect."""

    def __init__(self, n_in=13, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 6)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        # standardisation buffers (filled at fit time)
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_scale", torch.ones(6))

    def forward(self, x):
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_scale


def train():
    t_start = time.time()
    X, Y = build_dataset()

    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)

    model = DefectNet(n_in=X.shape[1])
    model.x_mean.copy_(Xt.mean(0))
    model.x_std.copy_(Xt.std(0).clamp_min(1e-12))
    model.y_scale.copy_(Yt.abs().mean(0).clamp_min(1e-16))

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = Xt.shape[0]
    idx_all = torch.randperm(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]

    v_dir = Xt[:, 3:6] / Xt[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-12)
    history = []

    for ep in range(EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(tr_idx.numel())]
        tot = 0.0
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            pred = model(Xt[b])
            data = ((pred - Yt[b]) ** 2).mean()
            small = (pred ** 2).mean()
            ortho = ((pred[:, 3:] * v_dir[b]).sum(1) ** 2).mean()
            ener = ((pred[:, 3:] * Xt[b, 3:6]).sum(1) ** 2).mean()
            loss = (data / model.y_scale.pow(2).mean()
                    + LAMBDA_SMALL * small
                    + LAMBDA_ORTHO * ortho
                    + LAMBDA_ENERGY * ener)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * b.numel()
        sched.step()

        if ep % 50 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xt[val_idx])
                rel = ((pv - Yt[val_idx]).norm(dim=1)
                       / Yt[val_idx].norm(dim=1).clamp_min(1e-30)).mean().item()
            history.append({"epoch": ep, "train_loss": tot / tr_idx.numel(),
                            "val_rel_defect_error": rel})
            logger.info("epoch %4d  loss=%.4e  val relative defect error=%.4f",
                        ep, tot / tr_idx.numel(), rel)

    path = os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt")
    torch.save(model.state_dict(), path)

    params = {
        "seed": SEED, "dt_work": DT_WORK, "dt_fine": DT_FINE,
        "t_final": T_FINAL, "tau_train": TAU_TRAIN, "tau_main": TAU_MAIN,
        "hidden": HIDDEN, "n_layers": N_LAYERS, "epochs": EPOCHS,
        "batch": BATCH, "lr": LR, "dtype": "float64",
        "lambda_small": LAMBDA_SMALL, "lambda_ortho": LAMBDA_ORTHO,
        "lambda_energy": LAMBDA_ENERGY,
        "n_samples": int(X.shape[0]),
        "wall_time_s": round(time.time() - t_start, 1),
        "history": history,
    }
    with open(os.path.join(CHECKPOINT_DIR, "corrector_b4_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    logger.info("saved %s (%.1f s)", path, time.time() - t_start)
    return model


if __name__ == "__main__":
    train()
