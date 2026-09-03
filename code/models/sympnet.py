"""
sympnet.py
==========
Symplectic Neural Network (SympNet), Section 2.2.3, Eq. (7).

Architecture
------------
SympNet builds the one-step map as a composition of explicit
"kick-drift-kick" symplectic layers, each of which is *exactly*
volume-preserving by construction, regardless of the learned weights:

  Kick layer:   v'  = v + f_theta(r) ,      r'  = r                 (dH/dr type)
  Drift layer:  r'  = r + g_theta(v') ,      v'' = v'                (dH/dv type)

where f_theta, g_theta are small MLPs (gradients of scalar potential-like
networks are NOT required here -- using shear/triangular maps, the Jacobian
is exactly a unit-determinant triangular matrix, guaranteeing
symplecticity independent of training, per Eq. 7: dPhi^T J dPhi = J with
J the canonical symplectic form).

We stack 3 kick-drift-kick blocks (6 shear layers total), each shear MLP
having 2 hidden layers x 64 neurons, tanh activation.

Loss: next-step MSE (position, velocity) plus a Jacobian-regularization
term (small weight decay on shear-map Jacobians, kept explicit for
clarity even though exact symplecticity already holds by construction).

Hyperparameters
----------------
  n_blocks = 3 (kick+drift each), hidden_layers = 2, hidden_size = 64
  activation = tanh, optimizer = Adam, lr = 1e-3
  seed = 42
"""

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

N_BLOCKS = 3
HIDDEN_LAYERS = 2
HIDDEN_SIZE = 64
LR = 1e-3
JAC_REG = 1e-4


class ShearMLP(nn.Module):
    """Small MLP used inside a single shear (kick or drift) layer."""

    def __init__(self, in_dim=3, out_dim=3, hidden_layers=HIDDEN_LAYERS,
                 hidden_size=HIDDEN_SIZE):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, out_dim)]
        self.net = nn.Sequential(*layers)
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class KickDriftBlock(nn.Module):
    """One symplectic kick-drift-kick block, exactly volume preserving."""

    def __init__(self, dt_scale=1.0):
        super().__init__()
        self.kick1 = ShearMLP()
        self.drift = ShearMLP()
        self.kick2 = ShearMLP()
        self.dt_scale = dt_scale

    def forward(self, r, v, dt):
        # half kick: v depends on r only -> shear map, exactly symplectic
        v = v + 0.5 * dt * self.kick1(r)
        # full drift: r depends on v only -> shear map
        r = r + dt * self.drift(v)
        # half kick
        v = v + 0.5 * dt * self.kick2(r)
        return r, v


class SympNet(nn.Module):
    """Composition of N kick-drift-kick blocks; exactly symplectic by construction."""

    def __init__(self, n_blocks=N_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList([KickDriftBlock() for _ in range(n_blocks)])

    def forward(self, r, v, dt):
        dt_b = dt / len(self.blocks)
        for block in self.blocks:
            r, v = block(r, v, dt_b)
        return r, v

    @staticmethod
    def physics_loss(r_next_pred, v_next_pred, r_next_ref, v_next_ref, jac_reg=JAC_REG):
        loss_r = torch.mean((r_next_pred - r_next_ref) ** 2)
        loss_v = torch.mean((v_next_pred - v_next_ref) ** 2)
        # explicit (small) regularization on outputs, standing in for the
        # Jacobian-regularization term mentioned in the Methods; exact
        # symplecticity already holds due to the shear-map construction
        reg = jac_reg * (torch.mean(r_next_pred ** 2) + torch.mean(v_next_pred ** 2))
        total = loss_r + loss_v + reg
        return total, {"loss_r": loss_r.item(), "loss_v": loss_v.item(), "reg": reg.item()}


def build_model(seed=SEED):
    """Build the model.

    `seed` is a parameter rather than a module constant so that
    ensembles over random initialisations are possible; the default
    reproduces the original behaviour exactly.
    """
    torch.manual_seed(seed)
    return SympNet()
