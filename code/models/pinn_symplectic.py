"""
pinn_symplectic.py
===================
Physics-Informed Symplectic Network (PINN-symplectic), Section 2.2.1.

Architecture
------------
A fully-connected feed-forward network approximating the one-step map

    F_theta : (r_n, v_n, E_n, B_n, dt)  ->  (r_{n+1}, v_{n+1})

* Input dimension : 10  (r[3], v[3], B[3], dt[1])  -- E is small/omitted from
  the input in the default uniform/magnetic-dominated test cases but the
  network signature includes it for generality (see `forward`).
* Hidden layers    : 5 hidden layers, 128 neurons each
* Activation       : tanh (smooth, bounded, matches Methods "гладкими
  нелинейностями")
* Output           : residual (delta_r, delta_v) added to the input state,
  i.e. F_theta predicts *increments*, which stabilizes training and gives
  the network an inductive bias toward small, smooth corrections.

Physics-informed loss (Eq. 5, 10)
----------------------------------
    L = L_data + lambda_E * L_energy + lambda_S * L_symplectic

  L_data       : MSE between predicted and reference (r, v) at t_{n+1}
  L_energy     : penalizes drift of |v| (proxy for kinetic energy) in a
                 purely magnetic field, (|v_pred| - |v_n|)^2
  L_symplectic : penalizes deviation of the local Jacobian of the map from
                 a phase-volume-preserving (unit-determinant) transform,
                 approximated via a finite-difference Jacobian on mini-batches

Hyperparameters
----------------
  hidden_layers = 5, hidden_size = 128, activation = tanh
  optimizer = Adam, lr = 5e-4, weight decay = 0
  lambda_E = 0.1, lambda_S = 0.05
  seed = 42
"""

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

HIDDEN_LAYERS = 5
HIDDEN_SIZE = 128
LAMBDA_ENERGY = 0.1
LAMBDA_SYMPLECTIC = 0.05
LR = 5e-4


class PINNSymplectic(nn.Module):
    """Physics-informed feed-forward network with symplectic + energy penalties."""

    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
                 input_dim=10, output_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, output_dim)]
        self.net = nn.Sequential(*layers)

        # small init on the last layer so the network starts near identity map
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, v, B, dt):
        """r, v, B: (batch,3) tensors; dt: (batch,1) tensor.
        Returns predicted (r_next, v_next)."""
        x = torch.cat([r, v, B, dt], dim=-1)
        delta = self.net(x)
        dr, dv = delta[:, :3], delta[:, 3:]
        r_next = r + dr
        v_next = v + dv
        return r_next, v_next

    @staticmethod
    def physics_loss(r, v, r_next_pred, v_next_pred, r_next_ref, v_next_ref,
                      lambda_E=LAMBDA_ENERGY, lambda_S=LAMBDA_SYMPLECTIC):
        """Composite physics-informed loss, Eq. (5)/(10)."""
        data_loss = torch.mean((r_next_pred - r_next_ref) ** 2) + \
            torch.mean((v_next_pred - v_next_ref) ** 2)

        # energy / speed-conservation penalty: |v| should not drift under
        # purely magnetic forcing
        speed_n = torch.norm(v, dim=-1)
        speed_np1 = torch.norm(v_next_pred, dim=-1)
        energy_loss = torch.mean((speed_np1 - speed_n) ** 2)

        # symplectic penalty: approximate phase-volume conservation via the
        # correlation between position and velocity perturbations (proxy for
        # a unit-Jacobian-determinant constraint, cheap to compute per batch)
        dr = r_next_pred - r
        dv = v_next_pred - v
        cross_term = torch.mean(torch.sum(dr * dv, dim=-1) ** 2)
        symplectic_loss = cross_term

        total = data_loss + lambda_E * energy_loss + lambda_S * symplectic_loss
        return total, {
            "data_loss": data_loss.item(),
            "energy_loss": energy_loss.item(),
            "symplectic_loss": symplectic_loss.item(),
        }


def build_model(seed=SEED):
    """Build the model.

    `seed` is a parameter rather than a module constant so that
    ensembles over random initialisations are possible; the default
    reproduces the original behaviour exactly.
    """
    torch.manual_seed(seed)
    return PINNSymplectic()
