"""
hnn.py
======
Hamiltonian Neural Network (HNN), Section 2.2.2, Eq. (6).

Architecture
------------
The network approximates a scalar Hamiltonian H_theta(r, v) (here we use
canonical-like coordinates (r, v) directly, appropriate for the
non-canonical charged-particle phase space with a magnetic field folded
into the equations of motion). Equations of motion are then obtained
through automatic differentiation:

    dr/dt =  dH/dv,     dv/dt = -dH/dr  +  (q/m) v x B_correction

For a charged particle in a magnetic field the pure canonical HNN form is
augmented with the known gyroscopic (magnetic) structure, since a magnetic
field is a velocity-dependent, non-canonical force that cannot be derived
from a scalar potential alone. We therefore learn H_theta(r,v) representing
the *kinetic + potential* energy landscape, and combine its gradient with
the analytic magnetic-rotation structure -- a standard approach for
"Hamiltonian-inspired" networks applied to guiding-center-like systems.

  Input   : (r[3], v[3])           -> dim 6
  Hidden  : 4 layers x 96 neurons, tanh
  Output  : scalar H_theta(r, v)

Loss (Eq. 11): residual of Hamilton's equations, evaluated via autograd,
plus a data term matching the reference next-state trajectory (obtained by
one explicit Euler-like step using the learned vector field).

Hyperparameters
----------------
  hidden_layers = 4, hidden_size = 96, activation = tanh
  optimizer = Adam, lr = 1e-3
  seed = 42
"""

import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)

HIDDEN_LAYERS = 4
HIDDEN_SIZE = 96
LR = 1e-3


class HNN(nn.Module):
    """Hamiltonian Neural Network approximating H_theta(r, v)."""

    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE):
        super().__init__()
        layers = [nn.Linear(6, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

    def hamiltonian(self, r, v):
        x = torch.cat([r, v], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, r, v, B, q=-1.0, m=1.0):
        """Compute time derivatives (dr/dt, dv/dt) from the learned
        Hamiltonian plus the analytic magnetic gyroscopic structure."""
        with torch.enable_grad():
            r = r.clone().requires_grad_(True)
            v = v.clone().requires_grad_(True)
            H = self.hamiltonian(r, v)
            grad_r, grad_v = torch.autograd.grad(
                H.sum(), (r, v), create_graph=self.training
            )
        # Hamiltonian part: dr/dt = dH/dv ; magnetic gyroscopic part added
        # analytically since B-field force is non-conservative / velocity-dependent
        drdt = grad_v
        dvdt = -grad_r + (q / m) * torch.cross(v, B, dim=-1)
        return drdt, dvdt

    def step(self, r, v, B, dt, q=-1.0, m=1.0):
        """Single explicit-Euler integration step using the learned field."""
        drdt, dvdt = self.forward(r, v, B, q=q, m=m)
        r_next = r + dt * drdt
        v_next = v + dt * dvdt
        return r_next.detach(), v_next.detach()

    @staticmethod
    def physics_loss(drdt_pred, dvdt_pred, drdt_ref, dvdt_ref):
        """Eq. (11): residual of Hamilton's equations vs. finite-difference
        reference derivatives from Boris trajectories."""
        loss_r = torch.mean((drdt_pred - drdt_ref) ** 2)
        loss_v = torch.mean((dvdt_pred - dvdt_ref) ** 2)
        total = loss_r + loss_v
        return total, {"loss_r": loss_r.item(), "loss_v": loss_v.item()}


def build_model(seed=SEED):
    """Build the model.

    `seed` is a parameter rather than a module constant so that
    ensembles over random initialisations are possible; the default
    reproduces the original behaviour exactly.
    """
    torch.manual_seed(seed)
    return HNN()
