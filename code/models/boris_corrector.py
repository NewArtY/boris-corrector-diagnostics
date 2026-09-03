"""
boris_corrector.py
===================

.. warning::

   **THIS MODULE DOES NOT PRODUCE ANY RESULT REPORTED IN THE ARTICLE.**

   The hybrid results of the Article (Figure 4 and the associated numbers in
   ``output_figures/figure4_numbers.json`` and
   ``output_figures/corrector_evaluation.json``) are produced by ``DefectNet``
   defined in ``training/train_corrector_b4.py``:

       ``DefectNet``          13 inputs -> 4 hidden layers x 128 -> 6,
                              52,102 parameters, float64,
                              checkpoint ``checkpoints/boris_corrector_b4.pt``

       ``BorisCorrectorNet``  (this file) 10 inputs -> 3 hidden layers x 64
                              -> 6, 9,414 parameters, float32,
                              checkpoint ``checkpoints/boris_corrector.pt``

   ``BorisCorrectorNet`` is retained only for the B1--B3 comparison runs in
   ``training/train.py``. The architecture described in Methods is
   ``DefectNet``, not this class. Reading this file to learn how the hybrid
   scheme of the Article works will give the wrong architecture, the wrong
   parameter count, the wrong precision and the wrong correction amplitude
   (``CORRECTION_SCALE = 0.05`` here; the measured relative amplitude of the
   correction actually used is 2.2e-3).

Hybrid Boris + Neural Corrector integrator, Section 2.2.4, Eq. (8), (12).

Idea
----
Use the classical Boris update as the base, physically-consistent,
(near-)symplectic propagation rule, then add a SMALL learned correction
produced by a compact MLP:

    (r_B, v_B)              = BorisStep(r_n, v_n, E_n, B_n, dt)
    (delta_r, delta_v)      = MLP_theta(r_n, v_n, B_n, dt)
    (r_{n+1}, v_{n+1})      = (r_B + delta_r, v_B + delta_v)

This compensates residual truncation error (especially phase error) of the
Boris step at larger dt, while inheriting Boris's stability because the
correction is constrained to be small.

Architecture
------------
  Input  : (r[3], v[3], B[3], dt[1]) -> 10
  Hidden : 3 layers x 64 neurons, tanh    (most compact of the 4 NN models)
  Output : (delta_r[3], delta_v[3]) -> 6, scaled by a small multiplier so
           the correction starts close to zero (residual-only correction)

Loss (Eq. 12): MSE of the corrected trajectory against reference data,
    L = || r_{n+1} - r_ref ||^2 + || v_{n+1} - v_ref ||^2 + lambda * L_phys
  L_phys includes:
    - smallness penalty on (delta_r, delta_v)
    - orthogonality penalty: correction should act mostly perpendicular to v
      (delta_v . v_hat ~ 0), preserving the Boris speed-conservation property
    - energy-change penalty: (|v_new| - |v_B|)^2

Hyperparameters
----------------
  hidden_layers = 3, hidden_size = 64, activation = tanh
  optimizer = Adam, lr = 1e-3
  lambda_phys = 0.05, correction_scale = 0.05
  seed = 42
"""

import torch
import torch.nn as nn

from models.boris import boris_step

SEED = 42
torch.manual_seed(SEED)

HIDDEN_LAYERS = 3
HIDDEN_SIZE = 64
LR = 1e-3
LAMBDA_PHYS = 0.05
CORRECTION_SCALE = 0.05


class BorisCorrectorNet(nn.Module):
    """Compact MLP producing a small correction to the Boris step."""

    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE,
                 correction_scale=CORRECTION_SCALE):
        super().__init__()
        layers = [nn.Linear(10, hidden_size), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, 6)]
        self.net = nn.Sequential(*layers)
        self.correction_scale = correction_scale

        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, v, B, dt):
        x = torch.cat([r, v, B, dt], dim=-1)
        out = self.correction_scale * self.net(x)
        delta_r, delta_v = out[:, :3], out[:, 3:]
        return delta_r, delta_v

    @staticmethod
    def physics_loss(r_boris, v_boris, delta_r, delta_v, r_next_ref, v_next_ref,
                      lambda_phys=LAMBDA_PHYS):
        r_pred = r_boris + delta_r
        v_pred = v_boris + delta_v

        data_loss = torch.mean((r_pred - r_next_ref) ** 2) + \
            torch.mean((v_pred - v_next_ref) ** 2)

        smallness = torch.mean(delta_r ** 2) + torch.mean(delta_v ** 2)

        v_hat = v_boris / (torch.norm(v_boris, dim=-1, keepdim=True) + 1e-8)
        orthogonality = torch.mean(torch.sum(delta_v * v_hat, dim=-1) ** 2)

        energy_change = torch.mean(
            (torch.norm(v_pred, dim=-1) - torch.norm(v_boris, dim=-1)) ** 2
        )

        phys = smallness + orthogonality + energy_change
        total = data_loss + lambda_phys * phys
        return total, {
            "data_loss": data_loss.item(),
            "smallness": smallness.item(),
            "orthogonality": orthogonality.item(),
            "energy_change": energy_change.item(),
        }


def boris_corrector_step_numpy(r, v, t, dt, field, net, q=-1.0, m=1.0, device="cpu"):
    """Numpy-facing convenience wrapper: one Boris+Corrector step."""
    import numpy as np
    r_b, v_b = boris_step(r, v, t, dt, field, q=q, m=m)
    B = field.B(r, t)

    with torch.no_grad():
        r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(0)
        v_t = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        B_t = torch.tensor(B, dtype=torch.float32, device=device).unsqueeze(0)
        dt_t = torch.tensor([[dt]], dtype=torch.float32, device=device)
        dr, dv = net(r_t, v_t, B_t, dt_t)
        dr = dr.squeeze(0).cpu().numpy()
        dv = dv.squeeze(0).cpu().numpy()

    return r_b + dr, v_b + dv


def build_model(seed=SEED):
    """Build the model.

    `seed` is a parameter rather than a module constant so that
    ensembles over random initialisations are possible; the default
    reproduces the original behaviour exactly.
    """
    torch.manual_seed(seed)
    return BorisCorrectorNet()
