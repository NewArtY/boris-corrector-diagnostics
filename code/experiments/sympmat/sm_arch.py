"""SympMat, written from Secs. III.B and III.C of Drimalas et al. 2025.

    G_i = I - beta_i u_i u_i^T J,        u_i normalised, beta_i scalar
    SympMat(z) = G_{4n} ... G_1 z,       4n = 8 reflectors for n = 2

G is symplectic for *any* beta and u, normalised or not, because J is
antisymmetric and therefore u^T J u = 0 identically:

    G^T J G = J + beta J u u^T J - beta J u u^T J + beta^2 (J u)(u^T J u) u^T J = J.

Normalising u only removes the redundancy beta <-> ||u||^2.  This is why the
paper can say the architecture "guarantees that said property will be respected
for any choice of parameters, even outside the training volume", and it is
checked numerically in `sm1_train.py` at every checkpoint.

Two variants, as in their Fig. 1:

  StandardSympMat     beta_i and u_i are free parameters; one model per (B, dt).
  ParametricSympMat   beta_i and u_i are the outputs of shallow tanh MLPs of the
                      physical parameter mu = B/B_0, hidden width 10 for beta and
                      20 for u (their Table I), one pair of MLPs per reflector
                      ("a series of shallow neural networks ... in each reflector
                      G_i").  The eight pairs are held as stacked tensors so that
                      one forward pass costs a handful of torch calls rather than
                      sixteen; the arithmetic is that of sixteen separate MLPs.

Everything is float64.  The standard model is reported to reach an MSE of
10^-30, which is not a statement one can check in float32.
"""
import numpy as np
import torch

DIM = 4
NREF = 8
J = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0, 0.0]], dtype=torch.float64)
EYE = torch.eye(DIM, dtype=torch.float64)


def _u(shape, fan_in, gen):
    """torch.nn.Linear's default initialisation, drawn from an explicit
    generator so that no global random state is touched."""
    bound = 1.0 / np.sqrt(fan_in)
    return (torch.rand(shape, generator=gen, dtype=torch.float64) * 2 - 1) * bound


def _reflectors_to_matrix(beta, u):
    """beta (..., NREF, K, 1), u (..., NREF, K, DIM) -> (..., K, DIM, DIM).

    The reflector axis is the fourth from the right throughout, so the same
    code serves one model, one model over a batch of field values, and a stack
    of models over a batch of field values.
    """
    u = u / u.norm(dim=-1, keepdim=True)
    uuT = u.unsqueeze(-1) * u.unsqueeze(-2)                 # (..., NREF, K, D, D)
    G = EYE - beta.unsqueeze(-1) * (uuT @ J)
    parts = torch.unbind(G, dim=-4)
    M = parts[0]
    for i in range(1, NREF):
        M = parts[i] @ M
    return M


class StandardSympMat(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        self.beta = torch.nn.Parameter(_u((NREF, 1, 1), 1, g))
        self.u = torch.nn.Parameter(_u((NREF, 1, DIM), DIM, g))

    def matrix(self):
        return _reflectors_to_matrix(self.beta, self.u)[0]


class ParametricSympMat(torch.nn.Module):
    WB, WU, NIN = 10, 20, 1

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        p = lambda *a: torch.nn.Parameter(_u(*a, g))
        self.W1b = p((NREF, self.WB, self.NIN), self.NIN)
        self.b1b = p((NREF, self.WB), self.NIN)
        self.W2b = p((NREF, 1, self.WB), self.WB)
        self.b2b = p((NREF, 1), self.WB)
        self.W1u = p((NREF, self.WU, self.NIN), self.NIN)
        self.b1u = p((NREF, self.WU), self.NIN)
        self.W2u = p((NREF, DIM, self.WU), self.WU)
        self.b2u = p((NREF, DIM), self.WU)

    def matrices(self, mu):
        """mu (K,) -> (K, DIM, DIM), one symplectic matrix per field value."""
        x = mu.reshape(1, -1, 1)
        hb = torch.tanh(x @ self.W1b.transpose(-1, -2) + self.b1b.unsqueeze(-2))
        beta = hb @ self.W2b.transpose(-1, -2) + self.b2b.unsqueeze(-2)
        hu = torch.tanh(x @ self.W1u.transpose(-1, -2) + self.b1u.unsqueeze(-2))
        u = hu @ self.W2u.transpose(-1, -2) + self.b2u.unsqueeze(-2)
        return _reflectors_to_matrix(beta, u)


class StackedParametricSympMat(torch.nn.Module):
    """A stack of independent ParametricSympMat models sharing one forward pass.

    The models never interact: their parameters are disjoint, the loss is the
    *sum* of the per-model mean squared errors so that each model's gradient is
    exactly the gradient it would have had alone, and Adam is elementwise.
    Training the stack is therefore identical to training the members
    separately, one Adam step at a time -- `sm1_train.py --equivalence` asserts
    that against the single-model class above, to fourteen digits.

    The reason for it is arithmetic, not statistical: at these sizes a step is
    dominated by the fixed cost of launching about a hundred torch kernels, and
    twenty-seven models cost very nearly what one costs.
    """

    def __init__(self, seeds):
        super().__init__()
        members = [ParametricSympMat(s) for s in seeds]
        names = [n for n, _ in members[0].named_parameters()]
        for n in names:
            stacked = torch.stack([dict(m.named_parameters())[n].detach()
                                   for m in members])
            setattr(self, n, torch.nn.Parameter(stacked))

    def matrices(self, mu):
        """mu (K,) -> (NMODEL, K, DIM, DIM)."""
        x = mu.reshape(1, 1, -1, 1)
        hb = torch.tanh(x @ self.W1b.transpose(-1, -2) + self.b1b.unsqueeze(-2))
        beta = hb @ self.W2b.transpose(-1, -2) + self.b2b.unsqueeze(-2)
        hu = torch.tanh(x @ self.W1u.transpose(-1, -2) + self.b1u.unsqueeze(-2))
        u = hu @ self.W2u.transpose(-1, -2) + self.b2u.unsqueeze(-2)
        return _reflectors_to_matrix(beta, u)
