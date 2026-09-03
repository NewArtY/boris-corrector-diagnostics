"""Training, exactly by their Table I and Sec. III.C.

  optimiser         ADAM
  loss              MSE
  schedule          stepwise, "halves the learning rate after increasing
                    intervals of epochs are covered"
  standard          batch 10, initial learning rate 4e-2, 100 pairs of states
                    drawn uniformly in the box [-1, 1]^4 of phase space,
                    1.5e3 epochs (their Fig. 2)
  parametric        batch 100, initial learning rate 4e-3, hidden widths 10
                    (beta) and 20 (u), tanh, 40 field values sampled uniformly
                    in B/B_0 in [0.5, 2.5] with 100 random state pairs each,
                    data rotated by a random angle every 50 epochs, 1e4 to
                    4e4 epochs

Two things their Table I does not print and this script had to choose.  Both
are recorded here as deviations, with the reason:

  epoch budget      20 000, the midpoint of the printed 1e4 to 4e4 range, the
                    same for every model and every seed.  Declared before the
                    first run.  What matters downstream is the training loss
                    reached, and that is reported for every model rather than
                    assumed.
  halving intervals "increasing intervals" fixes the shape but not the numbers.
                    The first halving is at epoch 500 and each interval is 1.15
                    times the previous, giving 13 halvings over 20 000 epochs
                    and a final learning rate of 4.9e-7.

The training set is generated from the analytical solution alone; no reference
integrator enters training, which is their point in Sec. III.C.

The nine step sizes times three seeds are trained as one stack of 27
independent models (see `sm_arch.StackedParametricSympMat`); `--equivalence`
asserts that the stack reproduces the single-model class step for step.

Usage:  python sm1_train.py --equivalence
        python sm1_train.py --standard
        python sm1_train.py --all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_common as S                                        # noqa: E402
import sm_arch as A                                          # noqa: E402

torch.set_num_threads(1)
CKPT = S.CKPT_DIR

EPOCHS_PARAM = 20000
EPOCHS_STD = 1500
LR_PARAM, BATCH_PARAM = 4e-3, 100
LR_STD, BATCH_STD = 4e-2, 10
AUGMENT_EVERY = 50
HALVE_FIRST, HALVE_RATIO = 500, 1.15


def halving_epochs(n_epochs, first=HALVE_FIRST, ratio=HALVE_RATIO):
    ms, t, iv = [], 0.0, float(first)
    while True:
        t += iv
        if t >= n_epochs:
            return ms
        ms.append(int(round(t)))
        iv *= ratio


def _rot_block(phi):
    c, s = np.cos(phi), np.sin(phi)
    R2 = np.array([[c, -s], [s, c]])
    return np.block([[R2, np.zeros((2, 2))], [np.zeros((2, 2)), R2]])


def make_data(dt, dt_index, rep):
    """100 pairs per field value from the analytical solution, their Sec. III.C."""
    bs = np.array(S.B_TRAIN)
    rng = np.random.default_rng(S.seed_of("pdata", dt_index, rep))
    X0 = rng.uniform(-1.0, 1.0, size=(len(bs), S.N_PAIRS_PER_B, 4))
    X1 = np.stack([X0[k] @ S.analytic_M(bs[k], dt).T for k in range(len(bs))])
    return X0.reshape(-1, 4), X1.reshape(-1, 4)


def train_stack(jobs, epochs=EPOCHS_PARAM, verbose=True, single=False):
    """jobs: list of (dt_index, rep).  Returns (matrices, histories, snaps, wall).

    `snaps` is the whole trajectory of the training: the 4n-reflector product
    evaluated at every field value, at every checkpoint where the full-batch
    loss is recorded.  It is kept because the crossing their Fig. 9 reports is
    not a property of the architecture alone -- it moves as the cube root of the
    training loss, since the SympMat curve is (per-step error) x (number of
    steps) and the Boris curve is second order -- so a reproduction has to be
    able to compare at *their* loss, 10^-8, and not only at whatever loss our
    budget happens to reach.  Storing the trajectory costs two megabytes per
    seed and removes the need to guess the right budget in advance.""" 
    bs = np.array(S.B_TRAIN)
    K, P = len(bs), S.N_PAIRS_PER_B
    n = K * P
    nm = len(jobs)

    x0 = np.stack([make_data(S.DT_LADDER[k], k, r)[0] for k, r in jobs])
    x1 = np.stack([make_data(S.DT_LADDER[k], k, r)[1] for k, r in jobs])
    x0t, x1t = torch.tensor(x0), torch.tensor(x1)
    bi = torch.tensor(np.repeat(np.arange(K), P))
    mu = torch.tensor(bs, dtype=torch.float64)
    ar = torch.arange(nm).unsqueeze(1)

    seeds = [S.seed_of("pinit", k, r) for k, r in jobs]
    if single:
        assert nm == 1
        model = A.ParametricSympMat(seeds[0])
        mats = lambda: model.matrices(mu).unsqueeze(0)
    else:
        model = A.StackedParametricSympMat(seeds)
        mats = lambda: model.matrices(mu)
    opt = torch.optim.Adam(model.parameters(), lr=LR_PARAM)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=halving_epochs(epochs), gamma=0.5)

    rb = [np.random.default_rng(S.seed_of("pbatch", k, r)) for k, r in jobs]
    ra = [np.random.default_rng(S.seed_of("paug", k, r)) for k, r in jobs]

    t0, hist, snaps, snap_ep = time.time(), [[] for _ in jobs], [], []
    for ep in range(epochs):
        if ep and ep % AUGMENT_EVERY == 0:
            R = torch.tensor(np.stack([_rot_block(g.uniform(0.0, 2 * np.pi))
                                       for g in ra]))
            x0t = torch.einsum("mij,mnj->mni", R, x0t)
            x1t = torch.einsum("mij,mnj->mni", R, x1t)
        perm = torch.tensor(np.stack([g.permutation(n) for g in rb]))
        for s in range(0, n, BATCH_PARAM):
            idx = perm[:, s:s + BATCH_PARAM]
            M = mats()[ar, bi[idx]]
            pred = (M @ x0t[ar, idx].unsqueeze(-1)).squeeze(-1)
            loss = ((pred - x1t[ar, idx]) ** 2).mean(dim=(1, 2)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 500 == 0 or ep == epochs - 1:
            with torch.no_grad():
                M = mats()[:, bi]
                full = ((torch.einsum("mnij,mnj->mni", M, x0t) - x1t) ** 2
                        ).mean(dim=(1, 2))
            for m in range(nm):
                hist[m].append([ep, float(full[m])])
            with torch.no_grad():
                snaps.append(mats().numpy().copy())
            snap_ep.append(ep)
            if verbose:
                print("ep %6d  lr %.2e  mse %.3e..%.3e  %.0fs"
                      % (ep, opt.param_groups[0]["lr"], float(full.min()),
                         float(full.max()), time.time() - t0), flush=True)
    with torch.no_grad():
        return (mats().numpy(), hist,
                {"M": np.asarray(snaps), "epoch": np.asarray(snap_ep)},
                time.time() - t0)


def train_standard(b, dt, rep=0, epochs=EPOCHS_STD, verbose=True):
    rng = np.random.default_rng(S.seed_of("sdata", 0, rep))
    X0 = rng.uniform(-1.0, 1.0, size=(S.N_PAIRS_PER_B, 4))
    X1 = X0 @ S.analytic_M(b, dt).T
    model = A.StandardSympMat(S.seed_of("sinit", 0, rep))
    opt = torch.optim.Adam(model.parameters(), lr=LR_STD)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=halving_epochs(epochs, first=50), gamma=0.5)
    rb = np.random.default_rng(S.seed_of("sbatch", 0, rep))
    x0t, x1t = torch.tensor(X0), torch.tensor(X1)
    hist = []
    for ep in range(epochs):
        perm = torch.tensor(rb.permutation(len(X0)))
        for s in range(0, len(X0), BATCH_STD):
            idx = perm[s:s + BATCH_STD]
            loss = ((x0t[idx] @ model.matrix().T - x1t[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 100 == 0 or ep == epochs - 1:
            with torch.no_grad():
                hist.append([ep, float((((x0t @ model.matrix().T) - x1t) ** 2).mean())])
            if verbose:
                print("  standard b=%g dt=%g ep=%5d mse=%.3e"
                      % (b, dt, ep, hist[-1][1]), flush=True)
    with torch.no_grad():
        return model.matrix().numpy(), hist


def summary(Mp, dt, k, rep, hist, wall):
    bs = np.array(S.B_TRAIN)
    out = {"final_mse": hist[-1][1], "loss_history": hist,
           "seed_init": S.seed_of("pinit", k, rep),
           "seed_data": S.seed_of("pdata", k, rep),
           "train_wall_s": wall,
           "max_symplectic_defect": max(S.sympl_defect(m) for m in Mp),
           "per_b": {}}
    for b in S.B_EVAL:
        j = int(np.argmin(np.abs(bs - b)))
        assert abs(bs[j] - b) < 1e-12
        Mt = S.analytic_M(b, dt)
        ev = np.linalg.eigvals(Mp[j])
        out["per_b"]["%g" % b] = {
            "frobenius_error": float(np.linalg.norm(Mp[j] - Mt)),
            "max_abs_error": float(np.max(np.abs(Mp[j] - Mt))),
            "eig_abs": sorted(float(v) for v in np.abs(ev)),
            "eig_abs_max_minus_1": float(np.max(np.abs(ev)) - 1.0),
            "symplectic_defect": S.sympl_defect(Mp[j])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rep", type=int, default=None,
                    help="train the nine step sizes of one seed only")
    ap.add_argument("--tag", default="",
                    help="write checkpoints into ckpt<tag>/ instead of ckpt/. "
                         "Because the halving milestones of `halving_epochs` "
                         "are generated from a fixed first interval and ratio "
                         "and merely truncated at the budget, a run with "
                         "--epochs E is an exact prefix of the run with a "
                         "larger budget; a shorter, tagged run is therefore "
                         "the same models caught earlier, not different ones.")
    ap.add_argument("--standard", action="store_true")
    ap.add_argument("--equivalence", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS_PARAM)
    a = ap.parse_args()
    global CKPT
    if a.tag:
        CKPT = CKPT + a.tag
    os.makedirs(CKPT, exist_ok=True)

    if a.equivalence:
        Ma, _, _, _ = train_stack([(4, 0)], epochs=25, verbose=False, single=True)
        Mb, _, _, _ = train_stack([(4, 0)], epochs=25, verbose=False, single=False)
        d = float(np.max(np.abs(Ma - Mb)))
        print("stack vs single after 25 epochs: max |dM| = %.3e" % d)
        assert d < 1e-13, d
        Mc, _, _, _ = train_stack([(4, 0), (7, 2)], epochs=25, verbose=False)
        d2 = float(np.max(np.abs(Mc[0] - Mb[0])))
        print("stack of 2 vs stack of 1, member 0: max |dM| = %.3e" % d2)
        assert d2 < 1e-13, d2
        return 0

    if a.standard:
        res = {}
        for b, dt in ((1.0, 2.0), (0.5, 0.25), (2.5, 0.25)):
            Mp, hist = train_standard(b, dt)
            Mt = S.analytic_M(b, dt)
            res["b%g_dt%g" % (b, dt)] = {
                "final_mse": hist[-1][1], "loss_history": hist,
                "frobenius_error": float(np.linalg.norm(Mp - Mt)),
                "max_abs_error": float(np.max(np.abs(Mp - Mt))),
                "symplectic_defect": S.sympl_defect(Mp),
                "seed_init": S.seed_of("sinit"), "seed_data": S.seed_of("sdata")}
        json.dump(res, open(os.path.join(CKPT, "standard.json"), "w"), indent=1)
        print("standard written")
        return 0

    if a.all or a.rep is not None:
        reps = range(S.N_SEEDS) if a.rep is None else [a.rep]
        jobs = [(k, r) for r in reps for k in range(len(S.DT_LADDER))]
        Mp, hist, snaps, wall = train_stack(jobs, epochs=a.epochs)
        res = {"epochs": a.epochs, "n_models": len(jobs), "train_wall_s": wall,
               "models": {}}
        for m, (k, r) in enumerate(jobs):
            np.savez_compressed(
                os.path.join(CKPT, "param_dt%d_rep%d.npz" % (k, r)),
                M=Mp[m], b=np.array(S.B_TRAIN), dt=S.DT_LADDER[k],
                M_trajectory=snaps["M"][:, m], snapshot_epoch=snaps["epoch"],
                snapshot_mse=np.array([h[1] for h in hist[m]]))
            res["models"]["dt%d_rep%d" % (k, r)] = summary(
                Mp[m], S.DT_LADDER[k], k, r, hist[m], wall)
        tag = "" if a.rep is None else "_rep%d" % a.rep
        json.dump(res, open(os.path.join(CKPT, "parametric%s.json" % tag), "w"),
                  indent=1)
        print("all written; final mse range %.3e .. %.3e"
              % (min(v["final_mse"] for v in res["models"].values()),
                 max(v["final_mse"] for v in res["models"].values())))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
