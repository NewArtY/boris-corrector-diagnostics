"""Shared machinery for wave W16 -- the ensemble over training seeds.

WHAT THIS DIRECTORY IS FOR
--------------------------
Every corrector number the manuscript prints stands on one checkpoint,
`checkpoints/boris_corrector_b4.pt`, produced by one run of
`training/train_corrector_b4.py` at seed 42.  `plan/reports/I0_platsdarm.md`
established at the start of the campaign that the withdrawn work's
generalisation claims did not survive retraining.  The question of this
directory is therefore not whether the corrector is good but whether the
committed checkpoint is **typical or lucky**, and the deliverable is a
percentile: where the committed run sits inside an ensemble of retrainings of
the same architecture at the same hyper-parameters.

Beside it, the four repetitions of the external architectures of W9.1 are
extended to ten, because W9.1 itself observed that at a spread of 2.3 to 3.6
in these channels four repetitions do not resolve a tail.

WHAT IS RETRAINED AND WHAT IS NOT
---------------------------------
Retrained: the corrector, by calling `training/train_corrector_b4.py`'s own
`build_dataset()` and `train()` unchanged, with two module globals redirected
and nothing else -- `SEED`, which is the only place the procedure consumes a
seed, and `CHECKPOINT_DIR`, so that the committed checkpoint is never written
to.  `sd2_train.py` refuses to run if the output directory is the bundle's own
checkpoint directory and checks the committed file's md5 before and after.
Also retrained: the three external architectures, at repetitions 4 to 9, by
calling `external_arch/ea1_train.py:train_one` and `score_section7` unchanged.

Not retrained and not written to: `checkpoints/`, `experiments/external_arch/`
(its `ea1_training.json` and `ckpt/` are the committed numbers of W9.1 and are
read only), `experiments/hpo/`, `experiments/map/`, `experiments/gtable/`,
`experiments/spectral/`.  Every file this directory writes lives under
`experiments/seeds/`.

WHAT IS MEASURED, AND WHY IT IS THE SAME FOUR CHANNELS
------------------------------------------------------
The measurement stand is W15's, imported and not rewritten: the batched
three-dimensional rollout and the field bridge from `../map/map_common.py`, the
four channels and their committed statistics from `../gtable/gt_common.py`, the
periodogram and the band from `../spectral/sw_common.py`, the JSON gatekeeper
from `../external_arch/ea_common.py`.  A change to any of them changes these
numbers too.  `sd1_calibration.py` reproduces the corrector and Boris rows of
Table~\\ref{tab:family} on this stand before anything is retrained.

    trajectory   root mean square of |r - r_ref| / r_L over the record
    phase        median of atan2(|v x v_ref|, v . v_ref) over the second half
                 -- never arccos: at theta ~ 1e-10 a double-precision arccos
                 returns exactly zero (W15 Section 3.4)
    energy       median of |(E - E_ref)/E_ref(0)| over the second half
    spectral     the integral of the PSD of the position error below
                 f/Omega_c^ref = 0.2, Hann window

THE ENSEMBLE IS NOT A SUBSTITUTE FOR THE COMMITTED RUN
------------------------------------------------------
A retrained corrector is a **different object** from the committed one.  The
ensemble answers "how much do the manuscript's numbers depend on luck in
training", and it is reported as that.  Which of the two goes into the text is
decided by the result and stated explicitly; it is never substituted silently.

The classical side is deterministic.  vps2, vps4, gl4, the Boris scheme, the
closed forms and the flop counts contain no random draw, and their numbers have
no spread.  That is written out in words rather than left as an empty cell.

SEED LEDGER, DECLARED BEFORE THE FIRST RUN
------------------------------------------
Blocks already occupied in this bundle:

    below 500_000          everything in the shipped bundle, including the
                           committed corrector at seed 42
    9_000_000..9_704_999   W9.1, `external_arch/ea_common.py:seed_of`
    11_000_000..13_999_909 W12, `hpo/hp_common.py:seed_of`
    13_000_000             W13, `spectral/sw_common.py:SPECTRAL_SEED`
    14_000_000             W14, `map/map_common.py:MAP_SEED`
    20260830               the initial-condition ensemble of `../stats/`

This directory forms every corrector seed in one place, `corrector_seed()`,
from a block no other script touches:

    16_000_000 + i ,   i = 0 .. N_CORRECTOR_SEEDS-1

The external architectures are **not** given a new block.  Their ledger already
has room: `ea_common.seed_of` was used by W9.1 at repetitions 0..3 only, and
W12 took data draws 10..13, 30..45 and 90 from it.  W16 takes repetitions
**4..9**, whose initialisation, shuffling and data seeds are free by
construction and disjoint from both.  Extending the existing ledger rather than
opening a second one is what keeps the ten repetitions one ensemble instead of
two.

Three seed accidents were caught earlier in this campaign, all of the same
kind: a generator rebuilt inside a loop.  No generator is constructed inside
any loop in this directory.  `training/train_corrector_b4.py:build_dataset`
builds its one generator at the top of the function, outside its loops, and is
called once per seed.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
for _p in (ROOT, os.path.join(EXP, "external_arch"), os.path.join(EXP, "map"),
           os.path.join(EXP, "gtable"), os.path.join(EXP, "spectral"),
           os.path.join(EXP, "classical")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import map_common as MC                                        # noqa: E402
import gt_common as G                                          # noqa: E402
import ea_common as EA                                         # noqa: E402
from ea_common import check_or_write                           # noqa: E402

clean = MC.clean

CKPT = os.path.join(HERE, "ckpt")
CKPT_EA = os.path.join(HERE, "ckpt_ea")
CACHE = os.path.join(HERE, "cache")
for _d in (CKPT, CKPT_EA, CACHE):
    os.makedirs(_d, exist_ok=True)

#: the bundle's own checkpoint directory.  Nothing here may write into it.
BUNDLE_CHECKPOINTS = os.path.join(ROOT, "checkpoints")
COMMITTED_CORRECTOR = os.path.join(BUNDLE_CHECKPOINTS, "boris_corrector_b4.pt")
COMMITTED_MD5 = "0fe271bdb54de8a720f11eec85ee01f5"

# ------------------------------------------------------------------ ledger --
SEED_BLOCK = 16_000_000
N_CORRECTOR_SEEDS = 16          # declared before the first run
COMMITTED_SEED = 42             # what the shipped checkpoint was trained at

#: the repetitions of the external-architecture ledger this wave adds.  W9.1
#: committed 0..3; W16 adds 4..9, for ten in all.
EA_REPS_COMMITTED = (0, 1, 2, 3)
EA_REPS_NEW = (4, 5, 6, 7, 8, 9)
EA_ARCHS = ("hnn", "sympnet", "pinn")


def corrector_seed(i):
    """The one place a corrector seed is formed."""
    assert 0 <= i < 1000
    return SEED_BLOCK + i


CORRECTOR_SEEDS = [corrector_seed(i) for i in range(N_CORRECTOR_SEEDS)]

# -------------------------------------------------------------------- axes --
DT = G.DT                                   # 0.3, the working step
HORIZONS = G.HORIZONS                       # H_paper 400, H_crossover 2120
FIELD_NAMES = MC.FIELD_NAMES
CHANNELS = G.CHANNELS                       # trajectory, phase, energy, spectral
N_IC = MC.N_IC                              # 8, from map_common, drawn there


# ============================================================ the corrector ==
def load_corrector(path):
    """One corrector checkpoint, lifted out of torch into plain numpy.

    Identical to `map_common.load_corrector_numpy` except that the path is an
    argument, so that the same evaluation code scores the committed checkpoint
    and every retrained one.
    """
    import torch
    torch.set_default_dtype(torch.float64)
    from training.train_corrector_b4 import DefectNet
    m = DefectNet(n_in=13)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return EA.lift_torch_mlp(m.net, m.x_mean.numpy(), m.x_std.numpy(),
                             m.y_scale.numpy())


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_committed_untouched():
    """The committed checkpoint must be bit-identical before and after every
    script in this directory.  W16 retrains; it does not replace."""
    got = md5(COMMITTED_CORRECTOR)
    if got != COMMITTED_MD5:
        raise SystemExit("the committed checkpoint changed: %s != %s"
                         % (got, COMMITTED_MD5))
    return got


def seed_ckpt(seed):
    return os.path.join(CKPT, "corrector_b4_s%d.pt" % seed)


# ============================================================ the references =
_REF = {}


def reference(fname, field, R0, V0, ts):
    """The reference orbit of a configuration on the grid `ts`.

    `../gtable/gt1_calibration.py:reference_orbit(tag="best")` verbatim: the
    closed form where there is one (uniform, B3, B4), DOP853 at rtol 3e-14
    where there is not (B1, B2).  Seed-independent, so it is computed once and
    cached on disk: the ensemble reruns the corrector, not the reference.
    """
    key = (fname, len(ts), float(ts[-1]))
    if key in _REF:
        return _REF[key]
    path = os.path.join(CACHE, "ref__%s__n%d.npz" % (fname, len(ts)))
    if os.path.exists(path):
        z = np.load(path)
        if z["ts"].shape == ts.shape and np.array_equal(z["ts"], ts):
            _REF[key] = (z["Rr"], z["Vr"])
            return _REF[key]
    from gt1_calibration import reference_orbit
    Rr, Vr = reference_orbit(fname, field, R0, V0, ts, tag="best")
    np.savez_compressed(path, ts=ts, Rr=Rr, Vr=Vr)
    _REF[key] = (Rr, Vr)
    return _REF[key]


# ========================================================== the measurement ==
def measure_run(fld, scheme, R0, V0, Rr, Vr, r_L, w0, mlp=None, n=None):
    """One scheme on one configuration, all four channels, both horizons.

    The rollout is `map_common.rollout`, the channel series
    `gt_common.channel_series` and the statistics `gt_common.summarise`, all
    unchanged.  Returns {horizon: {channel: {stat: (nb,) array}}}.
    """
    N = max(HORIZONS.values()) if n is None else n
    idx = np.arange(N)
    Rs, Vs, meta = MC.rollout(fld, scheme, R0, V0, DT, N - 1, idx, mlp=mlp)
    ch = G.channel_series(Rs, Vs, Rr[:N], Vr[:N], r_L)
    out = {}
    for hname, hn in HORIZONS.items():
        if hn > N:
            continue
        chn = {k: v[:hn] for k, v in ch.items()}
        sm = G.summarise(chn, DT, w0)
        rec = {}
        for c in CHANNELS:
            rec[c] = {"primary": sm[c]["primary"], "rms": sm[c]["rms"]}
            if "max" in sm[c]:
                rec[c]["max"] = sm[c]["max"]
            if c == "spectral":
                rec[c]["p_total"] = sm[c]["p_total"]
                rec[c]["frac_in_band"] = sm[c]["frac_in_band"]
                rec[c]["n_bins_in_band"] = sm[c]["n_bins_in_band"]
        rec["_run"] = {"n_steps": hn, "h": DT,
                       "n_nonfinite": meta["n_nonfinite"],
                       "flops_per_step": MC.flops_per_step(
                           scheme, meta.get("mean_iters")),
                       }
        rec["_run"]["total_flops"] = rec["_run"]["flops_per_step"] * hn
        if "mean_iters" in meta:
            rec["_run"]["mean_iters"] = meta["mean_iters"]
        out[hname] = rec
    return out


# ================================================================ statistics =
def quartiles(a):
    """Median and the interquartile range of a sample, with the sample size.

    The linear-interpolation quantile of numpy, stated so that the number can
    be recomputed: `numpy.percentile(a, [25, 50, 75])`.  The ratio max/min is
    reported beside it because that is the form W9.1 and W12 quote a spread in
    ("a factor of 3.4"), and a spread of orders is easier to read as a ratio
    than as a difference.
    """
    a = np.asarray(a, dtype=float)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return {"n": 0, "median": float("nan"), "q1": float("nan"),
                "q3": float("nan"), "iqr": float("nan"), "min": float("nan"),
                "max": float("nan"), "ratio_max_min": float("nan"),
                "iqr_ratio_q3_q1": float("nan")}
    q1, med, q3 = (float(x) for x in np.percentile(f, [25.0, 50.0, 75.0]))
    lo, hi = float(f.min()), float(f.max())
    return {"n": int(f.size), "median": med, "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": lo, "max": hi,
            "ratio_max_min": (hi / lo) if lo > 0 else float("inf"),
            "iqr_ratio_q3_q1": (q3 / q1) if q1 > 0 else float("inf")}


def percentile_of(value, sample):
    """Where one value sits inside a sample, reported four ways.

    `n_below` counts members strictly smaller than the value, `n_above` those
    strictly larger.  For an error channel, smaller is better, so a committed
    checkpoint in the FAVOURABLE tail is one with few members below it.
    `percentile_below` is the fraction of the sample strictly below; the
    mid-rank version, which splits ties, is reported beside it so that a tie
    does not have to be adjudicated by the choice of convention.

    The sample never contains the value being placed: the committed checkpoint
    is not one of the ensemble's own draws.
    """
    v = float(value)
    s = np.asarray(sample, dtype=float)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n == 0 or not math.isfinite(v):
        return {"n": n, "value": v, "n_below": 0, "n_above": 0,
                "percentile_below": float("nan"),
                "percentile_midrank": float("nan"),
                "rank_best_is_1": float("nan"),
                "ratio_to_median": float("nan")}
    below = int(np.sum(s < v))
    above = int(np.sum(s > v))
    ties = n - below - above
    med = float(np.median(s))
    return {"n": n, "value": v, "n_below": below, "n_above": above,
            "n_ties": ties,
            "percentile_below": 100.0 * below / n,
            "percentile_midrank": 100.0 * (below + 0.5 * ties) / n,
            "rank_best_is_1": below + 1,
            "ratio_to_median": (v / med) if med > 0 else float("nan"),
            "orders_to_median": (math.log10(med / v)
                                 if (med > 0 and v > 0) else float("nan"))}


# ============================================================ training cost ==
#: One flop per arithmetic operation, twenty per transcendental -- the model of
#: Section 9, imported from `../external_arch/ea_common.py` and not restated.
#: A forward pass of the corrector is 113,958 flops by that model, which is the
#: figure the manuscript prints; `ea1_train.py` asserts it.
CORRECTOR_WIDTHS = [13, 128, 128, 128, 128, 6]


def corrector_training_flops(n_samples=6000, epochs=400, batch=512,
                             dt_fine_substeps=150, n_coarse=400,
                             n_trajectories=15, boris_flops=113):
    """What one retraining of the corrector costs, in flops.

    Two terms, reported separately because they are paid at different times:

      data   the one-step defect is defined by propagating each sampled state
             over the working step with 150 Boris steps of h/150, and the
             sample itself is advanced along that fine reference, so the
             dataset costs n_trajectories * n_coarse * 150 Boris steps at the
             committed 113 flops per step.
      optim  a forward and a backward pass over every sample of every epoch.
             Reverse-mode costs about twice the forward pass, so one Adam
             step over a batch is 3 * batch * forward; the optimiser's own
             update is 5 flops per parameter per step and is included.

    The number the manuscript's cost column carries is the cost of *running*
    the corrector, 114,091 flops per step.  Training is paid once and the
    column does not contain it; W12 made the same point about the external
    architectures.  It is reported here so that the omission is visible.
    """
    fwd = EA.mlp_forward_flops(CORRECTOR_WIDTHS)
    n_par = sum(CORRECTOR_WIDTHS[i] * CORRECTOR_WIDTHS[i + 1]
                + CORRECTOR_WIDTHS[i + 1]
                for i in range(len(CORRECTOR_WIDTHS) - 1))
    data = n_trajectories * n_coarse * dt_fine_substeps * boris_flops
    steps_per_epoch = math.ceil(0.9 * n_samples / batch)
    adam_steps = epochs * steps_per_epoch
    optim = adam_steps * (3.0 * batch * fwd + 5.0 * n_par)
    return {"forward_flops": fwd, "n_parameters": n_par,
            "data_flops": float(data), "adam_steps": adam_steps,
            "optimisation_flops": float(optim),
            "total_flops": float(data + optim),
            "inference_flops_per_step": 114091.0,
            "run_flops_400_steps": 114091.0 * 400,
            "model": "1 flop per arithmetic op, 20 per transcendental "
                     "(Section 9); backward = 2x forward; Adam update 5 flops "
                     "per parameter per step"}


# ------------------------------------------------------------------ output --
def write(path, payload, force=False, rtol=1e-9):
    return check_or_write(path, json.loads(json.dumps(clean(payload))),
                          rtol=rtol, force=force)
