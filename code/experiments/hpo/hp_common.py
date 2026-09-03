"""W12.1 -- shared machinery for the hyper-parameter search on the three
external architectures of Section `sec:external`.

THE QUESTION THIS DIRECTORY ANSWERS
-----------------------------------
Section `sec:external` reports that vps4 is more accurate than the best of the
three learned architectures by a factor of 328 in the trajectory channel and
1122 in the energy channel, at 300 times lower cost.  The three were trained at
one configuration and one budget: the corrector's own 4400 Adam steps at batch
512 and lr 1e-3, four seeds, with two controls (four times the budget, twice
the width).  The obvious objection is that the baselines are undertrained and
that the paper dismantled a straw man.

This directory does not answer that by running a larger grid for its own sake.
It answers two measurable questions:

  (1) Over a hyper-parameter grid built where each architecture has its best
      chance, does the best reachable configuration close the gap?
  (2) How does the rollout error fall with training budget and with capacity,
      and what budget would the measured trend require in order to close a
      factor of 328?

A trend extrapolated to 10^6 Adam steps or 10^8 parameters is a stronger answer
than any finite grid, because it covers the configurations that were not run.

WHAT IS HELD FIXED
------------------
Everything that is not a hyper-parameter: the problem, the field, the step
size, the readout, the data-generating procedure, the flop model and the
scoring run of Section 7 are `experiments/external_arch/` verbatim, imported
rather than copied.  A change to `ea_common.py`, `ea_arch.py` or the scoring
function of `ea1_train.py` changes the numbers here too.

Nothing in `experiments/external_arch/` is written to.  The checkpoints and
JSON of wave W9.1 carry the numbers the manuscript prints and are not retrained.

THE TRAIN / VALIDATION SPLIT, DECLARED
--------------------------------------
`ea1_train.build_states(rep)` draws 15 trajectories -- three at each of the
five decay times of `TAU_TRAIN` -- with the initial radius and phase of each
drawn from the generator seeded by `ea_common.seed_of(7, 2, rep)`, and returns
the 400 coarse states along each, 6000 in all.  The split is therefore at the
level of the trajectory and not of the state, which is the only split that
means anything for data lying along orbits:

  training     6000 states from draw `rep in (10, 11, 12, 13)`, one draw per
               seed, exactly as the four repetitions of W9.1 used draws 0..3
  validation   6000 states from draw `rep = 90`, fifteen trajectories that
               appear in no training set, shared by every configuration and
               every seed so that the selection compares like with like

Draws 0..3 are the ones W9.1 committed; 10..13 and 90 are disjoint from them
and from each other.  Model selection uses the validation loss at the end of
training -- each architecture's own loss functional, since the three losses are
not comparable with one another, so selection is always within an architecture.
The rollout error on the run of Section 7 is never used for selection.  It is
reported as well, and where the report quotes the best configuration *by
rollout error* it says so explicitly: that is an oracle no practitioner has,
and it is quoted only because it bounds what the grid could have achieved.

SEED LEDGER
-----------
Three seed accidents were caught during this campaign (see
`external_arch/ea_common.py`), so seeds here are formed in one place, from one
block that no other script touches, and written into the output beside the
number they produced.

    11_000_000 + 1_000_000 * arch + 10_000 * cfg + 100 * rep + role

  arch  0 hnn, 1 sympnet, 2 pinn
  cfg   index of the configuration in `GRID[arch]`, 0..99
  rep   repetition, 0..99
  role  0 weight initialisation, 1 minibatch shuffling

The block runs from 11,000,000 to 13,999,909.  W9.1 uses 9,000,000..9,704,999
and the highest seed anywhere else in the bundle is 500,000, so no seed is
reused.  Generators are built once, before the loop that draws from them.

NUMBER OF SEEDS, DECLARED BEFORE THE RUN
----------------------------------------
Four, at every cell of the grid and every rung of the budget ladder.  Where the
compute did not fit, configurations were dropped and never seeds: the spread
over seeds at this problem is a factor of 3.4 in the trajectory channel (W9.1
Section 1.4) and a single-seed grid point would carry no information at all.

GATEKEEPING
-----------
One JSON per job under `runs/`, written on the first pass and compared on every
later one by `ea_common.check_or_write` at rtol 1e-6; a job whose numbers move
makes the script exit non-zero.  Sharding is a scheduling detail only: a job's
seeds depend on (arch, cfg, rep, role) and on nothing else, so the same job
gives the same file whatever `--shard` it was run under.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.normpath(os.path.join(HERE, "..", "external_arch"))
for _p in (EXT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ea_common as C          # noqa: E402
import ea_arch as A            # noqa: E402
import ea1_train as T          # noqa: E402

torch.set_default_dtype(torch.float64)

RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

# ---------------------------------------------------------------- ledger --
SEED_BLOCK = 11_000_000
ARCHS = ("hnn", "sympnet", "pinn")
ARCH_INDEX = {a: i for i, a in enumerate(ARCHS)}
ROLE = {"init": 0, "shuffle": 1, "data_init": 2, "data_shuffle": 3}


def seed_of(arch, cfg_index, rep, role):
    """The one place a seed is formed.  See the ledger in the module docstring."""
    ai = ARCH_INDEX[arch]
    ri = ROLE[role]
    assert 0 <= cfg_index < 100 and 0 <= rep < 100 and 0 <= ri < 10
    return SEED_BLOCK + 1_000_000 * ai + 10_000 * cfg_index + 100 * rep + ri


N_SEEDS = 4
DATA_REP_TRAIN = (10, 11, 12, 13)      # one data draw per seed
DATA_REP_VAL = 90                      # held out, shared by every job

#: The third resource, swept in `hp5_data.py`: how much training data there is.
#: Budget and capacity are not the only things a referee can say were too small.
#: The cut is at the level of the trajectory, as the train/validation split is:
#:
#:   small  five of the fifteen trajectories of the seed's own draw, one at
#:          each decay time -- a nested subset of the full set, 2000 states
#:   full   the fifteen trajectories of the seed's own draw, 6000 states,
#:          which is what every other job in this directory trains on
#:   large  sixty trajectories from four fresh draws, 24000 states, disjoint
#:          from the full set of every other seed and from the validation draw
#:
#: The number of Adam steps is held at the base budget across the three, so
#: this axis is data and nothing else.  Draws 30..45 belong to this sweep alone.
DATA_SWEEP = ("small", "full", "large")
DATA_SWEEP_SIZE = {"small": 2000, "full": 6000, "large": 24000}
DATA_REP_LARGE_BASE = 30               # draws 30+4r .. 33+4r for seed r

# ------------------------------------------------------------------ grid --
# Configurations are chosen where each architecture has a chance, not on a
# product grid for its own sake.  The reasoning is in the report; in short:
#
#  HNN      W9.1 measured that its field-matching loss and its rollout error
#           move in opposite directions, so the knob with a chance is not
#           capacity but conditioning -- the learning rate, and a *smaller*
#           network whose learned field is smoother.  The width sweep at fixed
#           depth is there so that the capacity trend can be extrapolated.
#  SympNet  the only one of the three that improved at four times the budget,
#           so it gets the widest grid and the longest ladder.  Its parameter
#           count grows linearly in the width, so capacity is bought both by
#           the width and by the number of gradient modules, and both are swept.
#  PINN     the only one whose W9.1 control improved with capacity (0.337 at
#           width 256 against 0.503 at 128), so its grid is a width sweep with
#           the learning rate swept at the width that helped.
#
# The first entry of each list is the configuration W9.1 trained, so that the
# grid contains the manuscript's own row as its anchor.
GRID = {
    "hnn": [
        ("L4W128_lr1e-3", dict(n_layers=4, width=128, lr=1e-3)),   # W9.1 anchor
        ("L4W64_lr1e-3",  dict(n_layers=4, width=64,  lr=1e-3)),
        ("L4W256_lr1e-3", dict(n_layers=4, width=256, lr=1e-3)),
        ("L6W128_lr1e-3", dict(n_layers=6, width=128, lr=1e-3)),
        ("L4W128_lr3e-4", dict(n_layers=4, width=128, lr=3e-4)),
        ("L4W128_lr3e-3", dict(n_layers=4, width=128, lr=3e-3)),
    ],
    "sympnet": [
        ("M10W256_lr1e-3", dict(n_modules=10, width=256, lr=1e-3)),  # W9.1 anchor
        ("M6W128_lr1e-3",  dict(n_modules=6,  width=128, lr=1e-3)),
        ("M16W256_lr1e-3", dict(n_modules=16, width=256, lr=1e-3)),
        ("M10W512_lr1e-3", dict(n_modules=10, width=512, lr=1e-3)),
        ("M10W256_lr3e-4", dict(n_modules=10, width=256, lr=3e-4)),
        ("M10W256_lr3e-3", dict(n_modules=10, width=256, lr=3e-3)),
    ],
    "pinn": [
        ("L4W128_lr1e-3", dict(n_layers=4, width=128, lr=1e-3)),   # W9.1 anchor
        ("L4W256_lr1e-3", dict(n_layers=4, width=256, lr=1e-3)),
        ("L4W512_lr1e-3", dict(n_layers=4, width=512, lr=1e-3)),
        ("L4W256_lr3e-4", dict(n_layers=4, width=256, lr=3e-4)),
        ("L4W256_lr3e-3", dict(n_layers=4, width=256, lr=3e-3)),
    ],
}

#: The configurations that form a clean capacity sweep -- everything but the
#: capacity held fixed -- so that the trend of the rollout error in the number
#: of parameters can be fitted and extrapolated.  Named here rather than picked
#: out of the results afterwards.
CAPACITY_SWEEP = {
    "hnn": ("L4W64_lr1e-3", "L4W128_lr1e-3", "L4W256_lr1e-3"),
    "sympnet": ("M6W128_lr1e-3", "M10W256_lr1e-3", "M16W256_lr1e-3",
                "M10W512_lr1e-3"),
    "pinn": ("L4W128_lr1e-3", "L4W256_lr1e-3", "L4W512_lr1e-3"),
}

#: Budget ladder, in multiples of the corrector's 4400 Adam steps.
#:
#: DEVIATION FROM WHAT WAS DECLARED, RECORDED AS ONE.  The ladder was declared
#: as x2, x4, x8, x16 (x32 for the SympNet, x8 for the PINN) on the
#: configuration the validation loss selects.  It was launched that way and
#: killed thirteen minutes in, when the first trace lines showed that the
#: machine -- a throttling laptop, an i7-13620H at its 2.4 GHz base clock after
#: an hour at full load -- was delivering a quarter of the throughput the
#: campaign was planned on.  Measured with the shipped benchmark, eight
#: concurrent workers share about nineteen SympNet steps a second in total,
#: which puts a single x32 run at nine hours.  What is run instead is below,
#: and the report says so in the same words.  Nothing was cut from the number
#: of seeds: four at every rung, as declared.
LADDER = {"hnn": (2, 8), "sympnet": (8,), "pinn": (2,)}

#: The ladder runs on the ANCHOR configuration -- entry 0 of each grid, the one
#: W9.1 trained and the manuscript prints -- and not on the one the validation
#: loss selected.  This too is a deviation and it is recorded as one.  The
#: reason is compute and the justification is measured: the configuration the
#: validation loss selected costs 1.6 times more per Adam step for the SympNet
#: and 2.1 times more for the PINN (for the HNN it differs only in the learning
#: rate and costs the same), and in the rollout channel all three differ from
#: their anchors by less than the spread over four seeds.  The PINN's selected
#: configuration is in fact the worse of the two.  The ladder's question is
#: also, strictly, a question about the
#: anchors: it asks whether the row the manuscript prints is undertrained.
LADDER_CFG_INDEX = 0

#: Fractions of the run at which the rollout is scored during training.  These
#: are NOT equivalent to full runs at that budget: the cosine schedule anneals
#: to zero at the end of whatever budget it was given, so an intermediate point
#: carries a learning rate that a completed run of that length would not have.
#: They measure something else, which is what they are here for -- whether
#: descending the loss moves the rollout error in the same direction.
TRACE_FRACTIONS = (0.05, 0.125, 0.25, 0.5, 0.75, 1.0)

SYMP_SUBSAMPLE = T.SYMP_SUBSAMPLE
LAMBDA_SYMP = T.LAMBDA_SYMP
VAL_PENALTY_STATES = 512       # states carrying the PINN Jacobian penalty at
                               # validation time, the first 512, deterministic

# classical targets, read from the file Table `tab:family` is printed from
CLASSICAL = os.path.normpath(os.path.join(EXT, "..", "classical", "verdict.json"))


def classical_rows():
    import json
    d = json.load(open(CLASSICAL, encoding="utf-8"))
    return d["schemes"], d["physical_signal"]


# ------------------------------------------------------- data, in-process --
_DATA = {}


def get_data(rep):
    if rep not in _DATA:
        _DATA[rep] = T.canonicalise(T.build_states(rep))
    return _DATA[rep]


def _take(d, mask):
    return {k: v[mask] for k, v in d.items()}


def _cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def training_set(rep, data_key):
    """The training set of one seed at one point of the data sweep.

    `build_states` lays its rows out as five decay times, three trajectories at
    each, four hundred coarse states along each, in that order, so trajectory
    `i // 400` and its index within its decay time is `(i // 400) % 3`.  The
    small set keeps the first trajectory at each decay time, which is a nested
    subset of the full set and covers the same five fields.
    """
    if data_key in (None, "full"):
        return get_data(DATA_REP_TRAIN[rep]), [DATA_REP_TRAIN[rep]]
    if data_key == "small":
        d = get_data(DATA_REP_TRAIN[rep])
        i = np.arange(d["x"].size)
        return _take(d, ((i // 400) % 3) == 0), [DATA_REP_TRAIN[rep]]
    if data_key == "large":
        reps = [DATA_REP_LARGE_BASE + 4 * rep + j for j in range(4)]
        return _cat([get_data(r) for r in reps]), reps
    raise ValueError(data_key)


# The reference of Section 7 is the same for every job.  scipy's DOP853 is
# deterministic on identical inputs, so memoising it changes no number; it
# removes a solve_ivp call from every one of the ~900 rollout scorings below.
_REF = {}
_dop853_raw = C.dop853


def _dop853_memo(tau, t_eval, r0=None, v0=None, rtol=1e-12, atol=1e-14):
    key = (float(tau), len(t_eval), float(t_eval[-1]),
           None if r0 is None else tuple(np.ravel(r0)),
           None if v0 is None else tuple(np.ravel(v0)), rtol, atol)
    if key not in _REF:
        _REF[key] = _dop853_raw(tau, t_eval, r0, v0, rtol, atol)
    return _REF[key]


C.dop853 = _dop853_memo


# ----------------------------------------------------------------- models --
def build_model(arch, cfg, gen):
    if arch == "hnn":
        return T.HNNTorch(gen, hidden=cfg["width"], n_layers=cfg["n_layers"])
    if arch == "pinn":
        return T.PINNTorch(gen, hidden=cfg["width"], n_layers=cfg["n_layers"])
    if arch == "sympnet":
        return T.SympNetTorch(gen, n_modules=cfg["n_modules"], width=cfg["width"])
    raise ValueError(arch)


def _tensors(arch, d):
    tt = lambda k: torch.tensor(d[k])                          # noqa: E731
    if arch == "hnn":
        return {"z": torch.stack([tt("x"), tt("y"), tt("px"), tt("py"),
                                  tt("s")], dim=1),
                "tgt": torch.stack([tt("dqx"), tt("dqy"), tt("dpx"),
                                    tt("dpy")], dim=1)}
    if arch == "sympnet":
        return {"q": torch.stack([tt("x"), tt("y")], dim=1),
                "p": torch.stack([tt("px"), tt("py")], dim=1),
                "qn": torch.stack([tt("xn"), tt("yn")], dim=1),
                "pn": torch.stack([tt("pxn"), tt("pyn")], dim=1),
                "s": tt("s")}
    return {"z": torch.stack([tt("x"), tt("y"), tt("vx"), tt("vy"), tt("s")],
                             dim=1),
            "tau_t": torch.stack([tt("tau"), tt("t")], dim=1),
            "dr": torch.stack([tt("xn") - tt("x"), tt("yn") - tt("y"),
                               tt("vxn") - tt("vx"), tt("vyn") - tt("vy")],
                              dim=1)}


def standardise(arch, model, tz):
    """Input standardisation and output scaling, exactly ea1_train.train_one."""
    with torch.no_grad():
        if arch in ("hnn", "pinn"):
            model.x_mean.copy_(tz["z"].mean(0))
            model.x_std.copy_(tz["z"].std(0).clamp_min(1e-12))
        if arch == "pinn":
            model.y_scale.copy_(tz["dr"].abs().mean(0).clamp_min(1e-16))


def loss_on(arch, model, tz, idx, scale=None):
    if arch == "hnn":
        return T.hnn_loss(model, tz["z"][idx], tz["tgt"][idx])
    if arch == "sympnet":
        return T.sympnet_loss(model, tz["q"][idx], tz["p"][idx], tz["s"][idx],
                              tz["qn"][idx], tz["pn"][idx], scale)
    r = T.pinn_residual(model, tz["z"][idx], tz["tau_t"][idx], C.DT)
    sub = idx[:SYMP_SUBSAMPLE]
    pen = T.pinn_symplectic_penalty(model, tz["z"][sub], tz["tau_t"][sub], C.DT)
    return r + LAMBDA_SYMP * pen


def validation_loss(arch, model, tzv, scale=None, chunk=1500):
    """The architecture's own loss on the held-out draw.

    Evaluated over all 6000 held-out states in chunks, sample-weighted so that
    the result is the mean over the whole set and not the mean of chunk means.
    For the PINN the Jacobian penalty is carried by the first
    `VAL_PENALTY_STATES` states, a fixed deterministic subset, because the
    penalty at training time is likewise carried by a subsample.
    """
    n = tzv["z"].shape[0] if arch != "sympnet" else tzv["q"].shape[0]
    was_training = model.training
    model.eval()          # the HNN builds a second-order graph while training
    tot, seen = 0.0, 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        idx = torch.arange(a, b)
        if arch == "hnn":
            v = float(T.hnn_loss(model, tzv["z"][idx], tzv["tgt"][idx]).item())
        elif arch == "sympnet":
            with torch.no_grad():
                v = float(T.sympnet_loss(model, tzv["q"][idx], tzv["p"][idx],
                                         tzv["s"][idx], tzv["qn"][idx],
                                         tzv["pn"][idx], scale).item())
        else:
            with torch.no_grad():
                v = float(T.pinn_residual(model, tzv["z"][idx],
                                          tzv["tau_t"][idx], C.DT).item())
        tot += v * (b - a)
        seen += b - a
    out = tot / seen
    if arch == "pinn":
        sub = torch.arange(VAL_PENALTY_STATES)
        pen = float(T.pinn_symplectic_penalty(model, tzv["z"][sub],
                                              tzv["tau_t"][sub], C.DT).item())
        out += LAMBDA_SYMP * pen
    if was_training:
        model.train()
    return out


# ------------------------------------------------------------------ flops --
def training_flops(arch, model, stepper, steps, batch):
    """An estimate of the arithmetic spent on training, on the flop model of
    Section 9: one forward pass per sample, and a reverse pass costed at twice
    the forward, which is the usual accounting for a dense network.

    It is an estimate and is labelled as one.  It is reported because the flop
    column of Table `tab:external` counts inference only, for every learned
    scheme in the paper including ours -- a concession that runs in the
    networks' favour and that is worth pricing when the objection under test is
    about training budget.
    """
    per_sample = stepper.flops_per_step()
    if arch == "hnn":
        per_sample = per_sample / 4.0          # one field evaluation, not RK4
    return float(3.0 * per_sample * batch * steps)


# ----------------------------------------------------------------- driver --
def run_job(arch, cfg_index, rep, mult, data_key=None, verbose=True):
    """Train one configuration at one seed, one budget and one data size.

    Returns the record that goes into `runs/<job id>.json`.
    """
    import time
    cfg_name, cfg = GRID[arch][cfg_index]
    steps = int(mult * C.ADAM_STEPS)
    if data_key is None:
        s_init = seed_of(arch, cfg_index, rep, "init")
        s_shuf = seed_of(arch, cfg_index, rep, "shuffle")
    else:
        # the data sweep draws from its own two roles, and shifts the
        # repetition index by ten per point so that the three points of the
        # sweep never share a seed with one another
        r_eff = rep + 10 * DATA_SWEEP.index(data_key)
        s_init = seed_of(arch, cfg_index, r_eff, "data_init")
        s_shuf = seed_of(arch, cfg_index, r_eff, "data_shuffle")
    gen = torch.Generator().manual_seed(s_init)
    shuffler = np.random.default_rng(s_shuf)         # built once, before the loop

    d, data_rep = training_set(rep, data_key)
    dv = get_data(DATA_REP_VAL)
    tz = _tensors(arch, d)
    tzv = _tensors(arch, dv)

    model = build_model(arch, cfg, gen)
    standardise(arch, model, tz)
    scale = None
    if arch == "sympnet":
        scale = float((((tz["qn"] - tz["q"]) ** 2).sum(1)
                       + ((tz["pn"] - tz["p"]) ** 2).sum(1)).mean())

    n = d["x"].size
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    marks = sorted({max(1, int(round(f * steps))) for f in TRACE_FRACTIONS})
    recent, trace = [], []
    t0 = time.time()
    for it in range(1, steps + 1):
        b = torch.from_numpy(shuffler.integers(0, n, size=C.BATCH))
        loss = loss_on(arch, model, tz, b, scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        recent.append(float(loss.item()))
        if len(recent) > 50:
            recent.pop(0)
        if it in marks:
            st = T.to_stepper(arch, model, C.TAU_PAPER)
            sc = T.score_section7(st)
            trace.append({"step": it,
                          "train_loss_mean50": float(np.mean(recent)),
                          "val_loss": validation_loss(arch, model, tzv, scale),
                          "traj": sc["pos_err_rms"],
                          "energy": sc["energy_err_median_2nd_half"]})
            if verbose:
                print("    %-8s %-16s rep%d x%-2d  it %6d  train %.4e  "
                      "val %.4e  traj %.4e  %6.1fs"
                      % (arch, cfg_name, rep, mult, it,
                         trace[-1]["train_loss_mean50"], trace[-1]["val_loss"],
                         trace[-1]["traj"], time.time() - t0), flush=True)
    wall = time.time() - t0

    st = T.to_stepper(arch, model, C.TAU_PAPER)
    sc = T.score_section7(st)
    n_par = int(sum(p_.numel() for p_ in model.parameters()))
    rec = {
        "arch": arch, "cfg": cfg_name, "cfg_index": cfg_index,
        "hyper": {k: cfg[k] for k in sorted(cfg)},
        # A job of the grid or the ladder writes the scalar draw index it has
        # always written; only a job of the data sweep, which trains on more
        # than one draw, carries the three extra fields added with it below.
        # The schema of an already-committed file is therefore unchanged, which
        # is what lets `check_or_write` keep gating it.
        "rep": rep, "data_rep": data_rep[0], "val_data_rep": DATA_REP_VAL,
        "budget_multiplier": mult, "adam_steps": steps, "batch": C.BATCH,
        "seed_init": s_init, "seed_shuffle": s_shuf,
        "n_parameters": n_par,
        "final_train_loss_mean50": float(np.mean(recent)),
        "val_loss": trace[-1]["val_loss"],
        "traj": sc["pos_err_rms"],
        "traj_final": sc["pos_err_final"],
        "energy": sc["energy_err_median_2nd_half"],
        "energy_max": sc["energy_err_max"],
        "diverged_at_step": sc["diverged_at_step"],
        # a run that left the domain returns no flop fields, so they are taken
        # from the stepper, which knows them whatever the trajectory did
        "flops_per_step": int(sc.get("flops_per_step", st.flops_per_step())),
        "flops_run": int(sc.get("flops_run", st.flops_per_step()
                                * int(round(C.T_FINAL / C.DT)))),
        "train_flops_estimate": training_flops(arch, model, st, steps, C.BATCH),
        "omega_h": A.measure_scheme_frequency(st),
        "trace": trace,
        "train_seconds": wall,
    }
    if data_key is not None:
        rec["data_key"] = data_key
        rec["data_rep"] = list(data_rep)
        rec["n_train_states"] = int(d["x"].size)
    return rec


def job_id(arch, cfg_index, rep, mult, data_key=None):
    cfg_name = GRID[arch][cfg_index][0]
    tag = "" if data_key is None else "__d" + data_key
    return "%s__%s__x%d%s__rep%d" % (arch, cfg_name, mult, tag, rep)


def job_path(arch, cfg_index, rep, mult, data_key=None):
    return os.path.join(RUNS, job_id(arch, cfg_index, rep, mult, data_key)
                        + ".json")


def grid_jobs():
    """Every (arch, cfg, rep) of the phase-1 grid, at the base budget."""
    out = []
    for arch in ARCHS:
        for ci in range(len(GRID[arch])):
            for rep in range(N_SEEDS):
                out.append((arch, ci, rep, 1))
    return out


def ladder_jobs(selection=None):
    """The budget ladder, on the anchor configuration (see LADDER_CFG_INDEX)."""
    out = []
    for arch in ARCHS:
        ci = LADDER_CFG_INDEX
        for mult in LADDER[arch]:
            for rep in range(N_SEEDS):
                out.append((arch, ci, rep, mult))
    return out


def data_jobs(selection=None):
    """The data sweep, on the anchor configuration, at the base budget.

    All three points, `full` included, are run inside the sweep's own seed
    roles.  The phase-one job of the same configuration is the same training
    set, but under the other pair of roles, and a sweep whose middle point came
    from a different ledger than its ends would confound the seed with the
    resource being swept.  The two `full` columns are a free check on each
    other and the report prints both.

    Compute cut, recorded: the sweep is run for the SympNet alone, on the
    anchor configuration.  The SympNet is the architecture the manuscript's
    factor of 328 is quoted against, so it is the one whose data axis decides
    anything; the other two are three and four times more expensive per Adam
    step and would have bought a weaker version of the same statement.
    """
    out = []
    for arch in ("sympnet",):
        ci = LADDER_CFG_INDEX
        for key in DATA_SWEEP:
            for rep in range(N_SEEDS):
                out.append((arch, ci, rep, 1, key))
    return out


#: rough per-job cost, used only to balance the shards; it changes no number.
#: Where the base-budget job of the same configuration has already been run,
#: its own elapsed time is the estimate, which is far better than any formula:
#: these are small matrix products and the cost is set as much by the number of
#: separate operations as by the arithmetic in them.
def cost_hint(job):
    """A job is (arch, cfg_index, rep, mult) or the same with a data key."""
    import json as _json
    arch, cfg_index, _rep, mult = job[:4]
    data_key = job[4] if len(job) > 4 else None
    # the number of Adam steps is fixed across the data sweep, so a larger
    # training set costs no more to train on; only its construction differs
    p = job_path(arch, cfg_index, 0, 1)
    if os.path.exists(p):
        try:
            base = float(_json.load(open(p, encoding="utf-8"))["train_seconds"])
            return base * mult
        except Exception:
            pass
    _n, cfg = GRID[arch][cfg_index]
    if arch == "sympnet":
        base = cfg["n_modules"] * cfg["width"] / (10 * 256.0) * 63.0
    else:
        w, L = cfg["width"], cfg["n_layers"]
        base = (35.0 if arch == "hnn" else 84.0) * (w / 128.0) ** 2 * (L / 4.0)
    return base * mult


def deal(jobs, nshards):
    """Longest-processing-time scheduling: the costliest job goes to the shard
    with the least work on it so far.  Deterministic given the hints, and the
    hints enter no number."""
    jobs = sorted(jobs, key=lambda j: (-cost_hint(j), job_id(*j)))
    load = [0.0] * nshards
    bins = [[] for _ in range(nshards)]
    for j in jobs:
        k = int(np.argmin(load))
        bins[k].append(j)
        load[k] += cost_hint(j)
    return bins, load


def run_shard(jobs, shard, nshards, force=False, verbose=True):
    """Run the jobs of one shard, gate each against its committed file."""
    mine = deal(jobs, nshards)[0][shard]
    bad = 0
    for k, job in enumerate(mine):
        arch, ci, rep, mult = job[:4]
        data_key = job[4] if len(job) > 4 else None
        path = job_path(arch, ci, rep, mult, data_key)
        if verbose:
            print("[shard %d/%d] %d/%d  %s" % (shard, nshards, k + 1, len(mine),
                                               os.path.basename(path)),
                  flush=True)
        rec = run_job(arch, ci, rep, mult, data_key=data_key, verbose=verbose)
        rc = C.check_or_write(path, rec, rtol=1e-6, force=force)
        bad += rc
    print("[shard %d/%d] done, %d job(s) failed to reproduce" %
          (shard, nshards, bad), flush=True)
    return 1 if bad else 0
