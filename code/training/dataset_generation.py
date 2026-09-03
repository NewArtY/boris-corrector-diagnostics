"""
dataset_generation.py
=======================
Generates reference ("ground truth") trajectory datasets using the
classical Boris integrator at a very small time step (0.001-0.01 of a
gyroperiod), as specified in Section 2.3 of the Methods:

  "Эталонные траектории частиц формировались с использованием
   высокоточного симплектического интегратора ... с малым шагом dt."

For each field configuration used in training (uniform, dipole,
stochastic -- the three configurations named explicitly in Section 2.3),
we integrate several randomized initial conditions and save:

  - state samples (r_n, v_n, B_n, dt) as network inputs
  - next-step reference targets (r_{n+1}, v_{n+1})
  - derivative estimates (dr/dt, dv/dt) for HNN training

Splits: 80/10/10 train/val/test, matching the article.

For CPU-only demonstration-grade training we use a modest but representative
number of steps (thousands, not 10^6) while preserving the same structure
and physics fidelity as the full-scale article dataset.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import set_global_seed, SEED, T_C, DT_REF, DATA_DIR, get_logger
from models.boris import integrate_boris
from fields import UniformField, DipoleField, StochasticField, DecayingField

logger = get_logger("dataset_generation")

# Number of reference steps generated per (field, initial condition) pair.
N_STEPS_PER_TRAJ = 2000
N_TRAJECTORIES_PER_FIELD = 5
TRAIN_FIELDS = {
    "uniform": UniformField(B0=1.0),
    "dipole": DipoleField(B0=1.0, r0=5.0),
    "stochastic": StochasticField(B0=1.0, eps=0.15, omega=0.5, seed=SEED),
}


def random_initial_conditions(n, rng):
    """Generate randomized initial (r, v) pairs with |v| ~ O(1) (normalized units)."""
    r0s = rng.uniform(-1.0, 1.0, size=(n, 3))
    r0s[:, 2] = 0.0  # start near the z=0 plane (typical gyro-motion setup)
    speeds = rng.uniform(0.8, 1.2, size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    v0s = np.zeros((n, 3))
    v0s[:, 0] = speeds * np.cos(angles)
    v0s[:, 1] = speeds * np.sin(angles)
    v0s[:, 2] = rng.uniform(-0.2, 0.2, size=n)
    return r0s, v0s


def generate_field_dataset(field, n_traj=N_TRAJECTORIES_PER_FIELD,
                            n_steps=N_STEPS_PER_TRAJ, dt=DT_REF, seed=SEED):
    """Generate a dataset of (state, next_state) pairs for one field config."""
    rng = np.random.default_rng(seed)
    r0s, v0s = random_initial_conditions(n_traj, rng)

    all_r, all_v, all_B, all_dt = [], [], [], []
    all_r_next, all_v_next = [], []
    all_drdt, all_dvdt = [], []

    for k in range(n_traj):
        rs, vs, ts = integrate_boris(r0s[k], v0s[k], 0.0, dt, n_steps, field)
        Bs = np.array([field.B(rs[i], ts[i]) for i in range(len(ts) - 1)])

        all_r.append(rs[:-1])
        all_v.append(vs[:-1])
        all_B.append(Bs)
        all_dt.append(np.full(len(ts) - 1, dt))
        all_r_next.append(rs[1:])
        all_v_next.append(vs[1:])
        all_drdt.append((rs[1:] - rs[:-1]) / dt)
        all_dvdt.append((vs[1:] - vs[:-1]) / dt)

    data = {
        "r": np.concatenate(all_r, axis=0),
        "v": np.concatenate(all_v, axis=0),
        "B": np.concatenate(all_B, axis=0),
        "dt": np.concatenate(all_dt, axis=0)[:, None],
        "r_next": np.concatenate(all_r_next, axis=0),
        "v_next": np.concatenate(all_v_next, axis=0),
        "drdt": np.concatenate(all_drdt, axis=0),
        "dvdt": np.concatenate(all_dvdt, axis=0),
    }
    return data


def split_dataset(data, train_frac=0.8, val_frac=0.1, seed=SEED):
    n = data["r"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    def subset(idx_):
        return {k: v[idx_] for k, v in data.items()}

    return subset(idx_train), subset(idx_val), subset(idx_test)


def build_and_save_all(force=False):
    set_global_seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    combined_train, combined_val, combined_test = [], [], []

    for name, field in TRAIN_FIELDS.items():
        out_path = os.path.join(DATA_DIR, f"{name}.npz")
        if os.path.exists(out_path) and not force:
            logger.info(f"Dataset for '{name}' already exists, skipping.")
            continue
        logger.info(f"Generating reference Boris dataset for field='{name}' "
                    f"(dt={DT_REF:.5g}, n_traj={N_TRAJECTORIES_PER_FIELD}, "
                    f"n_steps={N_STEPS_PER_TRAJ})")
        data = generate_field_dataset(field)
        train, val, test = split_dataset(data)
        np.savez(out_path, **{f"train_{k}": v for k, v in train.items()},
                 **{f"val_{k}": v for k, v in val.items()},
                 **{f"test_{k}": v for k, v in test.items()})
        logger.info(f"Saved {out_path}: total samples = {data['r'].shape[0]} "
                    f"(train={train['r'].shape[0]}, val={val['r'].shape[0]}, "
                    f"test={test['r'].shape[0]})")


def load_combined_dataset(split="train"):
    """Load and concatenate the dataset across all training fields for a given split."""
    all_data = {k: [] for k in ["r", "v", "B", "dt", "r_next", "v_next", "drdt", "dvdt"]}
    for name in TRAIN_FIELDS:
        path = os.path.join(DATA_DIR, f"{name}.npz")
        npz = np.load(path)
        for k in all_data:
            all_data[k].append(npz[f"{split}_{k}"])
    return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}


def generate_corrector_multiscale_dataset(field, n_traj=6, dt_coarse_list=None,
                                           substeps_per_coarse=32, seed=SEED):
    """Generate a dataset specifically for the Boris+Corrector model that
    spans MULTIPLE coarse time steps (not just the small reference dt).

    For each coarse dt in dt_coarse_list, we advance a fine-step reference
    trajectory (dt_fine = dt_coarse/substeps_per_coarse) to obtain the true
    next state after one coarse step, and pair it with (r_n, v_n, B_n,
    dt_coarse) as the corrector's training input. This teaches the
    correction network how the required correction scales with step size,
    which is essential for it to generalize to the larger steps used in the
    B4 decaying-field stress test.
    """
    if dt_coarse_list is None:
        dt_coarse_list = [0.02, 0.04, 0.08, 0.12, 0.16]
    substeps_per_coarse = max(substeps_per_coarse, int(64 * max(dt_coarse_list) / 0.16))

    rng = np.random.default_rng(seed)
    r0s, v0s = random_initial_conditions(n_traj, rng)

    all_r, all_v, all_B, all_dt = [], [], [], []
    all_r_next, all_v_next = [], []

    for dt_coarse in dt_coarse_list:
        dt_fine = dt_coarse / substeps_per_coarse
        n_coarse_steps = 40
        for k in range(n_traj):
            r, v, t = r0s[k].copy(), v0s[k].copy(), 0.0
            for _ in range(n_coarse_steps):
                B_here = field.B(r, t)
                r_start, v_start, t_start = r.copy(), v.copy(), t
                rs_fine, vs_fine, ts_fine = integrate_boris(
                    r_start, v_start, t_start, dt_fine, substeps_per_coarse, field)
                r_next_true, v_next_true = rs_fine[-1], vs_fine[-1]

                all_r.append(r_start); all_v.append(v_start); all_B.append(B_here)
                all_dt.append(dt_coarse)
                all_r_next.append(r_next_true); all_v_next.append(v_next_true)

                r, v, t = r_next_true, v_next_true, ts_fine[-1]

    data = {
        "r": np.array(all_r), "v": np.array(all_v), "B": np.array(all_B),
        "dt": np.array(all_dt)[:, None],
        "r_next": np.array(all_r_next), "v_next": np.array(all_v_next),
    }
    data["drdt"] = (data["r_next"] - data["r"]) / data["dt"]
    data["dvdt"] = (data["v_next"] - data["v"]) / data["dt"]
    return data


CORRECTOR_EXTRA_FIELDS = {
    "decaying": DecayingField(B0=1.0, tau=150.0),
}


def build_and_save_corrector_dataset(force=False):
    """Build and save the multi-scale dataset used to train Boris+Corrector.

    Includes the three main training fields PLUS the decaying field (B4),
    so the corrector also sees time-varying-B / induced-E samples during
    training and can generalize correctly to the B4 stress test used in
    Figure 4."""
    out_path = os.path.join(DATA_DIR, "corrector_multiscale.npz")
    if os.path.exists(out_path) and not force:
        logger.info("Multi-scale corrector dataset already exists, skipping.")
        return
    logger.info("Generating multi-scale Boris+Corrector training dataset "
                "(dt in [0.02, 0.16] for uniform/dipole/stochastic, "
                "dt in [0.1, 2.0] for decaying B4) ...")
    all_data = {k: [] for k in ["r", "v", "B", "dt", "r_next", "v_next", "drdt", "dvdt"]}
    for name, field in TRAIN_FIELDS.items():
        data = generate_corrector_multiscale_dataset(field, seed=SEED)
        for k in all_data:
            all_data[k].append(data[k])
    # decaying field: wider dt range matching the Figure-4 stress-test regime
    for name, field in CORRECTOR_EXTRA_FIELDS.items():
        data = generate_corrector_multiscale_dataset(
            field, seed=SEED, dt_coarse_list=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
            substeps_per_coarse=64)
        for k in all_data:
            all_data[k].append(data[k])
    combined = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    train, val, test = split_dataset(combined)
    np.savez(out_path, **{f"train_{k}": v for k, v in train.items()},
             **{f"val_{k}": v for k, v in val.items()},
             **{f"test_{k}": v for k, v in test.items()})
    logger.info(f"Saved {out_path}: total samples = {combined['r'].shape[0]}")


def load_corrector_dataset(split="train"):
    path = os.path.join(DATA_DIR, "corrector_multiscale.npz")
    npz = np.load(path)
    return {k: npz[f"{split}_{k}"] for k in ["r", "v", "B", "dt", "r_next", "v_next", "drdt", "dvdt"]}


if __name__ == "__main__":
    build_and_save_all()
    build_and_save_corrector_dataset()
