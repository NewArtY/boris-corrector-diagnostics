"""
common.py
=========
Shared physical constants, numerical parameters and utility helpers used
across this repository.

Physical setup: motion of a single particle of mass m and charge q in
prescribed electric (E) and magnetic (B) fields, evolved with

    dv/dt = (q/m) [E + v x B],      dr/dt = v

The reference ("ground truth") integrator is the classical Boris algorithm
run at a very small step (0.001-0.01 of a gyroperiod), and the learned
corrector is trained against Boris-generated trajectories.

A fixed global random seed (42) is used everywhere for reproducibility; see
the reproducibility section of README.md for what that does and does not buy.

Historical note: the wording of this docstring, and the four purely learned
integrators under models/, come from the first version of this work, titled
"Neural Integrators for Charged-Particle Motion", which was withdrawn. The
present paper uses the Boris pusher, the learned defect corrector and the
classical schemes of experiments/classical/; the rest is kept because the
legacy verification scripts read its output. README.md says which is which.
"""

import os
import random
import json
import logging
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# --------------------------------------------------------------------------
# Global reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED):
    """Fix all relevant RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # some CPU ops lack det. kernels
    os.environ["PYTHONHASHSEED"] = str(seed)


set_global_seed(SEED)

# --------------------------------------------------------------------------
# Physical constants (SI units), electron in a laboratory-scale magnetic field
# --------------------------------------------------------------------------
Q_E = -1.602176634e-19          # electron charge, C
M_E = 9.1093837015e-31          # electron mass, kg
B0 = 1.0                        # reference field magnitude, T (normalized configs use 1.0)
V0 = 1.0e6                      # reference speed scale, m/s (Table 3, "initial speed ~1e6 m/s")

OMEGA_C = abs(Q_E) * B0 / M_E    # cyclotron (gyro) frequency, rad/s (SI, informational)
T_C_SI = 2.0 * np.pi / OMEGA_C     # gyroperiod, s (SI, informational)
R_C = M_E * V0 / (abs(Q_E) * B0)  # cyclotron (Larmor) radius, m  (article: rc ~ 5.7 um)

# All simulation code in this repository works in NORMALIZED units, where
# q=-1, m=1, B0=1 so that omega_c = |q|*B0/m = 1 and the gyroperiod is
# simply T_C = 2*pi. This matches the default parameters used throughout
# fields/*.py and models/*.py (B0=1.0, q=-1.0, m=1.0). Physical (SI) scales
# are provided above (T_C_SI, R_C) purely for reference/context.
T_C = 2.0 * np.pi

# Reference integration step for "ground truth" Boris trajectories:
# 0.001 - 0.01 of one gyroperiod (Section 2.3), in normalized units
DT_REF_FRACTION = 0.005
DT_REF = DT_REF_FRACTION * T_C

# Coarser steps used to stress-test integrators (spanning > 3 orders of
# magnitude, as stated in the Methods: "от dt_min до dt_max")
DT_COARSE_FRACTIONS = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
DT_COARSE = DT_COARSE_FRACTIONS * T_C

# Number of gyroperiods for long-term stability studies (kept modest here
# for a demonstration-grade, CPU-only run; article uses 1000 gyroperiods)
N_GYROPERIODS_LONG = 60
N_GYROPERIODS_SHORT = 20

# --------------------------------------------------------------------------
# Checkpoints / output directories
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
FIGURE_DIR = os.path.join(ROOT_DIR, "output_figures")
DATA_DIR = os.path.join(ROOT_DIR, "training", "data")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Colorblind-friendly, publication-quality palette (Wong 2011 / Okabe-Ito)
# --------------------------------------------------------------------------
COLORS = {
    "boris": "#000000",            # black
    "boris_corrector": "#0072B2",  # blue
    "pinn_symplectic": "#D55E00",  # vermillion
    "hnn": "#009E73",              # bluish green
    "sympnet": "#CC79A7",          # reddish purple
    "reference": "#56B4E9",        # sky blue
    "physical": "#E69F00",         # orange
    "noise_floor": "#999999",      # grey
}

LABELS = {
    "boris": "Boris",
    "boris_corrector": "Boris + Corrector",
    "pinn_symplectic": "PINN-symplectic",
    "hnn": "HNN",
    "sympnet": "SympNet",
    "reference": "Reference (RK4, small dt)",
}


# --------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_run_params(logger: logging.Logger, params: dict):
    """Log a dictionary of run parameters in a readable, reproducible way."""
    logger.info("Run parameters: %s", json.dumps(params, default=str, indent=2))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
