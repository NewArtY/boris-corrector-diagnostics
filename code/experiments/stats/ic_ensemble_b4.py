"""
ic_ensemble_b4.py — Задача И1.3-3: ансамбль начальных условий.

Фиксированный корректор (поставленный boris_corrector_b4.pt, seed 42),
N начальных условий на поле B4. Отвечает на вопрос: выигрыш 118x —
типичное значение по фазовому пространству или удачная точка?

Начальные условия берутся из того же распределения, что и обучающая
выборка train_corrector_b4.build_dataset:
    rho   = 0.7 + 0.6*U      -> [0.7, 1.3]
    phase = 2*pi*U
    vpar  = 0.3*(U - 0.5)    -> [-0.15, 0.15]
Опорная точка статьи (rho=1, phase=0, vpar=0) лежит внутри этого диапазона.
"""
import os
import sys
import json
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, HERE)

torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR, get_logger
from training.train_corrector_b4 import DefectNet
from seed_sweep_b4 import evaluate

logger = get_logger("ic_ensemble")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

N_IC = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ENSEMBLE_SEED = 20260830


def main():
    model = DefectNet(n_in=13)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt"), map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(ENSEMBLE_SEED)
    rows = []
    t0 = time.time()

    # нулевая точка — та, что в статье
    specs = [{"rho": 1.0, "phase": 0.0, "vpar": 0.0, "is_paper_point": True}]
    for _ in range(N_IC):
        specs.append({"rho": 0.7 + 0.6 * rng.random(),
                      "phase": 2 * np.pi * rng.random(),
                      "vpar": 0.3 * (rng.random() - 0.5),
                      "is_paper_point": False})

    for i, s in enumerate(specs):
        rho, phase, vpar = s["rho"], s["phase"], s["vpar"]
        r0 = np.array([rho * np.cos(phase), rho * np.sin(phase), 0.0])
        v0 = np.array([-np.sin(phase), np.cos(phase), vpar])
        res = evaluate(model, r0=r0, v0=v0)
        rows.append({**s,
                     "traj_gain": res["traj_gain_projected"],
                     "energy_sep_hybrid": res["energy_separation_hybrid"],
                     "energy_sep_boris": res["energy_separation_boris"],
                     "energy_sep_raw": res["energy_separation_raw"],
                     "pos_rms_boris": res["boris"]["pos_err_rms"],
                     "pos_rms_hybrid": res["corrector_projected"]["pos_err_rms"],
                     "pos_rms_raw": res["corrector_raw"]["pos_err_rms"],
                     "e_err_boris": res["boris"]["energy_err_median_2nd_half"],
                     "e_err_hybrid": res["corrector_projected"]["energy_err_median_2nd_half"],
                     "e_err_raw": res["corrector_raw"]["energy_err_median_2nd_half"],
                     "physical_signal": res["physical_signal_median"]})
        if i % 20 == 0:
            logger.info("%d/%d  gain=%.1f  (%.0f s)", i, len(specs),
                        rows[-1]["traj_gain"], time.time() - t0)
            with open(os.path.join(OUT, "ic_ensemble_b4.json"), "w") as f:
                json.dump(rows, f, indent=2)

    with open(os.path.join(OUT, "ic_ensemble_b4.json"), "w") as f:
        json.dump(rows, f, indent=2)

    ens = [r for r in rows if not r["is_paper_point"]]
    g = np.array([r["traj_gain"] for r in ens])
    es = np.array([r["energy_sep_hybrid"] for r in ens])
    summary = {
        "n_ic": len(ens),
        "paper_point": {k: rows[0][k] for k in
                        ("traj_gain", "energy_sep_hybrid", "energy_sep_boris")},
        "traj_gain": {"mean": float(g.mean()), "std": float(g.std(ddof=1)),
                      "median": float(np.median(g)),
                      "q25": float(np.percentile(g, 25)),
                      "q75": float(np.percentile(g, 75)),
                      "min": float(g.min()), "max": float(g.max()),
                      "frac_hybrid_worse_than_boris": float((g < 1).mean())},
        "energy_sep_hybrid": {"mean": float(es.mean()), "std": float(es.std(ddof=1)),
                              "median": float(np.median(es)),
                              "q25": float(np.percentile(es, 25)),
                              "q75": float(np.percentile(es, 75)),
                              "min": float(es.min()), "max": float(es.max()),
                              "frac_below_signal": float((es > 1).mean())},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "ic_ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
