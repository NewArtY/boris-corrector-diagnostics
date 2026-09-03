"""
seed_sweep_baselines.py — Задача И1.3-2: устойчивость вывода о бейзлайнах.

Переобучает все четыре модели train.py с разными seed и пересчитывает
results_summary (B1-B4) для каждого. Ничего вне experiments/stats/ не пишет:
модели инжектируются прямо в integrator_runner._MODEL_CACHE, поставленный
checkpoints/*.pt не трогается.

Варьируется ТОЛЬКО случайность обучения (инициализация весов, перемешивание
батчей). Датасеты фиксированы (training/data/*.npz), поскольку И0 уже
установила, что данные не являются причиной расхождения.
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

from common import get_logger, FIGURE_DIR
import training.train as T
from training.dataset_generation import load_combined_dataset, load_corrector_dataset
import diagnostics.integrator_runner as IR
from diagnostics.integrator_runner import integrate, ALL_INTEGRATORS
from diagnostics.energy_drift import mean_abs_energy_drift, rms_energy_error
from fields import RadialField, WaveField, TiltedField, DecayingField

logger = get_logger("seed_sweep_baselines")
OUT = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

# те же параметры оценки, что в generate_results_summary.py
DT = 0.05
N_GYRO = 15
N_STEPS = int(N_GYRO * (2 * np.pi) / DT)
R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
EPS = 1e-16
FIELD_SPECS = [("B1", RadialField()), ("B2", WaveField()),
               ("B3", TiltedField()), ("B4", DecayingField(tau=40.0))]


def train_all(seed, train_data, corrector_data):
    """Обучает четыре модели с заданным seed. Возвращает dict модель->объект."""
    T.SEED = seed          # train_* читают SEED из глобалей модуля при вызове
    models, hist = {}, {}
    for key, fn, data in [
        ("pinn_symplectic", T.train_pinn, train_data),
        ("hnn", T.train_hnn, train_data),
        ("sympnet", T.train_sympnet, train_data),
        ("boris_corrector", T.train_boris_corrector, corrector_data),
    ]:
        t0 = time.time()
        m, h = fn(data, None)
        m.eval()
        models[key] = m
        hist[key] = {"final_loss": float(h[-1]), "min_loss": float(min(h)),
                     "last50_mean": float(np.mean(h[-50:])),
                     "last50_std": float(np.std(h[-50:])),
                     "wall_s": round(time.time() - t0, 1)}
        torch.save(m.state_dict(), os.path.join(CKPT, f"{key}_seed{seed}.pt"))
    return models, hist


def summarize(models):
    """Аналог generate_results_summary.main(), но на инжектированных моделях."""
    IR._MODEL_CACHE.clear()
    IR._MODEL_CACHE.update(models)
    out = {}
    for name in ALL_INTEGRATORS:
        out[name] = {}
        for tag, field in FIELD_SPECS:
            rs, vs, ts = integrate(name, field, R0, V0, 0.0, DT, N_STEPS)
            out[name][tag] = {
                "mean_abs_dE_E0": float(np.clip(mean_abs_energy_drift(vs), EPS, 1e12)),
                "rms_energy_error": float(np.clip(rms_energy_error(vs), EPS, 1e12)),
            }
    IR._MODEL_CACHE.clear()
    return out


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [42, 1, 7, 123, 2026]
    train_data = load_combined_dataset("train")
    corrector_data = load_corrector_dataset("train")
    logger.info("train samples=%d  corrector samples=%d",
                train_data["r"].shape[0], corrector_data["r"].shape[0])

    all_res = {}
    for s in seeds:
        t0 = time.time()
        models, hist = train_all(s, train_data, corrector_data)
        summ = summarize(models)
        all_res[str(s)] = {"summary": summ, "training": hist,
                           "wall_s": round(time.time() - t0, 1)}
        logger.info("seed %d done in %.0f s", s, time.time() - t0)
        with open(os.path.join(OUT, "seed_sweep_baselines.json"), "w") as f:
            json.dump(all_res, f, indent=2)

    # сводка распределений
    dist = {}
    for name in ALL_INTEGRATORS:
        dist[name] = {}
        for tag, _ in FIELD_SPECS:
            for metric in ("mean_abs_dE_E0", "rms_energy_error"):
                vals = np.array([all_res[str(s)]["summary"][name][tag][metric]
                                 for s in seeds])
                lv = np.log10(np.clip(vals, 1e-300, None))
                dist[name][f"{tag}.{metric}"] = {
                    "median": float(np.median(vals)),
                    "min": float(vals.min()), "max": float(vals.max()),
                    "spread_orders": float(lv.max() - lv.min()),
                    "values": [float(x) for x in vals],
                }
    with open(os.path.join(OUT, "baselines_distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)

    print("\n=== разброс по seed, порядков величины ===")
    for name in ALL_INTEGRATORS:
        for tag, _ in FIELD_SPECS:
            d = dist[name][f"{tag}.rms_energy_error"]
            print(f"{name:<18}{tag}  median={d['median']:.3e}  "
                  f"min={d['min']:.3e}  max={d['max']:.3e}  "
                  f"разброс={d['spread_orders']:.2f} порядка")


if __name__ == "__main__":
    main()
