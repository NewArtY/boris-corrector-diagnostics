"""
seed_lock_probe.py — довесок к И1.3-2.

Первый прогон seed_sweep_baselines дал разброс РОВНО НОЛЬ по всем моделям и
всем полям. Причина: в models/{hnn,sympnet,pinn_symplectic,boris_corrector}.py
функция build_model() вызывает torch.manual_seed(SEED) с ЖЁСТКО ЗАШИТЫМ
модульным SEED = 42. Она срабатывает после set_global_seed(...) и затирает
любой внешний засев, а поскольку глобальный ГСЧ сбрасывается непосредственно
перед созданием DataLoader, идентичным оказывается и перемешивание батчей.

Скрипт проверяет две вещи на модели HNN (самой дешёвой и самой проблемной):
  A. детерминизм при фиксированном seed: два обучения подряд в одном процессе
  B. чувствительность к seed при КОРРЕКТНОМ засеве (патчим models.hnn.SEED)
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

from common import get_logger
import models.hnn as MH
import training.train as T
from training.dataset_generation import load_combined_dataset
import diagnostics.integrator_runner as IR
from diagnostics.integrator_runner import integrate
from diagnostics.energy_drift import rms_energy_error
from fields import RadialField, WaveField, TiltedField, DecayingField

logger = get_logger("seed_lock_probe")
OUT = os.path.join(HERE, "results")
DT, N_STEPS = 0.05, int(15 * (2 * np.pi) / 0.05)
R0, V0 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
FIELDS = [("B1", RadialField()), ("B2", WaveField()),
          ("B3", TiltedField()), ("B4", DecayingField(tau=40.0))]


def flat_weights(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).numpy()


def eval_hnn(model):
    IR._MODEL_CACHE.clear()
    IR._MODEL_CACHE["hnn"] = model
    out = {}
    for tag, f in FIELDS:
        _, vs, _ = integrate("hnn", f, R0, V0, 0.0, DT, N_STEPS)
        out[tag] = float(np.clip(rms_energy_error(vs), 1e-16, 1e12))
    IR._MODEL_CACHE.clear()
    return out


def train_hnn_with_seed(train_data, seed):
    MH.SEED = seed        # то, что реально управляет инициализацией
    T.SEED = seed
    m, hist = T.train_hnn(train_data, None)
    m.eval()
    return m, float(hist[-1])


def main():
    train_data = load_combined_dataset("train")
    res = {}

    # A. детерминизм при одном и том же seed
    logger.info("=== A: детерминизм при seed=42 ===")
    wA, eA, lA = [], [], []
    for rep in range(2):
        m, loss = train_hnn_with_seed(train_data, 42)
        wA.append(flat_weights(m)); eA.append(eval_hnn(m)); lA.append(loss)
    dw = float(np.max(np.abs(wA[0] - wA[1])))
    res["determinism_same_seed"] = {
        "max_abs_weight_diff": dw,
        "identical": bool(dw == 0.0),
        "final_losses": lA,
        "rms_energy_error": eA,
    }
    logger.info("max|dw| между двумя прогонами seed=42: %.3e", dw)

    # B. чувствительность к seed при корректном засеве
    logger.info("=== B: разные seed с корректным засевом ===")
    per_seed = {}
    for s in [42, 1, 7]:
        m, loss = train_hnn_with_seed(train_data, s)
        per_seed[str(s)] = {"final_loss": loss, "rms_energy_error": eval_hnn(m),
                            "w_norm": float(np.linalg.norm(flat_weights(m)))}
        logger.info("seed %d: loss=%.3e  B4=%.3e", s, loss,
                    per_seed[str(s)]["rms_energy_error"]["B4"])
    res["seed_sensitivity"] = per_seed
    for tag, _ in FIELDS:
        v = np.array([per_seed[s]["rms_energy_error"][tag] for s in per_seed])
        lv = np.log10(np.clip(v, 1e-300, None))
        res.setdefault("spread_orders", {})[tag] = float(lv.max() - lv.min())

    with open(os.path.join(OUT, "seed_lock_probe.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
