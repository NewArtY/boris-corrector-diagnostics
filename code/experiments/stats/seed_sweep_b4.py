"""
seed_sweep_b4.py — Задача И1.3-1: устойчивость центрального результата по seed.

Переобучает корректор B4 с несколькими seed и для каждого пересчитывает
диагностику eval_corrector. Ничего вне experiments/stats/ не пишет:
поставленный checkpoints/boris_corrector_b4.pt не трогается.

Полная перезасевка: и выборка данных (rng), и инициализация сети, и
перемешивание батчей — всё от одного seed. Это честный ответ на вопрос
"что будет, если прогнать пайплайн заново с другим seed".
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

torch.set_default_dtype(torch.float64)

from common import set_global_seed, get_logger
from fields import DecayingField
from models.boris import boris_step, integrate_boris
from training.train_corrector_b4 import (
    DefectNet, DT_WORK, DT_FINE, T_FINAL, TAU_MAIN, TAU_TRAIN,
    N_TRAJ_PER_TAU, EPOCHS, BATCH, LR,
    LAMBDA_SMALL, LAMBDA_ORTHO, LAMBDA_ENERGY,
)

logger = get_logger("seed_sweep_b4")
OUT = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


def build_dataset(seed):
    """Копия training.train_corrector_b4.build_dataset с параметром seed."""
    X, Y = [], []
    rng = np.random.default_rng(seed)
    for tau in TAU_TRAIN:
        field = DecayingField(B0=1.0, tau=tau)
        for _ in range(N_TRAJ_PER_TAU):
            rho = 0.7 + 0.6 * rng.random()
            phase = 2 * np.pi * rng.random()
            vpar = 0.3 * (rng.random() - 0.5)
            r0 = np.array([rho * np.cos(phase), rho * np.sin(phase), 0.0])
            v0 = np.array([-np.sin(phase), np.cos(phase), vpar])
            n_coarse = int(round(T_FINAL / DT_WORK))
            r, v, t = r0.copy(), v0.copy(), 0.0
            for _ in range(n_coarse):
                r_b, v_b = boris_step(r, v, t, DT_WORK, field)
                rs_f, vs_f, _ = integrate_boris(r, v, t, DT_FINE, 150, field)
                r_ref, v_ref = rs_f[-1], vs_f[-1]
                B = np.atleast_1d(field.B(r, t)).ravel()
                E = np.atleast_1d(field.E(r, t)).ravel()
                X.append(np.concatenate([r, v, B, E, [DT_WORK]]))
                Y.append(np.concatenate([r_ref - r_b, v_ref - v_b]))
                r, v = r_ref, v_ref
                t += DT_WORK
    return np.asarray(X, float), np.asarray(Y, float)


def train_one(seed):
    t0 = time.time()
    set_global_seed(seed)
    X, Y = build_dataset(seed)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)

    model = DefectNet(n_in=X.shape[1])
    model.x_mean.copy_(Xt.mean(0))
    model.x_std.copy_(Xt.std(0).clamp_min(1e-12))
    model.y_scale.copy_(Yt.abs().mean(0).clamp_min(1e-16))

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = Xt.shape[0]
    idx_all = torch.randperm(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]
    v_dir = Xt[:, 3:6] / Xt[:, 3:6].norm(dim=1, keepdim=True).clamp_min(1e-12)

    for ep in range(EPOCHS):
        model.train()
        perm = tr_idx[torch.randperm(tr_idx.numel())]
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            pred = model(Xt[b])
            data = ((pred - Yt[b]) ** 2).mean()
            small = (pred ** 2).mean()
            ortho = ((pred[:, 3:] * v_dir[b]).sum(1) ** 2).mean()
            ener = ((pred[:, 3:] * Xt[b, 3:6]).sum(1) ** 2).mean()
            loss = (data / model.y_scale.pow(2).mean()
                    + LAMBDA_SMALL * small + LAMBDA_ORTHO * ortho
                    + LAMBDA_ENERGY * ener)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pv = model(Xt[val_idx])
        val_rel = ((pv - Yt[val_idx]).norm(dim=1)
                   / Yt[val_idx].norm(dim=1).clamp_min(1e-30)).mean().item()
    torch.save(model.state_dict(), os.path.join(CKPT, f"corrector_b4_seed{seed}.pt"))
    logger.info("seed %d trained in %.1f s, val_rel=%.4e", seed, time.time() - t0, val_rel)
    return model, val_rel, time.time() - t0


def integrate_corrected(field, r0, v0, dt, n_steps, model, project=True):
    rs = np.zeros((n_steps + 1, 3)); vs = np.zeros((n_steps + 1, 3))
    ts = np.zeros(n_steps + 1)
    r, v, t = np.array(r0, float), np.array(v0, float), 0.0
    rs[0], vs[0] = r, v
    with torch.no_grad():
        for i in range(1, n_steps + 1):
            r_b, v_b = boris_step(r, v, t, dt, field)
            B = np.atleast_1d(field.B(r, t)).ravel()
            E = np.atleast_1d(field.E(r, t)).ravel()
            x = torch.tensor(np.concatenate([r, v, B, E, [dt]]))[None, :]
            d = model(x).numpy()[0]
            dr, dv = d[:3], d[3:]
            if project:
                nb = np.linalg.norm(v_b)
                vh = v_b / max(nb, 1e-300)
                dv = dv - np.dot(dv, vh) * vh
                v_new = v_b + dv
                v_new *= nb / max(np.linalg.norm(v_new), 1e-300)
            else:
                v_new = v_b + dv
            r, v = r_b + dr, v_new
            t += dt
            rs[i], vs[i], ts[i] = r, v, t
    return rs, vs, ts


def evaluate(model, r0=None, v0=None, tau=TAU_MAIN, ref=None):
    """Диагностика как в diagnostics/eval_corrector.py."""
    field = DecayingField(B0=1.0, tau=tau)
    r0 = np.array([1.0, 0.0, 0.0]) if r0 is None else np.asarray(r0, float)
    v0 = np.array([0.0, 1.0, 0.0]) if v0 is None else np.asarray(v0, float)

    if ref is None:
        n_fine = int(round(T_FINAL / DT_FINE))
        rs_r, vs_r, ts_r = integrate_boris(r0, v0, 0.0, DT_FINE, n_fine, field)
    else:
        rs_r, vs_r, ts_r = ref
    E_ref = 0.5 * np.sum(vs_r ** 2, axis=1); E0 = E_ref[0]

    n_work = int(round(T_FINAL / DT_WORK))
    runs = {"boris": integrate_boris(r0, v0, 0.0, DT_WORK, n_work, field),
            "corrector_raw": integrate_corrected(field, r0, v0, DT_WORK, n_work, model, False),
            "corrector_projected": integrate_corrected(field, r0, v0, DT_WORK, n_work, model, True)}

    out = {}
    half = n_work // 2
    for k, (rs, vs, ts) in runs.items():
        Ei = np.interp(ts, ts_r, E_ref)
        E = 0.5 * np.sum(vs ** 2, axis=1)
        e_err = np.abs(E - Ei) / E0
        r_ref_i = np.vstack([np.interp(ts, ts_r, rs_r[:, j]) for j in range(3)]).T
        pos_err = np.linalg.norm(rs - r_ref_i, axis=1)
        out[k] = {"energy_err_median_2nd_half": float(np.median(e_err[half:])),
                  "energy_err_max": float(e_err.max()),
                  "pos_err_final": float(pos_err[-1]),
                  "pos_err_rms": float(np.sqrt(np.mean(pos_err ** 2)))}
    ts_b = runs["boris"][2]
    phys = float(np.median(np.abs((np.interp(ts_b, ts_r, E_ref) - E0) / E0)[half:]))
    out["physical_signal_median"] = phys
    out["traj_gain_projected"] = (out["boris"]["pos_err_rms"]
                                  / out["corrector_projected"]["pos_err_rms"])
    out["energy_separation_hybrid"] = phys / out["corrector_projected"]["energy_err_median_2nd_half"]
    out["energy_separation_boris"] = phys / out["boris"]["energy_err_median_2nd_half"]
    out["energy_separation_raw"] = phys / out["corrector_raw"]["energy_err_median_2nd_half"]
    return out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 1, 7, 123, 2026]
    all_res = {}
    for s in seeds:
        model, val_rel, wall = train_one(s)
        res = evaluate(model)
        res["val_rel_defect_error"] = val_rel
        res["train_wall_s"] = wall
        all_res[str(s)] = res
        logger.info("seed %d: gain=%.1fx  E_sep_hybrid=%.1f  pos_rms=%.3e",
                    s, res["traj_gain_projected"], res["energy_separation_hybrid"],
                    res["corrector_projected"]["pos_err_rms"])
        with open(os.path.join(OUT, "seed_sweep_b4.json"), "w") as f:
            json.dump(all_res, f, indent=2)
    print(json.dumps(all_res, indent=2))
