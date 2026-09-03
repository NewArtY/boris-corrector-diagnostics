"""
run_verify.py -- воспроизведение опубликованных чисел, итерация И0.

Запускает eval_corrector.main() и fig6_decaying_field_case.main() из
существующих чекпойнтов, перенаправляя ВСЕ выходные файлы в каталог этого
скрипта, чтобы поставленные output_figures/*.json остались нетронутыми.

Существующий код не редактируется: перенаправление сделано подменой
атрибута FIGURE_DIR в уже импортированных модулях.

    python verify_i0/run_verify.py --tag run1
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def redirect_figure_dir(target):
    """Подменить FIGURE_DIR во всех модулях, которые его связали при импорте."""
    os.makedirs(target, exist_ok=True)
    import common
    common.FIGURE_DIR = target
    import figures.plot_style as plot_style
    import diagnostics.eval_corrector as evalc
    import figures.fig6_decaying_field_case as fig6
    for mod in (plot_style, evalc, fig6):
        if hasattr(mod, "FIGURE_DIR"):
            mod.FIGURE_DIR = target
    return evalc, fig6


def timing_probe(n_steps=1000):
    """Секунды на n_steps шагов: чистый Boris против гибрида с сетью."""
    import numpy as np
    import torch
    from fields import DecayingField
    from models.boris import boris_step
    from diagnostics.eval_corrector import load_corrector
    from training.train_corrector_b4 import DT_WORK, TAU_MAIN

    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])

    # --- Boris ---
    r, v, t = r0.copy(), v0.copy(), 0.0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        r, v = boris_step(r, v, t, DT_WORK, field)
        t += DT_WORK
    t_boris = time.perf_counter() - t0

    # --- Boris + сеть с жёсткой проекцией ---
    model = load_corrector()
    r, v, t = r0.copy(), v0.copy(), 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            r_b, v_b = boris_step(r, v, t, DT_WORK, field)
            B = np.atleast_1d(field.B(r, t)).ravel()
            E = np.atleast_1d(field.E(r, t)).ravel()
            x = torch.tensor(np.concatenate([r, v, B, E, [DT_WORK]]))[None, :]
            d = model(x).numpy()[0]
            dr, dv = d[:3], d[3:]
            nb = np.linalg.norm(v_b)
            vh = v_b / max(nb, 1e-300)
            dv = dv - float(dv @ vh) * vh
            v_new = v_b + dv
            v_new *= nb / max(np.linalg.norm(v_new), 1e-300)
            r, v = r_b + dr, v_new
            t += DT_WORK
    t_hybrid = time.perf_counter() - t0

    return {
        "n_steps": n_steps,
        "seconds_boris": t_boris,
        "seconds_hybrid": t_hybrid,
        "hybrid_over_boris": t_hybrid / t_boris if t_boris > 0 else None,
        "us_per_step_boris": 1e6 * t_boris / n_steps,
        "us_per_step_hybrid": 1e6 * t_hybrid / n_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1", help="подкаталог для этого прогона")
    ap.add_argument("--timing", action="store_true", help="дополнительно замерить скорость шага")
    args = ap.parse_args()

    target = os.path.join(HERE, args.tag)
    evalc, fig6 = redirect_figure_dir(target)

    print(f"[verify] вывод перенаправлен в {target}")

    t0 = time.perf_counter()
    evalc.main()
    t_eval = time.perf_counter() - t0
    print(f"[verify] eval_corrector: {t_eval:.1f} c")

    t0 = time.perf_counter()
    fig6.main()
    t_fig = time.perf_counter() - t0
    print(f"[verify] fig6_decaying_field_case: {t_fig:.1f} c")

    meta = {"tag": args.tag, "seconds_eval_corrector": t_eval, "seconds_fig6": t_fig}
    if args.timing:
        meta["timing_probe"] = timing_probe()
        print("[verify] timing:", json.dumps(meta["timing_probe"], indent=1))
    with open(os.path.join(target, "_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
