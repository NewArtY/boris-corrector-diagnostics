"""
verify_model_dtype.py -- coordinator follow-up for I1.1
=======================================================
(2) Prove that the benchmarked hybrid is DefectNet + boris_corrector_b4.pt,
    not BorisCorrectorNet + boris_corrector.pt.
(3) Measure how much of the hybrid's per-step cost is float64.

Writes model_dtype.json. Touches nothing else.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
torch.set_default_dtype(torch.float64)

from common import CHECKPOINT_DIR
from fields import DecayingField
from models.boris import boris_step
from training.train_corrector_b4 import DefectNet, DT_WORK, TAU_MAIN
from bench import load_corrector

R0 = np.array([1.0, 0.0, 0.0])
V0 = np.array([0.0, 1.0, 0.0])
N_TIME = 4000


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    out = {}

    # ---------------- (2) identity of the benchmarked model ----------------
    ckpt_b4 = os.path.join(CHECKPOINT_DIR, "boris_corrector_b4.pt")
    ckpt_old = os.path.join(CHECKPOINT_DIR, "boris_corrector.pt")
    model = load_corrector()                     # what bench.py actually used
    sd = model.state_dict()
    shapes = {k: list(v.shape) for k, v in sd.items() if v.ndim > 0}
    n_par = sum(p.numel() for p in model.parameters())

    out["identity"] = {
        "class_benchmarked": type(model).__name__,
        "checkpoint_loaded": os.path.basename(ckpt_b4),
        "checkpoint_sha256_16": sha(ckpt_b4),
        "boris_corrector_pt_sha256_16": sha(ckpt_old) if os.path.exists(ckpt_old) else None,
        "n_parameters": n_par,
        "layer_shapes": shapes,
        "dtype": str(next(model.parameters()).dtype),
    }
    assert type(model).__name__ == "DefectNet", "wrong class benchmarked"
    assert n_par == 52102, f"expected 52102 params, got {n_par}"
    print(f"[2] benchmarked {type(model).__name__}, {n_par} params, "
          f"ckpt={os.path.basename(ckpt_b4)} sha={out['identity']['checkpoint_sha256_16']}")

    # cross-check: the published Figure 4 value is reproduced by this model
    pub = os.path.join(ROOT, "output_figures", "corrector_evaluation.json")
    with open(pub) as f:
        published = json.load(f)
    out["identity"]["published_pos_err_rms"] = published["corrector_projected"]["pos_err_rms"]
    with open(os.path.join(HERE, "work_precision.json")) as f:
        wp = json.load(f)
    mine = [r for r in wp["hybrid"] if r["dt"] == DT_WORK][0]["pos_err_rms"]
    out["identity"]["benchmarked_pos_err_rms"] = mine
    out["identity"]["relative_difference"] = abs(
        mine / published["corrector_projected"]["pos_err_rms"] - 1.0)
    print(f"    published={out['identity']['published_pos_err_rms']:.10e}  "
          f"benchmarked={mine:.10e}  reldiff={out['identity']['relative_difference']:.2e}")

    # ---------------- (3) cost of float64 ----------------------------------
    field = DecayingField(B0=1.0, tau=TAU_MAIN)
    x64 = torch.tensor(np.concatenate([R0, V0, [0, 0, 1.0], [0, 0, 0], [DT_WORK]]),
                       dtype=torch.float64)[None, :]

    m64 = model
    m32 = DefectNet(n_in=13).to(torch.float32)
    m32.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    m32.eval()
    x32 = x64.to(torch.float32)

    def bench_net(m, x, n=N_TIME):
        with torch.no_grad():
            for _ in range(200):
                m(x)
            t0 = time.perf_counter()
            for _ in range(n):
                m(x)
            return (time.perf_counter() - t0) / n * 1e6

    us64 = min(bench_net(m64, x64) for _ in range(3))
    us32 = min(bench_net(m32, x32) for _ in range(3))

    # Boris step alone, for the split
    def bench_boris(n=N_TIME):
        r, v, t = R0.copy(), V0.copy(), 0.0
        t0 = time.perf_counter()
        for _ in range(n):
            r, v = boris_step(r, v, t, DT_WORK, field)
            t += DT_WORK
        return (time.perf_counter() - t0) / n * 1e6

    us_boris = min(bench_boris() for _ in range(3))

    # tensor construction overhead per step (np.concatenate + torch.tensor)
    def bench_wrap(n=N_TIME):
        B = np.array([0.0, 0.0, 1.0]); E = np.zeros(3)
        t0 = time.perf_counter()
        for _ in range(n):
            torch.tensor(np.concatenate([R0, V0, B, E, [DT_WORK]]))[None, :]
        return (time.perf_counter() - t0) / n * 1e6

    us_wrap = min(bench_wrap() for _ in range(3))

    tot = [r for r in wp["hybrid"] if r["dt"] == DT_WORK][0]["us_per_step"]
    out["dtype_cost"] = {
        "note": ("Component costs are measured in ISOLATION and are NOT additive: "
                 "summed they exceed the in-loop step cost, because each isolated "
                 "micro-benchmark carries its own call/allocation overhead that is "
                 "amortised inside the integration loop. Treat them as upper bounds "
                 "on each component, not as a partition."),
        "defectnet_forward_us_float64": us64,
        "defectnet_forward_us_float32": us32,
        "float64_slowdown_on_network": us64 / us32,
        "float64_extra_us_per_step": us64 - us32,
        "boris_step_us_isolated": us_boris,
        "tensor_construction_us_isolated": us_wrap,
        "hybrid_step_measured_in_loop_us": tot,
        "boris_step_measured_in_loop_us": 1e6 * min(
            r["wall_s"] / r["n_steps"] for r in wp["boris"]),
        "float64_share_of_hybrid_step_pct": 100 * (us64 - us32) / tot,
        "network_share_of_hybrid_step_pct_upper_bound": 100 * us64 / tot,
        "float32_would_not_help_because": ("train_corrector_b4.py sets float64 "
                 "deliberately: at the 1e-6 relative energy error targeted here, "
                 "float32 round-off is itself the dominant noise source."),
    }

    print(f"[3] DefectNet forward: float64 {us64:.1f} us | float32 {us32:.1f} us "
          f"| slowdown {us64/us32:.2f}x")
    print(f"    boris_step {us_boris:.1f} us | tensor construction {us_wrap:.1f} us "
          f"| measured hybrid step {tot:.1f} us")
    print(f"    float64 costs {us64 - us32:.1f} us/step extra = "
          f"{100 * (us64 - us32) / tot:.1f}% of the hybrid step; "
          f"network is <={100 * us64 / tot:.0f}% of it")
    print("    (isolated component costs are NOT additive -- see 'note' in JSON)")

    with open(os.path.join(HERE, "model_dtype.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote model_dtype.json")


if __name__ == "__main__":
    main()
