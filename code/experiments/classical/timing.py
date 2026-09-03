"""timing.py -- wall-clock with spread, run last to minimise contention.
Flop counts (verdict.json) are load-independent and carry the conclusions;
these numbers are reported only as a secondary, implementation-dependent view."""
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
from common import CHECKPOINT_DIR
from fields import DecayingField
from models.boris import integrate_boris
from training.train_corrector_b4 import DefectNet, DT_WORK, T_FINAL, TAU_MAIN
import schemes as S
from run import integrate_hybrid, load_corrector

R0 = np.array([1.,0.,0.]); V0 = np.array([0.,1.,0.])
N_REP = 7
field = DecayingField(B0=1.0, tau=TAU_MAIN)
model = load_corrector()
n = int(round(T_FINAL / DT_WORK))
runners = {
 "shipped":   lambda: integrate_boris(R0, V0, 0.0, DT_WORK, n, field),
 "vps2":      lambda: S.integrate(S.make_vps2(field), R0, V0, DT_WORK, n),
 "vps4":      lambda: S.integrate(S.make_vps4(field), R0, V0, DT_WORK, n),
 "gl4":       lambda: S.integrate(S.make_gl4(field), R0, V0, DT_WORK, n),
 "hybrid":    lambda: integrate_hybrid(field, DT_WORK, n, model),
}
out = {}
for name, fn in runners.items():
    fn()  # warm-up
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts = np.array(sorted(ts))
    out[name] = {"median_s": float(np.median(ts)), "min_s": float(ts[0]),
                 "max_s": float(ts[-1]), "spread_pct": float(100*(ts[-1]-ts[0])/np.median(ts)),
                 "n_repeats": N_REP}
    print(f"{name:9s} median={np.median(ts):.4f}s  min={ts[0]:.4f}  max={ts[-1]:.4f}  "
          f"spread={100*(ts[-1]-ts[0])/np.median(ts):5.1f}%")
h = out["hybrid"]["median_s"]
for k in out: out[k]["cheaper_than_hybrid"] = h / out[k]["median_s"]
json.dump(out, open(os.path.join(HERE, "timing.json"), "w"), indent=2)
print("\nwrote timing.json")
