"""W0.2 companion: is 1.540 an asymptotic exponent, or a window average of a
curve that stalls once the driving field has decayed?

The paper configuration has tau = 1.2e5, and the campaign horizon 1e5 gyrations
is t = 6.28e5 = 5.24 tau.  The E field that does the work on the particle is
E = (B/2tau)(z x r) with B = B0 exp(-t/tau), so the energy-error growth rate is
proportional to exp(-t/tau): after a few tau no further error can accumulate and
the envelope must plateau.  If so the exponent 1.540 is not defined
asymptotically and the law p = (a+H)_+ (which requires an asymptotic H) does not
apply to it -- exactly the 'piecewise defect' caveat of P_LAW section 5, note 4.

Test: run the SAME shipped hybrid 10x longer (1e6 gyrations = 52 tau) and refit.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "symproj"))

import torch  # noqa: E402
torch.set_default_dtype(torch.float64)
import symproj as S  # noqa: E402

TWO_PI = 2.0 * np.pi
DT = S.DT_WORK
N_GYR = float(os.environ.get("PC_GYROS_LONG", 1e6))
N_STEPS = int(round(N_GYR * TWO_PI / DT))

out = {"setup": {"gyros": N_GYR, "n_steps": N_STEPS, "dt": DT,
                 "tau_paper": S.TAU_MAIN, "tau_quasistatic": 1.2e8,
                 "horizon_in_tau_paper": N_GYR * TWO_PI / DT * DT / S.TAU_MAIN},
       "runs": {}}
fwd = S.load_forward()

for cname, tau in (("paper", S.TAU_MAIN), ("quasistatic", 1.2e8)):
    t0 = time.time()
    d = S.run("proj", tau, DT, N_STEPS, fwd=fwd, base="shipped", n_samples=8000)
    t = d["t"]
    env = np.maximum.accumulate(d["env"])
    gy = t / TWO_PI
    rec = {"seconds": time.time() - t0,
           "exponent_last2dec_full_horizon": S.envelope_exponent(t, env),
           "env_final": float(env[-1]), "horizon_in_tau": float(t[-1] / tau),
           "half_decade_slopes": {}, "exponent_vs_horizon": {}}
    lg, le = np.log10(gy), np.log10(np.maximum(env, 1e-300))
    ok = env > 0
    for lo in np.arange(2.0, np.log10(N_GYR) - 0.4, 0.5):
        m = ok & (lg >= lo) & (lg < lo + 0.5)
        if m.sum() > 5:
            rec["half_decade_slopes"][f"{10**lo:.3g}-{10**(lo+0.5):.3g} gyr"] = {
                "slope": float(np.polyfit(lg[m], le[m], 1)[0]),
                "env_end": float(10 ** le[m][-1]),
                "t_over_tau": float(10 ** (lo + 0.5) * TWO_PI / tau)}
    # exponent as the campaign would report it, if the run had been stopped early
    for Hgy in (1e4, 3e4, 1e5, 3e5, 1e6):
        if Hgy > N_GYR:
            continue
        m = gy <= Hgy
        if m.sum() < 50:
            continue
        rec["exponent_vs_horizon"][f"{Hgy:.0e}"] = S.envelope_exponent(t[m], env[m])
    out["runs"][cname] = rec
    print(f"[{cname}] {rec['seconds']:.0f}s  full-horizon exponent="
          f"{rec['exponent_last2dec_full_horizon']:.4f}  env_final={rec['env_final']:.4e}  "
          f"T={rec['horizon_in_tau']:.1f} tau", flush=True)
    for k, v in rec["half_decade_slopes"].items():
        print(f"    {k:22s} slope={v['slope']:+.3f}  env={v['env_end']:.3e}  "
              f"t/tau={v['t_over_tau']:.2f}", flush=True)
    print("    exponent vs stopping horizon:",
          {k: round(v, 4) for k, v in rec["exponent_vs_horizon"].items()}, flush=True)
    np.savez_compressed(os.path.join(HERE, f"pc_stall_{cname}.npz"), t=t, env=env)

json.dump(out, open(os.path.join(HERE, "pc_stall.json"), "w"), indent=1)
print("done")
