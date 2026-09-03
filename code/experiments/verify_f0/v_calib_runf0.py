"""Recheck of F0.2 calibration + reduced independent rerun of the long-run
pipeline (small amplitudes, 1e4 gyrations, quasistatic config) to confirm the
plateau values of f0_summary_small.json through an independent execution.
Also: linearity-of-response check for the calibration methodology (point 8).
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_variational')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

import varint as V
from calibrate import calibrate, per_step_dv, SEEDS, T_SCALE, TARGET

out = {}
t0 = time.time()

# ------------------------------------------------ calibration reproduction
for mode in ("varnet", "additive"):
    amp, unit, check = calibrate(mode)
    out[mode] = {"amp": amp.tolist(), "achieved": check.tolist()}
    print(f"{mode}: amp={np.array2string(amp, precision=6)}")
    print(f"        achieved={np.array2string(check, precision=6)}")

# linearity of one-step response for varnet (is amp -> dv really linear?)
lin = []
for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
    amps = np.array(out["varnet"]["amp"]) * scale
    net = V.DeltaLNet(len(SEEDS), SEEDS, amps, hidden=32, t_scale=T_SCALE)
    got = per_step_dv(net, n_steps=500, mode="varnet")
    lin.append({"scale": scale, "mean_dv_over_scale": float(np.mean(got)/scale)})
    print(f"linearity varnet scale={scale}: dv/scale={np.mean(got)/scale:.6e}")
out["linearity"] = lin

# ------------------------------------------------ reduced long-run rerun
TAU_Q = 1.2e8
H = 0.3
N_GYR = 1e4
N_STEPS = int(round(N_GYR*2*np.pi/H))
cal = json.load(open(os.path.join(HERE, '..', 'f0_variational', 'calibration.json')))
rows = []
for fac in (1e-4, 1e-3):
    for mode in ("varnet", "additive"):
        key = "varnet" if mode == "varnet" else "additive"
        amps = np.array(cal[key]["amp_calibrated"]) * fac
        net = V.DeltaLNet(len(SEEDS), SEEDS, amps, hidden=32, t_scale=T_SCALE)
        d = V.integrate(mode, TAU_Q, H, N_STEPS,
                        net=net if mode == "varnet" else None,
                        dv_net=net if mode == "additive" else None,
                        n_samples=2000, n_ens=len(SEEDS))
        t = d["t"]
        emaxs = {}
        for Hg in (1e3, 1e4):
            m = t <= Hg*2*np.pi
            env = np.maximum.accumulate(d["env"], axis=1)
            emaxs[Hg] = [float(env[j][m][-1]) for j in range(len(SEEDS))]
        rows.append({"fac": fac, "mode": mode,
                     "Emax_1e3_med": float(np.median(emaxs[1e3])),
                     "Emax_1e4_med": float(np.median(emaxs[1e4]))})
        print(f"rerun fac={fac:g} {mode:8s}: Emax(1e3)med={np.median(emaxs[1e3]):.4g} "
              f"Emax(1e4)med={np.median(emaxs[1e4]):.4g}  ({time.time()-t0:.0f}s)")
out["rerun_small"] = rows

json.dump(out, open(os.path.join(HERE, "v_calib_runf0.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
