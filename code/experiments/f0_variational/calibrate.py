"""Ф0.2 step 2a -- amplitude calibration.

Both branches must perturb the step by the SAME relative velocity increment,
otherwise the comparison is rigged. Target is the measured working amplitude of
the trained DefectNet: |dv|/|v| = 2.22e-3 per step (И3.1, experiments/theory).

Method: for each candidate amplitude, step the perturbed map and the clean map
from the SAME state and measure |v_pert - v_clean| / |v_clean|, averaged along a
trajectory. Then rescale so the mean equals the target.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from training.train_corrector_b4 import DT_WORK, TAU_MAIN
import varint as V

TARGET = 2.22e-3          # measured relative correction amplitude of DefectNet
N_CAL = 4000              # steps used for calibration
SEEDS = [11, 22, 33, 44, 55]
H = DT_WORK
T_SCALE = 120.0           # t_final of the paper experiment, keeps t-feature O(1)


def per_step_dv(net, n_steps=N_CAL, tau=TAU_MAIN, mode='varnet'):
    """Mean relative per-step velocity perturbation vs the clean variational map."""
    n = net.n
    q = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    v = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    t = 0.0
    p = v - V.A_of(q, t, tau)
    acc = np.zeros(n)
    for i in range(n_steps):
        q_c, p_c, _ = V.var_step(q, p, t, H, tau, None)
        v_c = p_c + V.A_of(q_c, t + H, tau)
        if mode == 'varnet':
            q_p, p_p, _ = V.var_step(q, p, t, H, tau, net)
            v_p = p_p + V.A_of(q_p, t + H, tau)
        else:
            d1, d2 = net.grads(q, q_c, t)
            v_p = v_c + (d1 + d2)
        acc += np.linalg.norm(v_p - v_c, axis=-1) / np.linalg.norm(v_c, axis=-1)
        # advance on the CLEAN branch: the one-step response is then exactly
        # linear in amp, which is what "perturbation amplitude" has to mean.
        q, p = q_c, p_c
        t += H
    return acc / n_steps


def calibrate(mode):
    """Return per-seed amplitude that yields TARGET mean relative dv."""
    net = V.DeltaLNet(len(SEEDS), SEEDS, np.ones(len(SEEDS)),
                      hidden=32, t_scale=T_SCALE, q_scale=1.0)
    got = per_step_dv(net, mode=mode)
    amp = TARGET / got                      # response is linear in amp
    net2 = V.DeltaLNet(len(SEEDS), SEEDS, amp, hidden=32,
                       t_scale=T_SCALE, q_scale=1.0)
    check = per_step_dv(net2, mode=mode)
    return amp, got, check


if __name__ == "__main__":
    out = {"target_rel_dv_per_step": TARGET, "seeds": SEEDS, "h": H,
           "n_cal_steps": N_CAL, "t_scale": T_SCALE}
    for mode in ["varnet", "additive"]:
        t0 = time.time()
        amp, unit, check = calibrate(mode)
        out[mode] = {"amp_unit_response": unit.tolist(),
                     "amp_calibrated": amp.tolist(),
                     "achieved_rel_dv": check.tolist(),
                     "seconds": time.time() - t0}
        print(f"{mode:9s} amp={np.array2string(amp, precision=4)}")
        print(f"{'':9s} achieved={np.array2string(check, precision=4)}  "
              f"({time.time()-t0:.1f}s)")
    json.dump(out, open(os.path.join(HERE, "calibration.json"), "w"), indent=2)
    print("saved calibration.json")
