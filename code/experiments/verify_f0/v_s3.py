"""Rerun of the S3 broadband attacks (same seeds), output kept in verify_f0.
Verifies: truth ensemble mean, mean-field attack error (claim 2.42e-5, 2.4% of
signal), Monte-Carlo convergence at the hybrid budget (claim 3.21e-5 at N=418).
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_settings')))
import harness as H
from pfields import PerturbedDecaying
from s3_broadband import energy_curve, resample, make_field, DT_FINE

t0 = time.time()
rms = 1e-4
grid = np.linspace(0.0, 120.0, 601)
truth = []
for s in range(400):
    f, _ = make_field(rms, seed=s)
    ts, y, _ = energy_curve(f, DT_FINE)
    truth.append(resample(ts, y, grid))
    if s % 100 == 0:
        print(f"truth {s}  ({time.time()-t0:.0f}s)")
truth = np.array(truth)
mean_truth = truth.mean(axis=0)
sig = float(np.abs(mean_truth[len(grid)//2:]).max())
print(f"truth mean dE(T)={mean_truth[-1]:+.6e} sd={truth[:,-1].std():.4e} sig={sig:.6e}")

f_smooth = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
ts, y_mf, fl_mf = energy_curve(f_smooth, H.DT_WORK)
y_mf = resample(ts, y_mf, grid)
err_mf = float(np.abs(y_mf - mean_truth).max())
print(f"mean-field: err={err_mf:.6e} rel={err_mf/sig:.4f} flops={fl_mf:.3e}")

mc = []
for s in range(512):
    f, _ = make_field(rms, seed=10_000+s)
    ts, y, fl = energy_curve(f, H.DT_WORK)
    mc.append(resample(ts, y, grid))
mc = np.array(mc)
conv = {}
for n in (16, 128, 418, 512):
    e = float(np.abs(mc[:n].mean(axis=0) - mean_truth).max())
    conv[n] = e
    print(f"MC N={n}: err={e:.6e} rel={e/sig:.4f}")

json.dump({"mean_truth_final": float(mean_truth[-1]), "sig": sig,
           "err_mean_field": err_mf, "mc": conv},
          open(os.path.join(HERE, "v_s3.json"), "w"), indent=1)
print(f"done {time.time()-t0:.0f}s")
