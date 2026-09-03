"""Is the 'ignoring dB = 2.4% of signal' figure physics or ensemble noise?
Build two INDEPENDENT 400-member truth ensembles (disjoint seed ranges) at
rms=1e-4 and compare their means. If max|mean_A - mean_B| is of the same size
as the claimed mean-field attack error (2.42e-5), that error is sampling noise;
the true mean shift by the antithetic law is 36.02*(1e-4)^2 = 3.6e-7.
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_settings')))
from s3_broadband import energy_curve, resample, make_field, DT_FINE

t0 = time.time()
grid = np.linspace(0.0, 120.0, 601)
rms = 1e-4

def ensemble(seed0, n):
    acc = []
    for s in range(n):
        f, _ = make_field(rms, seed=seed0+s)
        ts, y, _ = energy_curve(f, DT_FINE)
        acc.append(resample(ts, y, grid))
    return np.array(acc)

A = ensemble(20000, 400); print(f"A done ({time.time()-t0:.0f}s)")
B = ensemble(30000, 400); print(f"B done ({time.time()-t0:.0f}s)")
mA, mB = A.mean(axis=0), B.mean(axis=0)
d = float(np.abs(mA - mB).max())
sd = float(A.std(axis=0)[-1])
print(f"max|mean_A - mean_B| over grid = {d:.4e}   (claimed attack err 2.42e-5)")
print(f"sd across realizations (final) = {sd:.4e}, SE of 400-mean = {sd/20:.4e}")
json.dump({"max_diff_two_truths": d, "sd_final": sd},
          open(os.path.join(HERE, "v_s3_noise.json"), "w"), indent=1)
