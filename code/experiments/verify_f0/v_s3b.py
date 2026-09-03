"""Independent recomputation of the S3b antithetic measurement (F0.1 point A1-A3).

(a) reproduce the shipped numbers (same seeds 0..23);
(b) fresh seeds 7000..7023 -> is the coefficient an ensemble property or a
    property of that particular set of 24 phase draws?
(c) larger sample (96 pairs) at the smallest structured amplitude -> does the
    3.4 sigma point survive more data?
(d) per-pair coefficients c_k = (pair_mean - smooth)/rms^2 -> how much of the
    "constant to 4 digits" is seed reuse?
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'f0_settings')))
from s3b_structured import dE_final, StructuredPerturbed, antithetic_mean

t00 = time.time()
smooth = StructuredPerturbed()
dE_smooth, _ = dE_final(smooth)
out = {"dE_smooth": dE_smooth}
print(f"smooth dE = {dE_smooth:+.12e}")

RMS = [3e-4, 1e-3, 3e-3, 1e-2]

def block(tag, seed0, n_pairs, rms_list, structured_list):
    rows = []
    for rms in rms_list:
        for structured in structured_list:
            m, se, vals = antithetic_mean(rms, n_pairs, structured=structured,
                                          seed0=seed0)
            shift = m - dE_smooth
            c = (vals - dE_smooth) / rms**2      # per-pair coefficient
            rows.append({"rms": rms, "structured": bool(structured),
                         "shift": shift, "stderr": se,
                         "sigma": shift / se,
                         "coef": shift / rms**2,
                         "coef_per_pair_mean": float(c.mean()),
                         "coef_per_pair_sd": float(c.std(ddof=1)),
                         "coef_per_pair": c.tolist()})
            print(f"[{tag}] rms={rms:.0e} {'S' if structured else 'U'} "
                  f"shift={shift:+.4e} +-{se:.1e} ({shift/se:+.2f}s) "
                  f"C={shift/rms**2:.4f}  sd(c_k)={c.std(ddof=1):.3f} "
                  f"({time.time()-t00:.0f}s)")
    return rows

out["repro_seed0"] = block("repro", 0, 24, RMS, [False, True])
out["fresh_seed7000"] = block("fresh", 7000, 24, RMS, [False, True])
out["big_structured_3e-4"] = block("big96", 200, 96, [3e-4], [True])
out["big_uniform_3e-4"] = block("big96u", 200, 96, [3e-4], [False])

json.dump(out, open(os.path.join(HERE, "v_s3b.json"), "w"), indent=1)
print(f"done {time.time()-t00:.0f}s -> v_s3b.json")
