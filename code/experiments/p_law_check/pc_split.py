"""W0.2 companion: WHERE does the law break on the trained hybrid?

P_LAW section 8 flags one step of the derivation as not proved, only checked
ensemble-wise:  "the envelope of |Re(conj(z0) S_N)| grows like max|S_N|".
That step is what turns the partial-sum exponent a+H into the envelope exponent
of the energy error.  Here it is tested directly on the trained hybrid: the
demodulated defect is summed and the growth exponents of |S|, |Re S| and |Im S|
are measured separately, in the theory-faithful (unperturbed) frame.
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pc_defect import (run_instrumented, envelope_exponent_from_series,  # noqa: E402
                       loglog_slope, sub_index, estimate_aH, DT, TWO_PI,
                       TAU_PAPER, TAU_QUASI, CAMPAIGN)
import symproj as S  # noqa: E402

N_GYR = float(os.environ.get("PC_GYROS", 100000))
N_STEPS = int(round(N_GYR * TWO_PI / DT))
out = {"setup": {"gyros": N_GYR, "n_steps": N_STEPS}, "runs": {}}
fwd = S.load_forward()

for cname, tau in (("paper", TAU_PAPER), ("quasistatic", TAU_QUASI)):
    t0 = time.time()
    base = run_instrumented("boris", tau, N_STEPS, fwd)
    res = run_instrumented("proj", tau, N_STEPS, fwd)
    n = N_STEPS
    t = np.arange(1, n + 1, dtype=float) * DT
    sub = sub_index(n, 600)
    dev = np.abs(res["signed"])
    u = base["zb"] / np.abs(base["zb"])
    w = res["kappa"] * np.conj(u)
    Sw = np.cumsum(w)
    ReS, ImS, absS = np.real(Sw), np.imag(Sw), np.abs(Sw)

    p_meas, _, _ = envelope_exponent_from_series(dev)
    rec = {"campaign": CAMPAIGN[f"{cname}/proj"], "p_measured": p_meas,
           "a_hat": None, "H_hat": None}
    a_h, H_h = estimate_aH(w[None, :])
    rec["a_hat"], rec["H_hat"] = a_h, H_h
    rec["a+H"] = max(0.0, a_h + H_h)

    for name, arr in (("|S|", absS), ("|Re S|", np.abs(ReS)),
                      ("|Im S|", np.abs(ImS))):
        e, _, env = envelope_exponent_from_series(arr)
        rec[f"envelope_exponent_{name}"] = e
        rec[f"pointwise_slope_{name}"] = loglog_slope(t[sub], arr[sub])
        rec[f"final_{name}"] = float(arr[-1])
    rec["ratio_|Re S|/|S|_final"] = float(abs(ReS[-1]) / max(absS[-1], 1e-300))
    # the two terms of dev separately
    e_lin, _, _ = envelope_exponent_from_series(np.abs(2.0 * ReS))
    e_quad, _, _ = envelope_exponent_from_series(absS ** 2)
    e_tot, _, _ = envelope_exponent_from_series(np.abs(2.0 * ReS + absS ** 2))
    rec["envelope_exponent_2|Re S|"] = e_lin
    rec["envelope_exponent_|S|^2"] = e_quad
    rec["envelope_exponent_full_reconstruction"] = e_tot
    rec["reconstruction_ratio_final"] = float(
        abs(2.0 * ReS[-1] + absS[-1] ** 2) / max(dev[-1], 1e-300))
    rec["law_step_check_env|Re S| minus env|S|"] = (
        rec["envelope_exponent_|Re S|"] - rec["envelope_exponent_|S|"])
    rec["seconds"] = time.time() - t0
    out["runs"][cname] = rec
    print(f"[{cname}] measured {p_meas:.4f} | a={a_h:+.4f} H={H_h:.4f} a+H={rec['a+H']:.4f}"
          f" | env|S|={rec['envelope_exponent_|S|']:.4f} env|ReS|={rec['envelope_exponent_|Re S|']:.4f}"
          f" env|ImS|={rec['envelope_exponent_|Im S|']:.4f}"
          f" | env2|ReS|={e_lin:.4f} env|S|^2={e_quad:.4f} env_total={e_tot:.4f}"
          f" ratio={rec['reconstruction_ratio_final']:.3f}  {rec['seconds']:.0f}s", flush=True)
    del base, res

json.dump(out, open(os.path.join(HERE, "pc_split.json"), "w"), indent=1)
print("done")
