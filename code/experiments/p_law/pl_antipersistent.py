"""PL-5: complete the H-range.  Anti-persistent phase memory H < 1/2, and the
analytic continuation of the exact constant C(a,H) = H(2H-1)B(a+1,2H-1)/(a+H)
below H = 1/2 (where gamma(m) is negative and the double integral is a finite part).
"""
import os, json, time
import numpy as np
from scipy.special import beta as Bfun

HERE = os.path.dirname(os.path.abspath(__file__))
H_STEP, TAU_Q, TWO_PI = 0.3, 1.2e8, 2 * np.pi
Z0 = 1j
t0 = time.time()


def fgn_batch(Hu, n, rng, nb):
    if abs(Hu - 0.5) < 1e-12:
        return (rng.standard_normal((nb, n)) + 1j * rng.standard_normal((nb, n))) / np.sqrt(2)
    k = np.arange(0, n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * Hu) - 2 * np.abs(k) ** (2 * Hu) + np.abs(k - 1) ** (2 * Hu))
    row = np.concatenate([g, g[-2:0:-1]])
    m = row.size
    lam = np.maximum(np.fft.fft(row).real, 0.0)
    amp = np.sqrt(lam / (2.0 * m))
    V = rng.standard_normal((nb, m)) + 1j * rng.standard_normal((nb, m))
    Y = np.fft.fft(amp[None, :] * V, axis=1)
    return (np.sqrt(2.0) * Y.real[:, :n] + 1j * np.sqrt(2.0) * Y.imag[:, :n]) / np.sqrt(2)


def C_theory(a, Hu):
    if abs(Hu - 0.5) < 1e-9:
        return 1.0 / (2 * a + 1.0)
    return Hu * (2 * Hu - 1) * Bfun(a + 1.0, 2 * Hu - 1.0) / (a + Hu)


def envelope_exponent(ts_, dev, n_samples=4000, win=100.0):
    stride = max(1, len(ts_) // n_samples)
    idx = np.arange(stride - 1, len(ts_), stride)
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts_[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / win) & (env > 0)
    return float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])


N = 1 << 19
NSEED, BATCH = 64, 8
KAPPA = 1e-9
t_n = np.arange(N) * H_STEP
ts = t_n + H_STEP
kidx = np.arange(1, N + 1, dtype=float)
n_probe = np.unique(np.round(np.logspace(np.log10(64), np.log10(N), 60)).astype(int)) - 1
rows = []
for Hu in (0.2, 0.3, 0.4):
    for a in (0.0, 0.25, 0.5, 0.75):
        acc = np.zeros(len(n_probe)); ex = []
        done = 0
        while done < NSEED:
            nb = min(BATCH, NSEED - done)
            rg = np.random.default_rng(70000 + done + int(1000 * Hu) + int(100 * a))
            u = fgn_batch(Hu, N, rg, nb)
            S = np.cumsum((KAPPA * kidx ** a)[None, :] * u, axis=1)
            acc += np.sum(np.abs(S[:, n_probe]) ** 2, axis=0)
            dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
            for b in range(nb):
                ex.append(envelope_exponent(ts, dev[b]))
            done += nb
        rms = np.sqrt(acc / NSEED); nn = n_probe + 1.0
        sel = nn > nn[-1] / 100.0
        q = float(np.polyfit(np.log10(nn[sel]), np.log10(rms[sel]), 1)[0])
        Cm = float(rms[-1] ** 2 / (KAPPA ** 2 * nn[-1] ** (2 * (a + Hu))))
        rows.append({"a": a, "H": Hu, "a_plus_H": a + Hu, "q_rms": q,
                     "q_minus_aH": q - (a + Hu),
                     "C_measured": Cm, "C_theory_continued": float(C_theory(a, Hu)),
                     "C_ratio": Cm / float(C_theory(a, Hu)),
                     "p_env_mean": float(np.mean(ex)), "p_env_std": float(np.std(ex))})
        print(rows[-1]["a"], rows[-1]["H"], round(q, 4), round(rows[-1]["C_ratio"], 4),
              round(rows[-1]["p_env_mean"], 4), f"{time.time()-t0:.0f}s", flush=True)
json.dump({"antipersistent": rows}, open(os.path.join(HERE, "pl_antipersistent.json"), "w"), indent=1)
