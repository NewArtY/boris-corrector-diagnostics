"""PL-2: is the 'edge shortfall' a window/realisation artefact or a property of the law?

  R1  exact reproduction of the verifier's C1 block (their code path, their seed 7)
  R2  the same estimator over 256 independent seeds -> bias and scatter
  R3  horizon convergence: does the estimator -> a+H as the run gets longer?
  R4  AR(1) in the co-rotating frame: crossover, not a Hurst exponent
  R5  the three sufficient conditions as limit points of p = (a+H)_+
  R6  lab-frame REAL kick (counter-rotating component present) - same law?
  R7  breakdown of the linear regime: dev ~ |S|^2 doubles the exponent
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H_STEP = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
Z0 = 1j
t0 = time.time()
out = {}


def phases(N):
    t_n = np.arange(N) * H_STEP
    th = 2 * np.arctan(H_STEP * np.exp(-t_n / TAU_Q) / 2)
    Phi = np.concatenate([[0.0], np.cumsum(th)])[:-1]
    return t_n, th, Phi


def envelope_exponent(ts_, dev, n_samples=4000, win=100.0):
    stride = max(1, len(ts_) // n_samples)
    idx = np.arange(stride - 1, len(ts_), stride)
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts_[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / win) & (env > 0)
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])


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


# ================================================== R1: reproduce the verifier verbatim
KAPPA_V = 3.4700895e-07          # verifier's calibrated kappa (vt_t3_trichotomy.json)
NG = 1e4
Nv = int(round(NG * TWO_PI / H_STEP))
t_n, th, Phi = phases(Nv)
rep = []
for a in (-0.25, 0.0, 0.25, 0.4):
    rg = np.random.default_rng(7)                       # THEIR seed, THEIR call order
    sig = KAPPA_V * ((np.arange(Nv) + 1.0) / Nv) ** a
    kk = sig * (rg.standard_normal(Nv) + 1j * rg.standard_normal(Nv)) / np.sqrt(2)
    S = np.cumsum(np.exp(1j * (Phi + th)) * kk)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    rep.append({"a": a, "a_plus_half": a + 0.5,
                "exponent_seed7": envelope_exponent(t_n + H_STEP, dev)})
out["R1_reproduce_verifier"] = rep
out["R1_verifier_reported"] = {"-0.25": 0.200, "0.0": 0.510, "0.25": 0.674, "0.4": 0.787}
print("R1", [round(r["exponent_seed7"], 4) for r in rep], f"{time.time()-t0:.0f}s", flush=True)

# ================================================== R2: seed scatter of the estimator
NSEED = 256
res = {a: [] for a in (-0.25, 0.0, 0.25, 0.4)}
rot = np.exp(1j * (Phi + th))
for s in range(NSEED):
    rg = np.random.default_rng(90000 + s)
    xi = (rg.standard_normal(Nv) + 1j * rg.standard_normal(Nv)) / np.sqrt(2)
    for a in res:
        sig = KAPPA_V * ((np.arange(Nv) + 1.0) / Nv) ** a
        S = np.cumsum(rot * sig * xi)
        dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
        res[a].append(envelope_exponent(t_n + H_STEP, dev))
r2 = []
for a, v in res.items():
    v = np.array(v)
    r2.append({"a": a, "a_plus_half": a + 0.5, "n_seeds": NSEED,
               "mean": float(v.mean()), "median": float(np.median(v)),
               "std": float(v.std()),
               "p05_p95": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
               "verifier_value_inside_90pct_band":
                   bool(np.percentile(v, 5) <= out["R1_verifier_reported"][str(a)]
                        <= np.percentile(v, 95))})
out["R2_seed_scatter_at_1e4_gyro"] = r2
# how correlated are the four estimates when ONE seed is reused (verifier's design)?
M = np.array([res[a] for a in (-0.25, 0.0, 0.25, 0.4)])
out["R2_correlation_across_a_same_seed"] = np.corrcoef(M).round(4).tolist()
print("R2", [round(x["mean"], 4) for x in r2], f"{time.time()-t0:.0f}s", flush=True)

# ================================================== R3: horizon convergence
r3 = []
for (a, Hu) in ((0.25, 0.5), (0.4, 0.5), (-0.25, 0.5), (0.0, 0.7), (0.25, 0.7)):
    for logN in (16, 18, 20, 22):
        if Hu != 0.5 and logN > 21:
            continue
        N = 1 << logN
        nb_tot = 64 if logN <= 20 else 16
        batch = 8 if logN <= 20 else 2
        t_nn, thn, Phin = phases(N)
        rotn = np.exp(1j * (Phin + thn))
        kidx = np.arange(1, N + 1, dtype=float)
        sig = 1e-9 * kidx ** a
        tsn = t_nn + H_STEP
        n_probe = np.unique(np.round(np.logspace(np.log10(64), np.log10(N), 60)).astype(int)) - 1
        acc = np.zeros(len(n_probe)); ex = []
        done = 0
        while done < nb_tot:
            nb = min(batch, nb_tot - done)
            rg = np.random.default_rng(500000 + 31 * done + logN * 7 + int(100 * a) + int(1000 * Hu))
            u = fgn_batch(Hu, N, rg, nb)
            S = np.cumsum(sig[None, :] * u, axis=1)
            acc += np.sum(np.abs(S[:, n_probe]) ** 2, axis=0)
            dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
            for b in range(nb):
                ex.append(envelope_exponent(tsn, dev[b]))
            done += nb
            del u, S, dev
        rms = np.sqrt(acc / nb_tot)
        nn = n_probe + 1.0
        sel = nn > nn[-1] / 100.0
        q = float(np.polyfit(np.log10(nn[sel]), np.log10(rms[sel]), 1)[0])
        r3.append({"a": a, "H": Hu, "a_plus_H": a + Hu, "log2N": logN,
                   "gyros": N * H_STEP / TWO_PI, "n_seeds": nb_tot,
                   "q_rms": q, "q_rms_minus_aH": q - (a + Hu),
                   "p_env_median": float(np.median(ex)),
                   "p_env_std": float(np.std(ex)),
                   "p_env_median_minus_aH": float(np.median(ex)) - (a + Hu)})
        print("  R3", a, Hu, logN, round(q, 4), round(float(np.median(ex)), 4),
              f"{time.time()-t0:.0f}s", flush=True)
out["R3_horizon_convergence"] = r3

json.dump(out, open(os.path.join(HERE, "pl_horizon.json"), "w"), indent=1)
print("saved part 1", time.time() - t0)
