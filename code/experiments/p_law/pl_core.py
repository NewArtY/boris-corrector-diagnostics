"""PL-1 (core): the two-parameter growth law for the exponential sum S_N.

Model (11_THEORY Lemma 3.1, re-derived independently):
    z_{n+1} = e^{-i th_n} z_n + kappa_n      ->   z_N = e^{-i Phi_N}(z_0 + S_N),
    S_N = sum_{k<N} e^{i Phi_{k+1}} kappa_k,   dev_N = |2Re(conj(z0)S_N)+|S_N|^2|/|z0|^2.

Define the DEMODULATED per-step defect  w_k := e^{i Phi_{k+1}} kappa_k.  Everything
about the growth exponent is a property of the partial sums of w.  Two parameters:

    a  : amplitude growth,   |w_k| envelope ~ k^a
    H  : Hurst exponent of the normalised sequence u_k = w_k/sigma_k
         (|sum_{k<=N} u_k| ~ N^H)

CLAIM:  p = (a + H)_+   in the linear regime |S_N| << |z_0|.

Sharper claim (exact constant, not just the exponent), for fractional Gaussian
noise u with Hurst H > 1/2 and sigma_k = kappa * k^a:

    Var(S_N) / (kappa^2 N^{2(a+H)})  ->  C(a,H) = H(2H-1) B(a+1, 2H-1) / (a+H)

and C(a,H) -> 1/(2a+1) as H -> 1/2+, which is exactly the white-noise answer.

This script:
  A. validates the fGn generator exactly (Var of fBm at N is N^{2H})
  B. validates C(a,H) numerically on a grid  (the decisive quantitative test)
  C. measures the ensemble-rms exponent on an (a,H) grid
  D. measures the CAMPAIGN-pipeline exponent (running-max envelope of dev,
     log-log fit over the last two decades) on the same grid, single seed
     (verifier style) and median over seeds
"""
import os, json, time
import numpy as np
from scipy.special import beta as Bfun

HERE = os.path.dirname(os.path.abspath(__file__))
H_STEP = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
Z0 = 1j

t0 = time.time()
out = {"model": "S_N = sum e^{i Phi_{k+1}} kappa_k ; w_k = e^{i Phi_{k+1}} kappa_k"}


# ----------------------------------------------------------------- fGn (Davies-Harte)
def fgn_batch(Hurst, n, rng, nb):
    """Return (nb, n) complex array u with E|u_k|^2 = 1, whose real and imaginary
    parts are independent unit-variance fractional Gaussian noises of Hurst H."""
    if abs(Hurst - 0.5) < 1e-12:
        return (rng.standard_normal((nb, n)) + 1j * rng.standard_normal((nb, n))) / np.sqrt(2)
    k = np.arange(0, n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * Hurst) - 2 * np.abs(k) ** (2 * Hurst)
               + np.abs(k - 1) ** (2 * Hurst))
    row = np.concatenate([g, g[-2:0:-1]])           # circulant first row, length 2n
    m = row.size
    lam = np.fft.fft(row).real
    lam = np.maximum(lam, 0.0)
    amp = np.sqrt(lam / (2.0 * m))
    V = rng.standard_normal((nb, m)) + 1j * rng.standard_normal((nb, m))
    Y = np.fft.fft(amp[None, :] * V, axis=1)
    # Re and Im each have covariance gamma/2 -> multiply by sqrt(2)
    g1 = np.sqrt(2.0) * Y.real[:, :n]
    g2 = np.sqrt(2.0) * Y.imag[:, :n]
    return (g1 + 1j * g2) / np.sqrt(2)              # E|u|^2 = 1


def C_theory(a, Hurst):
    if abs(Hurst - 0.5) < 1e-9:
        return 1.0 / (2 * a + 1.0)
    return Hurst * (2 * Hurst - 1) * Bfun(a + 1.0, 2 * Hurst - 1.0) / (a + Hurst)


# ================================================================= A. generator check
n_chk = 1 << 15
rng = np.random.default_rng(20260831)
A = []
for Hu in (0.5, 0.6, 0.7, 0.8, 0.9):
    u = fgn_batch(Hu, n_chk, rng, 256)
    # u is normalised to E|u|^2 = 1, so u.real has variance 1/2; rescale to a
    # unit-variance fGn before comparing with the exact identity Var(fBm(N)) = N^{2H}
    s = np.cumsum(np.sqrt(2.0) * u.real, axis=1)
    for N in (1 << 10, 1 << 13, 1 << 15):
        v = float(np.mean(s[:, N - 1] ** 2))
        A.append({"H": Hu, "N": N, "var_fBm_measured": v,
                  "var_fBm_exact_N^2H": float(N ** (2 * Hu)),
                  "ratio": v / N ** (2 * Hu)})
out["A_fgn_generator_check"] = A
out["A_note"] = ("Var(fBm(N)) = N^{2H} is exact for fGn; ratios should be 1 within "
                 "the ~1/sqrt(256) ensemble error (~6%).")

# ================================================================= B/C/D. the grid
N = 1 << 19                       # 524288 steps = 25033 gyro-orbits at h=0.3
                                  # (N*h/2pi = 25032.908; the 25046 that stood
                                  #  here until W6.2 was arithmetic in a comment
                                  #  and had reached the manuscript from it)
NSEED = 64
BATCH = 8
A_GRID = [-0.4, -0.25, 0.0, 0.25, 0.4, 0.6]
H_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
KAPPA = 1e-9                      # keeps |S_N| << |z0| = 1 (linear regime)

t_n = np.arange(N) * H_STEP
th = 2 * np.arctan(H_STEP * np.exp(-t_n / TAU_Q) / 2)
Phi = np.concatenate([[0.0], np.cumsum(th)])[:-1]
rot = np.exp(1j * (Phi + th))                        # e^{i Phi_{k+1}}
kidx = np.arange(1, N + 1, dtype=float)
ts = t_n + H_STEP

# sample grid for the ensemble-rms exponent (log-spaced N)
n_probe = np.unique(np.round(np.logspace(np.log10(64), np.log10(N), 60)).astype(int)) - 1


def envelope_exponent(ts_, dev, n_samples=4000, win=100.0):
    """Campaign pipeline, byte-for-byte the verifier's: stride-subsample, running max,
    log-log fit over the last `win` factor in time."""
    stride = max(1, len(ts_) // n_samples)
    idx = np.arange(stride - 1, len(ts_), stride)
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts_[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / win) & (env > 0)
    return float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])


grid = []
for Hu in H_GRID:
    # accumulators per a
    sum_S2 = {a: np.zeros(len(n_probe)) for a in A_GRID}
    env_ex = {a: [] for a in A_GRID}
    env_ex10 = {a: [] for a in A_GRID}
    first_seed_ex = {}
    done = 0
    while done < NSEED:
        nb = min(BATCH, NSEED - done)
        rg = np.random.default_rng(1000 + 17 * done + int(Hu * 1000))
        u = fgn_batch(Hu, N, rg, nb)
        for a in A_GRID:
            w = (KAPPA * kidx ** a)[None, :] * u
            S = np.cumsum(w, axis=1)
            sum_S2[a] += np.sum(np.abs(S[:, n_probe]) ** 2, axis=0)
            zs = Z0 + S
            dev = np.abs(np.abs(zs) ** 2 - 1.0)
            for b in range(nb):
                env_ex[a].append(envelope_exponent(ts, dev[b]))
                env_ex10[a].append(envelope_exponent(ts, dev[b], win=10.0))
            if done == 0:
                first_seed_ex[a] = env_ex[a][0]
        done += nb
    for a in A_GRID:
        rms = np.sqrt(sum_S2[a] / NSEED)
        nn = n_probe + 1.0
        sel = nn > nn[-1] / 100.0
        q = float(np.polyfit(np.log10(nn[sel]), np.log10(rms[sel]), 1)[0])
        # exact constant test at the last probe point
        Nl = nn[-1]
        C_meas = float(rms[-1] ** 2 / (KAPPA ** 2 * Nl ** (2 * (a + Hu))))
        grid.append({
            "a": a, "H": Hu, "a_plus_H": a + Hu,
            "q_rms_ensemble": q,
            "q_minus_aH": q - (a + Hu),
            "C_measured": C_meas, "C_theory": float(C_theory(a, Hu)),
            "C_ratio": C_meas / float(C_theory(a, Hu)),
            "p_env_seed0_verifier_style": first_seed_ex[a],
            "p_env_median_over_seeds": float(np.median(env_ex[a])),
            "p_env_iqr": [float(np.percentile(env_ex[a], 25)),
                          float(np.percentile(env_ex[a], 75))],
            "p_env_median_lastdecade": float(np.median(env_ex10[a])),
        })
    print(f"H={Hu} done  t={time.time()-t0:.0f}s", flush=True)

out["grid_setup"] = {"N_steps": N, "gyros": N * H_STEP / TWO_PI, "nseed": NSEED,
                     "kappa": KAPPA, "z0": "i", "a_grid": A_GRID, "H_grid": H_GRID}
out["BCD_grid"] = grid
out["runtime_s"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "pl_core.json"), "w"), indent=1)

print("\n a      H    a+H   q_rms   q-a-H   C_meas/C_th   p_env(seed0)  p_env(med)  p_env(1dec)")
for r in grid:
    print(f"{r['a']:+.2f} {r['H']:.2f} {r['a_plus_H']:+.3f} {r['q_rms_ensemble']:+.4f} "
          f"{r['q_minus_aH']:+.4f}   {r['C_ratio']:.4f}      "
          f"{r['p_env_seed0_verifier_style']:+.4f}    {r['p_env_median_over_seeds']:+.4f}   "
          f"{r['p_env_median_lastdecade']:+.4f}")
print(f"\nruntime {time.time()-t0:.0f}s")
