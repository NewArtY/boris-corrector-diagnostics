"""PL-6: the certification protocol.  Given a recorded per-step defect series,
recover (a, H) and PREDICT the long-horizon exponent from a SHORT run.

Protocol:
  1. demodulate:  w_k = e^{i Phi_{k+1}} kappa_k
  2. a_hat : slope of log rms|w| over log-spaced bins   (amplitude growth)
  3. u_k = w_k / (k^a_hat);  H_hat : slope of log|sum_{k<=n} u_k| (ensemble rms
     over log-spaced n)                                 (phase coherence)
  4. predict p = max(0, a_hat + H_hat)

Validation: synthetic w with known (a,H); estimate on the FIRST 1/16 of the run,
predict, then compare with the exponent actually measured on the FULL run.
"""
import os, json, time
import numpy as np

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


def estimate_aH(w, nbin=24):
    """w: (nb, n) demodulated defect.  Returns (a_hat, H_hat)."""
    nb, n = w.shape
    edges = np.unique(np.round(np.logspace(np.log10(max(32, n // 400)), np.log10(n), nbin)).astype(int))
    # --- a: rms|w| per log bin
    ctr, val = [], []
    for i in range(1, len(edges)):
        lo, hi = edges[i - 1], edges[i]
        if hi - lo < 8:
            continue
        ctr.append(np.sqrt(lo * hi))
        val.append(np.sqrt(np.mean(np.abs(w[:, lo:hi]) ** 2)))
    ctr, val = np.array(ctr), np.array(val)
    a_hat = float(np.polyfit(np.log10(ctr), np.log10(val), 1)[0])
    # --- H: partial sums of the amplitude-normalised sequence
    k = np.arange(1, n + 1, dtype=float)
    u = w / (k ** a_hat)[None, :]
    S = np.cumsum(u, axis=1)
    npts = np.unique(np.round(np.logspace(np.log10(max(32, n // 1000)), np.log10(n), 40)).astype(int)) - 1
    rms = np.sqrt(np.mean(np.abs(S[:, npts]) ** 2, axis=0))
    sel = (npts + 1) > (npts[-1] + 1) / 100.0
    H_hat = float(np.polyfit(np.log10(npts[sel] + 1.0), np.log10(rms[sel]), 1)[0])
    return a_hat, H_hat


def envelope_exponent(ts_, dev, n_samples=4000, win=100.0):
    stride = max(1, len(ts_) // n_samples)
    idx = np.arange(stride - 1, len(ts_), stride)
    run = np.maximum.reduceat(dev, np.arange(0, len(dev), stride))
    m = min(len(idx), len(run))
    tw, env = ts_[idx][:m], np.maximum.accumulate(run[:m])
    sel = (tw > tw[-1] / win) & (env > 0)
    return float(np.polyfit(np.log10(tw[sel]), np.log10(env[sel]), 1)[0])


N = 1 << 19
NSD = 32
KAP = 1e-9
t_n = np.arange(N) * H_STEP
ts = t_n + H_STEP
kidx = np.arange(1, N + 1, dtype=float)
SHORT = N >> 4                          # short run = 1/16 of the horizon

rows = []
for (a, Hu) in ((0.0, 0.5), (0.25, 0.5), (0.4, 0.5), (-0.25, 0.5),
                (0.0, 0.8), (0.25, 0.8), (0.0, 0.7), (0.5, 1.0)):
    if Hu == 1.0:
        w = np.exp(1j * np.pi / 4) * np.ones((NSD, N))
    else:
        rg = np.random.default_rng(80000 + int(100 * a) + int(1000 * Hu))
        w = fgn_batch(Hu, N, rg, NSD)
    w = w * (KAP * kidx ** a)[None, :]
    a_s, H_s = estimate_aH(w[:, :SHORT])
    a_f, H_f = estimate_aH(w)
    S = np.cumsum(w, axis=1)
    ex = [envelope_exponent(ts, np.abs(np.abs(Z0 + S[b]) ** 2 - 1.0)) for b in range(NSD)]
    rows.append({"a_true": a, "H_true": Hu, "p_true": a + Hu,
                 "a_hat_short": a_s, "H_hat_short": H_s,
                 "p_pred_from_short_run": max(0.0, a_s + H_s),
                 "a_hat_full": a_f, "H_hat_full": H_f,
                 "p_pred_from_full_run": max(0.0, a_f + H_f),
                 "p_measured_envelope_mean": float(np.mean(ex)),
                 "p_measured_std": float(np.std(ex)),
                 "short_run_gyros": SHORT * H_STEP / TWO_PI,
                 "full_run_gyros": N * H_STEP / TWO_PI})
    r = rows[-1]
    print(f"a={a:+.2f} H={Hu:.1f}  true p={a+Hu:.3f} | short: a={a_s:+.3f} H={H_s:.3f} "
          f"p={r['p_pred_from_short_run']:.3f} | full: p={r['p_pred_from_full_run']:.3f} "
          f"| envelope {np.mean(ex):.3f}+-{np.std(ex):.3f}  t={time.time()-t0:.0f}s", flush=True)

json.dump({"protocol": rows, "note":
           "a and H estimated from the demodulated defect on a run 16x shorter than "
           "the horizon; p = (a+H)_+ then predicts the envelope exponent measured on "
           "the full horizon."},
          open(os.path.join(HERE, "pl_protocol.json"), "w"), indent=1)
