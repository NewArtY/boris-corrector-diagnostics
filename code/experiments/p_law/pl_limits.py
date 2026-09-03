"""PL-3: limit points, AR(1), frame-dependence, nonlinear regime, and transfer
to the real Boris integrator.

  L1  AR(1) in the co-rotating frame: closed-form Var(S_N), analytic window-fit
      exponent vs measurement.  True H = 1/2; the 'continuum' is a crossover.
  L2  the three sufficient conditions as limit points of p = (a+H)_+
  L3  the clamp: a+H <= 0  =>  p = 0 (summable variance / bounded Weyl sums)
  L4  frame dependence: long memory in the LAB frame does NOT give p = a+H
  L5  linear-regime condition: |S_N| >> |z0| doubles the exponent
  L6  TRANSFER: same law in the full nonlinear Boris integrator (field B4)
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


NG = 1e4
N = int(round(NG * TWO_PI / H_STEP))
t_n, th, Phi = phases(N)
rot = np.exp(1j * (Phi + th))
ts = t_n + H_STEP
KAP = 3.4700895e-07

# ================================================== L1. AR(1) is a crossover, not H
def var_ar1(n, rho):
    """Var(sum_{k<n} w_k) for stationary complex AR(1), E|w|^2 = 1, gamma(m)=rho^|m|."""
    n = np.asarray(n, dtype=float)
    return n * (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ** n) / (1 - rho) ** 2


nn = np.unique(np.round(np.logspace(np.log10(N / 100.0), np.log10(N), 200)).astype(int))
l1 = []
for rho in (0.9, 0.99, 0.999):
    v = var_ar1(nn, rho)
    q_pred = float(np.polyfit(np.log10(nn), 0.5 * np.log10(v), 1)[0])
    # measurement with the campaign pipeline, ensemble
    ex = []
    for s in range(64):
        rg = np.random.default_rng(4242 + s)
        w = (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
        x = np.empty(N, complex)
        acc = w[0] / np.sqrt(1 - rho ** 2)          # stationary start
        for i in range(N):
            acc = rho * acc + w[i]
            x[i] = acc
        x *= KAP * np.sqrt(1 - rho ** 2)
        kk = np.real(x * np.conj(rot))              # co-rotating real kick (verifier's)
        S = np.cumsum(rot * kk)
        dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
        ex.append(envelope_exponent(ts, dev))
    l1.append({"rho": rho, "corr_time_steps": 1.0 / (1 - rho),
               "true_asymptotic_H": 0.5,
               "window_fit_exponent_from_closed_form_Var": q_pred,
               "measured_mean_64_seeds": float(np.mean(ex)),
               "measured_std": float(np.std(ex))})
    print("L1", rho, round(q_pred, 4), round(float(np.mean(ex)), 4), f"{time.time()-t0:.0f}s", flush=True)
out["L1_AR1_is_a_crossover"] = l1
out["L1_verifier_reported"] = {"0.9": 0.466, "0.99": 0.617, "0.999": 0.776}

# ================================================== L2/L3. limit points and clamp
l2 = []


def run_case(label, w, a=None, Hh=None, pred=None, nseed=1):
    S = np.cumsum(w)
    dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
    return {"case": label, "a": a, "H": Hh, "predicted_p": pred,
            "exponent": envelope_exponent(ts, dev)}


# (i) DC in the co-rotating frame  -> a=0, H=1
l2.append(run_case("DC in co-rotating frame (u_k = 1): a=0, H=1", KAP * np.ones(N) + 0j,
                   0.0, 1.0, 1.0))
# (ii) sustained resonance at omega_h in the LAB frame (same thing, other frame)
om_h0 = 2 * np.arctan(H_STEP / 2) / H_STEP
kk = KAP * np.sin(om_h0 * (t_n + 0.5 * H_STEP))
l2.append(run_case("lab-frame resonance at omega_h: a=0, H=1", rot * kk, 0.0, 1.0, 1.0))
# (iii) stationary incoherent -> a=0, H=1/2
exs = []
for s in range(64):
    rg = np.random.default_rng(777 + s)
    u = (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
    S = np.cumsum(KAP * u)
    exs.append(envelope_exponent(ts, np.abs(np.abs(Z0 + S) ** 2 - 1.0)))
l2.append({"case": "stationary incoherent: a=0, H=1/2", "a": 0.0, "H": 0.5,
           "predicted_p": 0.5, "exponent": float(np.mean(exs)),
           "std_64_seeds": float(np.std(exs))})
# (iv) summable variance (self-quenching): a < -1/2 -> clamp
kidx = np.arange(1, N + 1, dtype=float)
for a in (-0.5, -0.6, -0.75, -1.0):
    exs = []
    for s in range(32):
        rg = np.random.default_rng(31337 + s)
        u = (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
        S = np.cumsum(KAP * kidx ** a * u)
        exs.append(envelope_exponent(ts, np.abs(np.abs(Z0 + S) ** 2 - 1.0)))
    l2.append({"case": f"summable-variance incoherent a={a}", "a": a, "H": 0.5,
               "predicted_p": max(0.0, a + 0.5), "exponent": float(np.mean(exs)),
               "std_32_seeds": float(np.std(exs))})
# (v) bounded Weyl sums: H=0
beta = (np.sqrt(5) - 1) / 2                      # golden ratio, badly approximable
u = np.exp(2j * np.pi * beta * np.arange(N))
l2.append(run_case("Weyl sum, golden-ratio rotation (H=0): a=0", KAP * u, 0.0, 0.0, 0.0))
out["L2_limit_points"] = l2
print("L2 done", f"{time.time()-t0:.0f}s", flush=True)

# ================================================== L4. frame dependence
l4 = []
for Hu in (0.5, 0.9):
    for frame in ("co-rotating", "lab"):
        exs = []
        for s in range(24):
            rg = np.random.default_rng(555 + s + int(Hu * 100))
            u = fgn_batch(Hu, N, rg, 1)[0]
            if frame == "co-rotating":
                kk = np.real(u * np.conj(rot))    # long memory placed at omega_h
            else:
                kk = np.real(u)                   # long memory at zero frequency
            S = np.cumsum(KAP * rot * kk)
            exs.append(envelope_exponent(ts, np.abs(np.abs(Z0 + S) ** 2 - 1.0)))
        l4.append({"H_of_raw_noise": Hu, "frame_where_memory_lives": frame,
                   "predicted_p": (Hu if frame == "co-rotating" else 0.5),
                   "measured_mean": float(np.mean(exs)), "std": float(np.std(exs))})
out["L4_frame_dependence"] = l4
print("L4", [round(r["measured_mean"], 3) for r in l4], f"{time.time()-t0:.0f}s", flush=True)

# ================================================== L5. linear-regime condition
l5 = []
for kap in (3.47e-7, 3.47e-5, 3.47e-4, 1.5e-3):
    exs = []
    for s in range(24):
        rg = np.random.default_rng(2024 + s)
        u = (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
        S = np.cumsum(kap * u)
        dev = np.abs(np.abs(Z0 + S) ** 2 - 1.0)
        exs.append(envelope_exponent(ts, dev))
    Sf = np.abs(S[-1])
    l5.append({"kappa": kap, "final_|S_N|/|z0|": float(Sf),
               "exponent": float(np.mean(exs)), "linear_pred": 0.5,
               "quadratic_pred": 1.0})
out["L5_linear_regime"] = l5
print("L5", [round(r["exponent"], 3) for r in l5], f"{time.time()-t0:.0f}s", flush=True)

json.dump(out, open(os.path.join(HERE, "pl_limits.json"), "w"), indent=1)
print("saved", time.time() - t0)
