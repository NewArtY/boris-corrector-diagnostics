"""PL-4: corrected limit points, degeneracy condition, AR(1) horizon, nonlinear
regime, and TRANSFER of p = (a+H)_+ to the full nonlinear Boris integrator.

  M1  limit points done right (generic vs degenerate phase of z0; chirp-free
      resonance; phase-tracking drive)
  M2  degeneracy: if Re(conj(z0) S_N) vanishes identically the exponent doubles
  M3  AR(1) horizon sweep: the elevated exponent decays toward 1/2 as T grows
  M4  linear-regime breakdown: p -> 2(a+H)
  M5  full nonlinear Boris in field B4 with a prescribed-(a,H) per-step kick
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
H_STEP = 0.3
TAU_Q = 1.2e8
TWO_PI = 2 * np.pi
t0 = time.time()
out = {}


def phases(N, tau=TAU_Q):
    t_n = np.arange(N) * H_STEP
    th = 2 * np.arctan(H_STEP * np.exp(-t_n / tau) / 2)
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
Z0 = 1j


def dev_of(w, z0=Z0):
    S = np.cumsum(w)
    return np.abs(np.abs(z0 + S) ** 2 - np.abs(z0) ** 2) / np.abs(z0) ** 2


# ================================================== M1/M2. limit points, degeneracy
m1 = []
# DC in the co-rotating frame, generic phase relative to z0
for phi, tag in ((np.pi / 4, "generic phase pi/4"), (0.0, "degenerate: S _|_ z0")):
    w = KAP * np.exp(1j * phi) * np.ones(N)
    m1.append({"case": f"co-rotating DC, {tag}", "a": 0.0, "H": 1.0,
               "predicted_p": 1.0 if phi else 2.0,
               "exponent": envelope_exponent(ts, dev_of(w))})
# resonance at omega_h with NO chirp (tau -> infinity): true H = 1
t_c, th_c, Phi_c = phases(N, tau=1e30)
rot_c = np.exp(1j * (Phi_c + th_c))
om_h = 2 * np.arctan(H_STEP / 2) / H_STEP
kk = KAP * np.sin(om_h * (t_c + 0.5 * H_STEP))
m1.append({"case": "lab resonance at omega_h, NO chirp (tau=inf): a=0, H=1",
           "a": 0.0, "H": 1.0, "predicted_p": 1.0,
           "exponent": envelope_exponent(t_c + H_STEP, dev_of(rot_c * kk))})
# phase-tracking drive in the chirped field: still H = 1
kk = KAP * np.sin(Phi + 0.5 * th)
m1.append({"case": "phase-tracking drive sin(Phi_n) in the chirped field: a=0, H=1",
           "a": 0.0, "H": 1.0, "predicted_p": 1.0,
           "exponent": envelope_exponent(ts, dev_of(rot * kk))})
# fixed-frequency resonance WITH chirp: H is not 1 over this window (Fresnel stall)
kk = KAP * np.sin(om_h * (t_n + 0.5 * H_STEP))
m1.append({"case": "lab resonance at omega_h(0) WITH chirp (campaign config)",
           "a": 0.0, "H": "1 up to T*, 0 after", "predicted_p": "window average",
           "exponent": envelope_exponent(ts, dev_of(rot * kk))})
# growing-amplitude coherent drive: a=0.5, H=1 -> p=1.5
for a in (0.25, 0.5):
    kidx = np.arange(1, N + 1, dtype=float)
    w = KAP * (kidx / N) ** a * np.exp(1j * np.pi / 4)
    m1.append({"case": f"co-rotating DC with growing amplitude a={a}, H=1",
               "a": a, "H": 1.0, "predicted_p": 1.0 + a,
               "exponent": envelope_exponent(ts, dev_of(w))})
out["M1_limit_points_and_degeneracy"] = m1
print("M1", [round(r["exponent"], 4) for r in m1], f"{time.time()-t0:.0f}s", flush=True)

# ================================================== M3. AR(1) horizon sweep
def var_ar1(n, rho):
    n = np.asarray(n, dtype=float)
    return n * (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ** n) / (1 - rho) ** 2


m3 = []
for rho in (0.9, 0.99, 0.999):
    for gy in (1e3, 1e4, 1e5, 1e6, 1e7):
        Nx = int(round(gy * TWO_PI / H_STEP))
        nn = np.unique(np.round(np.logspace(np.log10(Nx / 100.0), np.log10(Nx), 200)).astype(int))
        q = float(np.polyfit(np.log10(nn), 0.5 * np.log10(var_ar1(nn, rho)), 1)[0])
        m3.append({"rho": rho, "gyros": gy, "N_steps": Nx,
                   "window_fit_exponent_closed_form": q})
out["M3_AR1_horizon_sweep"] = m3
out["M3_note"] = ("AR(1) has exponentially decaying memory: its true Hurst exponent is "
                  "exactly 1/2.  The elevated exponents are the ballistic->diffusive "
                  "crossover seen inside a finite fit window; they decay to 1/2 as T grows.")

# ================================================== M4. linear-regime breakdown
m4 = []
for kap in (3.47e-7, 3.47e-4, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1):
    exs, Sfin = [], []
    for s in range(16):
        rg = np.random.default_rng(2024 + s)
        u = (rg.standard_normal(N) + 1j * rg.standard_normal(N)) / np.sqrt(2)
        S = np.cumsum(kap * u)
        Sfin.append(abs(S[-1]))
        exs.append(envelope_exponent(ts, np.abs(np.abs(Z0 + S) ** 2 - 1.0)))
    m4.append({"kappa": kap, "median_final_|S_N|/|z0|": float(np.median(Sfin)),
               "exponent_mean": float(np.mean(exs)),
               "linear_pred_(a+H)": 0.5, "quadratic_pred_2(a+H)": 1.0})
out["M4_linear_regime_breakdown"] = m4
print("M4", [round(r["exponent_mean"], 3) for r in m4], f"{time.time()-t0:.0f}s", flush=True)

# ================================================== M5. full nonlinear Boris, field B4
def boris_run(kick_xy, r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0), tau=TAU_Q, B0=1.0):
    """kick_xy: (nb, N, 2) added to v after each Boris update. Returns (nb, N) |v|^2/|v0|^2."""
    nb, Nn, _ = kick_xy.shape
    q = -1.0
    r = np.tile(np.array(r0, float), (nb, 1))
    v = np.tile(np.array(v0, float), (nb, 1))
    E2 = np.empty((nb, Nn))
    for n in range(Nn):
        tt = n * H_STEP
        Bz = B0 * np.exp(-tt / tau)
        fac = Bz / (2.0 * tau)
        Ex = -fac * r[:, 1]
        Ey = fac * r[:, 0]
        hq = q * 0.5 * H_STEP
        vmx = v[:, 0] + hq * Ex
        vmy = v[:, 1] + hq * Ey
        tz = hq * Bz
        sz = 2 * tz / (1 + tz * tz)
        vpx = vmx + vmy * tz
        vpy = vmy - vmx * tz
        v[:, 0] = vmx + vpy * sz + hq * Ex
        v[:, 1] = vmy - vpx * sz + hq * Ey
        v[:, 0] += kick_xy[:, n, 0]
        v[:, 1] += kick_xy[:, n, 1]
        r[:, 0] += v[:, 0] * H_STEP
        r[:, 1] += v[:, 1] * H_STEP
        E2[:, n] = v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2
    return E2


NB_G = 3e3                                   # gyro-orbits for the Boris transfer test
NB = int(round(NB_G * TWO_PI / H_STEP))
tb, thb, Phib = phases(NB)
tsb = tb + H_STEP
Eref = np.exp(-tb / TAU_Q)                   # adiabatic reference E_kin/E0

# baseline: measure the actual numerical gyrophase, so the co-rotating frame is
# defined by the integrator itself, not by an assumed sign convention
base = boris_run(np.zeros((1, NB, 2)))
out["M5_baseline_floor_median"] = float(np.median(np.abs(base[0] - Eref)))
# recover zeta_n = v_x + i v_y of the baseline run
zb = np.empty(NB, complex)
rr = np.array([1.0, 0.0, 0.0]); vv = np.array([0.0, 1.0, 0.0])
for n in range(NB):
    tt = n * H_STEP
    Bz = np.exp(-tt / TAU_Q); fac = Bz / (2.0 * TAU_Q)
    hq = -0.5 * H_STEP
    Ex, Ey = -fac * rr[1], fac * rr[0]
    vmx, vmy = vv[0] + hq * Ex, vv[1] + hq * Ey
    tz = hq * Bz; sz = 2 * tz / (1 + tz * tz)
    vpx, vpy = vmx + vmy * tz, vmy - vmx * tz
    vv[0] = vmx + vpy * sz + hq * Ex
    vv[1] = vmy - vpx * sz + hq * Ey
    rr[0] += vv[0] * H_STEP; rr[1] += vv[1] * H_STEP
    zb[n] = vv[0] + 1j * vv[1]
uhat = zb / np.abs(zb)

KB = 2.0e-8
m5 = []
for (a, Hu) in ((0.0, 0.5), (0.25, 0.5), (0.4, 0.5), (0.0, 0.8), (0.25, 0.8), (0.0, 1.0)):
    NSD = 12 if Hu != 1.0 else 1
    rg = np.random.default_rng(31415 + int(100 * a) + int(1000 * Hu))
    if Hu == 1.0:
        w = np.exp(1j * np.pi / 4) * np.ones((1, NB))
    else:
        w = fgn_batch(Hu, NB, rg, NSD)
    sig = KB * ((np.arange(1, NB + 1) / NB) ** a)
    kap_lab = w * sig[None, :] * uhat[None, :]        # co-rotating -> lab
    kick = np.stack([kap_lab.real, kap_lab.imag], axis=2)
    E2 = boris_run(kick)
    exs = [envelope_exponent(tsb, np.abs(E2[b] - Eref)) for b in range(NSD)]
    m5.append({"a": a, "H": Hu, "predicted_p": a + Hu, "n_seeds": NSD,
               "exponent_mean": float(np.mean(exs)), "exponent_std": float(np.std(exs)),
               "final_dev": float(np.abs(E2[0, -1] - Eref[-1]))})
    print("  M5", a, Hu, round(float(np.mean(exs)), 4), f"{time.time()-t0:.0f}s", flush=True)
out["M5_full_nonlinear_boris"] = m5
out["M5_setup"] = {"gyros": NB_G, "h": H_STEP, "tau": TAU_Q, "kick_amplitude": KB,
                   "note": "kick placed in the frame co-rotating with the measured "
                           "numerical gyrophase of the unperturbed Boris run"}

json.dump(out, open(os.path.join(HERE, "pl_transfer.json"), "w"), indent=1)
print("saved", time.time() - t0)
