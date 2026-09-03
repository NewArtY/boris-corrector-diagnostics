"""W0.2 -- test the law p = (a+H)_+ against the one unexplained campaign number 1.540.

Two channels are measured on the SAME trained hybrid
(checkpoints/boris_corrector_b4.pt, modes 'proj'/'sym', base 'shipped'):

CHANNEL V -- the velocity defect, exactly as the linearised model defines it
    kappa_n = v_{n+1}^hybrid - Boris(v_n^hybrid; E(r_n), B(t_n))
  demodulated with the measured numerical gyrophase (P_LAW U3), in two frames:
    A: the unperturbed pure-Boris run  (theory-faithful reference propagator)
    B: the hybrid's own pre-projection Boris velocity (co-moving frame)
  then (a,H) by the P_LAW 6.2 protocol, p = (a+H)_+.

CHANNEL E -- the energy-increment defect.  Exactly, with no modelling step,
    dev_n = |sum_{k<=n} g_k|,   g_k = (E_k - E_{k-1})/E0 - (Ephys_k - Ephys_{k-1})/E0
  so the campaign's envelope IS the running max of a partial-sum modulus, and
  the same law applies to g: p = (a+H)_+ with a = amplitude-growth exponent of
  |g| and H = self-similarity exponent of its partial sums.  In the linear
  non-degenerate regime g_n = 2 Re(conj(z0) w_n) and the two channels coincide;
  channel E survives when the velocity defect is energy-neutral by construction.

The g-channel estimator is validated on synthetic defects with known (a,H).
"""
import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "symproj"))

import torch  # noqa: E402
torch.set_default_dtype(torch.float64)
import symproj as S  # noqa: E402

TWO_PI = 2.0 * np.pi
DT = S.DT_WORK                      # 0.3
TAU_PAPER = S.TAU_MAIN              # 1.2e5
TAU_QUASI = 1.2e8
N_GYR = float(os.environ.get("PC_GYROS", 100000))
N_STEPS = int(round(N_GYR * TWO_PI / DT))
t_start = time.time()

CAMPAIGN = {"paper/proj": 1.539542116411257, "paper/sym": 1.5403467739519194,
            "quasistatic/proj": 0.9772017548657451,
            "quasistatic/sym": 0.975730662079118}


# --------------------------------------------------------------------------
def run_instrumented(mode, tau, n_steps, fwd, dt=DT, B0=1.0,
                     r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0),
                     mu_tol=1e-15, mu_maxit=6):
    """Same arithmetic as symproj.run(base='shipped'), plus per-step recording."""
    rx, ry, rz = map(float, r0)
    vx, vy, vz = map(float, v0)
    t = 0.0
    k = -0.5 * dt
    inv_tau = 1.0 / tau
    E0 = 0.5 * (vx * vx + vy * vy + vz * vz)

    kap = np.empty(n_steps, complex)   # per-step velocity defect (lab, xy)
    zb = np.empty(n_steps, complex)    # unperturbed Boris post-update velocity
    zz = np.empty(n_steps, complex)    # actual post-update velocity
    rr_ = np.empty(n_steps, complex)   # position (xy)
    sgn = np.empty(n_steps)            # SIGNED (E - Ephys)/E0
    drm = np.empty(n_steps)            # |network position defect|
    dvpar = np.empty(n_steps)          # raw dv component along v_hat (killed by proj)
    norm_err = 0.0                     # max |  |v_{n+1}| - |v_Boris(v_n)| |
    buf = np.empty(13)
    mu = 0.0

    for i in range(n_steps):
        Bz = B0 * np.exp(-t * inv_tau)
        fac = 0.5 * Bz * inv_tau
        Ex = -fac * ry
        Ey = fac * rx

        vb0x, vb0y, vb0z = S.boris_kick(vx, vy, vz, Ex, Ey, Bz, k)
        target = np.sqrt(vb0x * vb0x + vb0y * vb0y + vb0z * vb0z)

        if mode == "boris":
            vnx, vny, vnz = vb0x, vb0y, vb0z
            vdx, vdy, vdz = vb0x, vb0y, vb0z
            drx = dry = drz = 0.0
            dpar = 0.0
        elif mode == "sym":
            a_ = 1.0 + 2.0 * mu
            S.features(rx, ry, rz, a_ * vx, a_ * vy, a_ * vz, Bz, Ex, Ey, dt, buf)
            d_frozen = fwd(buf).copy()

            def residual(m_):
                aa = 1.0 + 2.0 * m_
                bb = 1.0 / (1.0 - 2.0 * m_)
                bx_, by_, bz_ = S.boris_kick(aa * vx, aa * vy, aa * vz, Ex, Ey, Bz, k)
                nx_ = bb * (bx_ + d_frozen[3])
                ny_ = bb * (by_ + d_frozen[4])
                nz_ = bb * (bz_ + d_frozen[5])
                return (np.sqrt(nx_ * nx_ + ny_ * ny_ + nz_ * nz_) - target,
                        nx_, ny_, nz_, bx_, by_, bz_)

            m0 = mu
            f0 = residual(m0)[0]
            m1 = m0 - f0 / (4.0 * max(target, 1e-300))
            f1, nx, ny, nz, bx_, by_, bz_ = residual(m1)
            it = 2
            while abs(f1) > mu_tol * max(target, 1.0) and it < mu_maxit:
                den = f1 - f0
                if abs(den) < 1e-300:
                    break
                m2 = m1 - f1 * (m1 - m0) / den
                m0, f0 = m1, f1
                m1 = m2
                f1, nx, ny, nz, bx_, by_, bz_ = residual(m1)
                it += 1
            mu = m1
            vnx, vny, vnz = nx, ny, nz
            vdx, vdy, vdz = bx_, by_, bz_
            drx, dry, drz = d_frozen[0], d_frozen[1], d_frozen[2]
            dpar = ((d_frozen[3] * vb0x + d_frozen[4] * vb0y + d_frozen[5] * vb0z)
                    / max(target, 1e-300))
        else:                                   # 'proj' -- the shipped hybrid
            S.features(rx, ry, rz, vx, vy, vz, Bz, Ex, Ey, dt, buf)
            d = fwd(buf)
            dvx, dvy, dvz = d[3], d[4], d[5]
            nb = target
            inb = 1.0 / max(nb, 1e-300)
            hx, hy, hz = vb0x * inb, vb0y * inb, vb0z * inb
            dot = dvx * hx + dvy * hy + dvz * hz
            dpar = dot
            dvx -= dot * hx
            dvy -= dot * hy
            dvz -= dot * hz
            nvx = vb0x + dvx
            nvy = vb0y + dvy
            nvz = vb0z + dvz
            nn = np.sqrt(nvx * nvx + nvy * nvy + nvz * nvz)
            sc = nb / max(nn, 1e-300)
            vnx, vny, vnz = nvx * sc, nvy * sc, nvz * sc
            vdx, vdy, vdz = vb0x, vb0y, vb0z
            drx, dry, drz = d[0], d[1], d[2]

        rx += vdx * dt
        ry += vdy * dt
        rz += vdz * dt
        rx += drx
        ry += dry
        rz += drz

        vx, vy, vz = vnx, vny, vnz
        t += dt

        kap[i] = (vnx - vb0x) + 1j * (vny - vb0y)
        zb[i] = vb0x + 1j * vb0y
        zz[i] = vx + 1j * vy
        rr_[i] = rx + 1j * ry
        drm[i] = np.sqrt(drx * drx + dry * dry)
        dvpar[i] = dpar
        e = abs(np.sqrt(vnx * vnx + vny * vny + vnz * vnz) - target)
        if e > norm_err:
            norm_err = e
        Ecur = 0.5 * (vx * vx + vy * vy + vz * vz)
        sgn[i] = (Ecur - E0 * np.exp(-t * inv_tau)) / E0

    return {"kappa": kap, "zb": zb, "z": zz, "r": rr_, "signed": sgn,
            "dr": drm, "dv_parallel": dvpar, "E0": E0,
            "max_norm_violation": norm_err}


# --------------------------------------------------------------------------
def envelope_exponent_from_series(dev, dt=DT, n_samples=4000, decades=2.0):
    """symproj.run + symproj.envelope_exponent reproduced on a full series."""
    n = len(dev)
    stride = max(1, n // n_samples)
    starts = np.arange(0, n, stride)
    run = np.maximum.reduceat(dev, starts)
    ends = np.append(starts[1:], n)
    t = ends * dt
    env = np.maximum.accumulate(run)
    sel = (t > t[-1] / 10.0 ** decades) & (env > 0)
    if sel.sum() < 10:
        return float("nan"), t, env
    return float(np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)[0]), t, env


def estimate_aH(w, nbin=24, decades_a=None):
    """P_LAW 6.2 protocol.  w: (nb, n).  decades_a limits the a-fit window."""
    w = np.atleast_2d(w)
    nb, n = w.shape
    lo0 = max(32, n // 400) if decades_a is None else max(32, int(n / 10 ** decades_a))
    edges = np.unique(np.round(np.logspace(np.log10(lo0), np.log10(n), nbin)).astype(int))
    ctr, val = [], []
    for i in range(1, len(edges)):
        lo, hi = edges[i - 1], edges[i]
        if hi - lo < 8:
            continue
        ctr.append(np.sqrt(lo * hi))
        val.append(np.sqrt(np.mean(np.abs(w[:, lo:hi]) ** 2)))
    ctr, val = np.array(ctr), np.array(val)
    a_hat = float(np.polyfit(np.log10(ctr), np.log10(val), 1)[0])
    k = np.arange(1, n + 1, dtype=float)
    u = w / (k ** a_hat)[None, :]
    Su = np.cumsum(u, axis=1)
    npts = np.unique(np.round(np.logspace(np.log10(max(32, n // 1000)),
                                          np.log10(n), 40)).astype(int)) - 1
    rms = np.sqrt(np.mean(np.abs(Su[:, npts]) ** 2, axis=0))
    sel = ((npts + 1) > (npts[-1] + 1) / 100.0) & (rms > 0)
    H_hat = float(np.polyfit(np.log10(npts[sel] + 1.0), np.log10(rms[sel]), 1)[0])
    return a_hat, H_hat


def loglog_slope(x, y, decades=2.0):
    m = (x > x[-1] / 10.0 ** decades) & (y > 0) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    return float(np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0])


def sub_index(n, npts=400):
    return np.unique(np.round(np.logspace(np.log10(max(64, n // 4000)),
                                          np.log10(n), npts)).astype(int)) - 1


# --------------------------------------------------------------------------
def protocol_block(w, n, tag):
    """(a,H) by the protocol, on the full run and on the first 1/16."""
    SHORT = n >> 4
    a_s, H_s = estimate_aH(w[:SHORT][None, :])
    a_f, H_f = estimate_aH(w[None, :])
    a_w, H_w = estimate_aH(w[None, :], decades_a=2.0)
    Sw = np.cumsum(w)
    t = np.arange(1, n + 1, dtype=float) * DT
    sub = sub_index(n)
    return {
        "a_hat_short": a_s, "H_hat_short": H_s,
        "p_pred_short_(a+H)": max(0.0, a_s + H_s),
        "a_hat_full": a_f, "H_hat_full": H_f,
        "p_pred_full_(a+H)": max(0.0, a_f + H_f),
        "a_hat_last2dec": a_w, "H_hat_last2dec": H_w,
        "p_pred_last2dec_(a+H)": max(0.0, a_w + H_w),
        "slope_|S_n|_last2dec": loglog_slope(t[sub], np.abs(Sw[sub])),
        "|S_N|": float(abs(Sw[-1])),
        "|Re S_N|/|S_N|": float(abs(np.real(Sw[-1])) / max(abs(Sw[-1]), 1e-300)),
        "mean|Re w|/mean|w|": float(np.mean(np.abs(np.real(w))) /
                                    max(np.mean(np.abs(w)), 1e-300)),
    }


def analyse(tag, res, base):
    kap = res["kappa"]
    n = len(kap)
    t = np.arange(1, n + 1, dtype=float) * DT
    sub = sub_index(n)
    dev = np.abs(res["signed"])

    out = {"tag": tag, "n_steps": n, "gyros": n * DT / TWO_PI,
           "campaign_exponent": CAMPAIGN.get(tag)}
    p_meas, t_env, env = envelope_exponent_from_series(dev)
    out["p_measured_envelope"] = p_meas
    out["dev_final"] = float(dev[-1])
    out["max_norm_violation_|v|-|v_Boris|"] = res["max_norm_violation"]
    out["kappa_rms"] = float(np.sqrt(np.mean(np.abs(kap) ** 2)))
    out["dr_rms"] = float(np.sqrt(np.mean(res["dr"] ** 2)))
    out["dv_parallel_rms"] = float(np.sqrt(np.mean(res["dv_parallel"] ** 2)))

    # ---- trajectory / phase divergence from the unperturbed run
    dr_traj = np.abs(res["r"] - base["r"])
    out["traj_error_final"] = float(dr_traj[-1])
    out["traj_error_slope_last2dec"] = loglog_slope(t[sub], dr_traj[sub])
    dphi = np.unwrap(np.angle(res["z"]) - np.angle(base["z"]))
    out["gyrophase_drift_rad_final"] = float(dphi[-1])
    out["gyrophase_drift_slope_last2dec"] = loglog_slope(t[sub], np.abs(dphi[sub]))

    # ---- CHANNEL V: demodulated velocity defect
    uA = base["zb"] / np.abs(base["zb"])
    uB = res["zb"] / np.abs(res["zb"])
    for fr, u in (("V_frameA_unperturbed", uA), ("V_frameB_comoving", uB)):
        w = kap * np.conj(u)
        blk = protocol_block(w, n, fr)
        Sw = np.cumsum(w)
        dev_rec = np.abs(2.0 * np.real(Sw) + np.abs(Sw) ** 2)
        pr, _, _ = envelope_exponent_from_series(dev_rec)
        blk["p_reconstructed_envelope"] = pr
        blk["reconstruction_ratio_final"] = float(dev_rec[-1] / max(dev[-1], 1e-300))
        out[fr] = blk

    # ---- CHANNEL E: energy-increment defect (exact, dev = |sum g|)
    g = np.diff(np.concatenate([[0.0], res["signed"]]))
    blk = protocol_block(g.astype(complex), n, "E")
    Sg = np.cumsum(g)
    blk["identity_check_max|dev-|sum g||"] = float(np.max(np.abs(np.abs(Sg) - dev)))
    blk["g_rms"] = float(np.sqrt(np.mean(g ** 2)))
    blk["p_measured_envelope"] = p_meas
    out["E_energy_increment"] = blk
    return out, t_env, env


# --------------------------------------------------------------------------
def validate_g_channel():
    """Does the g-channel protocol recover a+H on synthetic defects?
    Non-degenerate additive defect, linear regime: g_n = 2 Re(conj(z0) w_n)."""
    rows = []
    N = 1 << 19
    kidx = np.arange(1, N + 1, dtype=float)
    z0 = 1j
    for (a, Hu) in ((0.0, 0.5), (0.25, 0.5), (0.4, 0.5), (0.0, 0.8),
                    (0.25, 0.8), (0.5, 1.0), (0.0, 1.0)):
        if Hu == 1.0:
            w = np.exp(1j * np.pi / 4) * np.ones(N)
        else:
            rg = np.random.default_rng(4242 + int(100 * a) + int(1000 * Hu))
            w = _fgn(Hu, N, rg)
        w = w * (1e-9 * kidx ** a)
        Sw = np.cumsum(w)
        sgn = 2.0 * np.real(np.conj(z0) * Sw) + np.abs(Sw) ** 2
        g = np.diff(np.concatenate([[0.0], sgn]))
        a_g, H_g = estimate_aH(g[None, :].astype(complex))
        p_env, _, _ = envelope_exponent_from_series(np.abs(sgn))
        rows.append({"a_true": a, "H_true": Hu, "p_true": a + Hu,
                     "a_hat_g": a_g, "H_hat_g": H_g,
                     "p_pred_g_channel": max(0.0, a_g + H_g),
                     "p_measured_envelope": p_env})
    return rows


def _fgn(Hu, n, rng):
    if abs(Hu - 0.5) < 1e-12:
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    k = np.arange(0, n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * Hu) - 2 * np.abs(k) ** (2 * Hu)
               + np.abs(k - 1) ** (2 * Hu))
    row = np.concatenate([g, g[-2:0:-1]])
    m = row.size
    lam = np.maximum(np.fft.fft(row).real, 0.0)
    amp = np.sqrt(lam / (2.0 * m))
    V = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    Y = np.fft.fft(amp * V)
    return (np.sqrt(2.0) * Y.real[:n] + 1j * np.sqrt(2.0) * Y.imag[:n]) / np.sqrt(2)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fwd = S.load_forward()
    results = {"setup": {"dt": DT, "gyros": N_GYR, "n_steps": N_STEPS,
                         "tau_paper": TAU_PAPER, "tau_quasistatic": TAU_QUASI,
                         "checkpoint": "boris_corrector_b4.pt", "base": "shipped",
                         "campaign_exponents": CAMPAIGN},
               "validation_g_channel": validate_g_channel(),
               "runs": {}}
    print("g-channel validation:", flush=True)
    for r in results["validation_g_channel"]:
        print(f"   true({r['a_true']:+.2f},{r['H_true']:.1f}) p={r['p_true']:.3f} -> "
              f"a={r['a_hat_g']:+.3f} H={r['H_hat_g']:.3f} p_pred={r['p_pred_g_channel']:.3f} "
              f"| envelope {r['p_measured_envelope']:.3f}", flush=True)

    store = {}
    for cname, tau in (("paper", TAU_PAPER), ("quasistatic", TAU_QUASI)):
        t0 = time.time()
        base = run_instrumented("boris", tau, N_STEPS, fwd)
        print(f"[{cname}] baseline boris {time.time()-t0:.1f}s", flush=True)
        for mode in ("proj", "sym"):
            t1 = time.time()
            res = run_instrumented(mode, tau, N_STEPS, fwd)
            rep, t_env, env = analyse(f"{cname}/{mode}", res, base)
            rep["seconds"] = time.time() - t1
            results["runs"][f"{cname}/{mode}"] = rep
            E = rep["E_energy_increment"]
            A = rep["V_frameA_unperturbed"]
            B = rep["V_frameB_comoving"]
            print(f"[{cname}/{mode}] {time.time()-t1:.0f}s p_meas={rep['p_measured_envelope']:.4f} "
                  f"(campaign {rep['campaign_exponent']:.4f})\n"
                  f"    E-channel : a={E['a_hat_full']:+.4f} H={E['H_hat_full']:.4f} "
                  f"a+H={E['p_pred_full_(a+H)']:.4f} | short a={E['a_hat_short']:+.4f} "
                  f"H={E['H_hat_short']:.4f} a+H={E['p_pred_short_(a+H)']:.4f} | "
                  f"2dec a+H={E['p_pred_last2dec_(a+H)']:.4f}\n"
                  f"    V-frameA  : a={A['a_hat_full']:+.4f} H={A['H_hat_full']:.4f} "
                  f"a+H={A['p_pred_full_(a+H)']:.4f} recon={A['p_reconstructed_envelope']:.3f} "
                  f"ratio={A['reconstruction_ratio_final']:.2e}\n"
                  f"    V-frameB  : a={B['a_hat_full']:+.4f} H={B['H_hat_full']:.4f} "
                  f"a+H={B['p_pred_full_(a+H)']:.4f} recon={B['p_reconstructed_envelope']:.3f} "
                  f"ratio={B['reconstruction_ratio_final']:.2e}\n"
                  f"    |v| violation={rep['max_norm_violation_|v|-|v_Boris|']:.2e} "
                  f"traj_err={rep['traj_error_final']:.3e} "
                  f"phase_drift={rep['gyrophase_drift_rad_final']:.3e} rad",
                  flush=True)
            step = max(1, N_STEPS // 4000)
            store[f"{cname}/{mode}/t_env"] = t_env
            store[f"{cname}/{mode}/env"] = env
            store[f"{cname}/{mode}/signed_sub"] = res["signed"][::step]
            store[f"{cname}/{mode}/kappa_sub"] = res["kappa"][::step]
            store[f"{cname}/{mode}/zb_base_sub"] = base["zb"][::step]
            del res
        del base

    json.dump(results, open(os.path.join(HERE, "pc_defect.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "pc_defect.npz"), **store)
    print("total", time.time() - t_start)
