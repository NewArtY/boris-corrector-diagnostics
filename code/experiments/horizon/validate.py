"""Validate fast.py against the published corrector_evaluation.json."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import fast as F

TAU = F.TAU_MAIN; DTW = F.DT_WORK; DTF = F.DT_FINE; TF = F.T_FINAL
Ws, bs, xm, xs, ysc, _ = F.load_net_numpy()
fwd = F.make_forward(Ws, bs, xm, xs, ysc)


def full_run(mode, dt, n_steps):
    """Same stepping as fast.run but stores everything (short runs only)."""
    rx, ry, rz = 1.0, 0.0, 0.0
    vx, vy, vz = 0.0, 1.0, 0.0
    t = 0.0; k = -0.5 * dt; inv_tau = 1.0 / TAU
    R = np.zeros((n_steps + 1, 3)); V = np.zeros((n_steps + 1, 3))
    T = np.zeros(n_steps + 1)
    R[0] = (rx, ry, rz); V[0] = (vx, vy, vz)
    x = np.empty(13)
    for i in range(1, n_steps + 1):
        Bz = np.exp(-t * inv_tau); fac = 0.5 * Bz * inv_tau
        Ex = -fac * ry; Ey = fac * rx
        if mode != 'boris':
            x[:] = (rx, ry, rz, vx, vy, vz, 0.0, 0.0, Bz, Ex, Ey, 0.0, dt)
            d = fwd(x)
        kEx = k * Ex; kEy = k * Ey
        vmx = vx + kEx; vmy = vy + kEy; vmz = vz
        tz = k * Bz; sz = 2.0 * tz / (1.0 + tz * tz)
        vpx = vmx + vmy * tz; vpy = vmy - vmx * tz
        vplx = vmx + vpy * sz; vply = vmy - vpx * sz
        vbx = vplx + kEx; vby = vply + kEy; vbz = vmz
        rbx = rx + vbx * dt; rby = ry + vby * dt; rbz = rz + vbz * dt
        if mode == 'boris':
            rx, ry, rz, vx, vy, vz = rbx, rby, rbz, vbx, vby, vbz
        else:
            dvx, dvy, dvz = d[3], d[4], d[5]
            if mode == 'proj':
                nb = np.sqrt(vbx**2 + vby**2 + vbz**2); inb = 1.0 / max(nb, 1e-300)
                hx, hy, hz = vbx*inb, vby*inb, vbz*inb
                dot = dvx*hx + dvy*hy + dvz*hz
                dvx -= dot*hx; dvy -= dot*hy; dvz -= dot*hz
                nvx, nvy, nvz = vbx+dvx, vby+dvy, vbz+dvz
                sc = nb / max(np.sqrt(nvx**2+nvy**2+nvz**2), 1e-300)
                vx, vy, vz = nvx*sc, nvy*sc, nvz*sc
            else:
                vx, vy, vz = vbx+dvx, vby+dvy, vbz+dvz
            rx, ry, rz = rbx+d[0], rby+d[1], rbz+d[2]
        t += dt
        R[i] = (rx, ry, rz); V[i] = (vx, vy, vz); T[i] = t
    return R, V, T


t0 = time.time()
n_fine = int(round(TF / DTF))
ts_r, rs_r, Es_r, E0 = F.fine_reference(TAU, DTF, n_fine, 1)
ts_r = np.concatenate([[0.0], ts_r]); rs_r = np.vstack([[1.0,0,0], rs_r])
Es_r = np.concatenate([[E0], Es_r])
t_fine = time.time() - t0

n_work = int(round(TF / DTW)); half = n_work // 2
out = {}
for mode, key in [('boris','boris'), ('raw','corrector_raw'), ('proj','corrector_projected')]:
    R, V, T = full_run(mode, DTW, n_work)
    Ei = np.interp(T, ts_r, Es_r)
    E = 0.5 * np.sum(V**2, axis=1)
    e_err = np.abs(E - Ei) / E0
    r_ref_i = np.vstack([np.interp(T, ts_r, rs_r[:, j]) for j in range(3)]).T
    pos = np.linalg.norm(R - r_ref_i, axis=1)
    out[key] = {"energy_err_median_2nd_half": float(np.median(e_err[half:])),
                "energy_err_max": float(e_err.max()),
                "pos_err_final": float(pos[-1]),
                "pos_err_rms": float(np.sqrt(np.mean(pos**2)))}

R, V, T = full_run('boris', DTW, n_work)
phys = float(np.median(np.abs((np.interp(T, ts_r, Es_r) - E0) / E0)[half:]))
out["physical_signal_median"] = phys
out["traj_gain_projected"] = out["boris"]["pos_err_rms"] / out["corrector_projected"]["pos_err_rms"]

pub = json.load(open(os.path.join(ROOT, "output_figures", "corrector_evaluation.json")))
print(f"fine reference: {n_fine} steps in {t_fine:.1f}s -> {t_fine/n_fine*1e6:.2f} us/step\n")
print(f"{'поле':<48}{'опубл.':>14}{'быстрая':>14}{'отн.разн.':>12}")
worst = 0.0
def walk(p, q, pre=""):
    global worst
    for k in p:
        if isinstance(p[k], dict): walk(p[k], q[k], pre + k + ".")
        else:
            a, b = float(p[k]), float(q[k])
            rel = abs(a-b)/max(abs(a),1e-300); worst = max(worst, rel)
            print(f"{pre+k:<48}{a:>14.6e}{b:>14.6e}{rel:>12.2e}")
walk(pub, out)
print(f"\nмаксимальное относительное расхождение: {worst:.3e}")
# analytic energy law check
E_an = E0 * np.exp(-ts_r / TAU)
print(f"аналитический закон E0*exp(-t/tau) против мелкого эталона: "
      f"max |отн.разн.| = {np.max(np.abs(Es_r-E_an)/E0):.3e}")
json.dump({"fast_vs_published_max_rel_diff": worst, "fast": out, "published": pub,
           "fine_ref_us_per_step": t_fine/n_fine*1e6,
           "analytic_vs_fine_max_absdiff_over_E0": float(np.max(np.abs(Es_r-E_an)/E0))},
          open(os.path.join(HERE, "validation.json"), "w"), indent=2)
