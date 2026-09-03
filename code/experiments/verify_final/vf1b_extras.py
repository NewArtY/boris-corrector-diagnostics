"""vf1b: (a) is the 'optimal' floor 4.585e-14 a real quantity or double-precision
noise?  Redo the run at r0_opt in 50-digit arithmetic (mpmath).  (b) T4 map with
the quasistatic chirp tau=1.2e8 (the v_caveat2 setting) to check T* plateau."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---------------------------------------------------------------- (a) mpmath
import mpmath as mp
mp.mp.dps = 50
TAU = mp.mpf("1.2e5"); H = mp.mpf("0.3"); Q = mp.mpf(-1)

def boris_mp(r0x, r0y, n):
    r = [mp.mpf(r0x), mp.mpf(r0y)]
    v = [mp.mpf(0), mp.mpf(1)]
    t = mp.mpf(0)
    devs = []
    for i in range(n):
        Bz = mp.e ** (-t / TAU)
        f = Bz / (2 * TAU)
        E = [-f * r[1], f * r[0]]
        k = Q * H / 2
        vm = [v[0] + k * E[0], v[1] + k * E[1]]
        tz = k * Bz
        sz = 2 * tz / (1 + tz * tz)
        vpx = vm[0] + vm[1] * tz
        vpy = vm[1] - vm[0] * tz
        v = [vm[0] + vpy * sz + k * E[0], vm[1] - vpx * sz + k * E[1]]
        r = [r[0] + v[0] * H, r[1] + v[1] * H]
        t += H
        sp2 = v[0] * v[0] + v[1] * v[1]
        devs.append(abs(sp2 - mp.e ** (-t / TAU)))
    return devs

n = 400
for lbl, x0, y0 in [("claimed_opt", "0.9999987", "0.1500041"),
                    ("gc_shift", "1.0", "0.15")]:
    devs = boris_mp(x0, y0, n)
    second = devs[n // 2:]
    second_sorted = sorted(second)
    med = second_sorted[len(second) // 2]
    out[f"mp_floor_{lbl}"] = {"median_2nd_half": float(med),
                              "max": float(max(devs))}

# true 2-D optimum in exact arithmetic? crude local probe around claimed opt
base = out["mp_floor_claimed_opt"]["median_2nd_half"]
probe = {}
for dx, dy in [(1e-6, 0), (-1e-6, 0), (0, 1e-6), (0, -1e-6)]:
    devs = boris_mp(mp.mpf("0.9999987") + mp.mpf(dx), mp.mpf("0.1500041") + mp.mpf(dy), n)
    s = sorted(devs[n // 2:]); probe[f"({dx},{dy})"] = float(s[len(s) // 2])
out["mp_probe_around_opt"] = probe

# ---------------------------------------------------------------- (b) T4 quasistatic
TAUQ = 1.2e8
H_ = 0.3
def t4_map(n_steps, kappa=3.5e-7, drive="fixed"):
    z0 = 1j; z = z0; t = 0.0; Phi = 0.0
    om_h0 = (2.0 / H_) * np.arctan(H_ / 2.0)
    ts = np.zeros(n_steps // 50); dev = np.zeros(n_steps // 50)
    j = 0
    run = 0.0
    for nn in range(n_steps):
        Om = np.exp(-t / TAUQ)
        thh = 2.0 * np.arctan(H_ * Om / 2.0)
        kick = kappa * (np.sin(om_h0 * t) if drive == "fixed" else np.sin(Phi))
        z = np.exp(-1j * thh) * z + kick
        Phi += thh; t += H_
        run = max(run, abs(abs(z) ** 2 - 1.0))
        if (nn + 1) % 50 == 0 and j < len(ts):
            ts[j] = t; dev[j] = run; j += 1
    return ts[:j], dev[:j]

TWO_PI = 2 * np.pi
n = int(round(3e4 * TWO_PI / H_))
ts, env = t4_map(n)
env = np.maximum.accumulate(env)
gyr = ts / TWO_PI
def at(g): return float(env[np.searchsorted(gyr, g)])
sel = (ts > ts[-1] / 100)
p = np.polyfit(np.log10(ts[sel]), np.log10(env[sel]), 1)[0]
out["T4_quasistatic_fixed_drive"] = {
    "emax_1e2": at(1e2), "emax_1e3": at(1e3), "emax_1e4": at(1e4),
    "emax_3e4": float(env[-1]),
    "growth_1e2_to_1e3": at(1e3) / at(1e2), "growth_1e3_to_1e4": at(1e4) / at(1e3),
    "fit_exponent_last2decades": float(p)}
ts, env = t4_map(n, drive="track")
env = np.maximum.accumulate(env)
gyr = ts / TWO_PI
sel = (ts > ts[-1] / 100)
p = np.polyfit(np.log10(ts[sel]), np.log10(env[sel]), 1)[0]
out["T4_quasistatic_tracking_drive"] = {
    "emax_1e4": at(1e4), "emax_3e4": float(env[-1]), "fit_exponent": float(p)}

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf1b_extras.json"), "w"), indent=1)
