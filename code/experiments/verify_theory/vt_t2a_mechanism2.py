"""VT-T2a part 2: the ACTUAL mechanism of the Boris energy floor in B4.

Exact algebra for the shipped Boris step (q=-1, m=1, B = Bz(t) zhat,
E = fac(t) * (zhat x r), fac = Bz/(2 tau)):

    |v_{n+1}|^2 - |v_n|^2 = 2k E(r_n,t_n).(v_n + v_+) + 2 k^2 |E|^2,  k = -h/2
                          = q h fac(t_n) * zhat.(r_n x vbar) + O(E^2),
      vbar = (v_n + v_+)/2.

So the energy is driven by a DISCRETE ANGULAR MOMENTUM  Lz_n = (r_n x vbar)_z.
In the continuum, for a gyro-orbit centred on the axis, canonical p_phi is
conserved and Lz(t) == 1 identically, which makes E = E0 exp(-t/tau) the
EXACT solution (not just adiabatic).  Any deviation of the discrete Lz from
1 is the entire source of energy error.

Predictions of this mechanism (to be tested):
  M1  Lz_n oscillates about 1 with amplitude ~ h*Omega/2 and ZERO secular part
  M2  hence dev(t) = (h/(2 tau)) (1 - cos(omega_h t)) : mean h/2tau,
      amplitude h/2tau, min 0, max h/tau  -- an OSCILLATION, not an offset
  M3  the zero secular part is protected by a discrete p_phi conserved by
      Boris in an axisymmetric field  -> this, not 'a reading convention',
      is why the envelope exponent is 0.000
  M4  the floor survives ANY velocity-time relabelling (already shown) and
      also survives the correctly resynchronised staggered readout.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_PI = 2 * np.pi


def kick_parts(v, E, Bz, h, q=-1.0):
    k = 0.5 * q * h
    vm = v + k * E
    tz = k * Bz
    sz = 2.0 * tz / (1.0 + tz * tz)
    vpx = vm[0] + vm[1] * tz
    vpy = vm[1] - vm[0] * tz
    vp = np.array([vm[0] + vpy * sz, vm[1] - vpx * sz, vm[2]])   # v_+
    return vp + k * E, vm, vp


def run(h, tau, t_final, B0=1.0, r0=(1.0, 0.0, 0.0), v0=(0.0, 1.0, 0.0)):
    n = int(round(t_final / h))
    r = np.array(r0, float); v = np.array(v0, float)
    ts = np.zeros(n + 1); sp2 = np.zeros(n + 1)
    Lz = np.zeros(n + 1); pphi = np.zeros(n + 1)
    rs = np.zeros((n + 1, 3))
    sp2[0] = v @ v; rs[0] = r
    Lz[0] = r[0] * v[1] - r[1] * v[0]
    pphi[0] = Lz[0] - 0.5 * B0 * (r[0] ** 2 + r[1] ** 2)
    t = 0.0
    for i in range(1, n + 1):
        Bz = B0 * np.exp(-t / tau)
        fac = 0.5 * Bz / tau
        E = np.array([-fac * r[1], fac * r[0], 0.0])
        vn = v
        v, vm, vp = kick_parts(v, E, Bz, h)
        vbar = 0.5 * (vn + vp)
        Lz[i] = r[0] * vbar[1] - r[1] * vbar[0]      # driver of the work term
        r = r + v * h
        t += h
        Bz1 = B0 * np.exp(-t / tau)
        pphi[i] = (r[0] * v[1] - r[1] * v[0]) - 0.5 * Bz1 * (r[0] ** 2 + r[1] ** 2)
        ts[i] = t; sp2[i] = v @ v; rs[i] = r
    return ts, sp2, Lz, pphi, rs


out = {}
TAU, H, TF = 1.2e5, 0.3, 120.0

# ---- M1: the discrete angular-momentum driver ----------------------------
ts, sp2, Lz, pphi, rs = run(H, TAU, TF)
half = len(ts) // 2
Lz2 = Lz[half:]
out["M1_discrete_Lz"] = {
    "mean_minus_1": float(np.mean(Lz2) - 1.0),
    "amplitude_(max-min)/2": float((np.max(Lz2) - np.min(Lz2)) / 2),
    "pred_amplitude_h_Omega_over_2": H / 2,
    "secular_slope_per_unit_time": float(np.polyfit(ts[half:], Lz2, 1)[0]),
}

# ---- M2: closed-form envelope prediction ---------------------------------
om_h = 2 * np.arctan(H / 2) / H
dev = np.abs(sp2 - np.exp(-ts / TAU))
pred = (H / (2 * TAU)) * (1.0 - np.cos(om_h * ts))
out["M2_closed_form"] = {
    "max_rel_discrepancy_vs_(h/2tau)(1-cos om_h t)":
        float(np.max(np.abs(dev - pred)) / (H / TAU)),
    "measured_max": float(np.max(dev)), "pred_max": float(np.max(pred)),
    "measured_median_2nd_half": float(np.median(dev[half:])),
    "pred_median_2nd_half": float(np.median(pred[half:])),
    "corr": float(np.corrcoef(dev[half:], pred[half:])[0, 1]),
}
# refine the frequency: best-fit omega
oms = om_h * np.linspace(0.98, 1.02, 4001)
res = [np.max(np.abs(dev - (H / (2 * TAU)) * (1 - np.cos(o * ts)))) for o in oms]
o_best = float(oms[int(np.argmin(res))])
out["M2_closed_form"]["best_fit_omega"] = o_best
out["M2_closed_form"]["omega_h"] = om_h
out["M2_closed_form"]["Omega_phys"] = 1.0
out["M2_closed_form"]["best_fit_residual_over_h_tau"] = float(np.min(res) / (H / TAU))

# ---- M3: is a discrete p_phi conserved? ----------------------------------
rows = []
for ng in (1e2, 1e3, 1e4):
    t_, s_, L_, pp_, _ = run(H, TAU, ng * TWO_PI)
    rows.append({"gyro": ng, "pphi_drift_max": float(np.max(np.abs(pp_ - pp_[0]))),
                 "pphi_0": float(pp_[0]),
                 "dev_max": float(np.max(np.abs(s_ - np.exp(-t_ / TAU))))})
out["M3_pphi"] = rows

# ---- M2b: Omega-dependence (B0 scan) -------------------------------------
# sampling-shift mechanism: median = h/(2 tau) independent of B0 AND of the
# oscillation.  Angular-momentum mechanism: amplitude = h Omega /(2 Omega tau)
# also = h/(2tau).  Distinguish instead by the OSCILLATION FREQUENCY = omega_h(B0)
rows = []
for B0 in (0.5, 1.0, 2.0):
    # keep h*Omega fixed?  no: keep h fixed, let theta_h change
    t_, s_, L_, pp_, _ = run(H, TAU, TF, B0=B0, v0=(0.0, B0, 0.0))
    # v0 chosen so the guiding centre stays on the axis: r_L = |v|/Omega = 1
    d_ = np.abs(s_ / (B0 ** 2) - np.exp(-t_ / TAU))     # normalise E0
    hh = len(t_) // 2
    omh = 2 * np.arctan(H * B0 / 2) / H
    rows.append({"B0": B0, "median_over_(h/2tau)": float(np.median(d_[hh:]) / (H / (2 * TAU))),
                 "max_over_(h/tau)": float(np.max(d_) / (H / TAU)),
                 "Lz_amp": float((np.max(L_[hh:]) - np.min(L_[hh:])) / 2),
                 "pred_Lz_amp_hB0_over_2": H * B0 / 2 * B0,
                 "omega_h": omh})
out["M2b_B0_scan"] = rows

# ---- M4: off-axis start -- is E0 exp(-t/tau) still the exact solution? ----
# guiding centre displaced: r0=(1,0,0), v0=(0,1,0) has GC at origin.
# use v0=(0.5,1,0): GC displaced by 0.5 in y.
rows = []
for lbl, r0, v0 in (("on-axis", (1.0, 0, 0), (0.0, 1.0, 0.0)),
                    ("off-axis", (1.0, 0, 0), (0.5, 1.0, 0.0))):
    t_, s_, L_, pp_, _ = run(H, TAU, TF, r0=r0, v0=v0)
    tf_, sf_, Lf_, ppf_, _ = run(H / 200, TAU, TF, r0=r0, v0=v0)
    E0 = s_[0]
    d_coarse = np.abs(s_ / E0 - np.exp(-t_ / TAU))
    d_fine = np.abs(sf_[::200] / E0 - np.exp(-t_ / TAU))
    d_vs_fine = np.abs(s_ - sf_[::200]) / E0
    hh = len(t_) // 2
    rows.append({"case": lbl,
                 "median_dev_vs_adiabatic_coarse": float(np.median(d_coarse[hh:])),
                 "median_dev_vs_adiabatic_FINE(h/200)": float(np.median(d_fine[hh:])),
                 "median_dev_vs_fine_reference": float(np.median(d_vs_fine[hh:])),
                 "h_over_2tau": H / (2 * TAU)})
out["M4_offaxis"] = rows

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt_t2a_mechanism2.json"), "w"), indent=1)
