"""EA2: the four probes of Section 6, run on the three external architectures.

Writes ea2_probes.json.  Rerunning compares against the committed file and
exits non-zero on any disagreement.

    python ea2_probes.py [--force] [--gyros N] [--quick]

THE FOUR PROBES, AS SECTION 6 DEFINES THEM AND AS THEY ARE CODED HERE
---------------------------------------------------------------------
P1  Drive the scheme with a defect held parallel to v at the amplitude of its
    own truncation error and check that the reported envelope growth exponent
    is 1.  The amplitude is measured, not assumed: eps_truncation is the median
    of |v_{n+1}^scheme - v_{n+1}^exact| / |v_n| over the first 400 steps, the
    exact propagation over one step being 150 Boris steps of h/150, which is
    the reference the corrector was trained against.  The kick is
    v <- v(1 + eps), the D4 drive of experiments/theory_check/t3_dichotomy.py.
    A scheme whose diagnostic reports 0 here is reporting its readout floor.

    Two amendments were forced by running the probe on foreign objects, and
    both are recorded in the output rather than hidden in the code:

    (a) The amplitude is capped.  Condition (i) of Section 4.3 is the linear
        regime |S_N| << |z_0|; a multiplicative parallel kick accumulates as
        (1+eps)^{2N} - 1, so eps must satisfy 2 eps N << 1 at the horizon.
        The Boris truncation amplitude at Omega h = 0.3 is 4e-3, which leaves
        the linear regime after four hundred steps.  eps_used is therefore
        min(eps_truncation, 0.05/(2N)) and both numbers are reported.
    (b) The response is isolated.  The probe reads the exponent of the
        reported energy error of the driven run, which presumes the scheme's
        own reported energy error to be below the response.  That holds for
        the Boris scheme, whose energy error is 1.25e-6, and fails for every
        learned map here by three to five orders of magnitude.  Each probe
        therefore carries an undriven twin in the same batch and reports the
        exponent of the difference beside the exponent of the raw series.

P2  Repeat the drive at the scheme frequency omega_h rather than at the natural
    frequency Omega, and sweep a detuning band whose width scales with the
    horizon.  omega_h has no closed form for a learned map, so it is measured
    from the unwrapped phase of v_x + i v_y on the unperturbed run; the same
    measurement on the Boris scheme is checked against 2 arctan(h Omega/2)/h
    before any learned number is believed.  The sweep is carried as one batch:
    all detunings and the undriven twin advance together, so a scan of 21
    frequencies costs one matmul per step rather than 21 runs.

P3  Recompute the same conclusion under a second readout convention.  Two
    operations, as Section 6 has them: the initial position moved half a step
    along v_0, and the energy read from the mean of the two velocities
    bracketing the instant rather than from the stored one.  The probe asks
    that the conclusion survive both, not that the level be unchanged.  The
    conclusion here is the pair (does the reported energy error stay below the
    physical signal, is the envelope growth exponent above 1/2).

P4  Replace random perturbation by a systematic one along v and pass the defect
    it produces to the channel-selection rule that Table 2 of Section 6 sets
    out.  That rule is not reimplemented here: `pb4_channel.channel_search` is
    imported and handed the five series it asks for, so this directory decides
    the fourth probe by exactly the rule the manuscript prints, and a change to
    the rule changes these verdicts.  Beside it, and only for the report, the
    two-channel reading the probe had before that rule existed is computed as
    well, so that the difference between the two can be stated.

    The probe runs in two modes.  Driven takes the injected parallel kick as
    the defect, which is what Section 6 describes.  Intrinsic takes the
    scheme's own defect against the Boris map with no drive at all, which is
    the mode in which the known failure was measured in Section 4.3, and it is
    the mode that answers whether that failure reproduces elsewhere.  Each
    architecture is additionally run with the projection of Section 4.3 placed
    after it, which makes its correction energy-neutral in the velocity and so
    violates condition (vi) deliberately.

VERDICTS
--------
Each probe returns a verdict from a rule written down before the runs:

  P1  pass if the envelope exponent of the isolated response lies in
      [0.85, 1.15]; "reports its readout floor" if below 0.20; "inconclusive"
      otherwise.
  P2  pass if the largest response over the sweep lands within one and a half
      grid steps of the measured omega_h, the on-resonance exponent is at
      least 0.85, and the response at the edge of the band is smaller by at
      least a factor of 3.
  P3  pass if the conclusion pair is the same under all three readings.
  P4  PASS, FAIL or NO EXPONENT, by lines 14 and 15 of Table 2 of Section 6.

A scheme that does not survive the horizon is not scored on the probes it
cannot reach.  Its survival step is reported instead, and that is the finding.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ea_common as C          # noqa: E402
import ea_arch as A            # noqa: E402
import ea1_train as T          # noqa: E402

OUT = os.path.join(HERE, "ea2_probes.json")
CKPT = os.path.join(HERE, "ckpt")

N_GYROS = 1.0e4                # reduced from the 1e5 of Section 7; see setup
GYROS_P2 = (1.0e3, 1.0e4)
N_DETUNE = 21
LINEAR_BUDGET = 0.05           # 2 eps N <= this, condition (i) of Section 4.3
REP_MAIN = 0
KAPPA_P2 = 1e-9                # the calibrated amplitude of t3_dichotomy.py
SCHEMES = ("boris", "hnn", "sympnet", "pinn")


# =====================================================================
def truncation_amplitude(stepper, tau, n=400, dt=C.DT):
    """eps: the scheme's own one-step velocity defect, relative to |v|."""
    x = np.array([C.R0[0]]); y = np.array([C.R0[1]])
    vx = np.array([C.V0[0]]); vy = np.array([C.V0[1]])
    t = 0.0
    rel = []
    for _ in range(n):
        xf, yf, vxf, vyf, tf = x[0], y[0], vx[0], vy[0], t
        for _m in range(150):
            xf, yf, vxf, vyf = C.boris_plane(xf, yf, vxf, vyf, tf, tau, dt / 150.0)
            tf += dt / 150.0
        xn, yn, vxn, vyn = stepper.step(x, y, vx, vy, t, dt)
        if not np.isfinite(vxn[0]) or not np.isfinite(vyn[0]):
            break
        rel.append(np.hypot(vxn[0] - vxf, vyn[0] - vyf) / np.hypot(vx[0], vy[0]))
        x, y, vx, vy = xn, yn, vxn, vyn
        t += dt
        if np.hypot(vx[0], vy[0]) > 1e3:
            break
    return float(np.median(rel)) if rel else float("nan")


def capped_eps(eps_trunc, n_steps):
    cap = LINEAR_BUDGET / (2.0 * n_steps)
    if not np.isfinite(eps_trunc):
        return cap, cap
    return min(eps_trunc, cap), cap


def verdict_p1(p):
    if not np.isfinite(p):
        return "not reached"
    if 0.85 <= p <= 1.15:
        return "pass"
    if p < 0.20:
        return "reports its readout floor"
    return "inconclusive"


# =====================================================================
def driven_run(stepper, tau, n_steps, eps):
    """The one rollout that the first and the fourth probe share.

    Both drive the scheme with the same parallel defect, and the first probe
    reads an exponent off the response while the fourth passes the defect to
    the channel search.  Running them together halves the cost of the pair.
    Member 0 is the undriven twin, member 1 the driven run.
    """
    scale = np.array([0.0, 1.0])

    def perturb(vx, vy, n, t):
        f = 1.0 + eps * scale
        return vx * f, vy * f

    return A.rollout(stepper, n_steps, tau=tau, nb=2, perturb=perturb,
                     record_defect=True, boris_compare=True)


def probe1(r, eps, eps_cap, eps_trunc):
    ex = A.envelope_exponents(r["t_env"], r["env"])
    alive = int(np.min(r["alive"]))
    if alive < 4096:
        return {"eps_truncation": eps_trunc, "eps_cap_linear_regime": eps_cap,
                "eps_used": eps, "alive_steps": [int(v) for v in r["alive"]],
                "verdict": "not reached", "verdict_raw": "not reached"}
    tr, er = A.response_envelope(r["signed"], alive)
    ex_r = A.envelope_exponents(tr, er)[0]
    dominates = bool(np.nanmax(r["env"][:alive, 0]) > np.nanmax(er[:, 0]))
    return {"eps_truncation": eps_trunc, "eps_cap_linear_regime": eps_cap,
            "eps_used": eps,
            "unperturbed": ex[0], "driven_raw": ex[1],
            "isolated_response": ex_r,
            "own_error_max": float(np.nanmax(r["env"][:, 0])),
            "response_max": float(np.nanmax(er[:, 0])),
            "own_error_dominates_response": dominates,
            "alive_steps": [int(v) for v in r["alive"]],
            "verdict_raw": verdict_p1(ex[1]["p_fit_last2dec"]),
            "verdict": verdict_p1(ex_r["p_fit_last2dec"])}


# =====================================================================
def probe2(stepper, tau, n_gyros, omega_h, kappa, dt=C.DT):
    """Sustained drive on a detuning band around the measured omega_h.

    The band is alpha T / omega_h wide, alpha being the chirp rate of the
    scheme frequency in the decaying field: that is the width Section 4.5
    measures, and it grows with the horizon.  A Fourier-limited floor
    3 pi / (T omega_h) keeps the grid meaningful when the chirp is narrower
    than the resolution of the run itself.
    """
    n_steps = int(round(n_gyros * C.TWO_PI / dt))
    alpha = 1.0 / ((1 + (dt / 2) ** 2) * tau)
    band_chirp = alpha * n_gyros * C.TWO_PI / omega_h
    band_fourier = 3.0 * np.pi / (n_gyros * C.TWO_PI) / omega_h
    band = 4.0 * max(band_chirp, band_fourier)
    # member 0 is the undriven twin; the rest carry the detuning grid
    rel = np.concatenate([[0.0], np.linspace(-band, band, N_DETUNE)])
    om = omega_h * (1.0 + rel)
    amp = np.full(N_DETUNE + 1, kappa)
    amp[0] = 0.0

    def perturb(vx, vy, n, t):
        return vx + amp * np.sin(om * (t - 0.5 * dt)), vy

    r = A.rollout(stepper, n_steps, tau=tau, nb=N_DETUNE + 1, perturb=perturb,
                  record_signed=True)
    alive = int(np.min(r["alive"]))
    if alive < 4096:
        return {"n_gyros": n_gyros, "omega_h_used": omega_h,
                "alive_steps_min": alive, "verdict": "not reached"}
    tr, resp = A.response_envelope(r["signed"], alive)
    rel = rel[1:]
    ex = A.envelope_exponents(tr, resp)
    emax = np.nanmax(resp, axis=0)
    j = int(np.nanargmax(emax)) if np.isfinite(emax).any() else -1
    j0 = int(np.nanargmin(np.abs(rel)))
    grid = float(rel[1] - rel[0])
    ok_peak = (j >= 0) and abs(rel[j]) <= 1.5 * abs(grid)
    p_on = ex[j0]["p_fit_last2dec"]
    edge = float(min(emax[0], emax[-1]))
    supp = float(emax[j0] / edge) if edge > 0 else float("inf")
    return {"n_gyros": n_gyros, "omega_h_used": omega_h, "kappa": kappa,
            "band_half_width_rel": float(band),
            "band_from_chirp": float(4.0 * band_chirp),
            "band_from_fourier_limit": float(4.0 * band_fourier),
            "grid_step_rel": grid,
            "rel_detune": [float(v) for v in rel],
            "emax_isolated_response": [float(v) for v in emax],
            "p_fit_isolated": [e["p_fit_last2dec"] for e in ex],
            "peak_rel_detune": float(rel[j]) if j >= 0 else float("nan"),
            "peak_emax": float(emax[j]) if j >= 0 else float("nan"),
            "emax_on_omega_h": float(emax[j0]),
            "p_on_omega_h": p_on,
            "half_decade_slopes_on_omega_h": ex[j0]["half_decade_slopes"],
            "undriven_emax": float(np.nanmax(r["env"][:, 0])),
            "suppression_at_band_edge": supp,
            "alive_steps_min": alive,
            "verdict": ("pass" if (ok_peak and np.isfinite(p_on) and p_on >= 0.85
                                   and supp >= 3.0)
                        else ("not reached" if not np.isfinite(p_on) else "fail"))}


# =====================================================================
def probe3(stepper, tau, n_steps, dt=C.DT):
    out = {}
    for label, half in (("synchronized", False), ("shifted_h_over_2", True)):
        r = A.rollout(stepper, n_steps, tau=tau, nb=1, half_step_start=half,
                      record_defect=True)
        ex = A.envelope_exponents(r["t_env"], r["env"])[0]
        sig = r["signed"][:, 0]
        alive = int(r["alive"][0])
        e0 = r["e0"]
        med = float(np.median(np.abs(sig[alive // 2:alive]))) if alive > 4 \
            else float("nan")
        phys = float(abs(1.0 - np.exp(-alive * dt / tau)))
        out[label] = {"energy_err_median_2nd_half": med,
                      "physical_signal": phys,
                      "below_signal": bool(med < phys),
                      "p_fit_last2dec": ex["p_fit_last2dec"],
                      "half_decade_slopes": ex["half_decade_slopes"],
                      "alive_steps": alive}
        if label == "synchronized":
            z = r["z_prekick"][:alive, 0]
            vmid = 0.5 * (z[:-1] + z[1:])
            ts = (np.arange(1, alive) + 0.5) * dt
            dev = np.abs(0.5 * np.abs(vmid) ** 2 - C.e_phys(ts, tau, e0)) / e0
            stride = max(1, dev.size // 4000)
            starts = np.arange(0, dev.size, stride)
            run = np.fmax.reduceat(dev, starts)
            tw = ts[np.minimum(starts + stride - 1, dev.size - 1)]
            env = np.fmax.accumulate(run)[:, None]
            exa = A.envelope_exponents(tw, env)[0]
            meda = float(np.median(dev[dev.size // 2:]))
            out["averaged_reading"] = {
                "energy_err_median_2nd_half": meda,
                "physical_signal": phys,
                "below_signal": bool(meda < phys),
                "p_fit_last2dec": exa["p_fit_last2dec"],
                "half_decade_slopes": exa["half_decade_slopes"],
                "alive_steps": alive}
    keys = ("synchronized", "shifted_h_over_2", "averaged_reading")
    concl = [(out[k]["below_signal"],
              bool(np.isfinite(out[k]["p_fit_last2dec"]) and
                   out[k]["p_fit_last2dec"] > 0.5)) for k in keys]
    out["conclusions"] = {k: list(c) for k, c in zip(keys, concl)}
    out["verdict"] = "pass" if len(set(concl)) == 1 else "fail"
    levels = [out[k]["energy_err_median_2nd_half"] for k in keys]
    out["level_spread_factor"] = float(max(levels) / min(levels)) \
        if min(levels) > 0 and np.isfinite(max(levels)) else float("inf")
    return out

# =====================================================================
#  the fourth probe, decided by the rule of Table 2 of Section 6
# =====================================================================
def _rule(kappa, u_ref, u_own, signed, tau, alive, dt=C.DT,
          deterministic=True):
    """`pb4_channel.channel_search` on our series, plus the horizon comparison.

    Nothing of the rule is reimplemented here.  This function assembles the
    five series the rule asks for and hands them over: the defect, the two
    reference directions, the reported error, and the physical speed
    $|v_{\\mathrm{ref}}| = \\exp(-t/2\\tau)$ at $|v_0| = 1$.  Lines 14 and 15
    of the table, the comparison against the horizon, are applied here in the
    same form `pb4_channel.bench_b` applies them.
    """
    PB = C.import_pb4()
    vphys = np.exp(-np.arange(1, alive + 1, dtype=float) * dt / (2.0 * tau))
    rep = PB.channel_search(kappa, u_ref, u_own, signed, vphys, tau,
                            alive // 16, alive, dt=dt,
                            deterministic=deterministic)
    p_meas, t_env, env = PB.envelope_exponent_from_series(np.abs(signed))
    rep["p_measured_full_horizon"] = p_meas
    rep["full_half_decade_slopes"] = PB.half_decade_slopes(t_env, env)
    rep["full_half_decade_spread"] = PB.slope_spread(
        rep["full_half_decade_slopes"])
    rep["full_curve_is_power_law"] = bool(
        rep["full_half_decade_spread"] <= PB.TOL_SPREAD)
    if rep["p_pred"] is None:
        rep["probe_verdict"] = rep["verdict"]
    elif not rep["full_curve_is_power_law"]:
        rep["probe_verdict"] = "NO EXPONENT (full horizon)"
    else:
        rep["probe_verdict"] = ("PASS" if abs(rep["p_pred"] - p_meas)
                                <= PB.TOL_P else "FAIL")
        rep["prediction_error"] = rep["p_pred"] - p_meas
    return rep


def _legacy_two_channels(kappa, u_ref, signed, alive, dt=C.DT):
    """The two-channel reading the probe had before Table 2 of Section 6.

    Kept beside the rule so that the report can say what the rule changed:
    the estimate on the demodulated velocity defect, the estimate on the
    increments of the reported energy error, and which of the two lands
    closer to the exponent measured at the horizon.
    """
    PC = C.import_pc()
    w = kappa * np.conj(u_ref)
    dev = np.abs(signed)
    p_meas, t_env, env = PC.envelope_exponent_from_series(dev, dt=dt)
    short = alive >> 4
    out = {"p_measured_envelope": float(p_meas),
           "half_decade_slopes": C.half_decade_slopes(t_env, env),
           "short_run_steps": short}
    aV_s, HV_s = PC.estimate_aH(w[:short][None, :])
    aV_f, HV_f = PC.estimate_aH(w[None, :])
    out["V_velocity_defect"] = {
        "a_hat_short": aV_s, "H_hat_short": HV_s,
        "p_pred_short": max(0.0, aV_s + HV_s),
        "a_hat_full": aV_f, "H_hat_full": HV_f,
        "p_pred_full": max(0.0, aV_f + HV_f)}
    g = np.diff(np.concatenate([[0.0], signed]))
    aE_s, HE_s = PC.estimate_aH(g[:short][None, :].astype(complex))
    aE_f, HE_f = PC.estimate_aH(g[None, :].astype(complex))
    Sg = np.cumsum(g)
    out["E_energy_increment"] = {
        "a_hat_short": aE_s, "H_hat_short": HE_s,
        "p_pred_short": max(0.0, aE_s + HE_s),
        "a_hat_full": aE_f, "H_hat_full": HE_f,
        "p_pred_full": max(0.0, aE_f + HE_f),
        "identity_check_max_abs_dev_minus_sum_g":
            float(np.max(np.abs(np.abs(Sg) - dev)))}
    miss = {ch: abs(out[ch]["p_pred_short"] - p_meas)
            for ch in ("V_velocity_defect", "E_energy_increment")}
    best = min(miss, key=miss.get)
    out["miss_from_short_run"] = {k: float(v) for k, v in miss.items()}
    out["channel_that_works"] = best if miss[best] <= 0.15 else None
    out["verdict"] = "pass" if miss[best] <= 0.15 else "fail"
    out["repair_needed"] = bool(miss["V_velocity_defect"] > 0.15 >=
                                miss["E_energy_increment"])
    return out


def probe4_driven(r, base, eps, tau, dt=C.DT):
    """Driven mode: the defect is the parallel kick that was injected.

    The response is isolated against the undriven twin, for the reason
    `A.response_envelope` gives, and the rule is then applied to the isolated
    series.  The reference direction is the unperturbed Boris run, the scheme's
    own direction is the Boris step taken from the scheme's own state, which is
    the pair `pb4_channel` uses for the corrector.
    """
    alive = int(min(np.min(r["alive"]), base["alive"][0]))
    if alive < 4096:
        return {"alive_steps": alive, "verdict": "not reached"}
    kap = eps * r["z_prekick"][:alive, 1]
    u_ref = base["z_prekick"][:alive, 0]
    u_ref = u_ref / np.abs(u_ref)
    u_own = r["z_boris"][:alive, 1]
    u_own = u_own / np.abs(u_own)
    sig = r["signed"][:alive, 1] - r["signed"][:alive, 0]
    out = {"alive_steps": alive, "eps": eps,
           "kappa_rms": float(np.sqrt(np.mean(np.abs(kap) ** 2))),
           "rule": _rule(kap, u_ref, u_own, sig, tau, alive, dt=dt),
           "two_channel_reading": _legacy_two_channels(kap, u_ref, sig, alive,
                                                       dt=dt)}
    out["verdict"] = out["rule"]["probe_verdict"]
    out["selected_channel"] = out["rule"]["selected"]
    return out


def probe4_intrinsic(stepper, tau, n_steps, base, dt=C.DT):
    """Intrinsic mode: the scheme's own defect, no drive.

    Section 4.3 measured the known failure this way, on the corrector's own
    defect against the Boris map with the corrector's own energy error scored
    against it.  This is the block that answers whether that failure and its
    repair reproduce on an architecture other than ours.
    """
    r = A.rollout(stepper, n_steps, tau=tau, nb=1, record_defect=True,
                  boris_compare=True)
    alive = int(min(r["alive"][0], base["alive"][0]))
    if alive < 4096:
        return {"alive_steps": alive, "verdict": "not reached"}
    kap = r["z_prekick"][:alive, 0] - r["z_boris"][:alive, 0]
    kap_rms = float(np.sqrt(np.mean(np.abs(kap) ** 2)))
    if kap_rms < 1e-280:
        # the Boris scheme against the Boris map, and the same scheme with the
        # projection, which is the identity on it
        return {"alive_steps": alive, "kappa_rms": kap_rms,
                "verdict": "no defect against the reference map"}
    u_ref = base["z_prekick"][:alive, 0]
    u_ref = u_ref / np.abs(u_ref)
    u_own = r["z_boris"][:alive, 0]
    u_own = u_own / np.abs(u_own)
    sig = r["signed"][:alive, 0]
    out = {"alive_steps": alive, "kappa_rms": kap_rms,
           "energy_neutrality_max_abs_d_speed":
               float(np.max(np.abs(np.abs(r["z_prekick"][:alive, 0])
                                   - np.abs(r["z_boris"][:alive, 0])))),
           "rule": _rule(kap, u_ref, u_own, sig, tau, alive, dt=dt),
           "two_channel_reading": _legacy_two_channels(kap, u_ref, sig, alive,
                                                       dt=dt)}
    out["verdict"] = out["rule"]["probe_verdict"]
    out["selected_channel"] = out["rule"]["selected"]
    return out


# =====================================================================
def steppers_for(tau):
    out = {"boris": A.BorisStepper(tau)}
    for arch in ("hnn", "sympnet", "pinn"):
        p = os.path.join(CKPT, "%s_r%d.npz" % (arch, REP_MAIN))
        if os.path.exists(p):
            out[arch] = T.load_stepper(p, tau)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gyros", type=float, default=N_GYROS)
    a = ap.parse_args()
    n_gyros = 30.0 if a.quick else a.gyros
    n_steps = int(round(n_gyros * C.TWO_PI / C.DT))
    gy2 = (30.0,) if a.quick else GYROS_P2

    out = {"setup": {
        "n_gyros": n_gyros, "n_steps": n_steps, "dt": C.DT,
        "tau_quasistatic": C.TAU_QUASI, "tau_decaying": C.TAU_PAPER,
        "repetition_probed": REP_MAIN,
        "linear_regime_budget_2_eps_N": LINEAR_BUDGET,
        "n_detune": N_DETUNE,
        "horizon_note":
            "Section 7 runs the corrector to 1e5 gyro-orbits.  The probes here "
            "stop at %g, a factor of %g shorter, because a learned map is "
            "evaluated once per step in numpy and the sweep of the second "
            "probe is 21 frequencies wide.  Every exponent below is a fit over "
            "the last two decades of that shorter run and carries the local "
            "half-decade slopes beside it, as Section 4.4 requires."
            % (n_gyros, 1e5 / n_gyros),
    }, "omega_h": {}, "probes": {}}

    for tau_label, tau in (("quasistatic", C.TAU_QUASI), ("decaying", C.TAU_PAPER)):
        st = steppers_for(tau)
        row = {n: A.measure_scheme_frequency(s) for n, s in st.items()}
        row["boris_closed_form"] = float(2 * np.arctan(C.DT / 2) / C.DT)
        row["boris_relative_measurement_error"] = \
            float(abs(row["boris"] - row["boris_closed_form"])
                  / row["boris_closed_form"])
        out["omega_h"][tau_label] = row
    assert out["omega_h"]["quasistatic"]["boris_relative_measurement_error"] < 1e-5, \
        "the omega_h measurement no longer reproduces the closed form on Boris"

    out["P4_projected"] = {}
    for tau_label, tau in (("quasistatic", C.TAU_QUASI), ("decaying", C.TAU_PAPER)):
        st = steppers_for(tau)
        # the unperturbed Boris run is the reference frame of every channel
        # search below, and it is the same run for every scheme at this tau
        base = A.rollout(A.BorisStepper(tau), n_steps, tau=tau, nb=1,
                         record_defect=True)
        for name in SCHEMES:
            if name not in st:
                continue
            s = st[name]
            t0 = time.time()
            key = "%s/%s" % (name, tau_label)
            eps_tr = truncation_amplitude(s, tau)
            eps, cap = capped_eps(eps_tr, n_steps)
            rec = {"eps_truncation": eps_tr, "eps_used": eps,
                   "omega_h": out["omega_h"][tau_label][name],
                   "flops_per_step": int(s.flops_per_step())}
            dr = driven_run(s, tau, n_steps, eps)
            rec["P1"] = probe1(dr, eps, cap, eps_tr)
            rec["P4"] = probe4_driven(dr, base, eps, tau)
            del dr
            rec["P3"] = probe3(s, tau, n_steps)
            rec["P4_intrinsic"] = probe4_intrinsic(s, tau, n_steps, base)
            if tau_label == "quasistatic":
                om = rec["omega_h"]
                rec["P2"] = {("gyros_%g" % g): probe2(s, tau, g, om, KAPPA_P2)
                             for g in gy2} if np.isfinite(om) else \
                    {"verdict": "not reached", "reason": "omega_h not measurable"}
            rec["seconds"] = time.time() - t0
            out["probes"][key] = rec
            p2v = rec.get("P2", {})
            p2s = ",".join(str(v.get("verdict")) for v in p2v.values()
                           if isinstance(v, dict)) or "-"
            print("[%-18s] eps=%.3e P1 %-26s P2 %-11s P3 %-5s "
                  "P4 %-24s P4int %-24s %.0fs"
                  % (key, eps, rec["P1"]["verdict"], p2s, rec["P3"]["verdict"],
                     rec["P4"]["verdict"], rec["P4_intrinsic"]["verdict"],
                     rec["seconds"]), flush=True)

            # ---- the same scheme with the projection of Section 4.3 after
            #      it, which makes its correction energy-neutral in the
            #      velocity and so violates condition (vi) on purpose
            ps = A.ProjectedStepper(s, tau)
            t1 = time.time()
            prec = probe4_intrinsic(ps, tau, n_steps, base)
            prec["seconds"] = time.time() - t1
            prec["flops_per_step"] = int(ps.flops_per_step())
            pkey = "%s+proj/%s" % (name, tau_label)
            out["P4_projected"][pkey] = prec
            print("[%-18s] P4 intrinsic %-24s channel %-6s "
                  "energy-neutral to %.2e  %.0fs"
                  % (pkey, prec["verdict"], str(prec.get("selected_channel")),
                     prec.get("energy_neutrality_max_abs_d_speed",
                              float("nan")), prec["seconds"]), flush=True)
            # a partial dump after every scheme, so that a run interrupted
            # part way leaves something readable behind
            with open(OUT + ".partial", "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
        del base

    if a.quick:
        print(json.dumps(out["probes"], indent=1)[:8000])
        return 0
    return C.check_or_write(OUT, out, rtol=1e-6, force=a.force)


if __name__ == "__main__":
    sys.exit(main())
