"""The channel-selection rule of the fourth probe, and its test.

Section 6 of the manuscript states that the fourth probe fails on the class of
schemes the paper is about: taken on the velocity defect, the two-parameter
estimate of Section 4.2 predicts p = 0 for the quasistatic run of the learned
corrector, whose measured envelope exponent is 0.9772, and 0.338 for the
decaying-field run, whose two-decade fit is 1.5395.  The repair was written as
prose: "take the two parameters on the increments of the reported energy error
as well", and "a scheme passes the fourth probe only when the channel the
estimate is taken on is the channel the error actually travels by".

This script turns that sentence into a decidable rule and measures what the
rule does.  It changes no number in the manuscript.  It re-runs the two
instrumented integrations of p_law_check/pc_defect.py, reproduces every field
of pc_defect.json and pc_split.json that Sections 4.3 and 6 quote, and exits
non-zero if any of them has moved.  Everything the rule adds is written to
pb4_channel.json beside this file.

THE RULE
--------
Inputs.  A short run of N0 = N/16 steps of the candidate scheme under the
systematic drive of the fourth probe, recording per step the velocity defect
kappa_k against the base method, the direction of the reference propagator,
and the reported relative energy error dev_n; the target horizon N; and the
explicit time scales of the problem.  Output: a channel, an exponent, and one
of three verdicts -- PASS, FAIL, NO EXPONENT.

  G0  extrapolation gate.  If any explicit coefficient of the equations
      changes by more than a factor of COEF_FACTOR between t(N0) and t(N),
      return NO EXPONENT.  The short run and the horizon are not the same
      problem, and no estimate taken on the first is a statement about the
      second.

  1   enumerate.  One channel per (defect, frame) pair the scheme offers,
      plus one per reported error series, that one being the sequence of its
      own increments.  Here: V(ref), the velocity defect demodulated in the
      frame of the reference propagator; V(own), the same defect demodulated
      in the frame of the scheme's own base step; E, the increments of the
      reported relative energy error.  Each channel carries a reconstruction
      map from its partial sums to the reported error:
          R_V(S) = | 2 |v_phys| Re S + |S|^2 | / |v_0|^2 ,
          R_E(S) = |S|   (an identity, exact to machine zero).

  2   parameters.  Section 4.2, unchanged: a_hat from the slope of
      log rms|c| over logarithmically spaced bins, H_hat from the slope of
      log rms|sum c k^-a_hat|, p_hat = max(0, a_hat + H_hat).

  G1  carriage gate.  rho = median over the last decade of the short run of
      |log10 R(S_n)/dev_n|.  Reject the channel if rho > TOL_REC decades.
      A channel that does not reproduce the error the diagnostic reports is
      not the channel the error travels by, whatever exponent it has.

  G2  cancellation gate.  Let p_rec be the envelope exponent of R(S_n) and
      p_terms the largest envelope exponent among the separate terms of R.
      Reject if p_rec > p_terms + TOL_P.  A reconstruction that grows faster
      than either of its terms is the residue of a cancellation between them,
      and the exponent of the channel is then not the exponent of the error.
      This is condition (ii) of Section 4.3 made measurable.

  3   select.  Among the surviving channels take the first in the fixed order
      V(ref), V(own), E.  V is tried first because it is a property of the
      scheme while E is a property of one reported diagnostic: a scheme whose
      error is entirely in the phase channel has dev = 0 and E sees nothing,
      which is the blindness of Section 2.  Where both survive they carry the
      same sequence, since g_n = 2 Re(z0bar w_n) in the linear non-degenerate
      regime, and V is the estimate on the complex sequence while E is the
      estimate on its real projection; the bench below measures the mean
      error of each.  A disagreement above TOL_P between survivors is
      reported with the result and is estimator noise, not a fact about the
      scheme.  If none survives, return NO EXPONENT.  Never return p = 0
      here: p = 0 is a claim that the error is bounded, and it may be made
      only from a channel that has passed G1.  The probe as first written
      returned p = 0 from a channel that fails G1 by 1.6 decades.

  G3  power-law gate, for a deterministic drive.  Local half-decade slopes of
      the envelope of dev over the last two decades of the short run
      (Section 4.4).  If they spread by more than TOL_SPREAD, return NO
      EXPONENT rather than a number.  Probe 4 drives the scheme with a
      systematic perturbation along v, so its own runs are deterministic and
      the gate always applies to them.  Under a stochastic drive the local
      slopes of a single realization spread by 0.67 to 2.71 across the eight
      cases of bench A below, so the gate carries no information there and is
      not applied; the exponent then carries the +-0.15 of Section 4.4 and
      the comparison is an ensemble one.

  4   verdict.  The probe passes when a channel is returned and the predicted
      exponent agrees with the exponent measured at the full horizon to
      TOL_P, the full-horizon envelope having itself passed the half-decade
      rule of Section 4.4.

THE THRESHOLDS are four, fixed once, and taken from quantities the paper has
already established: TOL_P = 0.15 is the single-realization interval M1 of
Section 4.4; TOL_SPREAD = 2 TOL_P; TOL_REC = 0.5 decades, a factor of three;
COEF_FACTOR = 2.  The script prints the margin by which every gate is passed
or failed, so that the verdicts can be seen not to sit on a threshold.

WHAT IS TESTED
--------------
Bench A, synthetic.  The eight cases of p_law/pl_protocol.py, where the fourth
probe already worked, are reproduced bit for bit as a control, and the rule is
then run on single realizations of the same cases plus two stalled ones.  The
rule must keep selecting the velocity channel there and must keep returning
the exponent it returned before, including the legitimate p = 0 of a stalled
case.

Bench B, the learned corrector.  The quasistatic run at tau = 1.2e8 and the
decaying-field run at tau = 1.2e5, 100,000 gyro-orbits each, the same two runs
Sections 4.3 and 6 report.

Run:  python pb4_channel.py           (about 15 min; four 2.1e6-step runs)
      PB4_GYROS=2000 python pb4_channel.py   smoke test, assertions skipped
Exit code 0 iff every stored field is reproduced and every rule check holds.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(EXP, "symproj"))
sys.path.insert(0, os.path.join(EXP, "p_law_check"))

from pc_defect import (run_instrumented, estimate_aH,               # noqa: E402
                       envelope_exponent_from_series, loglog_slope,
                       sub_index, DT, TWO_PI, TAU_PAPER, TAU_QUASI,
                       CAMPAIGN)

PC = os.path.join(EXP, "p_law_check")
PL = os.path.join(EXP, "p_law")

# ---- the four thresholds of the rule -------------------------------------
TOL_P = 0.15          # M1, Section 4.4: a single-realization exponent
TOL_SPREAD = 0.30     # 2 * TOL_P
TOL_REC = 0.5         # decades; a factor of 3.2
COEF_FACTOR = 2.0     # allowed change of a coefficient over the extrapolation
SHORT_DIV = 16        # the short run is 1/16 of the horizon (Section 4.2)

N_GYR = float(os.environ.get("PB4_GYROS", 100000))
N_STEPS = int(round(N_GYR * TWO_PI / DT))
FULL = abs(N_GYR - 100000.0) < 1e-9


# --------------------------------------------------------------------------
#  measurement helpers
# --------------------------------------------------------------------------
def envelope_series(dev, dt=DT, n_samples=4000):
    """Running maximum of dev on a logarithmically usable grid (pc_defect)."""
    _, t, env = envelope_exponent_from_series(dev, dt=dt, n_samples=n_samples)
    return t, env


def fit_exponent(t, env, decades=2.0):
    sel = (t > t[-1] / 10.0 ** decades) & (env > 0)
    if sel.sum() < 10:
        return float("nan")
    return float(np.polyfit(np.log10(t[sel]), np.log10(env[sel]), 1)[0])


def half_decade_slopes(t, env, decades=2.0):
    """Local slopes over the half-decade windows of the last `decades`."""
    out = []
    for i in range(int(2 * decades)):
        lo = t[-1] / 10.0 ** (decades - 0.5 * i)
        hi = t[-1] / 10.0 ** (decades - 0.5 * (i + 1))
        m = (t >= lo) & (t <= hi) & (env > 0)
        if m.sum() < 5:
            continue
        s = float(np.polyfit(np.log10(t[m]), np.log10(env[m]), 1)[0])
        out.append({"t_lo": float(lo), "t_hi": float(hi), "slope": s})
    return out


def slope_spread(sl):
    if len(sl) < 2:
        return float("nan")
    v = [s["slope"] for s in sl]
    return float(max(v) - min(v))


def env_exponent(x):
    e, _, _ = envelope_exponent_from_series(np.abs(x))
    return float(e)


# --------------------------------------------------------------------------
#  the rule
# --------------------------------------------------------------------------
def make_channels(kappa, u_ref, u_own, signed, vphys, v0abs=1.0):
    """The candidate channels of step 1, on whatever prefix is passed in."""
    ch = []
    for name, u in (("V(ref)", u_ref), ("V(own)", u_own)):
        w = kappa * np.conj(u)
        S = np.cumsum(w)
        lin = 2.0 * vphys * np.real(S) / v0abs ** 2
        quad = np.abs(S) ** 2 / v0abs ** 2
        ch.append({"name": name, "c": w, "S": S,
                   "recon": np.abs(lin + quad),
                   "terms": {"2|v_phys Re S|": np.abs(lin), "|S|^2": quad}})
    g = np.diff(np.concatenate([[0.0], signed]))
    S = np.cumsum(g)
    ch.append({"name": "E", "c": g.astype(complex), "S": S,
               "recon": np.abs(S), "terms": {"|S|": np.abs(S)}})
    return ch


def channel_search(kappa, u_ref, u_own, signed, vphys, tau, n_short, n_full,
                   dt=DT, deterministic=True):
    """Steps G0, 1, 2, G1, G2, 3, G3 of the rule.  Returns a report dict."""
    rep = {"n_short": int(n_short), "n_full": int(n_full),
           "short_gyros": n_short * dt / TWO_PI,
           "full_gyros": n_full * dt / TWO_PI}

    # ---- G0: extrapolation gate ------------------------------------------
    t_short, t_full = n_short * dt, n_full * dt
    change = float(np.exp((t_full - t_short) / tau))     # B(t) = B0 exp(-t/tau)
    rep["G0_coefficient_change"] = change
    rep["G0_limit"] = COEF_FACTOR
    rep["G0_pass"] = bool(change <= COEF_FACTOR)
    rep["G0_margin_factor"] = COEF_FACTOR / change

    # ---- 1, 2, G1, G2 ----------------------------------------------------
    dev = np.abs(signed[:n_short])
    t_env, env = envelope_series(dev)
    rows = []
    for c in make_channels(kappa[:n_short], u_ref[:n_short], u_own[:n_short],
                           signed[:n_short], vphys[:n_short]):
        a_hat, H_hat = estimate_aH(c["c"][None, :])
        p_hat = max(0.0, a_hat + H_hat)
        # G1 -- carriage
        m = (dev > 0) & (c["recon"] > 0)
        m &= np.arange(n_short) > n_short / 10.0
        rho = float(np.median(np.abs(np.log10(c["recon"][m] / dev[m]))))
        g1 = bool(rho <= TOL_REC)
        # G2 -- cancellation
        p_rec = env_exponent(c["recon"])
        p_terms = max(env_exponent(v) for v in c["terms"].values())
        g2 = bool(p_rec <= p_terms + TOL_P)
        rows.append({"channel": c["name"], "a_hat": a_hat, "H_hat": H_hat,
                     "p_hat": p_hat,
                     "G1_rho_decades": rho, "G1_pass": g1,
                     "G1_margin_decades": TOL_REC - rho,
                     "G2_p_reconstruction": p_rec, "G2_p_terms": p_terms,
                     "G2_pass": g2, "G2_margin": p_terms + TOL_P - p_rec,
                     "term_exponents": {k: env_exponent(v)
                                        for k, v in c["terms"].items()},
                     "survives": bool(g1 and g2)})
    rep["channels"] = rows

    # ---- G3: power law on the short run ----------------------------------
    sl = half_decade_slopes(t_env, env)
    rep["G3_half_decade_slopes_short"] = sl
    rep["G3_spread"] = slope_spread(sl)
    rep["G3_limit"] = TOL_SPREAD
    rep["G3_applies"] = bool(deterministic)
    rep["G3_pass"] = bool(rep["G3_spread"] <= TOL_SPREAD) if deterministic \
        else True
    rep["p_fit_short"] = fit_exponent(t_env, env)

    # ---- 3: select -------------------------------------------------------
    survivors = [r for r in rows if r["survives"]]
    rep["survivors"] = [r["channel"] for r in survivors]
    if not rep["G0_pass"]:
        rep["verdict"] = "NO EXPONENT"
        rep["reason"] = "G0: the coefficient changes by %.1f over the gap" % change
        rep["selected"] = None
        rep["p_pred"] = None
    elif not survivors:
        rep["verdict"] = "NO EXPONENT"
        rep["reason"] = "no channel passes G1 and G2"
        rep["selected"] = None
        rep["p_pred"] = None
    elif not rep["G3_pass"]:
        rep["verdict"] = "NO EXPONENT"
        rep["reason"] = ("G3: half-decade slopes spread by %.2f"
                         % rep["G3_spread"])
        rep["selected"] = survivors[0]["channel"]
        rep["p_pred"] = None
    else:
        spread_p = max(r["p_hat"] for r in survivors) - \
            min(r["p_hat"] for r in survivors)
        rep["p_hat_spread_over_survivors"] = spread_p
        rep["verdict"] = "EXPONENT"
        rep["reason"] = ("survivors disagree by %.2f" % spread_p) \
            if spread_p > TOL_P else ""
        rep["selected"] = survivors[0]["channel"]
        rep["p_pred"] = survivors[0]["p_hat"]
    return rep


# --------------------------------------------------------------------------
#  bench A -- synthetic
# --------------------------------------------------------------------------
def fgn_batch(Hu, n, rng, nb):
    """Verbatim from p_law/pl_protocol.py; the control below asserts it."""
    if abs(Hu - 0.5) < 1e-12:
        return (rng.standard_normal((nb, n))
                + 1j * rng.standard_normal((nb, n))) / np.sqrt(2)
    k = np.arange(0, n + 1, dtype=float)
    g = 0.5 * (np.abs(k + 1) ** (2 * Hu) - 2 * np.abs(k) ** (2 * Hu)
               + np.abs(k - 1) ** (2 * Hu))
    row = np.concatenate([g, g[-2:0:-1]])
    m = row.size
    lam = np.maximum(np.fft.fft(row).real, 0.0)
    amp = np.sqrt(lam / (2.0 * m))
    V = rng.standard_normal((nb, m)) + 1j * rng.standard_normal((nb, m))
    Y = np.fft.fft(amp[None, :] * V, axis=1)
    return (np.sqrt(2.0) * Y.real[:, :n]
            + 1j * np.sqrt(2.0) * Y.imag[:, :n]) / np.sqrt(2)


SYNTH_N = 1 << 19
SYNTH_KAP = 1e-9
SYNTH_CASES = [(0.0, 0.5), (0.25, 0.5), (0.4, 0.5), (-0.25, 0.5),
               (0.0, 0.8), (0.25, 0.8), (0.0, 0.7), (0.5, 1.0)]
STALLED_CASES = [(-0.6, 0.5), (-0.4, 0.2)]


def bench_a_control():
    """Reproduce pl_protocol.json's (a,H) columns bit for bit."""
    stored = json.load(open(os.path.join(PL, "pl_protocol.json"),
                            encoding="utf-8"))["protocol"]
    k = np.arange(1, SYNTH_N + 1, dtype=float)
    short = SYNTH_N >> 4
    bad = []
    rows = []
    for row in stored:
        a, Hu = row["a_true"], row["H_true"]
        if Hu == 1.0:
            w = np.exp(1j * np.pi / 4) * np.ones((32, SYNTH_N))
        else:
            rg = np.random.default_rng(80000 + int(100 * a) + int(1000 * Hu))
            w = fgn_batch(Hu, SYNTH_N, rg, 32)
        w = w * (SYNTH_KAP * k ** a)[None, :]
        a_s, H_s = estimate_aH(w[:, :short])
        a_f, H_f = estimate_aH(w)
        got = {"a_hat_short": a_s, "H_hat_short": H_s,
               "a_hat_full": a_f, "H_hat_full": H_f}
        for key, v in got.items():
            if v != row[key]:
                bad.append("pl_protocol.%s,%s.%s" % (a, Hu, key))
        rows.append({"a_true": a, "H_true": Hu, **got})
    return rows, bad


def bench_a_rule():
    """Run the rule on single realizations, velocity channel against E."""
    k = np.arange(1, SYNTH_N + 1, dtype=float)
    short = SYNTH_N >> 4
    t = np.arange(1, SYNTH_N + 1, dtype=float) * DT
    rows = []
    for (a, Hu) in SYNTH_CASES + STALLED_CASES:
        if Hu == 1.0:
            w = np.exp(1j * np.pi / 4) * np.ones(SYNTH_N)
        else:
            rg = np.random.default_rng(4242 + int(100 * a) + int(1000 * Hu))
            w = fgn_batch(Hu, SYNTH_N, rg, 1)[0]
        w = w * (SYNTH_KAP * k ** a)
        S = np.cumsum(w)
        signed = 2.0 * np.real(S) + np.abs(S) ** 2     # z0 = 1, demodulated
        kappa = w                                      # already demodulated
        u = np.ones(SYNTH_N, dtype=complex)
        vph = np.ones(SYNTH_N)
        det = (Hu == 1.0)
        rep = channel_search(kappa, u, u, signed, vph, np.inf, short, SYNTH_N,
                             deterministic=det)
        p_meas, t_env, env = envelope_exponent_from_series(np.abs(signed))
        sl_full = half_decade_slopes(t_env, env)
        per = {r["channel"]: r["p_hat"] for r in rep["channels"]}
        rows.append({"a_true": a, "H_true": Hu, "p_true": max(0.0, a + Hu),
                     "deterministic_drive": det,
                     "p_measured_envelope_full": p_meas,
                     "full_half_decade_spread": slope_spread(sl_full),
                     "verdict": rep["verdict"], "selected": rep["selected"],
                     "p_pred": rep["p_pred"],
                     "p_hat_per_channel": per,
                     "survivors": rep["survivors"],
                     "G1_rho": {r["channel"]: r["G1_rho_decades"]
                                for r in rep["channels"]},
                     "G2_margin": {r["channel"]: r["G2_margin"]
                                   for r in rep["channels"]}})
    return rows


# --------------------------------------------------------------------------
#  bench B -- the learned corrector
# --------------------------------------------------------------------------
def legacy_pc_defect_fields(kappa, zb_base, zb_own, signed, n):
    """The subset of pc_defect.json Sections 4.3 and 6 quote, recomputed with
    pc_defect's own arithmetic (v_phys taken as 1 in the reconstruction)."""
    out = {}
    dev = np.abs(signed)
    p_meas, _, _ = envelope_exponent_from_series(dev)
    out["p_measured_envelope"] = p_meas
    t = np.arange(1, n + 1, dtype=float) * DT
    sub = sub_index(n)
    for fr, zb in (("V_frameA_unperturbed", zb_base),
                   ("V_frameB_comoving", zb_own)):
        u = zb / np.abs(zb)
        w = kappa * np.conj(u)
        a_s, H_s = estimate_aH(w[:n >> 4][None, :])
        a_f, H_f = estimate_aH(w[None, :])
        Sw = np.cumsum(w)
        dev_rec = np.abs(2.0 * np.real(Sw) + np.abs(Sw) ** 2)
        pr, _, _ = envelope_exponent_from_series(dev_rec)
        out[fr] = {"a_hat_short": a_s, "H_hat_short": H_s,
                   "p_pred_short_(a+H)": max(0.0, a_s + H_s),
                   "a_hat_full": a_f, "H_hat_full": H_f,
                   "p_pred_full_(a+H)": max(0.0, a_f + H_f),
                   "slope_|S_n|_last2dec": loglog_slope(t[sub],
                                                       np.abs(Sw[sub])),
                   "|S_N|": float(abs(Sw[-1])),
                   "p_reconstructed_envelope": pr,
                   "reconstruction_ratio_final": float(dev_rec[-1]
                                                       / max(dev[-1], 1e-300))}
    g = np.diff(np.concatenate([[0.0], signed]))
    a_s, H_s = estimate_aH(g[:n >> 4][None, :].astype(complex))
    a_f, H_f = estimate_aH(g[None, :].astype(complex))
    Sg = np.cumsum(g)
    out["E_energy_increment"] = {
        "a_hat_short": a_s, "H_hat_short": H_s,
        "p_pred_short_(a+H)": max(0.0, a_s + H_s),
        "a_hat_full": a_f, "H_hat_full": H_f,
        "p_pred_full_(a+H)": max(0.0, a_f + H_f),
        "identity_check_max|dev-|sum g||": float(np.max(np.abs(np.abs(Sg)
                                                               - dev))),
        "slope_|S_n|_last2dec": loglog_slope(t[sub], np.abs(Sg[sub]))}
    return out


def compare(stored, got, path, bad, rtol=0.0):
    for key, want in stored.items():
        if key not in got:
            continue
        have = got[key]
        if isinstance(want, dict):
            compare(want, have, path + "." + key, bad, rtol)
        elif isinstance(want, float):
            ok = (have == want) if rtol == 0 else \
                abs(have - want) <= rtol * max(abs(want), 1e-300)
            if not ok:
                bad.append("%s.%s: %r vs stored %r" % (path, key, have, want))


def bench_b():
    import symproj as S
    fwd = S.load_forward()
    out, bad = {}, []
    stored_def = json.load(open(os.path.join(PC, "pc_defect.json"),
                                encoding="utf-8"))["runs"]
    stored_split = json.load(open(os.path.join(PC, "pc_split.json"),
                                  encoding="utf-8"))["runs"]
    for cname, tau in (("quasistatic", TAU_QUASI), ("paper", TAU_PAPER)):
        t0 = time.time()
        base = run_instrumented("boris", tau, N_STEPS, fwd)
        res = run_instrumented("proj", tau, N_STEPS, fwd)
        n = N_STEPS
        kappa, signed = res["kappa"], res["signed"]
        u_ref = base["zb"] / np.abs(base["zb"])
        u_own = res["zb"] / np.abs(res["zb"])
        vphys = np.exp(-np.arange(1, n + 1, dtype=float) * DT / (2.0 * tau))

        legacy = legacy_pc_defect_fields(kappa, base["zb"], res["zb"],
                                         signed, n)
        if FULL:
            compare(stored_def["%s/proj" % cname], legacy,
                    "pc_defect.%s/proj" % cname, bad)
            # the three envelope exponents Section 4.3 prints for the
            # decaying-field run, from pc_split.json
            u = u_ref
            Sw = np.cumsum(kappa * np.conj(u))
            got_split = {
                "envelope_exponent_|S|": env_exponent(np.abs(Sw)),
                "envelope_exponent_|Re S|": env_exponent(np.real(Sw)),
                "envelope_exponent_2|Re S|": env_exponent(2.0 * np.real(Sw)),
                "envelope_exponent_|S|^2": env_exponent(np.abs(Sw) ** 2),
                "envelope_exponent_full_reconstruction":
                    env_exponent(2.0 * np.real(Sw) + np.abs(Sw) ** 2),
                "a_hat": legacy["V_frameA_unperturbed"]["a_hat_full"],
                "H_hat": legacy["V_frameA_unperturbed"]["H_hat_full"]}
            compare(stored_split[cname], got_split,
                    "pc_split.%s" % cname, bad)
            out.setdefault("split_recomputed", {})[cname] = got_split

        rule = channel_search(kappa, u_ref, u_own, signed, vphys, tau,
                              n // SHORT_DIV, n)
        p_meas, t_env, env = envelope_exponent_from_series(np.abs(signed))
        rule["p_measured_full_horizon"] = p_meas
        rule["full_half_decade_slopes"] = half_decade_slopes(t_env, env)
        rule["full_half_decade_spread"] = slope_spread(
            rule["full_half_decade_slopes"])
        rule["full_curve_is_power_law"] = bool(
            rule["full_half_decade_spread"] <= TOL_SPREAD)
        if rule["p_pred"] is None:
            rule["probe_verdict"] = rule["verdict"]
        elif not rule["full_curve_is_power_law"]:
            rule["probe_verdict"] = "NO EXPONENT (full horizon)"
        else:
            rule["probe_verdict"] = ("PASS" if abs(rule["p_pred"] - p_meas)
                                     <= TOL_P else "FAIL")
            rule["prediction_error"] = rule["p_pred"] - p_meas
        rule["legacy_pc_defect_fields"] = legacy
        rule["seconds"] = time.time() - t0
        out[cname] = rule
        print("[%s] %s  selected=%s  p_pred=%s  p_measured=%.4f  (%.0f s)"
              % (cname, rule["probe_verdict"], rule["selected"],
                 rule["p_pred"], p_meas, rule["seconds"]), flush=True)
        for r in rule["channels"]:
            print("    %-7s a=%+.4f H=%+.4f p=%.4f | G1 rho=%.3f dec %s | "
                  "G2 p_rec=%.3f vs terms %.3f %s"
                  % (r["channel"], r["a_hat"], r["H_hat"], r["p_hat"],
                     r["G1_rho_decades"], "ok " if r["G1_pass"] else "REJECT",
                     r["G2_p_reconstruction"], r["G2_p_terms"],
                     "ok" if r["G2_pass"] else "REJECT"), flush=True)
        del base, res
    return out, bad


# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    bad = []
    results = {"setup": {"gyros": N_GYR, "n_steps": N_STEPS, "dt": DT,
                         "short_divisor": SHORT_DIV,
                         "thresholds": {"TOL_P": TOL_P,
                                        "TOL_SPREAD": TOL_SPREAD,
                                        "TOL_REC_decades": TOL_REC,
                                        "COEF_FACTOR": COEF_FACTOR}}}

    print("bench A control (pl_protocol.json reproduction)", flush=True)
    ctrl, bad_a = bench_a_control()
    results["bench_a_control"] = ctrl
    bad += bad_a
    print("   %s" % ("all four columns reproduce for all eight cases"
                     if not bad_a else "MISMATCH: %s" % bad_a), flush=True)

    print("bench A rule (single realizations)", flush=True)
    rows = bench_a_rule()
    results["bench_a_rule"] = rows
    for r in rows:
        print("   (%+.2f,%.1f) true %.3f  verdict %-11s ch=%-7s p=%s | "
              "measured %.3f | per-channel %s"
              % (r["a_true"], r["H_true"], r["p_true"], r["verdict"],
                 str(r["selected"]),
                 "%.3f" % r["p_pred"] if r["p_pred"] is not None else "  -  ",
                 r["p_measured_envelope_full"],
                 {k: round(v, 3) for k, v in r["p_hat_per_channel"].items()}),
              flush=True)

    print("bench B (learned corrector, %d gyro-orbits)" % N_GYR, flush=True)
    runs, bad_b = bench_b()
    results["bench_b"] = runs
    bad += bad_b

    # ---- the checks the report rests on ----------------------------------
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print("  %s %s: %s" % ("OK " if ok else "BAD", name, detail),
              flush=True)

    if FULL:
        q = runs["quasistatic"]
        p = runs["paper"]
        check("quasistatic: velocity channels rejected",
              all(not r["survives"] for r in q["channels"]
                  if r["channel"].startswith("V")),
              "V(ref) G1 rho = %.2f decades, V(own) G1 rho = %.2f decades"
              % (q["channels"][0]["G1_rho_decades"],
                 q["channels"][1]["G1_rho_decades"]))
        check("quasistatic: rule selects E and returns an exponent",
              q["selected"] == "E" and q["p_pred"] is not None,
              "selected %s, p_pred = %s" % (q["selected"], q["p_pred"]))
        check("quasistatic: prediction agrees with the measured exponent",
              q["p_pred"] is not None
              and abs(q["p_pred"] - q["p_measured_full_horizon"]) <= TOL_P,
              "p_pred %.4f vs measured %.4f, error %.4f"
              % (q["p_pred"], q["p_measured_full_horizon"],
                 q["p_pred"] - q["p_measured_full_horizon"]))
        check("quasistatic: the first version's p = 0 came from a rejected "
              "channel",
              abs(q["legacy_pc_defect_fields"]["V_frameA_unperturbed"]
                  ["p_pred_full_(a+H)"]) < 1e-12,
              "V(ref) full-run (a+H)_+ = %.4f"
              % q["legacy_pc_defect_fields"]["V_frameA_unperturbed"]
                ["p_pred_full_(a+H)"])
        check("decaying field: the rule returns no exponent",
              p["p_pred"] is None,
              "verdict %s -- %s" % (p["verdict"], p["reason"]))
        check("decaying field: the rule does not return 0.34",
              p["p_pred"] is None
              or abs(p["p_pred"] - 0.3376) > TOL_P,
              "V(ref) alone would have given %.4f"
              % max(0.0, runs["paper"]["channels"][0]["a_hat"]
                    + runs["paper"]["channels"][0]["H_hat"]))
        check("decaying field: the full-horizon curve is not a power law",
              not p["full_curve_is_power_law"],
              "half-decade slopes spread by %.2f over the last two decades"
              % p["full_half_decade_spread"])

    # bench A: the rule must keep the velocity channel and must return the
    # velocity channel's own estimate, unchanged.  Whether that estimate is
    # accurate is not tested here -- pl_protocol.json tests it, and the eight
    # rows of it are reproduced bit for bit above.
    for r in results["bench_a_rule"]:
        check("synthetic (%+.2f,%.1f): velocity channel kept"
              % (r["a_true"], r["H_true"]),
              r["selected"] == "V(ref)" and r["p_pred"] is not None
              and r["p_pred"] == r["p_hat_per_channel"]["V(ref)"],
              "selected %s, p_pred %s, survivors %s"
              % (r["selected"], r["p_pred"], r["survivors"]))
        if r["a_true"] + r["H_true"] <= 0.05:
            check("stalled (%+.2f,%.1f): p = 0 still returned, from a channel "
                  "that passes G1" % (r["a_true"], r["H_true"]),
                  r["p_pred"] is not None and r["p_pred"] < 0.15,
                  "verdict %s, selected %s, p_pred %s"
                  % (r["verdict"], r["selected"], r["p_pred"]))

    # the measured justification of the V-before-E order
    err_v = [abs(r["p_hat_per_channel"]["V(ref)"] - r["p_true"])
             for r in results["bench_a_rule"]]
    err_e = [abs(r["p_hat_per_channel"]["E"] - r["p_true"])
             for r in results["bench_a_rule"]]
    results["channel_order_evidence"] = {
        "mean_abs_error_V": float(np.mean(err_v)),
        "mean_abs_error_E": float(np.mean(err_e)),
        "max_abs_error_V": float(np.max(err_v)),
        "max_abs_error_E": float(np.max(err_e)),
        "n_cases": len(err_v)}
    check("V before E is the better order on the bench",
          np.mean(err_v) < np.mean(err_e),
          "mean |p_hat - p_true| over %d cases: V %.3f, E %.3f; worst case "
          "V %.3f, E %.3f" % (len(err_v), np.mean(err_v), np.mean(err_e),
                              np.max(err_v), np.max(err_e)))

    results["checks"] = checks
    results["stored_mismatches"] = bad
    results["seconds"] = time.time() - t0
    json.dump(results, open(os.path.join(HERE, "pb4_channel.json"), "w"),
              indent=1)

    failed = [c["check"] for c in checks if not c["pass"]]
    if bad:
        print("\nSTORED NUMBERS MOVED:\n  " + "\n  ".join(bad))
    if failed:
        print("\nRULE CHECKS FAILED:\n  " + "\n  ".join(failed))
    if bad or failed:
        return 1
    print("\nall stored fields reproduced; every rule check holds  (%.0f s)"
          % results["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
