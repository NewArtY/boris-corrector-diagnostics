"""sw2_spectra.py -- the residual spectrum in and out of the signal band.

The claim under test belongs to the first author:

    in the band of the slow physical signal, f/Omega_c < 0.2, the residual
    power of the classical scheme stands about six orders of magnitude above
    both corrected schemes, and the corrector pushes what is left out of that
    band into narrow harmonic lines at Omega_c and above.

What this script adds to it is vps4, a flop budget, and the closed-form
reference of `sw_common.py`.  Everything measured here is fixed by the
pre-registration before any of it was written: two residual series (position
and energy, both reported for every scheme), three power figures per scheme
(in band, out of band, total), three comparisons, five schemes, and a sweep of
the band edge so that the result is a function of the threshold rather than a
number at one threshold.

THE THREE COMPARISONS

  equal_total_flops   the basis of the gate.  Every scheme gets the flop
                      budget of the corrector's run and spends what it saves
                      on a finer step.
  (equal flops/step)  not a separate set of runs, and the accounting says why:
                      the per-step cost of an explicit scheme does not depend
                      on its step, so equalising flops per unit of simulated
                      time gives the same step sizes as equalising flops per
                      run at a fixed horizon.  The per-step ledger is written
                      into the output under "cost" and is the mechanism behind
                      the first comparison, not an independent measurement of
                      it.  This is reported rather than papered over.
  fixed_step          Omega h = 0.3 for every scheme: the first author's
                      setting, verbatim, and the setting Table 4 is scored in.

THREE HORIZONS, AND WHY THE BAND FORCES THEM

A record of N samples at spacing h resolves bins spaced 2 pi/(N h), so the
number of independent bins strictly inside f/Omega_c < 0.2 is one fifth of the
number of gyro-orbits in the record.  The window of Table 4 is 19.1
gyro-orbits and therefore carries 3.8 bins in the band the claim is about.
Twenty bins need 100 gyro-orbits.  Both are run, and so is a long record of
391 gyro-orbits, and the three are reported side by side because the
difference between them is itself part of the answer.

Writes sw2_spectra.json; exits non-zero if a rerun stops reproducing it.
Usage: python sw2_spectra.py [--force]
"""
import json
import os
import sys

import numpy as np

import sw_common as C
import schemes as S
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sw2_spectra.json")

SCHEMES = ["boris", "corrector", "vps2", "vps4", "gl4"]
CHANNELS = ["position", "energy"]

#: half-width of the window placed on each integer harmonic of Omega_c when
#: asking how much of the residual sits in narrow lines
HARMONIC_HALFWIDTH = 0.05
N_HARMONICS = 10


def equal_cost_multipliers():
    """The largest integer sub-stepping each scheme can afford per corrector step.

    Explicit schemes have a per-step cost that does not depend on the step, so
    m follows from the flop model directly.  gl4 is iterative: its per-step
    cost is priced from its measured mean iteration count, which depends on the
    step, so m is found by iterating the two until they agree.  A short
    calibration run of twenty output steps is enough because the count is flat.
    """
    budget = float(C.FLOPS_CORRECTOR)
    m = {"boris": int(budget // C.FLOPS_BORIS),
         "vps2": int(budget // C.FLOPS_VPS2),
         "vps4": int(budget // C.FLOPS_VPS4)}
    fps = {"boris": float(C.FLOPS_BORIS), "vps2": float(C.FLOPS_VPS2),
           "vps4": float(C.FLOPS_VPS4)}
    mg, trace = 100, []
    for _ in range(8):
        _, _, meta = C.run_classical("gl4", C.DT / mg, mg, 20)
        f = S.flops_gl4(meta["mean_iters"])
        trace.append({"m": mg, "mean_iters": meta["mean_iters"],
                      "flops_per_step": float(f)})
        nxt = int(budget // f)
        if nxt == mg:
            break
        mg = nxt
    m["gl4"] = mg
    fps["gl4"] = trace[-1]["flops_per_step"]
    m["corrector"] = 1
    fps["corrector"] = float(C.FLOPS_CORRECTOR)
    return m, fps, trace


def harmonic_content(series, dt, window="hann"):
    """How much of the residual sits in narrow lines at Omega_c and above.

    Returns the fraction of the total power inside +-0.05 of each integer
    multiple of Omega_c, the summed fraction over the first ten, and the three
    largest bins of the spectrum.  This is the second half of the claim: that
    the corrector does not remove the residual but moves it into lines.
    """
    nu, Sx = C.psd(series, dt, window)
    tot = float(Sx.sum())
    if tot <= 0:
        return {"per_harmonic": [], "sum_fraction": 0.0, "top_bins": []}
    per = []
    for k in range(1, N_HARMONICS + 1):
        msk = np.abs(nu - k) <= HARMONIC_HALFWIDTH
        per.append(float(Sx[msk].sum()) / tot if msk.any() else 0.0)
    order = np.argsort(Sx)[::-1][:3]
    top = [{"nu": float(nu[i]), "fraction": float(Sx[i]) / tot}
           for i in order]
    return {"per_harmonic": per, "sum_fraction": float(sum(per)),
            "top_bins": top}


def measure(R, V, Rr, Vr, n_use):
    """Every declared figure for one run, on both channels, under both windows."""
    ch = C.channels(R[:n_use], V[:n_use], Rr[:n_use], Vr[:n_use])
    rec = {}
    for name, series in ch.items():
        ok, _, _ = C.spectral_selfcheck(series, C.DT)
        bp = C.band_powers(series, C.DT)
        bh = C.band_powers(series, C.DT, window="blackmanharris")
        sweep, _ = C.band_sweep(series, C.DT, C.SWEEP_EDGES)
        rec[name] = {**bp,
                     "frac_in_band": bp["p_band"] / bp["p_total"]
                     if bp["p_total"] > 0 else float("nan"),
                     "time": C.time_metrics(series),
                     "harmonics": harmonic_content(series, C.DT),
                     "sweep_p_band": sweep,
                     "parseval_ok": bool(ok),
                     # the leakage control: same series, a window whose
                     # sidelobes are 61 dB lower
                     "bh": {"p_band": bh["p_band"], "p_out": bh["p_out"],
                            "p_total": bh["p_total"],
                            "frac_in_band": bh["p_band"] / bh["p_total"]
                            if bh["p_total"] > 0 else float("nan"),
                            "ratio_to_hann_band": bh["p_band"] / bp["p_band"]
                            if bp["p_band"] > 0 else float("nan")}}
    return rec


def main():
    force = "--force" in sys.argv
    m, fps, gl4_trace = equal_cost_multipliers()

    n_max = max(C.HORIZONS.values())
    ts = np.arange(n_max + 1) * C.DT
    print("building the closed-form basis at %d samples ..." % (n_max + 1))
    basis = C.bessel_basis(ts)
    Rr_all, Vr_all = C.exact_from_basis(basis, C.R0, C.V0)

    out = {"meta": {
        "claim_owner": "first author",
        "claim": "in f/Omega_c < 0.2 the classical residual power exceeds both "
                 "corrected schemes by about six orders of magnitude, and the "
                 "corrector moves what is left into narrow lines at Omega_c "
                 "and above",
        "band": C.BAND, "dt": C.DT, "tau": C.TAU,
        "r0": list(C.R0), "v0": list(C.V0),
        "reference": "closed form (Bessel order 0 in the Larmor frame), basis "
                     "in mpmath at 40 digits; see sw1_reference.json for why "
                     "DOP853 at rtol 1e-12 cannot decide the equal-cost case",
        "psd": "one-sided periodogram, one segment, Hann window; Parseval "
               "checked on every series, so the total power of the position "
               "channel is the square of its window-weighted root-mean-square "
               "error and the band ratio factors exactly into an amplitude "
               "ratio and a concentration factor.  Every in-band figure is "
               "repeated under a four-term Blackman-Harris window, whose "
               "sidelobes are 61 dB lower, as the control on leakage from the "
               "line at Omega_c into the band",
        "bins_in_band_identity": "n_bins = 0.2 * (gyro-orbits in the record)",
        "n_random_draws": 0,
    }}

    out["cost"] = {
        "flops_per_step": fps,
        "equal_cost_substeps": m,
        "equal_cost_step": {k: C.DT / v for k, v in m.items()},
        "flops_per_corrector_step": {k: fps[k] * m[k] for k in m},
        "budget_ratio": {k: fps[k] * m[k] / float(C.FLOPS_CORRECTOR)
                         for k in m},
        "gl4_fixed_point_trace": gl4_trace,
        "note": "equal flops per step and equal flops per run give the same "
                "step sizes at a fixed horizon, because the per-step cost of "
                "these schemes does not depend on the step",
    }
    print("cost ledger:")
    for k in SCHEMES:
        print("  %-10s %9.1f flop/step  m=%5d  h=%.4e  budget %.4f"
              % (k, fps[k], m[k], C.DT / m[k],
                 out["cost"]["budget_ratio"][k]))

    out["horizons"] = {}
    for hz, n in C.HORIZONS.items():
        gyro = n * C.DT / (2.0 * np.pi)
        rec = {"n_samples": n, "t_final": n * C.DT, "gyro_orbits": gyro,
               "df_over_omega_c": 2.0 * np.pi / (n * C.DT),
               "bins_in_band": int(np.sum(
                   2.0 * np.pi * np.arange(n // 2 + 1) / (n * C.DT) < C.BAND)),
               "runs": {}}
        print("\n%s: %d samples, t=%.1f, %.1f gyro-orbits, %d bins in band"
              % (hz, n, n * C.DT, gyro, rec["bins_in_band"]))
        for mode in ("fixed_step", "equal_total_flops"):
            rec["runs"][mode] = {}
            for name in SCHEMES:
                mm = 1 if mode == "fixed_step" else m[name]
                if name == "corrector":
                    R, V, meta = C.run_corrector(n)
                else:
                    R, V, meta = C.run_classical(name, C.DT / mm, mm, n)
                r = measure(R, V, Rr_all, Vr_all, n)
                r["_run"] = {"substeps": mm, "h": C.DT / mm,
                             "flops_total": meta["flops_per_step"] * mm * n,
                             **{k: v for k, v in meta.items()
                                if k != "flops_per_step"}}
                rec["runs"][mode][name] = r
                print("  %-16s %-10s band=%.4e out=%.4e total=%.4e  rms=%.4e"
                      % (mode, name, r["position"]["p_band"],
                         r["position"]["p_out"], r["position"]["p_total"],
                         r["position"]["time"]["rms"]))
        out["horizons"][hz] = rec

    # ------------------------------------------------------------ ratios ----
    # The decomposition of prediction P3.  For any pair the identity
    #     P_band(A)/P_band(B) = [P_tot(A)/P_tot(B)] * [c_A / c_B]
    # holds exactly, with c the in-band fraction; the first factor is the
    # square of the ratio of root-mean-square errors and the second is the
    # concentration factor.  It is an identity because the total power of the
    # position channel is the mean square position error by construction.
    ratios = {}
    for hz, rec in out["horizons"].items():
        ratios[hz] = {}
        for mode, runs in rec["runs"].items():
            ratios[hz][mode] = {}
            for ch in CHANNELS:
                base = runs["boris"][ch]
                d = {}
                for name in SCHEMES:
                    if name == "boris":
                        continue
                    o = runs[name][ch]
                    rb = base["p_band"] / o["p_band"] if o["p_band"] > 0 else float("inf")
                    rt = base["p_total"] / o["p_total"] if o["p_total"] > 0 else float("inf")
                    cc = ((base["p_band"] / base["p_total"])
                          / (o["p_band"] / o["p_total"])) if o["p_band"] > 0 else float("inf")
                    rb_bh = (base["bh"]["p_band"] / o["bh"]["p_band"]
                             if o["bh"]["p_band"] > 0 else float("inf"))
                    d[name] = {
                        "band_ratio": rb, "band_ratio_orders": _log10(rb),
                        "band_ratio_blackmanharris": rb_bh,
                        "band_ratio_orders_blackmanharris": _log10(rb_bh),
                        "total_ratio": rt, "total_ratio_orders": _log10(rt),
                        "amplitude_ratio": np.sqrt(rt) if np.isfinite(rt) else float("inf"),
                        "plain_rms_ratio": (base["time"]["rms"]
                                            / o["time"]["rms"]),
                        "concentration_factor": cc,
                        "concentration_orders": _log10(cc),
                        "identity_residual": abs(rb - rt * cc) / rb
                        if np.isfinite(rb) and rb > 0 else 0.0,
                    }
                ratios[hz][mode][ch] = d
    out["boris_over_scheme"] = ratios

    # ------------------------------------------------------------- gate -----
    g = {}
    for hz, rec in out["horizons"].items():
        r = rec["runs"]["equal_total_flops"]
        for ch in CHANNELS:
            g["%s_%s" % (hz, ch)] = {
                "p_band_vps4": r["vps4"][ch]["p_band"],
                "p_band_corrector": r["corrector"][ch]["p_band"],
                "vps4_below_corrector": bool(
                    r["vps4"][ch]["p_band"] < r["corrector"][ch]["p_band"]),
                "ratio_corrector_over_vps4":
                    r["corrector"][ch]["p_band"] / r["vps4"][ch]["p_band"]
                    if r["vps4"][ch]["p_band"] > 0 else float("inf"),
            }
    out["G0"] = g
    print("\nG0 (equal total flops, vps4 band power below the corrector's):")
    for k, v in g.items():
        print("  %-22s %s   corrector/vps4 = %.3e"
              % (k, "yes" if v["vps4_below_corrector"] else "NO",
                 v["ratio_corrector_over_vps4"]))

    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


def _log10(x):
    try:
        return float(np.log10(x)) if np.isfinite(x) and x > 0 else float("inf")
    except Exception:
        return float("inf")


if __name__ == "__main__":
    raise SystemExit(main())
