"""
s3_broadband.py -- S3/P2: broadband dB, realization unknown, statistics known.
==============================================================================
This is the only setting the plan expects to survive, because the asymmetry is
informational rather than numerical: no amount of step refinement recovers a
realization the integrator was never told.

The learned object would be the ENSEMBLE-AVERAGED dynamics (the plan, section
2: "обучается эффективная (усреднённая по реализациям) динамика"). So the
honest classical attacks target the same object:

  (b')  mean field    -- integrate with B_smooth only. dB has zero mean, so
                         this is the cheapest possible guess. Costs one run.
  (b)   Monte Carlo   -- sample N realizations from the PUBLISHED statistics,
                         integrate each, average. Costs N runs, error ~ 1/sqrt(N).
  (c)   smooth law    -- fit the ensemble-mean heating curve with a 2-3
                         parameter closed form. If this works, the effective
                         dynamics is analytic and nothing needs learning.

Attack (a) of the plan -- fitting the realization parametrically -- is not
implementable by construction: the realization is K independent phases and the
attacker is given only the PSD. That is the definition of the setting, and it
is recorded as such rather than run.

The decisive number: one hybrid run costs 4.56e7 flops (I1.1, on the real
52102-parameter DefectNet), which buys 418 vps4 runs at the working step. If
418-sample Monte Carlo already resolves the mean heating curve, a learned
effective model has nothing left to win -- it cannot be more accurate than the
ensemble it would have to be trained on.
"""
import json
import os
import numpy as np

import harness as H
from pfields import PerturbedDecaying, Broadband

HERE = os.path.dirname(os.path.abspath(__file__))

DT_FINE = 0.05          # truth: vps4 is 4th order, this is effectively exact
DT_WORK = H.DT_WORK
N_TRUTH = 400           # realizations defining the ensemble truth
RMS_SCAN = [1e-4, 3e-4, 1e-3, 3e-3]
N_SCAN = 24


def energy_curve(field, dt, t_final=H.T_FINAL):
    """Relative energy deviation on a uniform grid, plus its flop cost."""
    n = int(round(t_final / dt))
    rs, vs, ts, fl = H.run_scheme("vps4", field, dt, n)
    E = 0.5 * np.sum(vs ** 2, axis=1)
    return ts, (E - E[0]) / E[0], fl


def resample(ts, y, ts_out):
    return np.interp(ts_out, ts, y)


def make_field(rms, seed, n_modes=64):
    p = Broadband(w_lo=0.5, w_hi=5.0, n_modes=n_modes, rms=rms, seed=seed)
    return PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN, a=p.a, adot=p.adot), p


def scan_amplitude():
    """Pick dB so its heating is comparable to the smooth signal (7.5e-4)."""
    grid = np.linspace(0.0, H.T_FINAL, 601)
    rows = []
    for rms in RMS_SCAN:
        finals = []
        for s in range(N_SCAN):
            f, _ = make_field(rms, seed=1000 + s)
            ts, y, _ = energy_curve(f, DT_FINE)
            finals.append(resample(ts, y, grid))
        arr = np.array(finals)
        mean = arr.mean(axis=0)
        rows.append({"rms": rms,
                     "mean_dE_final": float(mean[-1]),
                     "mean_abs_dE_final": float(np.abs(arr[:, -1]).mean()),
                     "spread_dE_final": float(arr[:, -1].std())})
        print(f"  rms={rms:.1e}  <dE>={mean[-1]:+.3e}  "
              f"<|dE|>={np.abs(arr[:, -1]).mean():.3e}  "
              f"sd={arr[:, -1].std():.3e}")
    return rows


def main():
    out = {"meta": {"setting": "S3", "t_final": H.T_FINAL,
                    "dt_truth": DT_FINE, "dt_work": DT_WORK,
                    "n_truth_realizations": N_TRUTH,
                    "hybrid_flops_one_run":
                        H.HYBRID_FLOPS_PER_STEP * int(round(H.T_FINAL / DT_WORK))}}

    print("amplitude scan (heating vs smooth signal 7.497e-04):")
    out["amplitude_scan"] = scan_amplitude()

    # pick the amplitude whose mean |dE| is closest to the smooth signal
    target = 7.497e-4
    rms = min(out["amplitude_scan"],
              key=lambda r: abs(np.log10(max(r["mean_abs_dE_final"], 1e-300))
                                - np.log10(target)))["rms"]
    out["meta"]["rms_chosen"] = rms
    print(f"\nchosen dB rms = {rms:.2e}")

    grid = np.linspace(0.0, H.T_FINAL, 601)

    # ---------------------------------------------------- ensemble truth
    print(f"building truth ensemble ({N_TRUTH} realizations, dt={DT_FINE})...")
    truth = []
    for s in range(N_TRUTH):
        f, _ = make_field(rms, seed=s)
        ts, y, _ = energy_curve(f, DT_FINE)
        truth.append(resample(ts, y, grid))
    truth = np.array(truth)
    mean_truth = truth.mean(axis=0)
    sd_truth = truth.std(axis=0)
    sig = float(np.abs(mean_truth[len(grid) // 2:]).max())
    out["truth"] = {"mean_dE_final": float(mean_truth[-1]),
                    "sd_across_realizations_final": float(sd_truth[-1]),
                    "mean_signal_scale": sig}
    print(f"  ensemble mean dE(T) = {mean_truth[-1]:+.4e}"
          f"   sd across realizations = {sd_truth[-1]:.4e}")

    # ------------------------------------------------- attack b': mean field
    f_smooth = PerturbedDecaying(B0=1.0, tau=H.TAU_MAIN)
    ts, y_mf, fl_mf = energy_curve(f_smooth, DT_WORK)
    y_mf = resample(ts, y_mf, grid)
    err_mf = float(np.abs(y_mf - mean_truth).max())
    out["attack_mean_field"] = {"flops": fl_mf, "max_err_vs_truth_mean": err_mf,
                                "rel_err": err_mf / abs(sig)}
    print(f"  mean-field attack: err={err_mf:.4e} "
          f"({err_mf / abs(sig):.2f} of signal) at {fl_mf:.3e} flops")

    # --------------------------------------------------- attack b: Monte Carlo
    print("Monte-Carlo attack at the working step...")
    mc_runs, mc_flops = [], 0.0
    for s in range(512):
        f, _ = make_field(rms, seed=10_000 + s)     # samples, not the truth set
        ts, y, fl = energy_curve(f, DT_WORK)
        mc_runs.append(resample(ts, y, grid))
        mc_flops += fl
    mc_runs = np.array(mc_runs)
    per_run_flops = mc_flops / len(mc_runs)

    conv = []
    for n in (1, 4, 16, 64, 128, 256, 418, 512):
        est = mc_runs[:n].mean(axis=0)
        e = float(np.abs(est - mean_truth).max())
        conv.append({"n": n, "flops": per_run_flops * n,
                     "max_err_vs_truth_mean": e, "rel_err": e / abs(sig)})
        print(f"  N={n:4d}  flops={per_run_flops * n:.3e}  "
              f"err={e:.4e}  ({e / abs(sig):.3f} of signal)")
    out["attack_monte_carlo"] = {"per_run_flops": per_run_flops,
                                 "convergence": conv}

    # ------------------------------------------------- attack c: smooth law
    # Effective heating for weak broadband forcing should be diffusive:
    # <dE>(t) ~ alpha*t + beta*(adiabatic part already in the smooth field).
    A = np.vstack([grid, np.ones_like(grid)]).T
    coef, *_ = np.linalg.lstsq(A, mean_truth, rcond=None)
    fit = A @ coef
    res = float(np.abs(fit - mean_truth).max())
    out["attack_smooth_law"] = {"n_params": 2, "coef": coef.tolist(),
                                "max_residual": res, "rel_residual": res / abs(sig)}
    print(f"  2-parameter linear law: residual={res:.4e} "
          f"({res / abs(sig):.3f} of signal)")

    out["attack_parametric_fit"] = {
        "implemented": False,
        "reason": "not implementable by construction: the realization is 64 "
                  "independent phases and the attacker is given only the PSD"}

    with open(os.path.join(HERE, "s3_broadband.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote s3_broadband.json")


if __name__ == "__main__":
    main()
