"""HP4: the analysis of W12.1 -- trends, extrapolations, and equal cost.

    python hp4_report.py [--force]

Reads every job under `runs/` and `hp2_selection.json`, writes
`hp4_analysis.json`, and prints the tables the report carries.  Rerunning
recomputes and exits non-zero if the committed analysis no longer reproduces.
No number in the report is typed by hand; the tables below are its source.

WHAT IS COMPUTED
----------------
1. Grid summary.  Every configuration at the base budget, four seeds.

2. Budget trend.  For each architecture, a straight line through
   log10(rollout error) against log10(Adam steps), fitted separately to the
   median over seeds and to the best of the four seeds.  The second fit is the
   one that favours the architecture, and it is the one the extrapolation
   quotes.  The slope is the number that decides the question: with error
   ~ N^s, closing a factor F costs a budget factor F^(-1/s).  The ladder is run
   on the anchor configuration, `hp_common.LADDER_CFG_INDEX`, which is a
   recorded deviation from what was declared; the reason is in `hp_common`.

3. Capacity trend.  The same fit against log10(parameters) over the capacity
   sweep declared in `hp_common.CAPACITY_SWEEP`, and against the size of the
   training set over the sweep of `hp5_data.py`.  Budget, capacity and data are
   the three resources a referee can say were too small, and each gets a slope.

4. Extrapolation.  The Adam-step budget and the parameter count the fitted
   trends require in order to reach the vps4 row of Table `tab:family`,
   5.35e-5 Larmor radii and 2.64e-7 in energy.  Where a fitted slope is not
   negative the extrapolation is reported as unreachable, which is a stronger
   statement than any number: no amount of that resource closes the gap.

5. Equal cost.  The comparison of Section 7 is at a fixed step size, which is
   generous to the networks: they cost between 300 and 3000 times more per run.
   At equal cost the classical scheme takes proportionally smaller steps, so
   vps4 is re-run here at a ladder of step sizes down to the flop count of the
   learned schemes and the error is measured, not extrapolated.  The ladder
   also shows where double precision floors it.

6. The best checkpoint anywhere.  Along every training run the rollout was
   scored at six points.  The minimum over all of them, over every
   configuration, seed and budget of this campaign, is the smallest trajectory
   error this grid produced by any means, including means no practitioner has:
   it requires stopping at a step chosen by reading the test metric.  It is
   reported as an upper bound on what the search could have found.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hp_common as H          # noqa: E402
import ea_common as C          # noqa: E402

OUT = os.path.join(HERE, "hp4_analysis.json")
SEL = os.path.join(HERE, "hp2_selection.json")

#: step sizes at which vps4 is re-run for the equal-cost column
#: The last rung is chosen so that the flop count of the cheapest learned run
#: in this campaign falls inside the ladder rather than beyond its end.
VPS4_DT = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0005)


def load_all():
    out = []
    for p in sorted(glob.glob(os.path.join(H.RUNS, "*.json"))):
        out.append(json.load(open(p, encoding="utf-8")))
    return out


def loglog_fit(x, y):
    """Slope, intercept and R^2 of log10(y) on log10(x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n": int(m.sum())}
    lx, ly = np.log10(x[m]), np.log10(y[m])
    s, b = np.polyfit(lx, ly, 1)
    pred = s * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"slope": float(s), "intercept": float(b),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(m.sum())}


def required(fit, target, x_ref, y_ref):
    """The value of the resource at which the fitted line reaches `target`.

    Anchored on the measured point (x_ref, y_ref) rather than on the fitted
    intercept, so that the extrapolation starts from something that was
    actually run: x* = x_ref * (y_ref / target)^(-1/slope).
    """
    s = fit["slope"]
    if not np.isfinite(s) or s >= 0 or not np.isfinite(y_ref) or y_ref <= 0:
        return {"reachable": False, "slope": s,
                "reason": "the fitted trend does not decrease, so no amount of "
                          "this resource reaches the target"}
    factor = float((y_ref / target) ** (-1.0 / s))
    return {"reachable": True, "slope": s, "factor": factor,
            "value": float(x_ref * factor), "from_x": float(x_ref),
            "from_y": float(y_ref), "target": float(target)}


# ------------------------------------------------------- equal-cost vps4 ---
def vps4_ladder():
    """vps4 of `experiments/classical/schemes.py` at a ladder of step sizes.

    Same field, same window, same reference and same scoring as
    `classical/run.py`; the dt = 0.3 row must reproduce the committed
    workprecision row, and the function asserts that before returning.
    """
    ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
    CLS = os.path.join(ROOT, "experiments", "classical")
    for p in (ROOT, CLS):
        if p not in sys.path:
            sys.path.insert(0, p)
    from fields import DecayingField
    import schemes as S
    from scipy.integrate import solve_ivp

    R0 = np.array([1.0, 0.0, 0.0])
    V0 = np.array([0.0, 1.0, 0.0])
    field = DecayingField(B0=1.0, tau=C.TAU_PAPER)

    def ref(t_eval):
        def rhs(t, y):
            r, v = y[:3], y[3:]
            E = np.atleast_1d(field.E(r, t)).ravel()
            B = np.atleast_1d(field.B(r, t)).ravel()
            return np.concatenate([v, (C.Q / C.M) * (E + np.cross(v, B))])
        sol = solve_ivp(rhs, (0.0, C.T_FINAL), np.concatenate([R0, V0]),
                        method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t_eval)
        assert sol.success, sol.message
        return sol.y[:3].T, sol.y[3:].T

    rows = []
    for dt in VPS4_DT:
        n = int(round(C.T_FINAL / dt))
        ts = np.linspace(0.0, C.T_FINAL, n + 1)
        r_ref, v_ref = ref(ts)
        rs, vs, tt = S.integrate(S.make_vps4(field), R0, V0, dt, n)
        E = 0.5 * np.sum(vs ** 2, axis=1)
        E_ref = 0.5 * np.sum(v_ref ** 2, axis=1)
        half = len(ts) // 2
        pos = np.linalg.norm(rs - r_ref, axis=1)
        rows.append({"dt": float(dt), "n_steps": n,
                     "flops": float(n * S.FLOPS_PER_STEP["vps4"]),
                     "traj": float(np.sqrt(np.mean(pos ** 2))),
                     "energy": float(np.median(
                         (np.abs(E - E_ref) / E_ref[0])[half:]))})
    sch, _sig = H.classical_rows()
    assert abs(rows[0]["traj"] / sch["vps4"]["traj"] - 1.0) < 1e-9, \
        "the dt=0.3 rung no longer reproduces the vps4 row of tab:family"
    assert abs(rows[0]["flops"] / sch["vps4"]["flops"] - 1.0) < 1e-12
    return rows


def vps4_at_cost(rows, flops):
    """The vps4 rung whose flop count first reaches `flops`, interpolated in
    log-log between the two rungs that bracket it where they exist."""
    f = np.array([r["flops"] for r in rows])
    if flops <= f[0]:
        return {"bracketed": False, "note": "cheaper than vps4 at dt = 0.3"}
    if flops >= f[-1]:
        return {"bracketed": False, "flops": float(flops),
                "beyond_last_rung": True,
                "last_rung": rows[-1],
                "note": "beyond the measured ladder; the ladder already sits "
                        "at the double-precision floor there"}
    i = int(np.searchsorted(f, flops))
    a, b = rows[i - 1], rows[i]
    w = (np.log10(flops) - np.log10(a["flops"])) / \
        (np.log10(b["flops"]) - np.log10(a["flops"]))
    out = {"bracketed": True, "flops": float(flops),
           "dt": float(10 ** (np.log10(a["dt"]) + w * (np.log10(b["dt"])
                                                       - np.log10(a["dt"]))))}
    for k in ("traj", "energy"):
        out[k] = float(10 ** (np.log10(a[k]) + w * (np.log10(b[k])
                                                    - np.log10(a[k]))))
    return out


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    runs = load_all()
    if not os.path.exists(SEL):
        print("hp2_selection.json is missing -- run hp2_select.py first")
        return 2
    sel = json.load(open(SEL, encoding="utf-8"))
    sch, signal = H.classical_rows()
    tgt_traj = sch["vps4"]["traj"]
    tgt_en = sch["vps4"]["energy"]

    out = {"meta": {
        "n_jobs": len(runs),
        "n_seeds": H.N_SEEDS,
        "grid": {k: [c[0] for c in v] for k, v in H.GRID.items()},
        "ladder": {k: list(v) for k, v in H.LADDER.items()},
        "capacity_sweep": {k: list(v) for k, v in H.CAPACITY_SWEEP.items()},
        "selection": {k: v["cfg"] for k, v in sel["selection"].items()},
        "ladder_cfg": {k: H.GRID[k][H.LADDER_CFG_INDEX][0] for k in H.ARCHS},
        "data_sweep": {"points": list(H.DATA_SWEEP),
                       "sizes": dict(H.DATA_SWEEP_SIZE)},
        "targets": {"vps4_traj": tgt_traj, "vps4_energy": tgt_en,
                    "vps4_flops": sch["vps4"]["flops"],
                    "vps2_traj": sch["vps2"]["traj"],
                    "corrector_traj": sch["hybrid"]["traj"],
                    "physical_signal": signal},
    }}

    # ---------------------------------------------------------- 1. ladder --
    ladder = {}
    for arch in H.ARCHS:
        ci = H.LADDER_CFG_INDEX
        rungs = []
        for mult in (1,) + tuple(H.LADDER[arch]):
            rr = [r for r in runs if r["arch"] == arch and r["cfg_index"] == ci
                  and r["budget_multiplier"] == mult
                  and r.get("data_key", "full") == "full"
                  and "data_key" not in r]
            if len(rr) != H.N_SEEDS:
                print("missing rung: %s x%d has %d of %d seeds"
                      % (arch, mult, len(rr), H.N_SEEDS))
                return 2
            rr.sort(key=lambda r: r["rep"])
            traj = np.array([r["traj"] for r in rr])
            en = np.array([r["energy"] for r in rr])
            vl = np.array([r["val_loss"] for r in rr])
            best_ckpt = np.array([min(p["traj"] for p in r["trace"]) for r in rr])
            rungs.append({
                "multiplier": mult, "adam_steps": rr[0]["adam_steps"],
                "traj_median": float(np.median(traj)),
                "traj_min": float(traj.min()), "traj_max": float(traj.max()),
                "traj_by_seed": [float(v) for v in traj],
                "energy_median": float(np.median(en)),
                "energy_min": float(en.min()),
                "val_loss_median": float(np.median(vl)),
                "train_loss_median":
                    float(np.median([r["final_train_loss_mean50"] for r in rr])),
                "best_checkpoint_traj_min": float(best_ckpt.min()),
                "train_flops_estimate_median":
                    float(np.median([r["train_flops_estimate"] for r in rr])),
                "flops_run": rr[0]["flops_run"],
                "n_parameters": rr[0]["n_parameters"],
            })
        ladder[arch] = {"cfg": H.GRID[arch][ci][0], "cfg_index": ci,
                        "selected_cfg": sel["selection"][arch]["cfg"],
                        "rungs": rungs}
    # W9.1's own committed controls, read and not recomputed: one seed each at
    # four times the budget on the same anchor configurations.  They are an
    # independent rung of the same ladder, from a different seed ledger and a
    # different wave, and they belong beside ours.
    ea = json.load(open(os.path.join(H.EXT, "ea1_training.json"),
                        encoding="utf-8"))
    ladder["w9_1_controls"] = {
        k: {"adam_steps": v["adam_steps"], "n_parameters": v["n_parameters"],
            "traj": v["pos_err_rms"], "energy": v["energy_err_median_2nd_half"],
            "final_loss": v["final_loss"]}
        for k, v in ea["controls"].items()}
    out["ladder"] = ladder

    # ------------------------------------------------- 2. budget trend fit --
    trend = {}
    for arch in H.ARCHS:
        rg = ladder[arch]["rungs"]
        if len(rg) < 2:
            continue
        steps = [r["adam_steps"] for r in rg]
        fits = {
            "traj_median": loglog_fit(steps, [r["traj_median"] for r in rg]),
            "traj_best_seed": loglog_fit(steps, [r["traj_min"] for r in rg]),
            "energy_median": loglog_fit(steps, [r["energy_median"] for r in rg]),
            "energy_best_seed": loglog_fit(steps, [r["energy_min"] for r in rg]),
            "val_loss_median": loglog_fit(steps,
                                          [r["val_loss_median"] for r in rg]),
        }
        last = rg[-1]
        trend[arch] = {
            "fits": fits,
            "required_steps_traj": required(fits["traj_best_seed"], tgt_traj,
                                            last["adam_steps"], last["traj_min"]),
            "required_steps_traj_median": required(
                fits["traj_median"], tgt_traj, last["adam_steps"],
                last["traj_median"]),
            "required_steps_energy": required(fits["energy_best_seed"], tgt_en,
                                              last["adam_steps"],
                                              last["energy_min"]),
        }
    out["budget_trend"] = trend

    # ----------------------------------------------- 3. capacity trend fit --
    cap = {}
    for arch in H.ARCHS:
        names = H.CAPACITY_SWEEP[arch]
        pts = []
        for nm in names:
            c = sel["by_config"][arch][nm]
            pts.append({"cfg": nm, "n_parameters": c["n_parameters"],
                        "flops_per_step": c["flops_per_step"],
                        "traj_median": c["traj"]["median"],
                        "traj_min": c["traj"]["min"],
                        "energy_median": c["energy"]["median"],
                        "val_loss_median": c["val_loss"]["median"]})
        pars = [p["n_parameters"] for p in pts]
        fits = {"traj_median": loglog_fit(pars, [p["traj_median"] for p in pts]),
                "traj_best_seed": loglog_fit(pars, [p["traj_min"] for p in pts]),
                "val_loss_median": loglog_fit(
                    pars, [p["val_loss_median"] for p in pts])}
        j = int(np.argmin([p["traj_min"] for p in pts]))
        jm = int(np.argmin([p["traj_median"] for p in pts]))
        cap[arch] = {"points": pts, "fits": fits,
                     "required_parameters_traj": required(
                         fits["traj_best_seed"], tgt_traj,
                         pts[j]["n_parameters"], pts[j]["traj_min"]),
                     "required_parameters_traj_median": required(
                         fits["traj_median"], tgt_traj,
                         pts[jm]["n_parameters"], pts[jm]["traj_median"])}
    out["capacity_trend"] = cap

    # ------------------------------------------------------ 4. best of all --
    best = {}
    for arch in H.ARCHS:
        rr = [r for r in runs if r["arch"] == arch]
        j = int(np.argmin([r["traj"] for r in rr]))
        k = int(np.argmin([min(p["traj"] for p in r["trace"]) for r in rr]))
        kb = min(rr[k]["trace"], key=lambda p: p["traj"])
        best[arch] = {
            "n_runs": len(rr),
            "final_checkpoint": {
                "cfg": rr[j]["cfg"], "multiplier": rr[j]["budget_multiplier"],
                "rep": rr[j]["rep"], "adam_steps": rr[j]["adam_steps"],
                "traj": rr[j]["traj"], "energy": rr[j]["energy"],
                "flops_run": rr[j]["flops_run"],
                "vps4_traj_ratio": tgt_traj and rr[j]["traj"] / tgt_traj,
                "vps4_energy_ratio": rr[j]["energy"] / tgt_en,
                "vps4_flops_ratio": rr[j]["flops_run"] / sch["vps4"]["flops"]},
            "any_checkpoint_oracle": {
                "cfg": rr[k]["cfg"], "multiplier": rr[k]["budget_multiplier"],
                "rep": rr[k]["rep"], "step": kb["step"],
                "traj": kb["traj"], "energy": kb["energy"],
                "val_loss": kb["val_loss"],
                "val_loss_at_end": rr[k]["val_loss"],
                "traj_at_end": rr[k]["traj"],
                "vps4_traj_ratio": kb["traj"] / tgt_traj},
        }
    out["best_reached"] = best

    # ------------------------------------------------------ 5. equal cost --
    rows = vps4_ladder()
    eq = {"vps4_ladder": rows, "at_architecture_cost": {}}
    for arch in H.ARCHS:
        fl = ladder[arch]["rungs"][0]["flops_run"]
        eq["at_architecture_cost"][arch] = {
            "arch_flops_run": fl,
            "arch_traj_best_over_campaign": best[arch]["final_checkpoint"]["traj"],
            "vps4_at_that_cost": vps4_at_cost(rows, fl)}
    out["equal_cost"] = eq

    # ------------------------------------------- 6. loss / rollout mismatch --
    mism = {}
    for arch in H.ARCHS:
        rr = [r for r in runs if r["arch"] == arch]
        ratios = [r["traj"] / min(p["traj"] for p in r["trace"]) for r in rr]
        end_is_best = [r["traj"] <= min(p["traj"] for p in r["trace"]) * 1.0000001
                       for r in rr]
        mism[arch] = dict(sel["loss_vs_rollout"][arch])
        mism[arch].update({
            "end_over_best_checkpoint_median": float(np.median(ratios)),
            "end_over_best_checkpoint_max": float(np.max(ratios)),
            "fraction_of_runs_whose_end_is_their_best": float(np.mean(end_is_best)),
        })
    out["loss_vs_rollout"] = mism

    # ---------------------------------------------------- 7. the data axis --
    data = {}
    for arch in H.ARCHS:
        pts = []
        for key in H.DATA_SWEEP:
            rr = [r for r in runs if r["arch"] == arch
                  and r.get("data_key") == key]
            if len(rr) != H.N_SEEDS:
                continue
            rr.sort(key=lambda r: r["rep"])
            traj = np.array([r["traj"] for r in rr])
            pts.append({"data_key": key, "n_states": rr[0]["n_train_states"],
                        "cfg": rr[0]["cfg"],
                        "traj_median": float(np.median(traj)),
                        "traj_min": float(traj.min()),
                        "traj_by_seed": [float(v) for v in traj],
                        "energy_median": float(np.median(
                            [r["energy"] for r in rr])),
                        "val_loss_median": float(np.median(
                            [r["val_loss"] for r in rr]))})
        if len(pts) < 2:
            continue
        ns = [p["n_states"] for p in pts]
        fits = {"traj_median": loglog_fit(ns, [p["traj_median"] for p in pts]),
                "traj_best_seed": loglog_fit(ns, [p["traj_min"] for p in pts]),
                "val_loss_median": loglog_fit(
                    ns, [p["val_loss_median"] for p in pts])}
        j = int(np.argmin([p["traj_min"] for p in pts]))
        jm = int(np.argmin([p["traj_median"] for p in pts]))
        data[arch] = {"points": pts, "fits": fits,
                      "required_states_traj": required(
                          fits["traj_best_seed"], tgt_traj,
                          pts[j]["n_states"], pts[j]["traj_min"]),
                      "required_states_traj_median": required(
                          fits["traj_median"], tgt_traj,
                          pts[jm]["n_states"], pts[jm]["traj_median"])}
    out["data_trend"] = data

    # ------------------------------------- 8. what an unannealed point costs --
    # Every trace point except the last carries a learning rate that a completed
    # run of that length would have annealed away, so the trace is not a budget
    # ladder.  How far it is from one is measurable: compare the trace of the
    # longest run at the step count of the base budget with the completed base
    # run at the same seed.
    anneal = {}
    for arch in H.ARCHS:
        ci = H.LADDER_CFG_INDEX
        top = max(H.LADDER[arch])
        pairs, at_step = [], C.ADAM_STEPS
        for rep in range(H.N_SEEDS):
            short = [r for r in runs if r["arch"] == arch
                     and r["cfg_index"] == ci and r["budget_multiplier"] == 1
                     and r["rep"] == rep and "data_key" not in r]
            long = [r for r in runs if r["arch"] == arch
                    and r["cfg_index"] == ci and r["budget_multiplier"] == top
                    and r["rep"] == rep and "data_key" not in r]
            if not short or not long:
                continue
            hit = [p for p in long[0]["trace"]
                   if p["step"] == short[0]["adam_steps"]]
            if not hit:
                continue
            pairs.append({"rep": rep, "annealed": short[0]["traj"],
                          "truncated": hit[0]["traj"],
                          "ratio": hit[0]["traj"] / short[0]["traj"]})
        if pairs:
            anneal[arch] = {
                "at_step": at_step, "long_multiplier": top,
                "pairs": pairs,
                "ratio_median": float(np.median([p["ratio"] for p in pairs]))}
    out["annealing_gap"] = anneal

    # ---------------------------------------------- 9. the headline numbers --
    # The smallest trajectory error anywhere in the campaign, whatever
    # architecture, configuration, seed or budget produced it, against vps4 at
    # the step size of Section 7 and against vps4 at that run's own cost.
    allruns = [r for r in runs]
    j = int(np.argmin([r["traj"] for r in allruns]))
    b = allruns[j]
    eqc = vps4_at_cost(rows, b["flops_run"])
    out["headline"] = {
        "n_runs_in_campaign": len(allruns),
        "n_configurations": sum(len(v) for v in H.GRID.values()),
        "best_run": {"arch": b["arch"], "cfg": b["cfg"], "rep": b["rep"],
                     "adam_steps": b["adam_steps"],
                     "data_key": b.get("data_key", "full"),
                     "traj": b["traj"], "energy": b["energy"],
                     "flops_run": b["flops_run"],
                     "n_parameters": b["n_parameters"]},
        "against_vps4_same_step": {
            "traj_ratio": b["traj"] / tgt_traj,
            "energy_ratio": b["energy"] / tgt_en,
            "flops_ratio": b["flops_run"] / sch["vps4"]["flops"]},
        "against_vps4_equal_cost": {
            "vps4_dt": eqc.get("dt"), "vps4_traj": eqc.get("traj"),
            "vps4_energy": eqc.get("energy"),
            "traj_ratio": (b["traj"] / eqc["traj"]) if eqc.get("bracketed")
            else None},
        "manuscript_now_says": {"traj_ratio": 328.5, "energy_ratio": 1122.0,
                                "flops_ratio": 300.4},
    }

    rc = C.check_or_write(OUT, out, rtol=1e-6, force=a.force)
    print_tables(out, sel, sch)
    return rc


# ------------------------------------------------------------------ print --
def print_tables(out, sel, sch):
    h = out["headline"]
    b = h["best_run"]
    print("\n### Headline\n")
    print("%d runs over %d configurations, four seeds each.  The smallest "
          "trajectory error anywhere in the campaign is %.4e Larmor radii: %s, "
          "%s, seed %d, %d Adam steps, %s data, %d parameters, %.3e flops per "
          "run."
          % (h["n_runs_in_campaign"], h["n_configurations"], b["traj"],
             b["arch"], b["cfg"], b["rep"], b["adam_steps"], b["data_key"],
             b["n_parameters"], b["flops_run"]))
    s = h["against_vps4_same_step"]
    print("Against vps4 at the step size of Section 7: worse by %.1f in the "
          "trajectory channel and %.1f in the energy channel, at %.1f times "
          "the cost.  The manuscript's present figures are %.1f, %.0f and "
          "%.1f." % (s["traj_ratio"], s["energy_ratio"], s["flops_ratio"],
                     h["manuscript_now_says"]["traj_ratio"],
                     h["manuscript_now_says"]["energy_ratio"],
                     h["manuscript_now_says"]["flops_ratio"]))
    e = h["against_vps4_equal_cost"]
    if e["traj_ratio"]:
        print("At equal cost -- vps4 given the same %.3e flops, which buys it "
              "dt = %.3e -- vps4 reaches %.3e and the factor is %.3e."
              % (b["flops_run"], e["vps4_dt"], e["vps4_traj"],
                 e["traj_ratio"]))

    print("\n### Table 1 -- the grid at the base budget "
          "(4400 Adam steps, four seeds)\n")
    print("| arch | config | params | flops/step | val loss (med) | "
          "traj med | traj best | energy med |")
    print("|---|---|---|---|---|---|---|---|")
    for arch in H.ARCHS:
        for name, c in sel["by_config"][arch].items():
            star = " *" if name == sel["selection"][arch]["cfg"] else ""
            print("| %s | %s%s | %d | %d | %.3e | %.4e | %.4e | %.3e |"
                  % (arch, name, star, c["n_parameters"], c["flops_per_step"],
                     c["val_loss"]["median"], c["traj"]["median"],
                     c["traj"]["min"], c["energy"]["median"]))
    print("\n`*` marks the configuration the validation loss selected.")

    print("\n### Table 2 -- budget ladder on the anchor configuration\n")
    print("| arch | config | x | Adam steps | seeds | val loss (med) | "
          "traj med | traj best | energy med | train flops |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arch in H.ARCHS:
        L = out["ladder"][arch]
        for r in L["rungs"]:
            print("| %s | %s | x%d | %d | %d | %.3e | %.4e | %.4e | %.3e | "
                  "%.2e |"
                  % (arch, L["cfg"], r["multiplier"], r["adam_steps"],
                     H.N_SEEDS, r["val_loss_median"], r["traj_median"],
                     r["traj_min"], r["energy_median"],
                     r["train_flops_estimate_median"]))
        for k, v in sorted(out["ladder"]["w9_1_controls"].items()):
            if k.startswith(arch + "/") and "x4" in k:
                print("| %s | W9.1 control %s | x4 | %d | 1 | %.3e | %.4e | "
                      "%.4e | %.3e | -- |"
                      % (arch, k.split("/")[1], v["adam_steps"],
                         v["final_loss"], v["traj"], v["traj"], v["energy"]))

    print("\n### Table 3 -- fitted trends and what they require\n")
    print("| arch | resource | read | points | slope | R^2 | to reach vps4 |")
    print("|---|---|---|---|---|---|---|")

    def _row(arch, res, read, fit, req):
        s = ("%.3g (x%.3g)" % (req["value"], req["factor"])) \
            if req["reachable"] else "unreachable: the trend does not decrease"
        r2 = "--" if fit["n"] < 3 else "%.3f" % fit["r2"]
        print("| %s | %s | %s | %d | %+.3f | %s | %s |"
              % (arch, res, read, fit["n"], fit["slope"], r2, s))

    for arch in H.ARCHS:
        t = out["budget_trend"][arch]
        _row(arch, "Adam steps", "best seed", t["fits"]["traj_best_seed"],
             t["required_steps_traj"])
        _row(arch, "Adam steps", "median", t["fits"]["traj_median"],
             t["required_steps_traj_median"])
        c = out["capacity_trend"][arch]
        _row(arch, "parameters", "best seed", c["fits"]["traj_best_seed"],
             c["required_parameters_traj"])
        _row(arch, "parameters", "median", c["fits"]["traj_median"],
             c["required_parameters_traj_median"])
        d = out.get("data_trend", {}).get(arch)
        if d:
            _row(arch, "training states", "best seed",
                 d["fits"]["traj_best_seed"], d["required_states_traj"])
            _row(arch, "training states", "median", d["fits"]["traj_median"],
                 d["required_states_traj_median"])

    print("\n### Table 4 -- vps4 at the cost of a learned run\n")
    print("| dt | steps | flops | traj | energy |")
    print("|---|---|---|---|---|")
    for r in out["equal_cost"]["vps4_ladder"]:
        print("| %.4g | %d | %.3e | %.4e | %.4e |"
              % (r["dt"], r["n_steps"], r["flops"], r["traj"], r["energy"]))
    for arch, d in out["equal_cost"]["at_architecture_cost"].items():
        v = d["vps4_at_that_cost"]
        if v.get("bracketed"):
            print("%-8s run costs %.3e flops; vps4 at that cost runs at "
                  "dt = %.3e and reaches traj %.3e, energy %.3e"
                  % (arch, d["arch_flops_run"], v["dt"], v["traj"], v["energy"]))
        else:
            print("%-8s run costs %.3e flops; %s"
                  % (arch, d["arch_flops_run"], v.get("note", "")))

    print("\n### Table 5 -- best reached anywhere in the campaign\n")
    print("| arch | best final | x vps4 | best checkpoint (oracle) | x vps4 |")
    print("|---|---|---|---|---|")
    for arch in H.ARCHS:
        b = out["best_reached"][arch]
        print("| %s | %.4e (%s x%d) | %.0f | %.4e (%s x%d, step %d) | %.0f |"
              % (arch, b["final_checkpoint"]["traj"], b["final_checkpoint"]["cfg"],
                 b["final_checkpoint"]["multiplier"],
                 b["final_checkpoint"]["vps4_traj_ratio"],
                 b["any_checkpoint_oracle"]["traj"],
                 b["any_checkpoint_oracle"]["cfg"],
                 b["any_checkpoint_oracle"]["multiplier"],
                 b["any_checkpoint_oracle"]["step"],
                 b["any_checkpoint_oracle"]["vps4_traj_ratio"]))

    if out.get("data_trend"):
        print("\n### Table 5b -- the data axis\n")
        print("| arch | config | states | traj med | traj best | energy med | "
              "val loss med |")
        print("|---|---|---|---|---|---|---|")
        for arch, d in out["data_trend"].items():
            for p in d["points"]:
                print("| %s | %s | %d | %.4e | %.4e | %.3e | %.3e |"
                      % (arch, p["cfg"], p["n_states"], p["traj_median"],
                         p["traj_min"], p["energy_median"],
                         p["val_loss_median"]))
            f = d["fits"]["traj_median"]
            print("| %s | fit (median) | slope %+.3f | R^2 %.3f | | |"
                  % (arch, f["slope"], f["r2"]))

    if out.get("annealing_gap"):
        print("\n### Table 5c -- a truncated run against a completed one\n")
        print("| arch | at step | completed run | truncated longer run | ratio |")
        print("|---|---|---|---|---|")
        for arch, d in out["annealing_gap"].items():
            for p in d["pairs"]:
                print("| %s (x%d) | %d | %.4e | %.4e | %.2f |"
                      % (arch, d["long_multiplier"], d["at_step"],
                         p["annealed"], p["truncated"], p["ratio"]))
            print("| %s | median | | | %.2f |" % (arch, d["ratio_median"]))

    print("\n### Table 6 -- does the loss predict the rollout\n")
    print("| arch | Spearman(val, traj) across grid | within-run median | "
          "runs where it is negative | end / best checkpoint (med) |")
    print("|---|---|---|---|---|")
    for arch in H.ARCHS:
        m = out["loss_vs_rollout"][arch]
        print("| %s | %+.3f | %+.3f | %.0f%% | %.2f |"
              % (arch, m["across_configs_val_vs_traj"], m["within_run_median"],
                 100 * m["within_run_negative_fraction"],
                 m["end_over_best_checkpoint_median"]))


if __name__ == "__main__":
    sys.exit(main())
