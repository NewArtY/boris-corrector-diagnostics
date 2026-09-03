"""Their Fig. 9, and the crossing that Gate G0 of the preregistration is about.

Protocol, theirs: nine parametric SympMats trained at nine normalised step
sizes are used to carry an ensemble to the common final time omega_0 t_f = 8 in
a fixed uniform field; the L1 error against the analytical solution is plotted
against the phase advanced per step, omega_g dt = b omega_0 dt, for
B = 0.5 B_0 and B = 2.5 B_0; the same protocol is repeated with a Boris pusher.
They report a crossing at omega_g dt ~ 0.1, "about 0.015 of the gyroperiod",
and G0 asks for that crossing within a factor of two.

Everything in a uniform field is a linear map, so the autoregressive rollout is
a matrix power taken one step at a time -- no sampling, no integrator error of
ours between the scheme and the number.

Their paper says "a Boris pusher" and never says which one, and the words
leapfrog, staggered, half-step and synchronised do not occur in it, although it
cites Chin and Cator for the anatomy of Boris solvers.  All four members of that
anatomy are run here.  Two of them, B2B and BLF read where it stores, are the
two the preregistration is about; B1A and B1B are carried along because they
cost nothing and because Parker and Birdsall's solver is B1B.

Writes sm2_fig9.json.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_common as S                                        # noqa: E402

VARIANTS = ("B2B", "BLF_stored", "B1A", "B1B")
CKPT = S.CKPT_DIR


PAPER_MSE = 1e-8            # "reaching a MSE loss of ~10^-8", their Sec. III.C


def load_models(target_mse=None):
    """M_pred(b) for every (step size, seed).

    With `target_mse`, each model is taken at the earliest checkpoint at which
    its full-batch training loss had fallen to that level, instead of at the end
    of the declared budget.  The reason is in `sm1_train.train_stack`: their
    Fig. 9 crossing moves as the cube root of the training loss, so a
    reproduction that trains further than they did reports a crossing at a
    smaller step for a reason that has nothing to do with the method.  Their
    Sec. III.C prints the loss they reached, 10^-8; the epoch count is printed
    only as a range.  The printed loss is therefore the tighter constraint, and
    both readings are reported.
    """
    out, meta = {}, {}
    for k in range(len(S.DT_LADDER)):
        for r in range(S.N_SEEDS):
            f = os.path.join(CKPT, "param_dt%d_rep%d.npz" % (k, r))
            if not os.path.exists(f):
                raise SystemExit("missing checkpoint %s -- run sm1_train.py" % f)
            d = np.load(f)
            assert abs(float(d["dt"]) - S.DT_LADDER[k]) < 1e-15
            if target_mse is None:
                out[(k, r)] = (d["M"], d["b"])
                meta[(k, r)] = {"epoch": int(d["snapshot_epoch"][-1]),
                                "mse": float(d["snapshot_mse"][-1]),
                                "reached_target": True}
                continue
            mse = np.asarray(d["snapshot_mse"])
            hit = np.where(mse <= target_mse)[0]
            i = int(hit[0]) if len(hit) else int(len(mse) - 1)
            out[(k, r)] = (d["M_trajectory"][i], d["b"])
            meta[(k, r)] = {"epoch": int(d["snapshot_epoch"][i]),
                            "mse": float(mse[i]), "reached_target": bool(len(hit))}
    load_models.meta = meta
    return out


def sympmat_map(M, n):
    """M^n accumulated one step at a time: the autoregressive rollout, exactly."""
    A = np.eye(4)
    for _ in range(int(n)):
        A = M @ A
    return A


def _b_index(bs, b):
    j = int(np.argmin(np.abs(bs - b)))
    assert abs(bs[j] - b) < 1e-12, (bs[j], b)
    return j


def crossing(x, y_a, y_b):
    """The single crossing on the ladder, or None if there is not exactly one."""
    c = S.crossings(x, y_a, y_b)
    return c[0] if len(c) == 1 else None


def loglog_slope(x, y):
    return S.loglog_slope(x, y)


def per_particle_l1(Zp, Zt, b, metric):
    """Per-particle L1 over the four components, in canonical or in mechanical
    coordinates.  The state SympMat acts on is canonical -- their Eq. (4) -- so
    canonical is the metric of record; mechanical is carried as a sensitivity
    check, because the paper does not say which one Fig. 9 uses."""
    if metric == "mechanical":
        A = S.can_to_mech(b).T
        Zp, Zt = Zp @ A, Zt @ A
    return np.mean(np.abs(Zp - Zt), axis=1)


def crossing_distribution(tf, models, Z0, metric="canonical"):
    """The crossing that each particle of the ensemble would have reported on
    its own.  Their Fig. 9 says "a particle", their Sec. III.D uses 625 of
    them, and this is the spread between those two readings."""
    out = {}
    for b in S.B_EVAL:
        Zt = Z0 @ S.analytic_M(b, tf).T
        cur = {v: [] for v in VARIANTS}
        sym = {r: [] for r in range(S.N_SEEDS)}
        x = []
        for k, dt in enumerate(S.DT_LADDER):
            n = int(round(tf / dt))
            x.append(b * dt)
            for v in VARIANTS:
                cur[v].append(per_particle_l1(
                    Z0 @ S.boris_total_map(b, dt, n, v).T, Zt, b, metric))
            for r in range(S.N_SEEDS):
                M, bs = models[(k, r)]
                sym[r].append(per_particle_l1(
                    Z0 @ sympmat_map(M[_b_index(bs, b)], n).T, Zt, b, metric))
        o = {}
        for v in VARIANTS:
            yb = np.array(cur[v])                       # (rungs, particles)
            cs = []
            for i in range(Z0.shape[0]):
                for r in range(S.N_SEEDS):
                    c = S.crossings(x, yb[:, i], np.array(sym[r])[:, i])
                    if len(c) == 1:
                        cs.append(c[0])
            o[v] = {"n_resolved": len(cs),
                    "n_total": Z0.shape[0] * S.N_SEEDS,
                    "p10": float(np.percentile(cs, 10)) if cs else None,
                    "p50": float(np.percentile(cs, 50)) if cs else None,
                    "p90": float(np.percentile(cs, 90)) if cs else None}
        out["%g" % b] = o
    return out


def flop_crossing(fa, ya, fb, yb):
    """Where two error-versus-cost curves cross, both interpolated log-log.

    Their Fig. 9 puts the step size on the abscissa, which charges a SympMat
    step and a Boris step the same.  They are not the same: in a uniform field
    a trained SympMat is one 4x4 matrix-vector product, 28 flops by the model
    of Section 9, and a B2B step is two half drifts and a plane rotation with
    the Boris angle precomputed, 14.  On an equal-cost abscissa Boris is
    allowed twice as many steps.  This is the same measurement on that axis.
    """
    fa, ya, fb, yb = (np.log(np.asarray(v, float)) for v in (fa, ya, fb, yb))
    lo = max(fa.min(), fb.min())
    hi = min(fa.max(), fb.max())
    if not hi > lo:
        return None
    g = np.linspace(lo, hi, 2001)
    d = np.interp(g, np.sort(fa), ya[np.argsort(fa)]) - \
        np.interp(g, np.sort(fb), yb[np.argsort(fb)])
    s = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(s) != 1:
        return None
    i = s[0]
    t = -d[i] / (d[i + 1] - d[i])
    return float(np.exp(g[i] + t * (g[i + 1] - g[i])))


def errors_at(tf, models, Z0, metric="canonical"):
    """L1 error at the common final time, per field value, step size, scheme."""
    res = {}
    for b in S.B_EVAL:
        Zt = Z0 @ S.analytic_M(b, tf).T
        rows = []
        for k, dt in enumerate(S.DT_LADDER):
            n = tf / dt
            assert abs(n - round(n)) < 1e-12, (tf, dt)
            n = int(round(n))
            row = {"dt": dt, "omega_g_dt": b * dt, "n_steps": n,
                   "gyro_orbits": b * tf / S.TWO_PI}
            for v in VARIANTS:
                row[v] = float(np.mean(per_particle_l1(
                    Z0 @ S.boris_total_map(b, dt, n, v).T, Zt, b, metric)))
            row["BLF_centred"] = row["B2B"]
            sm = []
            for r in range(S.N_SEEDS):
                M, bs = models[(k, r)]
                sm.append(float(np.mean(per_particle_l1(
                    Z0 @ sympmat_map(M[_b_index(bs, b)], n).T, Zt, b, metric))))
            row["sympmat"] = sm
            row["sympmat_median"] = float(np.median(sm))
            rows.append(row)
        res["%g" % b] = rows
    return res


def analyse(res):
    """Crossings and slopes, per field value, per Boris variant, per seed."""
    out = {}
    for b, rows in res.items():
        x = np.array([r["omega_g_dt"] for r in rows])
        o = {"omega_g_dt": [float(v) for v in x],
             "n_steps": [r["n_steps"] for r in rows],
             "gyro_orbits": rows[0]["gyro_orbits"],
             "sympmat_median": [r["sympmat_median"] for r in rows],
             "sympmat_slope": [loglog_slope(x, [r["sympmat"][i] for r in rows])
                               for i in range(S.N_SEEDS)],
             "crossings": {}}
        for v in VARIANTS + ("BLF_centred",):
            yb = np.array([r[v] for r in rows])
            o[v] = [float(t) for t in yb]
            o[v + "_slope"] = loglog_slope(x, yb)
            # The asymptotic slope, over the five finest rungs.  Their text says
            # "Boris still demonstrates second-order dependence of the error with
            # dt", and of the four members of the Chin-Cator family only B2B does:
            # BLF read where it stores, B1A and B1B are all first order in the
            # position.  That sentence is therefore the only thing in the paper
            # that says which Boris was run.
            o[v + "_slope_fine"] = loglog_slope(x[-5:], yb[-5:])
            c = [crossing(x, yb, [r["sympmat"][i] for r in rows])
                 for i in range(S.N_SEEDS)]
            o["crossings"][v] = {
                "per_seed": c,
                "median": None if any(t is None for t in c) else float(np.median(c)),
                "min": None if any(t is None for t in c) else float(min(c)),
                "max": None if any(t is None for t in c) else float(max(c))}
        # P2: the two defensible readouts of one stored leapfrog run
        o["blf_readout_ratio"] = [float(r["BLF_stored"] / r["BLF_centred"])
                                  for r in rows]
        out[b] = o
    return out


def one_reading(models, Z0, tag):
    """Everything Fig. 9 is asked for, for one set of trained matrices."""
    res = errors_at(S.TF_MAIN, models, Z0)
    ana = analyse(res)
    o = {"curves": res, "analysis": ana,
         "analysis_mechanical_metric": analyse(
             errors_at(S.TF_MAIN, models, Z0, metric="mechanical")),
         "per_particle_crossings": crossing_distribution(S.TF_MAIN, models, Z0),
         "training": {"dt%d_rep%d" % (k, r): load_models.meta[(k, r)]
                      for (k, r) in models}}
    ss = {}
    for b in S.B_EVAL:
        rows = []
        for k, dt in enumerate(S.DT_LADDER):
            Zt = Z0 @ S.analytic_M(b, dt).T
            e = [float(np.mean(per_particle_l1(
                Z0 @ models[(k, r)][0][_b_index(models[(k, r)][1], b)].T,
                Zt, b, "canonical"))) for r in range(S.N_SEEDS)]
            rows.append({"dt": dt, "omega_g_dt": b * dt, "per_seed": e,
                         "median": float(np.median(e))})
        ss["%g" % b] = rows
    o["single_step_l1"] = ss

    iso = {}
    for b, rows in res.items():
        nsteps = np.array([r["n_steps"] for r in rows], float)
        fB = nsteps * S.flops_boris_step_uniform()
        fS = nsteps * S.flops_sympmat_step_uniform()
        wg = np.array([r["omega_g_dt"] for r in rows])
        e = {}
        for v in VARIANTS:
            yb = [r[v] for r in rows]
            cs = [flop_crossing(fB, yb, fS, [r["sympmat"][i] for r in rows])
                  for i in range(S.N_SEEDS)]
            ok = [c for c in cs if c is not None]
            if not ok:
                e[v] = {"flops": None}
                continue
            f = float(np.median(ok))
            n_at = f / S.flops_sympmat_step_uniform()
            srt = np.argsort(nsteps)
            e[v] = {"flops": f, "sympmat_n_steps": n_at,
                    "sympmat_omega_g_dt": float(np.exp(np.interp(
                        np.log(n_at), np.log(nsteps[srt]), np.log(wg[srt])))),
                    "per_seed_flops": cs}
        iso["%s" % b] = e
    o["iso_flop_crossings"] = iso

    gate = {}
    for b in ana:
        for v in ("B2B", "BLF_stored"):
            c = ana[b]["crossings"][v]["median"]
            gate["b%s_%s" % (b, v)] = {
                "crossing_omega_g_dt": c,
                "ratio_to_paper": None if c is None else c / 0.1,
                "within_factor_2": bool(c is not None and 0.5 <= c / 0.1 <= 2.0),
                "in_gyroperiods": None if c is None else c / S.TWO_PI}
    o["G0"] = gate

    print("\n=== %s ===" % tag)
    print("%-6s %-12s %10s %10s %8s %12s"
          % ("B", "scheme", "crossing", "vs 0.1", "slope", "iso-flop"))
    for b in ana:
        for v in VARIANTS + ("BLF_centred",):
            c = ana[b]["crossings"][v]["median"]
            f = iso.get(b, {}).get(v, {}).get("sympmat_omega_g_dt")
            print("%-6s %-12s %10s %10s %8.2f %12s"
                  % (b, v, "n/a" if c is None else "%.4f" % c,
                     "n/a" if c is None else "%.2f" % (c / 0.1),
                     ana[b][v + "_slope"],
                     "" if f is None else "%.4f" % f))
        print("%-6s %-12s %10s %10s %8.2f  step L1 %.2e"
              % (b, "sympmat", "", "", np.median(ana[b]["sympmat_slope"]),
                 np.median([r["median"] for r in ss[b]])))
    return o


def main(force=False):
    Z0 = S.ensemble()
    final = load_models()
    out = {"tf": S.TF_MAIN,
           "ensemble_seed": S.seed_of("ensemble"),
           "n_ensemble": S.N_ENSEMBLE,
           "n_seeds": S.N_SEEDS,
           "paper_crossing_omega_g_dt": 0.1,
           "paper_single_step_l1_order": 1e-4,
           "paper_training_mse": PAPER_MSE,
           "at_declared_budget": one_reading(final, Z0, "declared budget"),
           "at_paper_training_loss": one_reading(
               load_models(PAPER_MSE), Z0, "matched to their MSE 1e-8")}

    ns = np.array([S.TF_MAIN / dt for dt in S.DT_LADDER])
    out["flops_per_trajectory"] = {
        "n_steps": [float(n) for n in ns],
        "boris": [float(n * S.flops_boris_step_uniform()) for n in ns],
        "sympmat_uniform_field": [
            float(n * S.flops_sympmat_step_uniform()) for n in ns],
        "sympmat_plus_one_matrix_build": [
            float(n * S.flops_sympmat_step_uniform()
                  + S.flops_parametric_build()) for n in ns]}
    out["G0"] = {"at_declared_budget": out["at_declared_budget"]["G0"],
                 "at_paper_training_loss": out["at_paper_training_loss"]["G0"]}

    # ---- where the crossing sits as a function of the training loss --------
    # The SympMat curve of Fig. 9 is (per-step error) x (number of steps), so it
    # falls as 1/dt; the Boris curve rises as dt^2.  Equating them gives
    # dt^3 proportional to the per-step error, so the crossing they report is a
    # reading of how long the network was trained and not only of what the
    # architecture is.  The exponent is measured here rather than asserted.
    sweep = {}
    for target in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
        m = load_models(target)
        meta = dict(load_models.meta)
        res = errors_at(S.TF_MAIN, m, Z0)
        ana = analyse(res)
        row = {"target_mse": target,
               "n_models_reaching_target": sum(1 for v in meta.values()
                                               if v["reached_target"]),
               "median_epoch": float(np.median([v["epoch"] for v in meta.values()])),
               "median_mse": float(np.median([v["mse"] for v in meta.values()]))}
        for b in ana:
            bf = float(b)
            per_rung = []
            for k, dt in enumerate(S.DT_LADDER):
                Zt = Z0 @ S.analytic_M(bf, dt).T
                per_rung.append(np.median([np.mean(per_particle_l1(
                    Z0 @ m[(k, r_)][0][_b_index(m[(k, r_)][1], bf)].T,
                    Zt, bf, "canonical")) for r_ in range(S.N_SEEDS)]))
            row["b%s_step_l1" % b] = float(np.median(per_rung))
            for v in ("B2B", "BLF_stored"):
                row["b%s_%s" % (b, v)] = ana[b]["crossings"][v]["median"]
        sweep["%g" % target] = row
    out["crossing_vs_training_loss"] = sweep
    for b in ("0.5", "2.5"):
        for v in ("B2B", "BLF_stored"):
            xs = [(sweep[t]["b%s_step_l1" % b], sweep[t]["b%s_%s" % (b, v)])
                  for t in sweep
                  if sweep[t]["b%s_%s" % (b, v)] is not None]
            if len(xs) >= 3:
                out.setdefault("crossing_loss_exponent", {})["b%s_%s" % (b, v)] = \
                    S.loglog_slope([p[0] for p in xs], [p[1] for p in xs])
    print("\ncrossing vs per-step L1 error, exponent (1/3 expected):")
    for k, v in out.get("crossing_loss_exponent", {}).items():
        print("   %-18s %.3f" % (k, v))
    return S.check_or_write(S.outpath("sm2_fig9.json"), out,
                            rtol=1e-6, atol=1e-15, force=force)


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
