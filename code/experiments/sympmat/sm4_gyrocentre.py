"""The gyrocentre, measured in SympMat the way Section 3.3 measures it in Boris.

Two claims of their Sec. III.D are quantitative and can be checked directly:

  "this artificial drift of the guiding center appears to account for most of
   the error, since SympMat captures the gyroradius and gyrofrequency
   accurately"

and Table II, where the doubly degenerate unit eigenvalue of the analytical
transfer matrix splits in the trained model.  The guiding centre is exactly the
object Section 3.3 of the manuscript locates in Boris by a two-handle scan, so
the same scan run on SympMat is the direct bridge between the two papers.

Part 1  Error decomposition of the autoregressive rollout at omega_0 dt = 2.0,
        their Fig. 5-8 setting, out to 10^5 iterations: the L1 position error is
        split into a guiding-centre displacement, a gyroradius error and a
        phase error, and the share of the first is reported.
Part 2  The spectrum of the trained matrix against the analytical
        exp(+-i theta), 1, 1 -- our Table II.
Part 3  The two handles of Section 3.3, on one test particle whose speed is
        exactly 1, over twenty gyro-orbits, at three rungs of the ladder.  See
        `two_handle_scan` for the two readings of handle B and for why the
        secular one is the one that can be sharp in a uniform field.  For B1A
        and B1B the closed form fixes the answer, so the instrument is
        calibrated on the same run that measures SympMat.

Writes sm4_gyrocentre.json.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_common as S                                        # noqa: E402
import sm2_fig9 as F                                         # noqa: E402

DECOMP_STEPS = (10, 100, 1000, 10000, 100000)
#: The decomposition is taken at their omega_0 dt = 2.0, the setting of their
#: Figs. 5-8, and at omega_0 dt = 0.5.  The second rung is there because our
#: training does not converge at the first: at omega_0 dt = 2.0 two of the three
#: seeds end at MSE 5e-5 and 1e-4 rather than the 1e-8 the paper reports, and a
#: statement about what SympMat's error is made of should not be read off a
#: model that failed to train.  Both are reported.
DECOMP_RUNGS = (0, 2)
SCAN_N_BETA = 61
SCAN_N_DELTA = 61
SCAN_ORBITS = 20.0
SCAN_RUNGS = (0, 2, 4)


def _decompose(Zp, Zt, b):
    """Guiding centre, gyroradius and phase error of a predicted canonical state."""
    gp, gt = S.guiding_centre(Zp, b), S.guiding_centre(Zt, b)
    Vp = Zp @ S.can_to_mech(b).T
    Vt = Zt @ S.can_to_mech(b).T
    rho_p, rho_t = Zp[:, :2] - gp, Zt[:, :2] - gt
    sp, st = np.linalg.norm(rho_p, axis=1), np.linalg.norm(rho_t, axis=1)
    ang = np.arctan2(rho_p[:, 1], rho_p[:, 0]) - np.arctan2(rho_t[:, 1], rho_t[:, 0])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    return {
        "gyrocentre_shift": float(np.mean(np.linalg.norm(gp - gt, axis=1))),
        "gyroradius_error": float(np.mean(np.abs(sp - st))),
        "phase_error_rad": float(np.mean(np.abs(ang))),
        "phase_error_length": float(np.mean(st * np.abs(ang))),
        "position_error": float(np.mean(np.linalg.norm(Zp[:, :2] - Zt[:, :2], axis=1))),
        "speed_error": float(np.mean(np.abs(np.linalg.norm(Vp[:, 2:], axis=1)
                                            - np.linalg.norm(Vt[:, 2:], axis=1)))),
        "l1": S.l1_error(Zp, Zt)}


def _gc(Z, b):
    """(X_c, Y_c) from canonical states, vectorised over any leading shape."""
    V = Z @ S.can_to_mech(b).T
    return np.stack([V[..., 0] + V[..., 3] / b, V[..., 1] - V[..., 2] / b], axis=-1)


def _series(maps, Z):
    """(N, K, 4) from a list of N maps and K initial states."""
    A = np.stack(maps)
    return np.einsum("nij,kj->nki", A, Z)


def two_handle_scan(b, dt, kind, M=None, variant=None):
    """Section 3.3's two handles, on one test particle whose speed is exactly 1.

    handle A   shift the sampling time of the reference by delta over [-h, h].
    handle B   move the initial position along v_0 by beta over
               [-h||v_0||, +h||v_0||] on 61 points -- Section 3.3's own range,
               which at h = 0.3 and unit speed is its [-0.3, 0.3] -- with the
               reference left where it was.  The predicted minimum h||v_0||/2
               therefore falls on grid point 45 of 61 exactly.

    Two readings of handle B are reported.  The first is Section 3.3's own, the
    median position error over the run.  The second is the secular part of the
    gyrocentre: the exact flow conserves (x + v_y/b, y - v_x/b), a discrete map
    does not, and the instantaneous value of that expression on a Boris orbit
    oscillates at order theta about the centre of the discrete circle, so the
    quantity an initial shift can cancel is its mean over the run, not its
    instantaneous value.  The run is therefore 20 gyro-orbits long, and the
    second reading is |mean_n GC_pred - GC_true|.  For B1A and B1B the closed
    form says that reading must vanish at beta = -+ h||v_0||/2, which calibrates
    the instrument on the same run that measures SympMat.
    """
    h = dt
    v0 = np.array([0.0, 1.0])
    r0 = np.array([-1.0 / b, 0.0])                 # guiding centre at the origin
    W0 = np.array([r0[0], r0[1], v0[0], v0[1]])
    c2m, m2c = S.can_to_mech(b), S.mech_to_can(b)
    Z0 = W0 @ m2c.T
    n = int(round(SCAN_ORBITS * S.TWO_PI / (b * h)))

    if kind == "sympmat":
        acc = S.accumulate(M, M, range(1, n + 1))
        maps_can = [acc[i] for i in range(1, n + 1)]
    else:
        first, rep, ro = S.boris_step_matrices(b, h)[variant]
        acc = S.accumulate(first, rep, range(1, n + 1))
        maps_can = [m2c @ ro @ acc[i] @ c2m for i in range(1, n + 1)]
    ts = np.arange(1, n + 1) * h

    # ---- handle A ---------------------------------------------------------
    deltas = np.linspace(-h, h, SCAN_N_DELTA)
    Zp = _series(maps_can, Z0[None, :])[:, 0, :]
    medA = np.array([float(np.median(np.linalg.norm(
        Zp[:, :2] - np.stack([S.analytic_M(b, t + d) @ Z0 for t in ts])[:, :2],
        axis=1))) for d in deltas])
    base = medA[int(np.argmin(np.abs(deltas)))]

    # ---- handle B ---------------------------------------------------------
    span = h * np.linalg.norm(v0)
    betas = np.linspace(-span, span, SCAN_N_BETA)
    vhat = v0 / np.linalg.norm(v0)
    Wshift = np.tile(W0, (SCAN_N_BETA, 1))
    Wshift[:, :2] += betas[:, None] * vhat[None, :]
    Zp2 = _series(maps_can, Wshift @ m2c.T)               # (N, nbeta, 4)
    ref0 = np.stack([S.analytic_M(b, t) @ Z0 for t in ts])
    medB = np.median(np.linalg.norm(Zp2[:, :, :2] - ref0[:, None, :2], axis=2), axis=0)

    gcp = _gc(Zp2, b).mean(axis=0)                        # (nbeta, 2)
    gct = _gc(ref0, b).mean(axis=0)
    secG = np.linalg.norm(gcp - gct[None, :], axis=1)

    jA, jB, jG = int(np.argmin(medA)), int(np.argmin(medB)), int(np.argmin(secG))
    pred = h * np.linalg.norm(v0) / 2.0
    mid = SCAN_N_BETA // 2
    return {"n_steps": n, "orbits": SCAN_ORBITS,
            "baseline_median": float(base),
            "handleA_gain": float(base / medA[jA]),
            "handleA_argmin_over_h": float(deltas[jA] / h),
            "handleB_gain": float(base / medB[jB]),
            "handleB_argmin_over_predicted": float(betas[jB] / pred),
            "handleB_predicted_argmin": float(pred),
            "handleB_mirror_ratio": float(medB[SCAN_N_BETA - 1 - jB] / medB[jB]),
            "gc_secular_baseline": float(secG[mid]),
            "gc_secular_gain": float(secG[mid] / secG[jG]) if secG[jG] > 0 else None,
            "gc_argmin": float(betas[jG]),
            "gc_argmin_over_predicted": float(betas[jG] / pred)}


def run(models, Z0):
    out = {"decomposition": {}, "spectrum": {}, "scan": {},
           "decomposition_rungs": [S.DT_LADDER[k] for k in DECOMP_RUNGS],
           "ensemble_seed": S.seed_of("ensemble")}

    for k in DECOMP_RUNGS:
        dt = S.DT_LADDER[k]
        for b in S.B_EVAL:
            dec, spec = [], []
            for r in range(S.N_SEEDS):
                M, bs = models[(k, r)]
                Mb = M[F._b_index(bs, b)]
                acc = S.accumulate(Mb, Mb, DECOMP_STEPS)
                for nstep in DECOMP_STEPS:
                    d = _decompose(Z0 @ acc[nstep].T,
                                   Z0 @ S.analytic_M(b, nstep * dt).T, b)
                    d.update({"seed": r, "n_steps": nstep, "dt": dt})
                    d["gyrocentre_share"] = (d["gyrocentre_shift"]
                                             / d["position_error"])
                    # Section 7 quotes the range of this ratio, so it is
                    # stored rather than derived in prose: the audit of
                    # numbers traces printed values to a file, and a ratio
                    # computed only while writing has no file to trace to.
                    d["gyrocentre_over_gyroradius"] = (
                        d["gyrocentre_shift"] / d["gyroradius_error"])
                    dec.append(d)
                ev = np.linalg.eigvals(Mb)
                # Sort the eigenvalues once and read both quantities off that
                # order.  Sorting the moduli and the arguments independently
                # destroys the pairing, and the pairing is the whole point
                # here: it is what separates the gyration pair, which stays on
                # the unit circle, from the degenerate pair, which leaves it.
                ev = ev[np.lexsort((np.abs(ev), np.angle(ev)))]
                th = b * dt
                spec.append({"seed": r, "dt": dt,
                             "abs": [float(v) for v in np.abs(ev)],
                             "arg": [float(v) for v in np.angle(ev)],
                             "analytic_abs": [1.0, 1.0, 1.0, 1.0],
                             "analytic_arg": sorted([0.0, 0.0,
                                                     float(th % (2 * np.pi) - 2 * np.pi
                                                           if th % (2 * np.pi) > np.pi
                                                           else th % (2 * np.pi)),
                                                     -float(th % (2 * np.pi) - 2 * np.pi
                                                            if th % (2 * np.pi) > np.pi
                                                            else th % (2 * np.pi))]),
                             "max_abs_minus_1": float(np.max(np.abs(ev)) - 1.0),
                             "unit_pair_split": float(
                                 np.ptp(np.sort(np.abs(ev))[1:3]))})
            out["decomposition"]["dt%g_b%g" % (dt, b)] = dec
            out["spectrum"]["dt%g_b%g" % (dt, b)] = spec

            # Boris, same decomposition at the same step, for scale
            n = int(round(S.TF_MAIN / dt))
            out["decomposition"]["dt%g_b%g_boris" % (dt, b)] = {
                v: _decompose(Z0 @ S.boris_total_map(b, dt, n, v).T,
                              Z0 @ S.analytic_M(b, S.TF_MAIN).T, b)
                for v in F.VARIANTS}

    # ---- Part 3, the two handles, over the whole ladder --------------------
    for b in S.B_EVAL:
        rows = []
        for k in SCAN_RUNGS:
            d = S.DT_LADDER[k]
            row = {"dt": d, "omega_g_dt": b * d, "rung": k}
            for v in F.VARIANTS:
                row[v] = two_handle_scan(b, d, "boris", variant=v)
            row["sympmat"] = [
                two_handle_scan(b, d, "sympmat",
                                M=models[(k, r)][0][F._b_index(models[(k, r)][1], b)])
                for r in range(S.N_SEEDS)]
            rows.append(row)
        out["scan"]["%g" % b] = rows

    print("%-9s %-5s %8s %10s %12s %12s %10s"
          % ("w0 dt", "B", "n", "|dGC|", "|drho|", "rho*dphi", "GC share"))
    for k in DECOMP_RUNGS:
        for b in S.B_EVAL:
            key = "dt%g_b%g" % (S.DT_LADDER[k], b)
            for d in out["decomposition"][key]:
                if d["seed"] != 2:
                    continue
                print("%-9g %-5s %8d %10.3e %12.3e %12.3e %10.3f"
                      % (d["dt"], b, d["n_steps"], d["gyrocentre_shift"],
                         d["gyroradius_error"], d["phase_error_length"],
                         d["gyrocentre_share"]))
    print("\ntwo-handle scan at omega_0 dt = %g over %g gyro-orbits"
          % (S.DT_LADDER[SCAN_RUNGS[1]], SCAN_ORBITS))
    print("%-5s %-12s %10s %10s %10s %10s"
          % ("B", "scheme", "gainA", "gcGain", "argmin/pred", "want"))
    for b in S.B_EVAL:
        row = out["scan"]["%g" % b][1]
        for v in F.VARIANTS:
            e = row[v]
            print("%-5s %-12s %10.2f %10.3e %10.4f %10.4f"
                  % (b, v, e["handleA_gain"], e["gc_secular_gain"] or 0.0,
                     e["gc_argmin_over_predicted"], 1.0))
        e = row["sympmat"][0]
        print("%-5s %-12s %10.2f %10.3e %10.4f %10.4f"
              % (b, "sympmat", e["handleA_gain"], e["gc_secular_gain"] or 0.0,
                 e["gc_argmin_over_predicted"], 1.0))
    # The ratio of the guiding-centre shift to the gyroradius error is quoted
    # in Section 7, and it is not a constant: the shift drifts linearly in the
    # step count while the gyroradius error stays bounded, so the ratio grows.
    # Quoting a range over all step counts at once would hide that.  The
    # summary below is stored so the printed values trace to a file.
    by_steps = {}
    for key, dec in out["decomposition"].items():
        if key.endswith("_boris"):
            continue
        for d in dec:
            r = d["gyrocentre_over_gyroradius"]
            if r == r:                      # drop the overflowed 1e5 cases
                by_steps.setdefault(str(d["n_steps"]), []).append(r)
    out["gyrocentre_over_gyroradius_by_steps"] = {
        n: {"cases": len(v),
            "min": float(np.min(v)),
            "median": float(np.median(v)),
            "max": float(np.max(v))}
        for n, v in sorted(by_steps.items(), key=lambda kv: int(kv[0]))}
    return out


def main(force=False):
    Z0 = S.ensemble()
    out = {}
    for tag, target in (("at_declared_budget", None),
                        ("at_paper_training_loss", F.PAPER_MSE)):
        print("\n=== %s ===" % tag)
        out[tag] = run(F.load_models(target), Z0)
    return S.check_or_write(S.outpath("sm4_gyrocentre.json"), out,
                            rtol=1e-6, atol=1e-15, force=force)


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
