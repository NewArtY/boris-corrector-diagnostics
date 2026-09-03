"""P3: does the crossing survive a long horizon?

Their Fig. 9 is taken at omega_0 t_f = 8, which is 0.64 gyro-orbits at
B = 0.5 B_0 and 3.18 at B = 2.5 B_0.  They add one sentence of their own
qualification: "This crossing appears to persist with increasing t_f; for
example, at t_f = 1000.0 the crossing of the curves remain similar, although
less distinct."

The preregistration predicts that beyond a hundred gyro-orbits the crossing
disappears or inverts, on the strength of our own horizon curve, whose
inversion is at 100.98 gyro-orbits (experiments/horizon/crossover.json).  This
script runs their protocol unchanged at omega_0 t_f = 8, 80, 200, 800, 1000 and
8000 -- every one an exact integer number of steps at every rung of the ladder
-- which spans 0.64 to 637 gyro-orbits at B = 0.5 B_0 and 3.2 to 3183 at
B = 2.5 B_0, and includes their own t_f = 1000.

Nothing else changes: same ensemble, same models, same metric, same ladder.

Writes sm3_horizon.json.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sm_common as S                                        # noqa: E402
import sm2_fig9 as F                                         # noqa: E402

HORIZONS = (8.0, 80.0, 200.0, 800.0, 1000.0, 8000.0)


def _boris_snapshots(b, dt, ns):
    """The Boris family at every horizon, in canonical coordinates."""
    mats = S.boris_step_matrices(b, dt)
    c2m, m2c = S.can_to_mech(b), S.mech_to_can(b)
    snaps = {}
    for fam, readouts in (("BLF_stored", ("BLF_stored", "B2B")),
                          ("B1A", ("B1A",)), ("B1B", ("B1B",))):
        first, rep, _ = mats[fam]
        acc = S.accumulate(first, rep, ns.values())
        for name in readouts:
            ro = mats[name][2]
            snaps[name] = {tf: m2c @ ro @ acc[ns[tf]] @ c2m for tf in ns}
    return snaps


def run(models, Z0, boris_cache):
    curves = {}
    for b in S.B_EVAL:
        rows = []
        for k, dt in enumerate(S.DT_LADDER):
            ns = {tf: int(round(tf / dt)) for tf in HORIZONS}
            for tf in HORIZONS:
                assert abs(tf / dt - ns[tf]) < 1e-9, (tf, dt)
            snaps = dict(boris_cache[(b, k)])
            for r in range(S.N_SEEDS):
                M, bs = models[(k, r)]
                Mb = M[F._b_index(bs, b)]
                acc = S.accumulate(Mb, Mb, ns.values())
                snaps["sympmat%d" % r] = {tf: acc[ns[tf]] for tf in ns}
            row = {"dt": dt, "omega_g_dt": b * dt,
                   "n_steps": {"%g" % tf: ns[tf] for tf in HORIZONS}}
            for name, d in snaps.items():
                row[name] = {"%g" % tf: S.l1_error(
                    Z0 @ d[tf].T, Z0 @ S.analytic_M(b, tf).T) for tf in HORIZONS}
            row["BLF_centred"] = row["B2B"]
            rows.append(row)
        curves["%g" % b] = rows

    ana = {}
    for b, rows in curves.items():
        x = np.array([r["omega_g_dt"] for r in rows])
        per_h = {}
        for tf in HORIZONS:
            t = "%g" % tf
            d = {"gyro_orbits": float(b) * tf / S.TWO_PI,
                 "crossings": {}, "slopes": {},
                 "sympmat_median": [float(np.median([r["sympmat%d" % i][t]
                                                     for i in range(S.N_SEEDS)]))
                                    for r in rows]}
            for v in F.VARIANTS + ("BLF_centred",):
                yb = [r[v][t] for r in rows]
                d[v] = [float(q) for q in yb]
                d["slopes"][v] = S.loglog_slope(x, yb)
                cs = [S.crossings(x, yb, [r["sympmat%d" % i][t] for r in rows])
                      for i in range(S.N_SEEDS)]
                d["crossings"][v] = {
                    "per_seed": cs,
                    "n_crossings": [len(c) for c in cs],
                    "median_first": (float(np.median([c[0] for c in cs]))
                                     if all(len(c) >= 1 for c in cs) else None)}
            d["sympmat_slope"] = [S.loglog_slope(
                x, [r["sympmat%d" % i][t] for r in rows]) for i in range(S.N_SEEDS)]
            per_h[t] = d
        ana[b] = per_h
    return {"curves": curves, "analysis": ana}


def _report(tag, ana):
    print("\n=== %s ===" % tag)
    print("%-5s %8s %9s %12s %12s %10s %10s"
          % ("B", "w0 tf", "orbits", "cross B2B", "cross BLF", "slopeB2B", "slopeSM"))
    for b in ana:
        for tf in HORIZONS:
            d = ana[b]["%g" % tf]
            c2 = d["crossings"]["B2B"]["median_first"]
            cl = d["crossings"]["BLF_stored"]["median_first"]
            print("%-5s %8g %9.1f %12s %12s %10.2f %10.2f"
                  % (b, tf, d["gyro_orbits"],
                     "none" if c2 is None else "%.4f" % c2,
                     "none" if cl is None else "%.4f" % cl,
                     d["slopes"]["B2B"], float(np.median(d["sympmat_slope"]))))


def main(force=False):
    Z0 = S.ensemble()
    cache = {}
    for b in S.B_EVAL:
        for k, dt in enumerate(S.DT_LADDER):
            ns = {tf: int(round(tf / dt)) for tf in HORIZONS}
            cache[(b, k)] = _boris_snapshots(b, dt, ns)
    out = {"horizons": list(HORIZONS),
           "our_crossover_gyrations": 100.98381139180759,
           "ensemble_seed": S.seed_of("ensemble")}
    for tag, target in (("at_declared_budget", None),
                        ("at_paper_training_loss", F.PAPER_MSE)):
        out[tag] = run(F.load_models(target), Z0, cache)
        _report(tag, out[tag]["analysis"])
    return S.check_or_write(S.outpath("sm3_horizon.json"), out,
                            rtol=1e-6, atol=1e-15, force=force)


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
