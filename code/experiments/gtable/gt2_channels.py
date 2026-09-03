"""gt2_channels.py -- the four channels, five schemes, one configuration per
invocation, at the working step Omega h = 0.3.

    python gt2_channels.py --field B4_decaying      one configuration
    python gt2_channels.py                          all five, in sequence
    python gt2_channels.py --field uniform --force  overwrite deliberately

Writes gt2_channels__<field>.json beside this file, one per configuration, and
exits non-zero if a rerun no longer reproduces a committed one.  Sharding by
configuration is scheduling only: the five are independent.

WHAT IS MEASURED
----------------
For every (scheme, horizon, initial condition) the four channels of the
pre-registration -- trajectory, phase, total energy, spectral power in
nu < 0.2 Omega_c^ref -- each under its committed statistic and each under the
common root-mean-square, with the supporting figures (maximum, final value,
in-band fraction, bin count, Parseval residual, the residual of the polar
identity of Section~\\ref{sec:channels}) beside them.  Nothing is selected
after the fact: the whole cross is written out and `gt3_gtable.py` reads it.

WHAT IS NOT VARIED
------------------
The step.  It is fixed at the working Omega h = 0.3 by the pre-registration;
the step axis is the map of W14 and is not re-run here.  The corrector is one
committed checkpoint and is not retrained, fine-tuned or re-seeded; the
ensemble over seeds is W16.

THE SPECTRAL CHANNEL IS MEASURED TWICE
--------------------------------------
Once with the declared Omega_c^ref -- the initial gyrofrequency -- and once
with the time mean of |B| along the reference orbit.  Both are
scheme-independent; the first is the declaration, the second is the
sensitivity check announced in `gt_common.py` before the runs.  In three of
the five configurations they are the same number to within a part in a
thousand; in B2 the gyrofrequency swings by ten per cent along the orbit and
the check is not decorative there.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import gt_common as G
import map_common as MC
from ea_common import check_or_write
from gt1_calibration import reference_orbit

HERE = os.path.dirname(os.path.abspath(__file__))

OUTPUTS = ["gt2_channels__uniform.json", "gt2_channels__B1_radial.json",
           "gt2_channels__B2_wave.json", "gt2_channels__B3_tilted.json",
           "gt2_channels__B4_decaying.json"]


def slice_series(ch, n):
    return {k: v[:n] for k, v in ch.items()}


def run_field(fname, force=False):
    t_start = time.time()
    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    field = fields[fname]
    fld = fast[fname]
    R0, V0 = MC.initial_conditions(MC.N_IC)
    r_L = MC.larmor_radii(field, R0, V0)
    mlp = MC.load_corrector_numpy()

    N = max(G.HORIZONS.values())
    ts = np.arange(N) * G.DT
    idx = np.arange(N)

    print("[%s] building the reference" % fname, flush=True)
    Rr, Vr = reference_orbit(fname, field, R0, V0, ts, tag="best")

    w0 = G.reference_gyrofrequency(field, R0)
    w_along = G.gyrofrequency_along(field, Rr, ts)

    # the physical energy signal of this configuration on each window: what
    # the reference's own energy does.  Where it is zero the ratio
    # "signal over error" is undefined and only the ratio of errors, which is
    # G, is defined.  The two are not the same statement and are not mixed.
    Eref = 0.5 * np.sum(Vr ** 2, axis=-1)

    out = {"meta": {
        "field": fname,
        "field_class": type(field).__name__,
        "field_description": getattr(field, "description", ""),
        "corrector_saw_this_field_in_training": MC.TRAINED_ON[fname],
        "closed_form": MC.CLOSED_FORM[fname],
        "reference": ("closed form" if MC.CLOSED_FORM[fname]
                      else "DOP853 rtol 3e-14 atol 1e-16"),
        "dt": G.DT, "schemes": G.SCHEMES, "channels": G.CHANNELS,
        "horizons_samples": G.HORIZONS, "horizons_t": G.HORIZON_T,
        "n_initial_conditions": MC.N_IC,
        "n_random_draws": G.N_RANDOM_DRAWS,
        "larmor_radii": r_L,
        "omega_c_ref_per_ic": w0,
        "omega_c_ref_declaration":
            "|q/m| |B(r_0, 0)|, declared in gt_common.py before any run",
        "omega_c_alt_per_ic_H_paper":
            w_along[:G.HORIZONS["H_paper"]].mean(axis=0),
        "omega_c_alt_per_ic_H_crossover": w_along.mean(axis=0),
        "omega_c_alt_declaration":
            "time mean of |q/m||B| along the reference orbit; the declared "
            "sensitivity alternative, also scheme-independent",
        "band_edge_ratio": G.BAND_EDGE_RATIO,
        "phase_definition": "atan2(|v x v_ref|, v . v_ref), radians",
        "corrector_checkpoint": "boris_corrector_b4.pt -- ONE checkpoint; "
                                "every corrector figure here is that single "
                                "run, not an ensemble over seeds (W16)",
    }}

    for hname, n in G.HORIZONS.items():
        half = n // 2
        rel = np.abs(Eref[:n] - Eref[0][None, :]) / Eref[0][None, :]
        out["meta"]["physical_signal_median_2nd_half_%s" % hname] = \
            np.median(rel[half:], axis=0)
        out["meta"]["bins_in_band_%s" % hname] = int(np.sum(
            G.TWO_PI * np.arange(n // 2 + 1) / (n * G.DT)
            < G.BAND_EDGE_RATIO * w0[0]))
        out["meta"]["gyro_orbits_%s" % hname] = n * G.DT / G.TWO_PI

    runs = {}
    for s in G.SCHEMES:
        t0 = time.time()
        Rs, Vs, meta = MC.rollout(fld, s, R0, V0, G.DT, N - 1, idx, mlp=mlp)
        wall = time.time() - t0
        ch = G.channel_series(Rs, Vs, Rr, Vr, r_L)
        fps = MC.flops_per_step(s, meta.get("mean_iters"))
        for hname, n in G.HORIZONS.items():
            chn = slice_series(ch, n)
            sm = G.summarise(chn, G.DT, w0)
            # the same spectral channel under the alternative Omega_c
            w_alt = w_along[:n].mean(axis=0)
            pb_alt = np.array([
                G.band_power(chn["position_vector"][:, i, :], G.DT,
                             w_alt[i])["p_band"] for i in range(MC.N_IC)])
            sm["spectral"]["p_band_omega_c_alt"] = pb_alt
            sm["phase"]["polar_identity_max_rel_residual"] = \
                G.polar_identity_residual(Vs[:n], Vr[:n], chn["phase"])
            rec = {c: {k: v for k, v in sm[c].items()} for c in G.CHANNELS}
            rec["_run"] = {
                "n_steps": n, "h": G.DT, "t_final": (n - 1) * G.DT,
                "gyro_orbits": n * G.DT / G.TWO_PI,
                "flops_per_step": fps, "total_flops": fps * n,
                "n_nonfinite": meta["n_nonfinite"],
            }
            if "mean_iters" in meta:
                rec["_run"]["mean_iters"] = meta["mean_iters"]
                rec["_run"]["max_iters"] = meta["max_iters"]
            runs["%s|%s" % (hname, s)] = rec
        print("  %-10s %5.1f s  traj(H_paper,ic0)=%.4e  theta=%.4e  "
              "E=%.4e  P_band=%.4e"
              % (s, wall,
                 runs["H_paper|%s" % s]["trajectory"]["primary"][0],
                 runs["H_paper|%s" % s]["phase"]["primary"][0],
                 runs["H_paper|%s" % s]["energy"]["primary"][0],
                 runs["H_paper|%s" % s]["spectral"]["primary"][0]),
              flush=True)

    out["runs"] = runs
    out["meta"]["wall_s"] = time.time() - t_start
    path = os.path.join(HERE, "gt2_channels__%s.json" % fname)
    return check_or_write(path, json.loads(json.dumps(G.clean(out))),
                          force=force)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, choices=G.FIELD_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rc = 0
    for n in ([a.field] if a.field else G.FIELD_NAMES):
        rc |= run_field(n, force=a.force)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
