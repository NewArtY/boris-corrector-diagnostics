"""sw3_ensemble.py -- is the band result a property of one initial condition?

`sw2_spectra.py` runs the initial condition of \\ref{sec:app_setups}, which is
the one every number in Section 7 is taken on.  A claim about where the
residual sits in frequency should not rest on it alone, so the same measurement
is repeated over eight initial conditions drawn from the distribution the
corrector was trained on -- rho in [0.7, 1.3], v_parallel in [-0.15, 0.15],
uniform phase -- which is the ensemble of \\ref{sec:app_setups}.

Only the fixed-step comparison is run here, at Omega h = 0.3 over the window of
Table 4.  That is the first author's setting verbatim, and it is the setting in
which his claim is stated; the equal-cost gate is decided in `sw2_spectra.py`
by a margin of twenty orders of magnitude, which no spread over eight initial
conditions can touch.

A second reading is priced here as well.  The residual of the position channel
can be taken as the error *vector*, which is what the pre-registration fixes
and what makes the total power the mean square position error, or as the scalar
|dr|.  The two are not the same measurement: taking the norm rectifies an
oscillation at Omega_c and folds it onto DC and 2 Omega_c, which moves an
in-band power by up to seven orders for a scheme whose residual is one clean
line.  Both are reported so that the claim is tested under the reading most
favourable to it as well as under the declared one.

SEEDS.  Eight draws, the number declared before the run in the module docstring
of `sw_common.py`.  One generator, built once, outside every loop, from
`C.SPECTRAL_SEED`.  The drawn states are written into the output beside the
numbers they produced.

Writes sw3_ensemble.json; exits non-zero if a rerun stops reproducing it.
Usage: python sw3_ensemble.py [--force]
"""
import json
import os
import sys

import numpy as np

import sw_common as C
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sw3_ensemble.json")

SCHEMES = ["boris", "corrector", "vps2", "vps4", "gl4"]


def draw_initial_conditions():
    """The eight states, from one generator built once outside every loop."""
    rng = np.random.default_rng(C.SPECTRAL_SEED)
    rho = rng.uniform(0.7, 1.3, C.N_ENSEMBLE_DRAWS)
    phase = rng.uniform(0.0, 2.0 * np.pi, C.N_ENSEMBLE_DRAWS)
    vpar = rng.uniform(-0.15, 0.15, C.N_ENSEMBLE_DRAWS)
    out = []
    for i in range(C.N_ENSEMBLE_DRAWS):
        r0 = np.array([rho[i] * np.cos(phase[i]), rho[i] * np.sin(phase[i]), 0.0])
        v0 = np.array([-np.sin(phase[i]), np.cos(phase[i]), vpar[i]])
        out.append((r0, v0, {"rho": float(rho[i]), "phase": float(phase[i]),
                             "v_par": float(vpar[i])}))
    return out


def main():
    force = "--force" in sys.argv
    n = C.HORIZONS["H1_paper"]
    ts = np.arange(n + 1) * C.DT
    basis = C.bessel_basis(ts)          # the grid is shared by every draw

    out = {"meta": {
        "what": "eight initial conditions at Omega h = 0.3 over the window of "
                "Table 4, the first author's setting verbatim",
        "seed": C.SPECTRAL_SEED, "n_random_draws": C.N_ENSEMBLE_DRAWS,
        "distribution": "rho U[0.7,1.3], phase U[0,2pi), v_par U[-0.15,0.15], "
                        "the ensemble of the appendix",
        "band": C.BAND, "dt": C.DT, "n_samples": n,
        "reference": "closed form, basis in mpmath at 40 digits",
    }, "draws": [], "per_draw": []}

    for k, (r0, v0, tag) in enumerate(draw_initial_conditions()):
        Rr, Vr = C.exact_from_basis(basis, r0, v0)
        rec = {"draw": k, **tag, "r0": list(r0), "v0": list(v0), "schemes": {}}
        for name in SCHEMES:
            if name == "corrector":
                R, V, _ = C.run_corrector(n, r0=r0, v0=v0)
            else:
                R, V, _ = C.run_classical(name, C.DT, 1, n, r0=r0, v0=v0)
            ch = C.channels(R[:n], V[:n], Rr[:n], Vr[:n])
            entry = {}
            for cname, series in ch.items():
                bp = C.band_powers(series, C.DT)
                bh = C.band_powers(series, C.DT, window="blackmanharris")
                entry[cname] = {**bp, "p_band_blackmanharris": bh["p_band"],
                                "frac_in_band": bp["p_band"] / bp["p_total"]}
            scal = np.linalg.norm(ch["position"], axis=1)[:, None]
            entry["position_scalar_norm"] = C.band_powers(scal, C.DT)
            rec["schemes"][name] = entry
        out["per_draw"].append(rec)
        out["draws"].append(tag)
        print("draw %d rho=%.3f vpar=%+.3f  Boris band=%.4e  corrector band=%.4e"
              % (k, tag["rho"], tag["v_par"],
                 rec["schemes"]["boris"]["position"]["p_band"],
                 rec["schemes"]["corrector"]["position"]["p_band"]))

    # ------------------------------------------------------------- summary --
    summ = {}
    for ch in ("position", "energy", "position_scalar_norm"):
        summ[ch] = {}
        for name in SCHEMES:
            if name == "boris":
                continue
            vals = []
            for rec in out["per_draw"]:
                b = rec["schemes"]["boris"][ch]["p_band"]
                o = rec["schemes"][name][ch]["p_band"]
                vals.append(float(np.log10(b / o)) if o > 0 else float("inf"))
            fin = [v for v in vals if np.isfinite(v)]
            summ[ch][name] = {
                "band_ratio_orders": vals,
                "median": float(np.median(fin)) if fin else float("inf"),
                "min": float(np.min(fin)) if fin else float("inf"),
                "max": float(np.max(fin)) if fin else float("inf"),
                "n_draws_above_six_orders": int(sum(1 for v in vals if v >= 6.0)),
            }
    out["boris_over_scheme_orders"] = summ

    print("\nBoris over scheme, in-band power, orders of magnitude, "
          "over eight initial conditions:")
    for ch in ("position", "energy", "position_scalar_norm"):
        print(" %s" % ch)
        for name, v in summ[ch].items():
            print("   %-10s median %7.3f   range %7.3f .. %7.3f   "
                  "draws reaching six orders: %d/8"
                  % (name, v["median"], v["min"], v["max"],
                     v["n_draws_above_six_orders"]))

    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
