"""sw1_reference.py -- is the reference the floor of the measurement?

The pre-registration of wave W13 puts this check before every conclusion.  The
reference of Section 7 is DOP853 at rtol 1e-12, and W12 measured vps4 at equal
cost reaching 6.21e-12 Larmor radii against it.  If the residual of a scheme
falls to the error of the reference, the number that comes out is a property of
the reference.

Three things are established here, in this order.

1. CALIBRATION.  The scalar plane integrators of `sw_common.py` reproduce the
   four classical trajectory errors of Table 4 and the corrector's, against the
   same DOP853 reference, to the last printed digit.  A mismatch here means the
   rest of the directory is measuring some other scheme.

2. THE REFERENCE'S OWN ERROR.  The problem has a closed form (see the module
   docstring of `sw_common.py`), so the error of DOP853 can be measured rather
   than assumed, at three tolerances and at two horizons.

3. DO THE BAND POWERS MOVE?  Every band power of Section 3 is recomputed
   against four references -- DOP853 at rtol 1e-12, 1e-13 and 3e-14, and the
   closed form -- and each scheme is labelled reference-limited or not by
   whether its in-band power moves by more than one per cent between the
   tightest DOP853 and the closed form.

Writes sw1_reference.json; exits non-zero if a rerun stops reproducing it.
Usage: python sw1_reference.py [--force]
"""
import json
import os
import sys

import numpy as np

import sw_common as C
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sw1_reference.json")

#: the four trajectory errors Table 4 prints, and the corrector's, all at
#: Omega h = 0.3 over t = 120 against DOP853 rtol 1e-12 / atol 1e-14.
TABLE4 = {"boris": 0.4166534686545755,
          "vps2": 0.010583014451054718,
          "vps4": 5.35394015973843e-05,
          "gl4": 0.0007745714317733591,
          "corrector": 0.0034688730013653656}

REFS = [("dop853_rtol1e-12", dict(rtol=1e-12, atol=1e-14)),
        ("dop853_rtol1e-13", dict(rtol=1e-13, atol=1e-16)),
        ("dop853_rtol3e-14", dict(rtol=3e-14, atol=1e-16))]

#: equal-total-flop sub-stepping.  m is the largest integer for which m steps
#: of the scheme cost no more than one corrector step; gl4's per-step cost
#: depends on its iteration count and is solved for in `sw2_spectra.py`, which
#: reaches m = 146 at six iterations.
EQUAL_COST_M = {"boris": 1009, "vps2": 1253, "vps4": 417, "gl4": 146}


def run_scheme(name, n_out, mode):
    """One run.  mode is 'fixed' (Omega h = 0.3) or 'equalcost'."""
    if name == "corrector":
        return C.run_corrector(n_out)
    m = 1 if mode == "fixed" else EQUAL_COST_M[name]
    return C.run_classical(name, C.DT / m, m, n_out)


def main():
    force = "--force" in sys.argv
    out = {"meta": {
        "what": "reference adequacy for the W13 spectral probe",
        "tau": C.TAU, "dt": C.DT, "r0": list(C.R0), "v0": list(C.V0),
        "band": C.BAND,
        "closed_form": "zeta'' + (Bz^2/4) zeta = 0 -> Bessel order 0; "
                       "basis in mpmath at 40 digits, combination in float64",
        "n_random_draws": 0,
    }}

    # ---------------------------------------------------------- 1. calibration
    n1 = C.HORIZONS["H1_paper"]
    ts1 = np.arange(n1 + 1) * C.DT
    ref_paper = C.dop853_ref(ts1, rtol=1e-12, atol=1e-14)

    def refs_paper_pos(_ts):
        return ref_paper[0]
    cal = {}
    runs_h1 = {}
    for name, target in TABLE4.items():
        R, V, meta = run_scheme(name, n1, "fixed")
        runs_h1[name] = (R, V, meta)
        rms = float(np.sqrt(np.mean(np.sum((R - ref_paper[0]) ** 2, axis=1))))
        cal[name] = {"pos_err_rms": rms, "table4": target,
                     "rel_diff": abs(rms - target) / target}
    out["calibration_vs_table4"] = cal
    worst = max(v["rel_diff"] for v in cal.values())
    if worst > 1e-9:
        print("CALIBRATION FAILED: worst relative difference %.3e" % worst)
        return 1
    print("calibration: five schemes reproduce Table 4 to %.1e" % worst)

    # ------------------------------------------- 2. the reference's own error
    basis1 = C.bessel_basis(ts1)
    exact1 = C.exact_from_basis(basis1, C.R0, C.V0)
    ic_residual = float(np.abs(exact1[0][0] - C.R0).max()
                        + np.abs(exact1[1][0] - C.V0).max())
    n3 = C.HORIZONS["H3_long"]
    ts3 = np.arange(n3 + 1) * C.DT
    basis3 = C.bessel_basis(ts3)
    exact3 = C.exact_from_basis(basis3, C.R0, C.V0)

    # what the float64 reconstruction of the closed form costs, priced against
    # the same closed form carried end to end in mpmath at 49 samples spread
    # over the longest record, and the same again at 60 digits
    spot = np.linspace(0, n3, 49).astype(int)
    b_spot = C.bessel_basis(ts3[spot])
    r_f64 = C.exact_from_basis(b_spot, C.R0, C.V0)[0]
    r_mp40 = C.exact_reference_mp(ts3[spot], C.R0, C.V0, dps=40)[0]
    r_mp60 = C.exact_reference_mp(ts3[spot], C.R0, C.V0, dps=60)[0]
    out["closed_form_check"] = {
        "initial_condition_residual": ic_residual,
        "float64_reconstruction_max_abs": float(np.abs(r_f64 - r_mp40).max()),
        "dps40_vs_dps60_max_abs": float(np.abs(r_mp40 - r_mp60).max()),
        "n_spot_samples": int(len(spot)),
    }
    print("closed form: IC residual %.1e, float64 reconstruction %.1e, "
          "dps40 vs dps60 %.1e"
          % (ic_residual, out["closed_form_check"]["float64_reconstruction_max_abs"],
             out["closed_form_check"]["dps40_vs_dps60_max_abs"]))

    ref_err = {}
    for hz, ts, ex in (("H1_paper", ts1, exact1), ("H3_long", ts3, exact3)):
        ref_err[hz] = {}
        for tag, kw in REFS:
            Rr, _ = C.dop853_ref(ts, **kw)
            d = np.linalg.norm(Rr - ex[0], axis=1)
            ref_err[hz][tag] = {
                "pos_err_rms": float(np.sqrt(np.mean(d ** 2))),
                "pos_err_max": float(d.max()),
                "pos_err_final": float(d[-1]),
            }
            print("%-10s %-18s rms=%.3e max=%.3e"
                  % (hz, tag, ref_err[hz][tag]["pos_err_rms"],
                     ref_err[hz][tag]["pos_err_max"]))
    out["reference_own_error"] = ref_err

    # ------------------------------- 2b. the equal-cost figure the paper prints
    # \ref{sec:app_setups}: "at the flop count of the cheapest learned run it
    # reaches 6.2e-12 Larmor radii, which is where double precision floors it,
    # so the equal-cost factor is 1.2e9."  That step is h = 9.99e-4; the
    # nearest step that divides the output grid is h = 1.0e-3.  The claim that
    # double precision is what floors it is exactly the claim the closed form
    # can now adjudicate.
    w12 = {}
    for m_sub, tag in ((300, "h1.0e-3"), (417, "h7.19e-4_corrector_budget")):
        R, V, _ = C.run_classical("vps4", C.DT / m_sub, m_sub, n1)
        d_exact = np.linalg.norm(R - exact1[0], axis=1)
        d_paper = np.linalg.norm(R - refs_paper_pos(ts1), axis=1)
        w12[tag] = {
            "h": C.DT / m_sub,
            "pos_err_rms_vs_closed_form": float(np.sqrt(np.mean(d_exact ** 2))),
            "pos_err_rms_vs_dop853_rtol1e-12":
                float(np.sqrt(np.mean(d_paper ** 2))),
        }
        w12[tag]["inflation_of_the_paper_reference"] = (
            w12[tag]["pos_err_rms_vs_dop853_rtol1e-12"]
            / w12[tag]["pos_err_rms_vs_closed_form"])
        print("vps4 at h=%.3e: true rms %.4e, against DOP853 rtol 1e-12 %.4e "
              "(inflated %.1fx)"
              % (w12[tag]["h"], w12[tag]["pos_err_rms_vs_closed_form"],
                 w12[tag]["pos_err_rms_vs_dop853_rtol1e-12"],
                 w12[tag]["inflation_of_the_paper_reference"]))
    # the equal-cost factor of \ref{sec:external}, recomputed: the best of the
    # ninety-six searched learned runs is 7.21e-3 Larmor radii
    w12["sympnet_best_traj"] = 7.21e-3
    w12["equal_cost_factor_paper"] = 1.16e9
    w12["equal_cost_factor_closed_form"] = (
        7.21e-3 / w12["h1.0e-3"]["pos_err_rms_vs_closed_form"])
    print("equal-cost factor against the best learned run: paper 1.16e9, "
          "closed form %.3e" % w12["equal_cost_factor_closed_form"])
    out["w12_equal_cost_remeasured"] = w12

    # --------------------------------------------- 3. do the band powers move?
    def measure(R, V, refs, n_use):
        # the same convention as sw2_spectra.py: the record is the first n_use
        # samples, so that the two files compare the same bins
        rec = {}
        for tag, (Rr, Vr) in refs.items():
            ch = C.channels(R[:n_use], V[:n_use], Rr[:n_use], Vr[:n_use])
            rec[tag] = {k: C.band_powers(v, C.DT) for k, v in ch.items()}
        return rec

    refs1 = {tag: C.dop853_ref(ts1, **kw) for tag, kw in REFS}
    refs1["exact"] = exact1

    moved = {}
    for name in TABLE4:
        R, V, _ = runs_h1[name]
        moved["H1_fixed_" + name] = measure(R, V, refs1, n1)
        if name == "corrector":
            continue
        R, V, _ = run_scheme(name, n1, "equalcost")
        moved["H1_equalcost_" + name] = measure(R, V, refs1, n1)

    # the two schemes the pre-registration singles out as at risk, at the long
    # horizon where the reference has had time to accumulate
    refs3 = {tag: C.dop853_ref(ts3, **kw) for tag, kw in REFS}
    refs3["exact"] = exact3
    for name in ("vps4", "gl4"):
        R, V, _ = run_scheme(name, n3, "equalcost")
        moved["H3_equalcost_" + name] = measure(R, V, refs3, n3)

    verdict = {}
    for key, rec in moved.items():
        v = {}
        for ch in ("position", "energy"):
            a = rec["dop853_rtol3e-14"][ch]["p_band"]
            b = rec["exact"][ch]["p_band"]
            c = rec["dop853_rtol1e-12"][ch]["p_band"]
            rel_tight = abs(a - b) / max(b, 1e-300)
            rel_paper = abs(c - b) / max(b, 1e-300)
            v[ch] = {"p_band_exact": b,
                     "p_band_dop853_paper": c,
                     "p_band_dop853_tight": a,
                     "rel_shift_paper_ref": rel_paper,
                     "rel_shift_tight_ref": rel_tight,
                     "reference_limited": bool(rel_tight > 0.01)}
        verdict[key] = v
    out["band_power_vs_reference"] = moved
    out["verdict"] = verdict

    n_lim = sum(1 for k, v in verdict.items()
                if v["position"]["reference_limited"])
    print("\nreference-limited on the position channel: %d of %d runs"
          % (n_lim, len(verdict)))
    for k, v in sorted(verdict.items()):
        print("  %-28s paper-ref shift %10.3e   tight-ref shift %10.3e  %s"
              % (k, v["position"]["rel_shift_paper_ref"],
                 v["position"]["rel_shift_tight_ref"],
                 "REFERENCE-LIMITED" if v["position"]["reference_limited"]
                 else ""))

    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
