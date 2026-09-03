"""gt1_calibration.py -- the W15 stand, calibrated before it is used, and the
reference adjudicated per configuration and per channel.

    python gt1_calibration.py [--force]

Writes gt1_calibration.json beside this file and exits non-zero if a rerun no
longer reproduces the committed one.  Nothing is written outside this
directory and nothing is retrained.

WHAT IS CHECKED HERE, AND WHY EACH
-----------------------------------
1.  THE BAND IS DECLARED AND ITS AMBIGUITY IS PRICED.  Omega_c^ref is the
    initial gyrofrequency of the configuration (see `gt_common.py`); this
    section records it, records how far |B| actually moves along the reference
    orbit in each configuration, and records the alternative -- the time mean
    along the reference orbit -- that `gt3_gtable.py` carries as the declared
    sensitivity check.  The choice was written down before any run.

2.  THE STAND REPRODUCES Table~\\ref{tab:family}.  All five rows of the
    manuscript's own table are recomputed here through the W15 record, against
    the manuscript's DOP853 reference at rtol 1e-12, and compared with the
    printed values and with the committed `../map/mp1_calibration.json`.  This
    is the calibration the pre-registration asks for on the two channels that
    already existed.

3.  THE STAND REPRODUCES A COMMITTED SPECTRAL NUMBER.  The in-band ratios of
    `../spectral/sw2_spectra.json` for B4 at H1_paper are recomputed here by a
    different route: W13 integrates the planar reduction in scalar Python and
    W15 integrates the three-dimensional field classes in the batched bridge
    of W14.  The two must agree to rounding, and the amount by which they do
    is the calibration of the spectral channel.

4.  THE STAND REPRODUCES THE ONE COMMITTED PHASE NUMBER, AND THE arccos TRAP
    IS EXHIBITED.  Section~\\ref{sec:channels} reports a median angle of
    38.11 degrees over the second half of the Boris run in B4;
    `../theory_check/t1_boris_channels.json` holds the measurement.  It is
    reproduced three ways -- t1's own recipe, the same reference with the
    two-argument arctangent, and the closed form with the arctangent -- and
    then the angle at which arccos stops being a measurement is shown
    directly, because three of the five schemes hold theta below it.

5.  THE REFERENCE IS ADJUDICATED PER CONFIGURATION AND PER CHANNEL.  W13 found
    four of its eleven runs limited by the reference rather than by the
    scheme, and found it only by checking.  Here each configuration's second
    reference is put through the same four channels as if it were a scheme,
    which gives a floor in the units of each channel; `gt3_gtable.py` marks
    every cell within a factor of ten of it.
"""
import json
import os
import sys
import time

import numpy as np

import gt_common as G
import map_common as MC
import sw_common as SW
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gt1_calibration.json")

#: the printed values of Table~\ref{tab:family}, transcribed from
#: article/main.tex.  They are targets, not inputs.
TAB_FAMILY_TRAJ = {"boris": 0.417, "corrector": 3.47e-3, "vps2": 1.06e-2,
                   "vps4": 5.35e-5, "gl4": 7.75e-4}
TAB_FAMILY_ENERGY = {"boris": 1.25e-6, "corrector": 1.64e-5, "vps2": 5.57e-6,
                     "vps4": 2.64e-7, "gl4": 4.18e-8}
TAB_FAMILY_SIGNAL = 7.497e-4

#: the committed files this script calibrates against
SW2 = os.path.join(G.EXP, "spectral", "sw2_spectra.json")
MP1 = os.path.join(G.EXP, "map", "mp1_calibration.json")
T1 = os.path.join(G.EXP, "theory_check", "t1_boris_channels.json")


# ------------------------------------------------------------------ part 1 --
def declared_band(fields, R0, V0):
    """Omega_c^ref, and what it costs to have to declare it."""
    rec = {}
    for fname in G.FIELD_NAMES:
        field = fields[fname]
        w0 = G.reference_gyrofrequency(field, R0)
        d = {"omega_c_ref_per_ic": w0,
             "omega_c_ref_ic0": float(w0[0]),
             "band_edge_ic0": float(G.BAND_EDGE_RATIO * w0[0])}
        for hname, N in G.HORIZONS.items():
            ts = np.arange(N) * G.DT
            Rr, _ = reference_orbit(fname, field, R0, V0, ts, tag="best")
            w = G.gyrofrequency_along(field, Rr, ts)
            d[hname] = {
                "omega_c_along_reference_mean_ic0": float(w[:, 0].mean()),
                "omega_c_along_reference_min_ic0": float(w[:, 0].min()),
                "omega_c_along_reference_max_ic0": float(w[:, 0].max()),
                "relative_spread_ic0": float(
                    (w[:, 0].max() - w[:, 0].min()) / w[:, 0].mean()),
                "mean_over_initial_ic0": float(w[:, 0].mean() / w0[0]),
                # the declared sensitivity alternative, scheme-independent
                "omega_c_alt_mean_per_ic": w.mean(axis=0),
            }
        rec[fname] = d
    return rec


# ----------------------------------------------------------------- helpers --
_SOLS = {}


def reference_orbit(fname, field, R0, V0, ts, tag="best"):
    """The reference of a configuration on the grid `ts`, for every member.

    tag = "best"   the closed form where there is one, else DOP853 rtol 3e-14
          "paper"  DOP853 rtol 1e-12, the reference Table~\\ref{tab:family} is
                   scored against
          "second" the independent second reference used to price the first:
                   DOP853 rtol 3e-14 where the first is a closed form, and
                   DOP853 rtol 1e-13 where the first is DOP853 rtol 3e-14
    """
    nb = R0.shape[0]
    n = len(ts)
    Rr = np.empty((n, nb, 3)); Vr = np.empty((n, nb, 3))
    closed = MC.CLOSED_FORM[fname] is not None
    if tag == "best" and closed:
        for i in range(nb):
            r, v = MC.exact(fname, field, ts, R0[i], V0[i])
            Rr[:, i], Vr[:, i] = r, v
        return Rr, Vr
    if tag == "best":
        kw = dict(rtol=3e-14, atol=1e-16)
    elif tag == "paper":
        kw = dict(rtol=1e-12, atol=1e-14)
    elif tag == "second":
        kw = (dict(rtol=3e-14, atol=1e-16) if closed
              else dict(rtol=1e-13, atol=1e-15))
    else:
        raise ValueError(tag)
    key = (fname, kw["rtol"], float(ts[-1]))
    if key not in _SOLS:
        _SOLS[key] = [MC.dop853(field, np.array([0.0, float(ts[-1])]),
                                R0[i], V0[i], **kw) for i in range(nb)]
    for i in range(nb):
        r, v = MC.dop853_at(_SOLS[key][i], ts)
        Rr[:, i], Vr[:, i] = r, v
    return Rr, Vr


# ------------------------------------------------------------------ part 2 --
def calibrate_tab_family(fields, fast, R0, V0, mlp):
    """All five rows of Table~\\ref{tab:family}, recomputed on this stand.

    Scored exactly as the table is: B4, Omega h = 0.3, t = 120, against DOP853
    at rtol 1e-12, root mean square of the position error in Larmor radii and
    the median relative energy error over the second half.  The table's record
    is N+1 = 401 samples, t = 0 .. 120; the W15 record is N = 400 samples,
    t = 0 .. 119.7.  Both are reported so that the cost of the convention is a
    measured number and not an assumption.
    """
    fname = "B4_decaying"
    field = fields[fname]
    n = G.HORIZONS["H_paper"]                       # 400
    ts_401 = np.arange(n + 1) * G.DT
    idx = np.arange(n + 1)
    r_L = MC.larmor_radii(field, R0, V0)
    Rr, Vr = reference_orbit(fname, field, R0, V0, ts_401, tag="paper")

    E_ref = 0.5 * np.sum(Vr[:, 0] ** 2, axis=1)
    half = (n + 1) // 2
    signal = float(np.median(np.abs(E_ref - E_ref[0])[half:] / E_ref[0]))

    cal = {}
    worst_t = worst_e = 0.0
    for s in G.SCHEMES:
        Rs, Vs, meta = MC.rollout(fast[fname], s, R0, V0, G.DT, n, idx,
                                  mlp=mlp)
        ch = G.channel_series(Rs, Vs, Rr, Vr, r_L)
        rms401 = float(np.sqrt(np.mean(ch["trajectory"][:, 0] ** 2)))
        en401 = float(np.median(np.abs(ch["energy"][half:, 0])))
        h400 = n // 2
        rms400 = float(np.sqrt(np.mean(ch["trajectory"][:n, 0] ** 2)))
        en400 = float(np.median(np.abs(ch["energy"][h400:n, 0])))
        rt = abs(rms401 - TAB_FAMILY_TRAJ[s]) / TAB_FAMILY_TRAJ[s]
        re = abs(en401 - TAB_FAMILY_ENERGY[s]) / TAB_FAMILY_ENERGY[s]
        worst_t = max(worst_t, rt); worst_e = max(worst_e, re)
        cal[s] = {
            "trajectory_rms_401": rms401,
            "table_family_trajectory": TAB_FAMILY_TRAJ[s],
            "rel_diff_trajectory_vs_printed": rt,
            "energy_median_2nd_half_401": en401,
            "table_family_energy": TAB_FAMILY_ENERGY[s],
            "rel_diff_energy_vs_printed": re,
            "trajectory_rms_400_W15_record": rms400,
            "energy_median_2nd_half_400_W15_record": en400,
            "record_convention_rel_diff_trajectory":
                abs(rms400 - rms401) / rms401,
            "record_convention_rel_diff_energy":
                abs(en400 - en401) / max(en401, 1e-300),
            "flops_per_step": MC.flops_per_step(s, meta.get("mean_iters")),
            "total_flops_400_steps": MC.flops_per_step(
                s, meta.get("mean_iters")) * n,
        }
        if "mean_iters" in meta:
            cal[s]["mean_iters"] = meta["mean_iters"]
        print("  %-10s traj %.10e (printed %.3g, rel %.2e)"
              % (s, rms401, TAB_FAMILY_TRAJ[s], rt), flush=True)

    # the same rows against the committed mp1_calibration.json, leaf by leaf
    mp1 = json.load(open(MP1, encoding="utf-8"))["calibration_vs_tab_family"]
    agree = {}
    for s in G.SCHEMES:
        a = cal[s]["trajectory_rms_401"]; b = mp1[s]["pos_err_rms"]
        c = cal[s]["energy_median_2nd_half_401"]
        d = mp1[s]["energy_err_median_2nd_half"]
        agree[s] = {"trajectory_rel_diff_vs_mp1": abs(a - b) / b,
                    "energy_rel_diff_vs_mp1": abs(c - d) / d}
    return {
        "rows": cal,
        "vs_committed_mp1_calibration": agree,
        "worst_rel_diff_trajectory_vs_printed": worst_t,
        "worst_rel_diff_energy_vs_printed": worst_e,
        "physical_signal_median_2nd_half": signal,
        "table_family_signal_printed": TAB_FAMILY_SIGNAL,
        "signal_rel_diff": abs(signal - TAB_FAMILY_SIGNAL) / TAB_FAMILY_SIGNAL,
        "note": "the printed table carries three significant figures, so a "
                "relative difference of order 1e-3 against it is the printing "
                "and not the stand; the comparison against mp1_calibration."
                "json is the one carried to full precision",
    }


# ------------------------------------------------------------------ part 3 --
def calibrate_spectral(fields, fast, R0, V0, mlp):
    """The in-band ratios of `../spectral/sw2_spectra.json`, recomputed here.

    W13 measures them on the planar reduction integrated in scalar Python;
    W15 measures them on the three-dimensional field class integrated in the
    batched bridge, against the same closed form, with Omega_c^ref = 1 as B4
    gives.  The two routes share the periodogram and nothing else.
    """
    fname = "B4_decaying"
    field = fields[fname]
    n = G.HORIZONS["H_paper"]
    ts = np.arange(n) * G.DT
    idx = np.arange(n)
    r_L = MC.larmor_radii(field, R0, V0)
    Rr, Vr = reference_orbit(fname, field, R0, V0, ts, tag="best")
    w0 = G.reference_gyrofrequency(field, R0)

    pb = {}
    for s in G.SCHEMES:
        Rs, Vs, _ = MC.rollout(fast[fname], s, R0, V0, G.DT, n - 1, idx,
                               mlp=mlp)
        ch = G.channel_series(Rs, Vs, Rr, Vr, r_L)
        bp = G.band_power(ch["position_vector"][:, 0, :], G.DT, w0[0])
        pb[s] = bp

    sw2 = json.load(open(SW2, encoding="utf-8"))
    ref = sw2["boris_over_scheme"]["H1_paper"]["fixed_step"]["position"]
    swruns = sw2["horizons"]["H1_paper"]["runs"]["fixed_step"]
    rec = {}
    for s in G.SCHEMES:
        d = {"p_band_W15": pb[s]["p_band"],
             "p_band_W13": swruns[s]["position"]["p_band"],
             "p_total_W15": pb[s]["p_total"],
             "p_total_W13": swruns[s]["position"]["p_total"],
             "n_bins_in_band": pb[s]["n_bins_in_band"]}
        d["p_band_rel_diff"] = abs(d["p_band_W15"] - d["p_band_W13"]) \
            / d["p_band_W13"]
        if s != G.BASE:
            r = pb[G.BASE]["p_band"] / pb[s]["p_band"]
            d["band_ratio_orders_W15"] = float(np.log10(r))
            d["band_ratio_orders_W13"] = ref[s]["band_ratio_orders"]
            d["band_ratio_orders_abs_diff"] = abs(
                d["band_ratio_orders_W15"] - d["band_ratio_orders_W13"])
        rec[s] = d
        print("  %-10s p_band W15 %.10e  W13 %.10e  rel %.2e"
              % (s, d["p_band_W15"], d["p_band_W13"],
                 d["p_band_rel_diff"]), flush=True)
    rec["_worst_band_ratio_orders_abs_diff"] = max(
        rec[s]["band_ratio_orders_abs_diff"] for s in G.SCHEMES
        if s != G.BASE)
    rec["_note"] = (
        "the two routes are not expected to agree bit for bit: W13 integrates "
        "the planar reduction with a scalar inner loop and W15 the "
        "three-dimensional field class with the batched bridge, so they "
        "differ by the order in which the same arithmetic is done.  The "
        "agreement measured above is what calibrates the spectral channel of "
        "this directory")
    return rec


# ------------------------------------------------------------------ part 4 --
def calibrate_phase(fields, fast, R0, V0, mlp):
    """The one committed phase number, and the arccos trap that would ruin it.

    `../theory_check/t1_boris_channels.py` measures the median angle over the
    second half of the Boris run in B4 as 38.110 degrees, using a Boris
    reference at h/150 and arccos.  Reproduced here three ways.
    """
    fname = "B4_decaying"
    field = fields[fname]
    n = G.HORIZONS["H_paper"]
    ts = np.arange(n + 1) * G.DT
    idx = np.arange(n + 1)
    half = (n + 1) // 2

    # (a) t1's own reference: the same Boris scheme at h/150, read at the
    #     matched integer times.  Reproduced through this stand's rollout.
    fine = MC.rollout(fast[fname], "boris", R0, V0, G.DT / 150, n * 150,
                      np.arange(n * 150 + 1)[::150])
    Rf, Vf = fine[0], fine[1]
    Rw, Vw, _ = MC.rollout(fast[fname], "boris", R0, V0, G.DT, n, idx)

    def theta_atan2(Va, Vb):
        return np.arctan2(np.linalg.norm(np.cross(Va, Vb), axis=-1),
                          np.sum(Va * Vb, axis=-1))

    def theta_arccos(Va, Vb):
        c = (np.sum(Va * Vb, axis=-1)
             / (np.linalg.norm(Va, axis=-1) * np.linalg.norm(Vb, axis=-1)))
        return np.arccos(np.clip(c, -1.0, 1.0))

    t1 = json.load(open(T1, encoding="utf-8"))
    target = t1["theta_median_2nd_half_deg"]["measured"]

    got = {}
    for tag, th in (("t1_recipe_boris_h_over_150_arccos",
                     theta_arccos(Vw[:, 0], Vf[:, 0])),
                    ("same_reference_atan2", theta_atan2(Vw[:, 0], Vf[:, 0]))):
        v = float(np.degrees(np.median(np.abs(th[half:]))))
        got[tag] = {"deg": v, "abs_diff_vs_t1": abs(v - target),
                    "rel_diff_vs_t1": abs(v - target) / target}

    Rr, Vr = reference_orbit(fname, field, R0, V0, ts, tag="best")
    th_cf = theta_atan2(Vw[:, 0], Vr[:, 0])
    v = float(np.degrees(np.median(np.abs(th_cf[half:]))))
    got["closed_form_reference_atan2"] = {
        "deg": v, "abs_diff_vs_t1": abs(v - target),
        "rel_diff_vs_t1": abs(v - target) / target,
        "note": "a different reference from t1's, so this is not expected to "
                "agree to rounding; the h/150 Boris reference carries its own "
                "phase lag, smaller by 150^2"}

    # the trap, shown rather than asserted: every scheme's angle under both
    # formulas, against the closed form
    trap = {}
    for s in G.SCHEMES:
        Rs, Vs, _ = MC.rollout(fast[fname], s, R0, V0, G.DT, n, idx, mlp=mlp)
        ta = theta_atan2(Vs[:, 0], Vr[:, 0])
        tc = theta_arccos(Vs[:, 0], Vr[:, 0])
        ma = float(np.median(np.abs(ta[half:])))
        mc = float(np.median(np.abs(tc[half:])))
        trap[s] = {"theta_atan2_rad": ma, "theta_arccos_rad": mc,
                   "arccos_over_atan2": mc / ma if ma > 0 else float("inf"),
                   "arccos_is_exactly_zero": bool(mc == 0.0)}
        print("  %-10s theta atan2 %.6e  arccos %.6e" % (s, ma, mc),
              flush=True)
    return {
        "t1_target_deg": target,
        "article_deg": t1["theta_median_2nd_half_deg"]["article"],
        "reproductions": got,
        "arccos_vs_atan2_per_scheme": trap,
        "resolution_note": (
            "arccos of a double-precision cosine cannot resolve an angle "
            "below about sqrt(2 eps) = 2.1e-8 rad; three of the five schemes "
            "sit below that, so the phase channel of this wave is measured "
            "with the two-argument arctangent throughout and arccos appears "
            "nowhere outside this calibration"),
        "sqrt_2eps_rad": float(np.sqrt(2.0 * np.finfo(float).eps)),
    }


# ------------------------------------------------------------------ part 5 --
def reference_floors(fields, R0, V0):
    """What the reference of each configuration is worth, in each channel.

    The second reference is put through the four channels exactly as a scheme
    is: the primary reference plays the role of the numerical solution and the
    second reference the role of the truth.  What comes out is a floor in the
    units of each channel, which is the only form in which "the residual has
    fallen to the level of the reference" can be checked.
    """
    rec = {}
    for fname in G.FIELD_NAMES:
        field = fields[fname]
        r_L = MC.larmor_radii(field, R0, V0)
        w0 = G.reference_gyrofrequency(field, R0)
        closed = MC.CLOSED_FORM[fname] is not None
        d = {"closed_form": MC.CLOSED_FORM[fname],
             "reference_primary": ("closed form" if closed
                                   else "DOP853 rtol 3e-14 atol 1e-16"),
             "reference_second": ("DOP853 rtol 3e-14 atol 1e-16" if closed
                                  else "DOP853 rtol 1e-13 atol 1e-15")}
        for hname, N in G.HORIZONS.items():
            ts = np.arange(N) * G.DT
            Ra, Va = reference_orbit(fname, field, R0, V0, ts, tag="best")
            Rb, Vb = reference_orbit(fname, field, R0, V0, ts, tag="second")
            ch = G.channel_series(Ra, Va, Rb, Vb, r_L)
            sm = G.summarise(ch, G.DT, w0)
            d[hname] = {c: {"floor_primary_statistic": G.stat(sm[c]["primary"]),
                            "floor_rms": G.stat(sm[c]["rms"])}
                        for c in G.CHANNELS}
        rec[fname] = d
        print("  %-12s floor(traj,H_paper,ic0) = %.3e"
              % (fname,
                 rec[fname]["H_paper"]["trajectory"]
                 ["floor_primary_statistic"]["ic0"]), flush=True)

    # The floor that actually applies.  Where the reference of `gt2` is a
    # closed form, its own error is not the distance to DOP853 -- that is the
    # error of DOP853 -- but the cost of evaluating the closed form in
    # float64.  It is measured here against the same closed form carried end
    # to end in mpmath at forty digits, at the canonical initial condition and
    # at both horizons, in all four channels.  Three orders below the
    # conservative figure above, and it is this one `gt3_gtable.py` uses for
    # the three configurations that have a closed form.
    tight = {}
    for fname in G.FIELD_NAMES:
        if MC.CLOSED_FORM[fname] is None:
            continue
        field = fields[fname]
        r_L = MC.larmor_radii(field, R0[:1], V0[:1])
        w0 = G.reference_gyrofrequency(field, R0[:1])
        d = {}
        for hname, N in G.HORIZONS.items():
            ts = np.arange(N) * G.DT
            Ra, Va = MC.exact(fname, field, ts, R0[0], V0[0])
            Rm, Vm = MC.exact_mp(fname, field, ts, R0[0], V0[0], dps=40)
            ch = G.channel_series(Ra[:, None, :], Va[:, None, :],
                                  Rm[:, None, :], Vm[:, None, :], r_L)
            sm = G.summarise(ch, G.DT, w0)
            d[hname] = {c: float(sm[c]["primary"][0]) for c in G.CHANNELS}
            print("  %-12s %-12s tight float64-vs-mpmath40 traj %.3e "
                  "phase %.3e"
                  % (fname, hname, d[hname]["trajectory"],
                     d[hname]["phase"]), flush=True)
        tight[fname] = d
    return {"per_configuration": rec,
            "closed_form_float64_vs_mpmath40_ic0": tight,
            "ref_limit_factor": G.REF_LIMIT_FACTOR,
            "which_floor_applies":
                "gt2_channels.py scores the three closed-form configurations "
                "against the float64 closed form, so the floor that applies "
                "to them is closed_form_float64_vs_mpmath40_ic0 and not the "
                "distance to DOP853, which is DOP853's error.  B1 and B2 have "
                "no closed form, are scored against DOP853 at rtol 3e-14, and "
                "take the per_configuration floor",
            "method": "the second reference is scored through the same four "
                      "channels as a scheme, so the floor is in the units of "
                      "each channel; a cell within ref_limit_factor of it is "
                      "marked in gt3_gtable.json rather than reported as a "
                      "property of the scheme"}


# --------------------------------------------------------------------- main -
def main():
    force = "--force" in sys.argv
    t0 = time.time()
    fields = MC.make_fields()
    fast = MC.make_fast_fields(fields)
    R0, V0 = MC.initial_conditions(MC.N_IC)
    mlp = MC.load_corrector_numpy()

    out = {"meta": {
        "what": "calibration of the W15 stand and the reference, per "
                "configuration and per channel",
        "dt": G.DT, "horizons_samples": G.HORIZONS,
        "horizons_t": G.HORIZON_T,
        "gyro_orbits": {k: v / G.TWO_PI for k, v in G.HORIZON_T.items()},
        "bins_in_band_identity": "n_bins = 0.2 * (gyro-orbits in the record)",
        "schemes": G.SCHEMES, "fields": G.FIELD_NAMES,
        "channels": G.CHANNELS,
        "band_edge_ratio": G.BAND_EDGE_RATIO,
        "omega_c_ref_declaration":
            "Omega_c^ref = |q/m| |B(r_0, 0)|, the initial gyrofrequency of the "
            "configuration, declared in gt_common.py before any run; "
            "scheme-independent by construction",
        "phase_definition":
            "theta = atan2(|v x v_ref|, v . v_ref); never arccos",
        "n_initial_conditions": MC.N_IC,
        "n_random_draws": G.N_RANDOM_DRAWS,
        "seeds": "none drawn here; the initial conditions are "
                 "../map/map_common.py:initial_conditions with MAP_SEED",
        "corrector_checkpoint": "checkpoints/boris_corrector_b4.pt -- one "
                                "checkpoint, trained on B4 at Omega h = 0.3; "
                                "every corrector number in this directory is "
                                "that one run and not an ensemble",
        "rank_statistic": G.RANK_STAT,
    }}

    print("1. the declared band", flush=True)
    out["declared_band"] = declared_band(fields, R0, V0)
    for f in G.FIELD_NAMES:
        b = out["declared_band"][f]
        print("  %-12s Omega_c^ref = %.10f   spread along the reference "
              "(H_paper) = %.2e"
              % (f, b["omega_c_ref_ic0"],
                 b["H_paper"]["relative_spread_ic0"]), flush=True)

    print("2. Table tab:family, all five rows", flush=True)
    out["calibration_vs_tab_family"] = calibrate_tab_family(
        fields, fast, R0, V0, mlp)

    print("3. the committed spectral numbers of W13", flush=True)
    out["calibration_vs_sw2_spectra"] = calibrate_spectral(
        fields, fast, R0, V0, mlp)

    print("4. the committed phase number, and arccos", flush=True)
    out["calibration_vs_t1_phase"] = calibrate_phase(
        fields, fast, R0, V0, mlp)

    print("5. the reference, per configuration and per channel", flush=True)
    out["reference_floors"] = reference_floors(fields, R0, V0)

    out["meta"]["wall_s"] = time.time() - t0
    return check_or_write(OUT, json.loads(json.dumps(G.clean(out))),
                          force=force)


if __name__ == "__main__":
    raise SystemExit(main())
