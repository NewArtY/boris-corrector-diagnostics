"""RC2: the decisive discriminator, run identically in both systems.

Two knobs, one question.

  KNOB 1 (read-out).  Best single global time offset delta applied to the
  reference when the reported deviation is formed.  If the observed floor is a
  read-out artefact, some delta must kill it.

  KNOB 2 (initial data).  Shift the initial condition by an O(h/2) amount and
  re-measure the floor.  If the floor is a dynamical property of the discrete
  orbit, some shift must kill it; a pure read-out artefact cannot care.

The floor is reported dimensionlessly as  dev0 / A_ref  (the scheme's own error
in units of the pure h/2 read-out artefact), so magnetic and LL numbers are
directly comparable.

Also: the SHAPE of the deviation series (envelope / median).  A constant
sampling offset gives envelope == median (ratio 1); an oscillation about zero
gives ratio 2.

Output: rc2_decisive.json
"""
import json, math, os
import numpy as np
import rc_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ================================================================ SYSTEM M ===
TAU, TF, H = 1.2e5, 120.0, 0.3
fB, ffac = rc.mk_field("exp", TAU)


def Qref_mag(t):
    return np.exp(-np.asarray(t, float) / TAU)


def mag_floor(scheme, h, r0):
    t, Q, rho = rc.run_mag(scheme, fB, ffac, h, TF, r0=r0)
    m = rc.metrics(t, Q, Qref_mag, None, h)
    half = len(t) // 2
    d = Q[half:] - Qref_mag(t[half:])
    return t, Q, m, {"signed_min": float(d.min()), "signed_max": float(d.max()),
                     "median_abs": float(np.median(np.abs(d))),
                     "envelope_over_median": float(np.max(np.abs(d)) /
                                                   max(np.median(np.abs(d)), 1e-300))}


# --- knob 1: best read-out offset -------------------------------------------
k1 = {}
for sch in rc.MAG_SCHEMES:
    t, Q, m, shape = mag_floor(sch, H, (1.0, 0.0, 0.0))
    bo = rc.best_offset(t, Q, Qref_mag, H)
    k1[sch] = {"dev0": m["dev0"], "A_ref": m["A_ref"],
               "floor_over_A_ref": m["floor_in_units_of_artefact"],
               **bo, **shape}
OUT["M_knob1_readout_offset"] = k1

# --- knob 2: initial-data shift ---------------------------------------------
# discrete guiding centre of the shipped map sits at r_n - M v_n with
# M = h R (R - I)^{-1}, R = rotation by -theta_h about z, theta_h = 2 atan(h/2).
th_h = 2 * math.atan(H / 2)
Rm = np.array([[math.cos(th_h), math.sin(th_h)], [-math.sin(th_h), math.cos(th_h)]])
Mm = H * Rm @ np.linalg.inv(Rm - np.eye(2))
r0_gc = np.array([Mm[0, 1] * 1.0, Mm[1, 1] * 1.0, 0.0])   # M @ (0,1)

k2 = {"scan": [], "named": {}}
for lbl, r0 in (("baseline r0=(1,0,0)", (1.0, 0.0, 0.0)),
                ("r0=(1,h/2,0)", (1.0, H / 2.0, 0.0)),
                ("r0 = M v0 (discrete GC on axis)", tuple(r0_gc))):
    t, Q, m, shape = mag_floor("boris_shipped", H, r0)
    bo = rc.best_offset(t, Q, Qref_mag, H)
    k2["named"][lbl] = {"r0": [float(x) for x in r0], "dev0": m["dev0"],
                        "A_ref": m["A_ref"],
                        "floor_over_A_ref": m["floor_in_units_of_artefact"],
                        "readout_collapse_factor": bo["collapse_factor"],
                        **shape}
base = k2["named"]["baseline r0=(1,0,0)"]["dev0"]
for b in np.linspace(-0.3, 0.3, 61):
    t, Q, m, shape = mag_floor("boris_shipped", H, (1.0, float(b), 0.0))
    k2["scan"].append({"b": float(b), "dev0": m["dev0"],
                       "floor_over_A_ref": m["floor_in_units_of_artefact"]})
vals = np.array([r["floor_over_A_ref"] for r in k2["scan"]])
k2["scan_summary"] = {"min": float(vals.min()), "max": float(vals.max()),
                      "argmin_b": float(k2["scan"][int(vals.argmin())]["b"]),
                      "max_over_min": float(vals.max() / vals.min()),
                      "h_v_over_2": H * 1.0 / 2.0}
OUT["M_knob2_initial_data"] = k2

# ================================================================ SYSTEM L ===
r_e, c_l = 2.8179403262e-15, 2.99792458e8
omega = 2 * math.pi * c_l / 1.0e-6
EPS_PHYS = (2 * r_e * omega / (3 * c_l)) * 857.0
SETS = {"physical": dict(alpha=0.36, eps=EPS_PHYS, theta0=0.8, T=8.0, h=0.05),
        "generic": dict(alpha=1.0, eps=0.1, theta0=0.3, T=2.0, h=0.05)}


def ll_floor(sysL, scheme, h, T, theta0):
    t, thn, Q = sysL.run(scheme, h, T, theta0=theta0)
    m = rc.metrics(t, Q, sysL.Qref, sysL.dQref_dt, h)
    half = len(t) // 2
    d = Q[half:] - sysL.Qref(t[half:])
    shape = {"signed_min": float(d.min()), "signed_max": float(d.max()),
             "median_abs": float(np.median(np.abs(d))),
             "envelope_over_median": float(np.max(np.abs(d)) /
                                           max(np.median(np.abs(d)), 1e-300))}
    return t, thn, Q, m, shape


LL_OUT = {}
for pname, S in SETS.items():
    sysL = rc.LL(S["alpha"], S["eps"], S["theta0"])
    h, T = S["h"], S["T"]
    rec = {"knob1_readout_offset": {}, "knob2_initial_data": {"scan": [], "named": {}}}
    for sch in rc.LL_SCHEMES:
        t, thn, Q, m, shape = ll_floor(sysL, sch, h, T, S["theta0"])
        bo = rc.best_offset(t, Q, sysL.Qref, h)
        # exact time lag of the numerical trajectory (scalar autonomous ODE:
        # the numerical point always lies ON the true orbit)
        half = len(t) // 2
        s_n = sysL.t_of_theta(thn[half:]) - t[half:]
        rec["knob1_readout_offset"][sch] = {
            "dev0": m["dev0"], "A_ref": m["A_ref"],
            "floor_over_A_ref": m["floor_in_units_of_artefact"],
            **bo, **shape,
            "s_exact_median": float(np.median(s_n)),
            "s_exact_min": float(s_n.min()), "s_exact_max": float(s_n.max()),
            "two_s_over_h": float(2 * abs(np.median(s_n)) / h)}
    # knob 2: shift the initial condition, reference follows (autonomous system)
    for lbl, th0 in (("baseline", S["theta0"]),
                     ("theta0 = theta_exact(h/2)  (exact time translation by h/2)",
                      float(rc.LL(S["alpha"], S["eps"], S["theta0"]).theta_exact(h / 2))),
                     ("theta0 + h/2", S["theta0"] + h / 2)):
        s2 = rc.LL(S["alpha"], S["eps"], th0)
        t, thn, Q, m, shape = ll_floor(s2, "euler", h, T, th0)
        bo = rc.best_offset(t, Q, s2.Qref, h)
        rec["knob2_initial_data"]["named"][lbl] = {
            "theta0": th0, "dev0": m["dev0"], "A_ref": m["A_ref"],
            "floor_over_A_ref": m["floor_in_units_of_artefact"],
            "readout_collapse_factor": bo["collapse_factor"], **shape}
    lo, hi = (0.2, 1.6) if pname == "physical" else (0.05, 0.7)
    for sch in ("euler", "trapezoid", "rk4"):
        vals = []
        for th0 in np.linspace(lo, hi, 121):
            s2 = rc.LL(S["alpha"], S["eps"], float(th0))
            t, thn, Q, m, shape = ll_floor(s2, sch, h, T, float(th0))
            vals.append((float(th0), m["floor_in_units_of_artefact"]))
        v = np.array([x[1] for x in vals])
        rec["knob2_initial_data"]["scan"].append(
            {"scheme": sch, "theta0_range": [lo, hi],
             "min": float(v.min()), "max": float(v.max()),
             "argmin_theta0": float(vals[int(v.argmin())][0]),
             "max_over_min": float(v.max() / v.min())})
    LL_OUT[pname] = rec
OUT["LL"] = LL_OUT

with open(os.path.join(HERE, "rc2_decisive.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)

print("== M knob1 (read-out offset), h=0.3 ==")
for k, v in OUT["M_knob1_readout_offset"].items():
    print(f"{k:24s} dev0={v['dev0']:.4e} floor/A={v['floor_over_A_ref']:9.3e} "
          f"best_delta/h={v['argmin_delta_over_h']:+7.4f} collapse={v['collapse_factor']:11.4e} "
          f"env/med={v['envelope_over_median']:.4f}")
print("== M knob2 (initial data) ==")
for k, v in OUT["M_knob2_initial_data"]["named"].items():
    print(f"{k:34s} dev0={v['dev0']:.4e} floor/A={v['floor_over_A_ref']:9.3e} "
          f"readout_collapse={v['readout_collapse_factor']:.3e} env/med={v['envelope_over_median']:.4f}")
print("  scan:", OUT["M_knob2_initial_data"]["scan_summary"])
for pname, rec in OUT["LL"].items():
    print(f"== LL {pname} knob1 ==")
    for k, v in rec["knob1_readout_offset"].items():
        print(f"{k:12s} dev0={v['dev0']:.4e} floor/A={v['floor_over_A_ref']:9.3e} "
              f"best_delta/h={v['argmin_delta_over_h']:+10.3e} collapse={v['collapse_factor']:11.4e} "
              f"env/med={v['envelope_over_median']:.4f} 2|s|/h={v['two_s_over_h']:9.3e}")
    print(f"== LL {pname} knob2 ==")
    for k, v in rec["knob2_initial_data"]["named"].items():
        print(f"  {k[:52]:52s} floor/A={v['floor_over_A_ref']:.5f}")
    for r in rec["knob2_initial_data"]["scan"]:
        print("  scan", r)
print("wrote rc2_decisive.json")
