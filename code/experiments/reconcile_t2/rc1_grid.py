"""RC1: both contested metrics, both systems, one harness.

For every (system, parameter set, h, scheme) it reports
  dev0      -- the scheme's own error against the exact reference at t_n
  A_ref     -- median |Qref(t_n + h/2) - Qref(t_n)| : the READ-OUT artefact,
               a functional of the reference solution alone
  A_signed  -- dev(h/2) - dev(0), the quantity the second-system agent called
               "the artefact"
  R_true    -- signal/dev0     (the theory verifier's ratio: the "600x")
  R_art     -- signal/A_ref    (the second-system agent's ratio)
  and both R's divided by 2 t_med / h.

Output: rc1_grid.json
"""
import json, math, os
import numpy as np
import rc_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {"note": "all numbers computed here; nothing taken from either agent"}

# ---------------------------------------------------------------- system M --
TAU, TF = 1.2e5, 120.0
fB, ffac = rc.mk_field("exp", TAU)


def Qref_mag(t):
    return np.exp(-np.asarray(t, float) / TAU)


def dQref_mag(t):
    return -np.exp(-np.asarray(t, float) / TAU) / TAU


rows = []
for h in (0.6, 0.3, 0.15, 0.05):
    for sch in rc.MAG_SCHEMES:
        t, Q, rho = rc.run_mag(sch, fB, ffac, h, TF)
        m = rc.metrics(t, Q, Qref_mag, dQref_mag, h)
        m.update({"system": "magnetic", "params": f"exp tau={TAU:g} T={TF:g}",
                  "h": h, "scheme": sch,
                  "h_over_2tau": h / (2 * TAU)})
        rows.append(m)
OUT["magnetic"] = rows

# ---------------------------------------------------------------- system L --
r_e, c_l, lam_um = 2.8179403262e-15, 2.99792458e8, 1.0e-6
omega = 2 * math.pi * c_l / lam_um
EPS_PHYS = (2 * r_e * omega / (3 * c_l)) * 857.0
SETS = {"physical": dict(alpha=0.36, eps=EPS_PHYS, theta0=0.8, T=8.0),
        "generic": dict(alpha=1.0, eps=0.1, theta0=0.3, T=2.0)}

rows = []
for pname, S in SETS.items():
    sysL = rc.LL(S["alpha"], S["eps"], S["theta0"])
    for h in (0.1, 0.05, 0.025, 0.0125):
        for sch in rc.LL_SCHEMES:
            t, thn, Q = sysL.run(sch, h, S["T"])
            m = rc.metrics(t, Q, sysL.Qref, sysL.dQref_dt, h)
            m.update({"system": "LL", "params": pname, "h": h, "scheme": sch,
                      "alpha": S["alpha"], "eps": S["eps"], "theta0": S["theta0"],
                      "T": S["T"], "Lambda": sysL.Lam,
                      "theta_star": sysL.theta_star})
            rows.append(m)
OUT["LL"] = rows
OUT["LL_params"] = {k: dict(v, Lambda=2 * (v["alpha"] - v["eps"]),
                            theta_star=math.acosh(math.sqrt(v["alpha"] / v["eps"])))
                    for k, v in SETS.items()}

# ------------------------------------------------- scheme-spread summaries --
def spread(rows, key, filt):
    sel = [r for r in rows if filt(r)]
    v = np.array([r[key] for r in sel], float)
    return {"schemes": [r["scheme"] for r in sel], "values": [float(x) for x in v],
            "max_over_min": float(np.max(np.abs(v)) / np.min(np.abs(v))),
            "rel_spread": float((np.max(v) - np.min(v)) / np.mean(v))}


summ = {}
for key in ("A_ref", "A_signed", "dev0", "R_true_over_2th", "R_art_over_2th"):
    summ[f"magnetic_h0.3_{key}"] = spread(OUT["magnetic"], key,
                                          lambda r: r["h"] == 0.3)
    summ[f"LL_physical_h0.05_{key}"] = spread(OUT["LL"], key,
                                              lambda r: r["h"] == 0.05 and r["params"] == "physical")
OUT["scheme_spread"] = summ

with open(os.path.join(HERE, "rc1_grid.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)

hdr = f"{'sys':9s} {'params':9s} {'h':>7s} {'scheme':22s} {'dev0':>11s} {'A_ref':>11s} {'A_signed':>11s} {'R_true':>11s} {'R/2th':>9s} {'R_art':>11s} {'Rart/2th':>9s} {'floor/A':>10s}"
print(hdr)
for grp in ("magnetic", "LL"):
    for r in OUT[grp]:
        print(f"{grp:9s} {str(r['params'])[:9]:9s} {r['h']:7.4f} {r['scheme']:22s} "
              f"{r['dev0']:11.4e} {r['A_ref']:11.4e} {r['A_signed']:11.4e} "
              f"{r['R_true']:11.4e} {r['R_true_over_2th']:9.4f} "
              f"{r['R_art']:11.4e} {r['R_art_over_2th']:9.4f} {r['floor_in_units_of_artefact']:10.3e}")
print("wrote rc1_grid.json")
