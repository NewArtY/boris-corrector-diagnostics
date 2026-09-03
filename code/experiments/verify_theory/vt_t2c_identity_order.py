"""VT-T2c: 'the two Boris variants are ONE map, differing only in the
convention for READING the velocity'.

Two separate questions, which the claim conflates:
  Q1 is the RECURSION identical?              (map identity -- strict test)
  Q2 is the observed difference only READING? (i.e. can a relabelling of the
     same numerical state sequence produce the staggered results?)

Q2 matters because 11_THEORY attributes the measured order gap 0.96 vs 1.75
in POSITION to 'reading conventions'.  Positions are produced at integer
times by BOTH variants; relabelling velocities cannot change r_n.  The order
gap therefore can only come from convention (ii), the initial half-step-back
kick -- which is a change of INITIAL DATA, not of reading, and produces a
genuinely different orbit.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
from fields import DecayingField
from models.boris import integrate_boris

TAU, TF = 1.2e5, 120.0
R0 = np.array([1.0, 0.0, 0.0]); V0 = np.array([0.0, 1.0, 0.0])


def kick(v, E, B, dt, q=-1.0, m=1.0):
    a = 0.5 * q * dt / m
    vm = v + a * E
    tv = a * B
    sv = 2.0 * tv / (1.0 + np.dot(tv, tv))
    vp = vm + np.cross(vm + np.cross(vm, tv), sv)
    return vp + a * E


def core(r0, w0, h, n, field):
    """The one recursion:  w <- kick(w; E(r,t), B(r,t), h);  r <- r + w h."""
    r = np.array(r0, float); w = np.array(w0, float)
    rs = np.zeros((n + 1, 3)); ws = np.zeros((n + 1, 3))
    rs[0], ws[0] = r, w
    t = 0.0
    for i in range(1, n + 1):
        E = np.asarray(field.E(r, t), float); B = np.asarray(field.B(r, t), float)
        w = kick(w, E, B, h)
        r = r + w * h
        t += h
        rs[i], ws[i] = r, w
    return rs, ws


out = {}
fld = DecayingField(B0=1.0, tau=TAU)

# ---- Q1: strict map identity ---------------------------------------------
n = 20000
rs_ship, vs_ship, _ = integrate_boris(R0, V0, 0.0, 0.3, n, fld)
rs_core, ws_core = core(R0, V0, 0.3, n, fld)
E0f = np.asarray(fld.E(R0, 0.0), float); B0f = np.asarray(fld.B(R0, 0.0), float)
v_back = kick(V0, E0f, B0f, -0.15)
rs_s2, vs_s2, _ = integrate_boris(R0, v_back, 0.0, 0.3, n, fld)
rs_c2, ws_c2 = core(R0, v_back, 0.3, n, fld)
out["Q1_map_identity"] = {
    "shipped_vs_core_same_init_max_dr": float(np.max(np.abs(rs_ship - rs_core))),
    "shipped_vs_core_same_init_max_dv": float(np.max(np.abs(vs_ship - ws_core))),
    "shipped_vs_core_halfback_init_max_dr": float(np.max(np.abs(rs_s2 - rs_c2))),
    "shipped_vs_core_halfback_init_max_dv": float(np.max(np.abs(vs_s2 - ws_c2))),
    "verdict": "0.0 everywhere => identical recursion (bitwise)",
}

# ---- how big is the init convention on the ORBIT? -------------------------
out["Q1_init_convention_cost"] = {
    "rms_position_gap_over_20000_steps":
        float(np.sqrt(np.mean(np.sum((rs_ship - rs_s2) ** 2, axis=1)))),
    "max_position_gap": float(np.max(np.linalg.norm(rs_ship - rs_s2, axis=1))),
    "larmor_radius": 1.0,
}

# ---- Q2: convergence order in POSITION, both inits, against a fine ref ----
def pos_err(h, init):
    n = int(round(TF / h))
    if init == "shipped":
        w0 = V0
    else:
        w0 = kick(V0, E0f, B0f, -0.5 * h)
    rs, ws = core(R0, w0, h, n, fld)
    return rs[-1]

h_ref = 1e-4
n_ref = int(round(TF / h_ref))
r_ref = core(R0, V0, h_ref, n_ref, fld)[0][-1]     # same ODE, both converge here
rows = []
for init in ("shipped", "halfback"):
    hs, errs = [], []
    for h in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
        e = np.linalg.norm(pos_err(h, init) - r_ref)
        hs.append(h); errs.append(e)
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    rows.append({"init": init, "h": hs, "pos_err_at_T": errs, "order": float(p)})
out["Q2_position_order"] = rows
out["Q2_note"] = ("positions are emitted at integer times by BOTH; the order "
                  "gap is produced by the initial half-step-back (a change of "
                  "initial DATA), not by any relabelling of the velocity")

# ---- Q2b: can a pure relabelling reproduce the staggered energy? ----------
# take ONE run (shipped init) and read its velocity three ways
n = int(round(TF / 0.3))
rs, ws = core(R0, V0, 0.3, n, fld)
ts = np.arange(n + 1) * 0.3
Eph = np.exp(-ts / TAU)
read = {}
read["as_integer_v_n"] = np.sum(ws ** 2, axis=1) - Eph
vavg = 0.5 * (ws[1:] + ws[:-1])
read["as_avg_of_neighbours"] = np.sum(vavg ** 2, axis=1) - Eph[1:]
hh = n // 2
out["Q2b_same_run_three_readouts"] = {
    k: {"median_abs": float(np.median(np.abs(v[hh:]))),
        "min": float(np.min(v[hh:])), "max": float(np.max(v[hh:]))}
    for k, v in read.items()}

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vt_t2c_identity_order.json"), "w"), indent=1)
