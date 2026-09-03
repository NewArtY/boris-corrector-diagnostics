"""vf2: P1.2 -- does the staggered (averaging) readout floor hide the secular
growth of the trained hybrid?  Recomputed from the shipped symproj data with my
own envelope fit (same definition as the campaign: slope of log10(env) vs
log10(t) for t > t_end/100, env = running max)."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.abspath(os.path.join(HERE, "..", "symproj"))
out = {}

def myfit(t, env):
    e = np.maximum.accumulate(env)
    sel = (t > t[-1] / 100.0) & (e > 0)
    p = np.polyfit(np.log10(t[sel]), np.log10(e[sel]), 1)
    return float(p[0])

zq_sh = np.load(os.path.join(SP, "env_quasistatic_shipped.npz"))
zq_st = np.load(os.path.join(SP, "env_quasistatic_staggered.npz"))

H, OM = 0.3, 1.0
floor_theory = np.sin(np.arctan(H * OM / 2)) ** 2
out["floor_theory_sin2"] = float(floor_theory)

for tag, z, key in [("shipped/proj", zq_sh, "shipped/proj"),
                    ("staggered/proj", zq_st, "staggered/proj"),
                    ("staggered/boris", zq_st, "staggered/boris"),
                    ("shipped/boris", zq_sh, "shipped/boris"),
                    ("staggered/sym", zq_st, "staggered/sym")]:
    t = z[f"{key}/t"]; env = z[f"{key}/env"]; ee = z[f"{key}/e_err"]
    e = np.maximum.accumulate(env)
    TWO_PI = 2 * np.pi
    gyr = t / TWO_PI
    def at(g):
        return float(e[np.searchsorted(gyr, g, side="right") - 1])
    out[tag] = {
        "my_exponent": myfit(t, env),
        "env_1e3": at(1e3), "env_1e4": at(1e4), "env_1e5": at(1e5),
        "e_err_first": float(ee[0]), "e_err_last": float(ee[-1]),
        "e_err_min": float(np.min(ee)), "e_err_max": float(np.max(ee)),
    }

# how much of the floor does the hidden growth reach, and when would it emerge
g_ship = out["shipped/proj"]; g_stag = out["staggered/proj"]
p = g_ship["my_exponent"]
E5 = g_ship["env_1e5"]
fl = out["staggered/boris"]["env_1e5"]
out["hidden_growth"] = {
    "shipped_growth_at_1e5": E5,
    "staggered_floor": fl,
    "growth_over_floor": E5 / fl,
    "emergence_horizon_gyr_extrapolated": float(1e5 * (fl / E5) ** (1.0 / p)),
    "staggered_env_total_relative_rise": g_stag["env_1e5"] / g_stag["env_1e3"] - 1.0,
}
# is staggered/proj e_err literally floor +/- hidden growth?
t = zq_st["staggered/proj/t"]; ee = zq_st["staggered/proj/e_err"]
eb = zq_st["staggered/boris/e_err"]
out["stag_proj_minus_stag_boris"] = {
    "max_abs": float(np.max(np.abs(ee - eb))),
    "at_end": float(ee[-1] - eb[-1])}

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "vf2_symproj_floor.json"), "w"), indent=1)
