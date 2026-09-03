"""verdict.py -- the operating-point slice of the work-precision study.

Reduces workprecision.json to the one step size at which the hybrid works,
Omega_c dt = 0.3, and expresses every scheme as a ratio to the hybrid.  That
slice is what Section 7 of the paper quotes and what Figure 3 asserts against:
the 417.9x flop advantage of vps4, the 64.8x trajectory and 62.2x energy
factors, the 2845 by which vps4 sits below the physical signal, and the 2.9 of
vps2.

Reads:   workprecision.json   (written by run.py in this directory)
Writes:  verdict.json

Provenance note.  Until wave W6.2 this directory shipped verdict.json with no
script that produced it: run.py writes only workprecision.json and timing.py
writes only timing.json.  A data file with no producing script is exactly what
Appendix A.7 of the paper says does not exist here, so the derivation was
written down.  It reproduces the shipped verdict.json exactly, every field of
it, from the shipped workprecision.json -- which is the evidence that this is
the derivation that was used and not a new one.  (The wall-clock fields are
carried through unchanged from workprecision.json and, like every timing in
this bundle, are a property of the machine.)

Usage: python verdict.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.3
ORDER = ["shipped", "staggered", "imr", "vps2", "gl4", "vps4", "hybrid"]

wp = json.load(open(os.path.join(HERE, "workprecision.json"), encoding="utf-8"))
signal = wp["meta"]["physical_signal_median"]

rows = {}
for r in wp["runs"]:
    if r["dt"] == DT:
        rows[r["scheme"]] = r
missing = [s for s in ORDER if s not in rows]
if missing:
    raise SystemExit("workprecision.json has no dt=%g row for: %s"
                     % (DT, ", ".join(missing)))

hyb = rows["hybrid"]
out = {"physical_signal": signal, "dt": DT, "schemes": {}}
for name in ORDER:
    r = rows[name]
    traj = r["pos_err_rms"]
    energy = r["energy_err_median_2nd_half"]
    out["schemes"][name] = {
        "traj": traj,
        "energy": energy,
        # how far below the physical signal this scheme's energy error sits
        "below_signal": signal / energy,
        "flops": r["flops"],
        "wall_s": r["wall_s"],
        # > 1 means the scheme beats the hybrid on that channel
        "traj_vs_hybrid": hyb["pos_err_rms"] / traj,
        "energy_vs_hybrid": hyb["energy_err_median_2nd_half"] / energy,
        "flops_cheaper_than_hybrid": hyb["flops"] / r["flops"],
    }
    print("%-10s traj=%.4e energy=%.4e  vs hybrid: traj %8.3f  energy %8.3f  "
          "flops %8.1f" % (name, traj, energy,
                           out["schemes"][name]["traj_vs_hybrid"],
                           out["schemes"][name]["energy_vs_hybrid"],
                           out["schemes"][name]["flops_cheaper_than_hybrid"]))

json.dump(out, open(os.path.join(HERE, "verdict.json"), "w", encoding="utf-8"),
          indent=2)
print("\nwrote verdict.json")
