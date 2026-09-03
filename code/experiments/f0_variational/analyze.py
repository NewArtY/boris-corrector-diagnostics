"""Final Ф0.2 comparison table.

The power-law exponent alone is misleading when a curve saturates: a plateau at
a catastrophic level also yields exponent 0. Every row therefore carries the
absolute envelope value at each horizon, an explicit saturation flag, and the
inside/outside ratio at matched injected error.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [("large", "f0_summary.json"), ("small", "f0_summary_small.json")]
HZ = ["1e+03", "1e+04", "1e+05"]
rows = []

for tag, fn in FILES:
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    ng = d["config"]["n_gyrations"]
    for cname, modes in d["results"].items():
        for mode, r in modes.items():
            by_amp = {}
            for m in r["members"]:
                a = m.get("amp_factor", "base")
                by_amp.setdefault(a, []).append(m)
            for a, ms in sorted(by_amp.items(), key=lambda kv: str(kv[0])):
                ex = np.array([m["envelope_powerlaw_exponent"] for m in ms])
                row = {"file": tag, "n_gyr": ng, "config": cname, "mode": mode,
                       "amp_factor": a, "n_members": len(ms),
                       "exponent_med": float(np.nanmedian(ex)),
                       "exponent_min": float(np.nanmin(ex)),
                       "exponent_max": float(np.nanmax(ex))}
                for h in HZ:
                    v = [m["horizons"][h]["energy_err_max"]
                         for m in ms if h in m["horizons"]]
                    if v:
                        row[f"Emax_{h}"] = float(np.median(v))
                # saturation: envelope within a factor 3 of total energy, or
                # growth over the last decade below 1.3x
                e5 = row.get("Emax_1e+05"); e4 = row.get("Emax_1e+04")
                row["saturated"] = bool(
                    (e5 is not None and e5 > 0.3) or
                    (e5 is not None and e4 is not None and e5 / max(e4, 1e-300) < 1.3
                     and e5 > 1e-3))
                rows.append(row)

print(f"{'set':6s} {'config':14s} {'mode':9s} {'amp':>8s} {'expo(med)':>10s} "
      f"{'E@1e3':>10s} {'E@1e4':>10s} {'E@1e5':>10s}  sat")
for r in rows:
    print(f"{r['file']:6s} {r['config'][:14]:14s} {r['mode']:9s} "
          f"{str(r['amp_factor']):>8s} {r['exponent_med']:10.3f} "
          f"{r.get('Emax_1e+03', float('nan')):10.3e} "
          f"{r.get('Emax_1e+04', float('nan')):10.3e} "
          f"{r.get('Emax_1e+05', float('nan')):10.3e}  {r['saturated']}")

# inside/outside ratio at matched amplitude
print("\ninside(varnet) / outside(additive) at matched injected error:")
cmp_rows = []
for tag, _ in FILES:
    for cname in set(r["config"] for r in rows if r["file"] == tag):
        for a in set(r["amp_factor"] for r in rows
                     if r["file"] == tag and r["config"] == cname
                     and r["mode"] != "base"):
            v = [r for r in rows if r["file"] == tag and r["config"] == cname
                 and r["amp_factor"] == a and r["mode"] == "varnet"]
            w = [r for r in rows if r["file"] == tag and r["config"] == cname
                 and r["amp_factor"] == a and r["mode"] == "additive"]
            if not v or not w:
                continue
            for h in HZ:
                k = f"Emax_{h}"
                if k in v[0] and k in w[0]:
                    ratio = v[0][k] / max(w[0][k], 1e-300)
                    cmp_rows.append({"file": tag, "config": cname,
                                     "amp_factor": a, "horizon": h,
                                     "varnet": v[0][k], "additive": w[0][k],
                                     "ratio_in_over_out": ratio})
                    print(f"  {tag:6s} {cname[:14]:14s} amp={str(a):>8s} {h}: "
                          f"var={v[0][k]:.3e} add={w[0][k]:.3e} "
                          f"ratio={ratio:.3f}")

json.dump({"rows": rows, "inside_vs_outside": cmp_rows},
          open(os.path.join(HERE, "analysis.json"), "w"), indent=2)
print("\nsaved analysis.json")
