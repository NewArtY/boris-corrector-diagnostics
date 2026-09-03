"""RC5: which of the two metrics survives which axis of generalisation.

Same magnetic run, five decay laws, four schemes, metric A (R_art = signal/A_ref)
and metric B (R_true = signal/dev0), both divided by 2 t_med / h.

Expectation under the reconciliation:
  * R_art / (2t/h) is IDENTICAL across schemes (A_ref has no scheme in it) but
    still depends on the field law -- it is 1 for laws with f'(0) != 0 and 1/2
    for laws with f'(0) = 0, exactly the verifier's closed form
        R = (1 - f(t_med)) / ((h/2)|f'(t_med)|).
  * R_true / (2t/h) depends on BOTH, and equals 1 only for the shipped Boris map.

Output: rc5_fieldlaws.json
"""
import json, os
import numpy as np
import rc_common as rc

HERE = os.path.dirname(os.path.abspath(__file__))
TF, H = 120.0, 0.3
LAWS = [("exp", 1.2e5, 1.0), ("pow", 1.2e5, 1.0), ("pow", 1.2e5, 3.0),
        ("lin", 1.2e5, 1.0), ("cos", 1.2e5, 1.0), ("gauss", 3000.0, 1.0)]

rows = []
for law, tau, beta in LAWS:
    fB, ffac = rc.mk_field(law, tau, beta)

    def Qref(t, fB=fB):
        return np.asarray(fB(t), float)

    for sch in rc.MAG_SCHEMES:
        t, Q, rho = rc.run_mag(sch, fB, ffac, H, TF)
        m = rc.metrics(t, Q, Qref, None, H)
        rows.append({"law": law, "beta": beta, "tau": tau, "scheme": sch,
                     "dev0": m["dev0"], "A_ref": m["A_ref"],
                     "R_true": m["R_true"], "R_true_over_2th": m["R_true_over_2th"],
                     "R_art": m["R_art"], "R_art_over_2th": m["R_art_over_2th"],
                     "floor_over_A_ref": m["floor_in_units_of_artefact"]})

summ = {}
for law, tau, beta in LAWS:
    key = f"{law}(beta={beta},tau={tau:g})"
    sel = [r for r in rows if r["law"] == law and r["beta"] == beta and r["tau"] == tau]
    a = np.array([r["R_art_over_2th"] for r in sel])
    b = np.array([r["R_true_over_2th"] for r in sel])
    summ[key] = {"R_art_over_2th_values": [float(x) for x in a],
                 "R_art_scheme_spread_max_over_min": float(a.max() / a.min()),
                 "R_true_over_2th_values": [float(x) for x in b],
                 "R_true_scheme_spread_max_over_min": float(b.max() / b.min()),
                 "schemes": [r["scheme"] for r in sel]}

OUT = {"rows": rows, "by_law": summ, "h": H, "T": TF}
with open(os.path.join(HERE, "rc5_fieldlaws.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)

print(f"{'law':16s} {'scheme':22s} {'R_true/(2t/h)':>14s} {'R_art/(2t/h)':>14s} {'floor/A':>11s}")
for r in rows:
    print(f"{r['law']+str(r['beta']):16s} {r['scheme']:22s} "
          f"{r['R_true_over_2th']:14.6f} {r['R_art_over_2th']:14.6f} {r['floor_over_A_ref']:11.4e}")
print()
for k, v in summ.items():
    print(f"{k:28s} R_art spread {v['R_art_scheme_spread_max_over_min']:.6f}  "
          f"R_art value {v['R_art_over_2th_values'][0]:.4f}  "
          f"R_true spread {v['R_true_scheme_spread_max_over_min']:.4g}")
print("wrote rc5_fieldlaws.json")
