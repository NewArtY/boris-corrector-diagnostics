"""mp3_maps.py -- the three maps, built from the five grid shards.

Reads mp2_grid__<field>.json for the five configurations and mp1_calibration.json
for the per-configuration reference floor, and writes mp3_maps.json.  Nothing
is integrated here; this file only reduces what mp2 measured.

    A   corrector beats Boris on the trajectory, at the same step
    B   corrector beats vps4 at equal total flops
    C   the energy channel does not show the trajectory error already made

Every cell carries the count over the eight initial conditions, the median and
the extremes, and a flag saying whether the residual has fallen to within a
factor of ten of what the reference in that configuration is worth.  A cell
that is reference-limited is reported as such and is not read as a property of
the scheme -- the carry-over W13 insisted on.

Usage: python mp3_maps.py [--force]
"""
import json
import os
import sys

import numpy as np

import map_common as C
from ea_common import check_or_write

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mp3_maps.json")

#: a residual within this factor of what the reference is worth is not a
#: measurement of the scheme.  Declared here; W13 used a one-per-cent shift of
#: a band power for the same purpose.
REF_LIMIT_FACTOR = 10.0


def load():
    shards = {}
    for f in C.FIELD_NAMES:
        p = os.path.join(HERE, "mp2_grid__%s.json" % f)
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run mp2_grid.py --field %s" % (p, f))
        shards[f] = json.load(open(p, encoding="utf-8"))
    cal = json.load(open(os.path.join(HERE, "mp1_calibration.json"),
                         encoding="utf-8"))
    budget = {}
    for f in C.FIELD_NAMES:
        p = os.path.join(HERE, "mp5_equalcost__%s.json" % f)
        if os.path.exists(p):
            budget[f] = json.load(open(p, encoding="utf-8"))
    return shards, cal, budget


_SPECIAL = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}


def arr(x):
    """The JSON writer turns non-finite values into their names; turn them
    back, so that a run that diverged is an infinity rather than a hole."""
    return np.array([_SPECIAL[v] if isinstance(v, str) else float(v)
                     for v in x])


def stat(v):
    v = np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return {"median": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_finite": 0, "ic0": float(v[0])}
    return {"median": float(np.median(v[ok])), "min": float(v[ok].min()),
            "max": float(v[ok].max()), "n_finite": int(ok.sum()),
            "ic0": float(v[0])}


def main():
    force = "--force" in sys.argv
    shards, cal, budget = load()
    floors = {f: {h: cal["reference_per_configuration"][f]["floor_estimate_" + h]
                  for h in C.HORIZONS} for f in C.FIELD_NAMES}

    out = {"meta": {
        "what": "the three maps of wave W14, reduced from the grid",
        "claims": {
            "A": "the learned corrector is more accurate than the Boris "
                 "scheme on the trajectory, at the same step",
            "B": "the learned corrector is more accurate than vps4 at equal "
                 "total flops",
            "C": "the energy diagnostic does not show the trajectory error "
                 "that has already accumulated"},
        "primary_metric": "root mean square of the position error in Larmor "
                          "radii against the best reference of that "
                          "configuration",
        "reference_floor_per_configuration": floors,
        "ref_limit_factor": REF_LIMIT_FACTOR,
        "C_pos_threshold_larmor": C.C_POS_THRESHOLD,
        "C_energy_threshold": C.C_ENERGY_THRESHOLD,
        "n_initial_conditions": C.N_IC,
        "one_corrector_checkpoint": True,
        "corrector_trained_at": {"field": "B4_decaying", "dt": C.DT_TRAIN},
    }}

    mapA, mapB, mapC, cells = {}, {}, {}, {}
    afortiori_check = {}

    for f in C.FIELD_NAMES:
        g = shards[f]["grid"]
        eq = shards[f]["equal_cost_vps4"]
        for h in C.HORIZONS:
            floor = floors[f][h]
            for dt in C.DT_GRID:
                key = "%s|%s|%g" % (h, f, dt)
                pos = {s: arr(g["%s|%g|%s" % (h, dt, s)]["position"]["rms"])
                       for s in C.SCHEMES}
                en = {s: arr(g["%s|%g|%s" % (h, dt, s)]["energy"]
                             ["median_2nd_half"]) for s in C.SCHEMES}
                en_rms = {s: arr(g["%s|%g|%s" % (h, dt, s)]["energy"]["rms"])
                          for s in C.SCHEMES}
                cells[key] = {
                    "position_rms": {s: stat(pos[s]) for s in C.SCHEMES},
                    "energy_median_2nd_half": {s: stat(en[s])
                                               for s in C.SCHEMES},
                    "reference_limited": {
                        s: bool(np.nanmedian(pos[s])
                                < REF_LIMIT_FACTOR * floor)
                        for s in C.SCHEMES},
                    "total_flops": {s: g["%s|%g|%s" % (h, dt, s)]["total_flops"]
                                    for s in C.SCHEMES},
                }

                # ----------------------------------------------------- map A
                gain = pos["boris"] / pos["corrector"]
                holds = pos["corrector"] < pos["boris"]
                mapA[key] = {
                    "gain_boris_over_corrector": stat(gain),
                    "holds_n_of_%d" % C.N_IC: int(np.nansum(holds)),
                    "holds_at_ic0": bool(holds[0]),
                    "boris_rms": stat(pos["boris"]),
                    "corrector_rms": stat(pos["corrector"]),
                    "reference_limited_either": bool(
                        cells[key]["reference_limited"]["boris"]
                        or cells[key]["reference_limited"]["corrector"]),
                    "corrector_at_its_training_step": bool(dt == C.DT_TRAIN),
                    "corrector_in_its_training_field": C.TRAINED_ON[f],
                }

                # ----------------------------------------------------- map B
                ek = "%s|%g" % (h, dt)
                if ek in eq:
                    v4 = arr(eq[ek]["position"]["rms"])
                    mode = "measured"
                    m = eq[ek]["substeps_per_corrector_step"]
                    afortiori_check[key] = {
                        "vps4_equal_cost_rms": stat(v4),
                        "vps4_same_step_rms": stat(pos["vps4"]),
                        "equal_cost_not_worse_n_of_%d" % C.N_IC:
                            int(np.nansum(v4 <= pos["vps4"])),
                    }
                else:
                    v4 = pos["vps4"]
                    mode = "inferred_from_same_step"
                    m = C.equal_cost_substeps("vps4")
                holdsB = pos["corrector"] < v4
                mapB[key] = {
                    "mode": mode,
                    "substeps": m,
                    "corrector_rms": stat(pos["corrector"]),
                    "vps4_reference_rms": stat(v4),
                    "vps4_same_step_rms": stat(pos["vps4"]),
                    "ratio_corrector_over_vps4": stat(pos["corrector"] / v4),
                    "holds_n_of_%d" % C.N_IC: int(np.nansum(holdsB)),
                    "holds_at_ic0": bool(holdsB[0]),
                    "vps4_reference_limited": bool(
                        np.nanmedian(v4) < REF_LIMIT_FACTOR * floor),
                    "flops_ratio_corrector_over_vps4_same_step": (
                        C.FLOPS["corrector"] / C.FLOPS["vps4"]),
                }

                # ----------------------------------------------------- map C
                rec = {}
                for s in C.SCHEMES:
                    blind = pos[s] / np.where(en_rms[s] > 0, en_rms[s], np.nan)
                    binary = (pos[s] >= C.C_POS_THRESHOLD) & \
                             (en[s] <= C.C_ENERGY_THRESHOLD)
                    rec[s] = {
                        "position_rms": stat(pos[s]),
                        "energy_rms": stat(en_rms[s]),
                        "energy_median_2nd_half": stat(en[s]),
                        "blindness_pos_over_energy": stat(blind),
                        "energy_blind_n_of_%d" % C.N_IC: int(np.nansum(binary)),
                        "energy_blind_at_ic0": bool(binary[0]),
                    }
                # does the energy channel rank the schemes as the trajectory
                # does?  A diagnostic that saw the trajectory error would.
                rhos = []
                for i in range(C.N_IC):
                    rhos.append(C.spearman([en[s][i] for s in C.SCHEMES],
                                           [pos[s][i] for s in C.SCHEMES]))
                rec["_rank_agreement_energy_vs_position"] = stat(
                    np.array(rhos))
                rec["_n_schemes_energy_blind_at_ic0"] = int(sum(
                    rec[s]["energy_blind_at_ic0"] for s in C.SCHEMES))
                mapC[key] = rec

    # ------------------------------------- the corrector's budget, spent well
    # Claim B is about vps4.  This is the same budget spent on the two cheaper
    # classical schemes, from mp5_equalcost.py, which is the question the first
    # author asks first: one corrector step is 1009 Boris steps.
    budget_tbl = {}
    for f, b in budget.items():
        eqc = shards[f]["equal_cost_vps4"]
        for k, rec in b["equal_cost"].items():
            h, dts, s = k.split("|")
            key = "%s|%s|%s" % (h, f, dts)
            row = budget_tbl.setdefault(key, {})
            corr = arr(shards[f]["grid"]["%s|%s|corrector" % (h, dts)]
                       ["position"]["rms"])
            v = arr(rec["position"]["rms"])
            row["corrector_rms"] = stat(corr)
            row[s + "_equal_cost_rms"] = stat(v)
            row[s + "_substeps"] = rec["substeps_per_corrector_step"]
            row["ratio_corrector_over_" + s] = stat(corr / v)
            row[s + "_energy_median"] = stat(arr(rec["energy"]
                                                 ["median_2nd_half"]))
            if "%s|%s" % (h, dts) in eqc:
                v4 = arr(eqc["%s|%s" % (h, dts)]["position"]["rms"])
                row["vps4_equal_cost_rms"] = stat(v4)
                row["ratio_corrector_over_vps4"] = stat(corr / v4)
    out["equal_cost_budget"] = budget_tbl

    out["cells"] = cells
    out["map_A"] = mapA
    out["map_B"] = mapB
    out["map_C"] = mapC
    out["equal_cost_monotonicity_check"] = afortiori_check

    # ------------------------------------------------------------- summaries
    nA = sum(1 for k, v in mapA.items() if v["holds_n_of_8"] >= 5)
    nB = sum(1 for k, v in mapB.items() if v["holds_n_of_8"] >= 5)
    a_cells = {k: v for k, v in mapA.items() if v["holds_n_of_8"] >= 5}
    out["summary"] = {
        "n_cells": len(mapA),
        "A_holds_in_cells": nA,
        "A_holds_where": sorted(a_cells),
        "A_holds_fraction": nA / len(mapA),
        "B_holds_in_cells": nB,
        "B_holds_where": sorted(k for k, v in mapB.items()
                                if v["holds_n_of_8"] >= 5),
        "C_energy_blind_cells_any_scheme": sum(
            1 for k, v in mapC.items()
            if any(v[s]["energy_blind_n_of_8"] >= 5 for s in C.SCHEMES)),
        "C_energy_blind_scheme_cells": sum(
            sum(1 for s in C.SCHEMES if v[s]["energy_blind_n_of_8"] >= 5)
            for v in mapC.values()),
        "C_scheme_cells_total": len(mapC) * len(C.SCHEMES),
        "rank_agreement_median_over_all_cells": float(np.nanmedian(
            [v["_rank_agreement_energy_vs_position"]["median"]
             for v in mapC.values()])),
        "rank_agreement_negative_cells": int(sum(
            1 for v in mapC.values()
            if v["_rank_agreement_energy_vs_position"]["median"] < 0)),
    }

    # ------------------------------------------------- pre-registration status
    st = {}
    # P1 -- A holds only near the training step
    a_at_train = [k for k in a_cells if k.endswith("|0.3")]
    a_off_train = [k for k in a_cells if not k.endswith("|0.3")]
    st["P1"] = {
        "prediction": "A holds only in the neighbourhood of the training step",
        "A_cells_at_dt_0.3": sorted(a_at_train),
        "A_cells_away_from_dt_0.3": sorted(a_off_train),
        "status": ("confirmed" if a_cells and not a_off_train
                   else ("failed" if a_off_train else "failed")),
    }
    st["P2"] = {
        "prediction": "B holds nowhere on the grid",
        "B_cells": out["summary"]["B_holds_where"],
        "status": "confirmed" if nB == 0 else "failed",
    }
    st["P3"] = {
        "prediction": "C holds widely and is the real subject of the map",
        "cells_with_an_energy_blind_scheme":
            out["summary"]["C_energy_blind_cells_any_scheme"],
        "cells_total": len(mapC),
        "status": "confirmed" if out["summary"][
            "C_energy_blind_cells_any_scheme"] > 0 else "failed",
    }
    st["P4"] = {
        "prediction": "negative control: if A holds widely or B holds "
                      "anywhere, the predictions are wrong and that is the "
                      "leading result",
        "A_holds_fraction": out["summary"]["A_holds_fraction"],
        "B_holds_in_cells": nB,
        "status": ("failed" if (nB > 0 or nA / len(mapA) > 0.5)
                   else "confirmed"),
    }
    out["prereg_status"] = st

    # ------------------------------------------------------ headline numbers
    hl = {}
    ga = {k: v["gain_boris_over_corrector"] for k, v in mapA.items()}
    worst_a = min(ga, key=lambda k: ga[k]["median"])
    hl["A_worst_cell"] = {"cell": worst_a, "gain": ga[worst_a]}
    hl["A_holding_cells"] = {k: {"gain_median": mapA[k]
                                 ["gain_boris_over_corrector"]["median"],
                                 "gain_ic0": mapA[k]
                                 ["gain_boris_over_corrector"]["ic0"],
                                 "gain_min": mapA[k]
                                 ["gain_boris_over_corrector"]["min"],
                                 "gain_max": mapA[k]
                                 ["gain_boris_over_corrector"]["max"],
                                 "holds_n_of_8": mapA[k]["holds_n_of_8"]}
                            for k in sorted(a_cells)}
    rb = {k: v["ratio_corrector_over_vps4"] for k, v in mapB.items()}
    best_b = min(rb, key=lambda k: rb[k]["median"])
    hl["B_closest_cell"] = {"cell": best_b, "ratio": rb[best_b],
                            "mode": mapB[best_b]["mode"],
                            "orders_of_magnitude":
                                float(np.log10(rb[best_b]["median"]))}
    hl["B_orders_range"] = [
        float(np.log10(min(v["median"] for v in rb.values()))),
        float(np.log10(max(v["median"] for v in rb.values())))]
    # the single most extreme demonstration of C
    best_c, best_v = None, -1.0
    for k, v in mapC.items():
        for s in C.SCHEMES:
            p = v[s]["position_rms"]["median"]
            e = v[s]["energy_median_2nd_half"]["median"]
            if e <= C.C_ENERGY_THRESHOLD and np.isfinite(p) and p > best_v:
                best_v, best_c = p, (k, s)
    hl["C_most_extreme"] = {
        "cell": best_c[0], "scheme": best_c[1],
        "position_rms_larmor": mapC[best_c[0]][best_c[1]]["position_rms"],
        "energy_median_2nd_half": mapC[best_c[0]][best_c[1]]
        ["energy_median_2nd_half"],
        "energy_rms": mapC[best_c[0]][best_c[1]]["energy_rms"],
        "blindness": mapC[best_c[0]][best_c[1]]["blindness_pos_over_energy"],
    }
    out["headline"] = hl

    rc = check_or_write(OUT, json.loads(json.dumps(C.clean(out))), force=force)
    report(out)
    return rc


# ---------------------------------------------------------------- the tables
def report(out):
    mapA, mapB, mapC = out["map_A"], out["map_B"], out["map_C"]
    cells = out["cells"]
    print("\n" + "=" * 78)
    print("MAP A -- corrector vs Boris on the trajectory, same step")
    print("gain = boris rms / corrector rms, median over 8 initial "
          "conditions; > 1 means the corrector wins")
    for h in C.HORIZONS:
        print("\n  %s (%.2f gyro-orbits)" % (h, C.HORIZONS[h] / C.TWO_PI))
        print("    %-8s" % "Omega h" + "".join("%14s" % f
                                               for f in C.FIELD_NAMES))
        for dt in C.DT_GRID:
            row = "    %-8g" % dt
            for f in C.FIELD_NAMES:
                v = mapA["%s|%s|%g" % (h, f, dt)]
                mark = "*" if v["holds_n_of_8"] >= 5 else " "
                row += "%13.3e%s" % (v["gain_boris_over_corrector"]["median"],
                                     mark)
            print(row)

    print("\n" + "=" * 78)
    print("MAP B -- corrector vs vps4 at equal total flops")
    print("ratio = corrector rms / vps4 rms at the same total flop count; "
          "< 1 means the corrector wins.  m = measured, i = inferred from the "
          "same-step run")
    for h in C.HORIZONS:
        print("\n  %s" % h)
        print("    %-8s" % "Omega h" + "".join("%14s" % f
                                               for f in C.FIELD_NAMES))
        for dt in C.DT_GRID:
            row = "    %-8g" % dt
            for f in C.FIELD_NAMES:
                v = mapB["%s|%s|%g" % (h, f, dt)]
                tag = "m" if v["mode"] == "measured" else "i"
                mark = "*" if v["holds_n_of_8"] >= 5 else tag
                row += "%13.3e%s" % (v["ratio_corrector_over_vps4"]["median"],
                                     mark)
            print(row)

    print("\n" + "=" * 78)
    print("MAP C -- the energy channel against the trajectory error")
    print("per cell: how many of the five schemes are energy-blind "
          "(position rms >= %g r_L while the energy error is <= %g), and the "
          "rank agreement between the two channels across the five schemes"
          % (C.C_POS_THRESHOLD, C.C_ENERGY_THRESHOLD))
    for h in C.HORIZONS:
        print("\n  %s" % h)
        print("    %-8s" % "Omega h"
              + "".join("%16s" % f for f in C.FIELD_NAMES))
        for dt in C.DT_GRID:
            row = "    %-8g" % dt
            for f in C.FIELD_NAMES:
                v = mapC["%s|%s|%g" % (h, f, dt)]
                nblind = sum(1 for s in C.SCHEMES
                             if v[s]["energy_blind_n_of_8"] >= 5)
                rho = v["_rank_agreement_energy_vs_position"]["median"]
                row += "%11d/5 %s" % (nblind,
                                      ("%+.2f" % rho) if rho == rho else " nan")
            print(row)

    print("\n" + "=" * 78)
    print("BLINDNESS RATIO -- position rms in r_L divided by energy rms, "
          "median over the eight initial conditions, at H_crossover")
    print("    %-10s" % "scheme" + "".join("%14s" % f for f in C.FIELD_NAMES))
    for dt in (0.3, 0.5):
        print("  Omega h = %g" % dt)
        for s in C.SCHEMES:
            row = "    %-10s" % s
            for f in C.FIELD_NAMES:
                v = mapC["H_crossover|%s|%g" % (f, dt)][s]
                row += "%14.3e" % v["blindness_pos_over_energy"]["median"]
            print(row)

    print("\n" + "=" * 78)
    print("THE STEP AXIS -- position rms in Larmor radii, median over the "
          "eight initial conditions, H_paper.  R marks a cell within a factor "
          "of ten of what the reference in that configuration is worth")
    print("    %-8s%-14s" % ("Omega h", "") + "".join("%14s" % f
                                                      for f in C.FIELD_NAMES))
    for s in ("boris", "corrector", "vps4"):
        print("  %s" % s)
        for dt in C.DT_GRID:
            row = "    %-8g%-14s" % (dt, "")
            for f in C.FIELD_NAMES:
                v = cells["H_paper|%s|%g" % (f, dt)]
                mark = "R" if v["reference_limited"][s] else " "
                row += "%13.3e%s" % (v["position_rms"][s]["median"], mark)
            print(row)

    if out.get("equal_cost_budget"):
        print("\n" + "=" * 78)
        print("THE CORRECTOR'S OWN FLOP BUDGET, SPENT ON A CLASSICAL SCHEME")
        print("position rms in Larmor radii, median over eight initial "
              "conditions, at equal total flops")
        print("  %-14s %-9s %-6s %12s %12s %12s %12s"
              % ("horizon", "field", "Om h", "corrector",
                 "boris x1009", "vps2 x1253", "vps4 x417"))
        for k in sorted(out["equal_cost_budget"]):
            h, f, dt = k.split("|")
            r = out["equal_cost_budget"][k]
            print("  %-14s %-9s %-6s %12.4e %12.4e %12.4e %12.4e"
                  % (h, f, dt, r["corrector_rms"]["median"],
                     r.get("boris_equal_cost_rms", {}).get("median",
                                                           float("nan")),
                     r.get("vps2_equal_cost_rms", {}).get("median",
                                                          float("nan")),
                     r.get("vps4_equal_cost_rms", {}).get("median",
                                                          float("nan"))))

    print("\n" + "=" * 78)
    print("PRE-REGISTRATION STATUS")
    for k in ("P1", "P2", "P3", "P4"):
        print("  %s  %s" % (k, out["prereg_status"][k]["status"]))
    print("\nsummary: %s" % json.dumps(
        {k: v for k, v in out["summary"].items()
         if not k.endswith("_where")}, indent=1))
    if out["summary"]["A_holds_in_cells"]:
        print("A holds in: %s" % ", ".join(out["summary"]["A_holds_where"]))
    if out["summary"]["B_holds_in_cells"]:
        print("B holds in: %s" % ", ".join(out["summary"]["B_holds_where"]))

    hl = out["headline"]
    print("\n" + "=" * 78)
    print("HEADLINE NUMBERS")
    print("  A holds in %d of %d cells:" % (out["summary"]["A_holds_in_cells"],
                                            out["summary"]["n_cells"]))
    for k, v in hl["A_holding_cells"].items():
        print("     %-32s gain median %8.2f  ic0 %8.2f  range %.2f..%.2f  "
              "%d/8 initial conditions"
              % (k, v["gain_median"], v["gain_ic0"], v["gain_min"],
                 v["gain_max"], v["holds_n_of_8"]))
    print("  A fails worst at %s: gain %.3e (the corrector is %.3g times "
          "less accurate than the Boris scheme it corrects)"
          % (hl["A_worst_cell"]["cell"], hl["A_worst_cell"]["gain"]["median"],
             1.0 / hl["A_worst_cell"]["gain"]["median"]))
    print("  B never holds; the corrector's closest approach is %s, where it "
          "is still %.3e times (%.1f orders) less accurate than vps4 at the "
          "same total flops [%s]"
          % (hl["B_closest_cell"]["cell"],
             hl["B_closest_cell"]["ratio"]["median"],
             hl["B_closest_cell"]["orders_of_magnitude"],
             hl["B_closest_cell"]["mode"]))
    print("  B spans %.1f to %.1f orders of magnitude over the map"
          % tuple(hl["B_orders_range"]))
    c = hl["C_most_extreme"]
    print("  C, the most extreme cell: %s, %s -- trajectory off by %.4e "
          "Larmor radii while the energy error reads %.3e"
          % (c["cell"], c["scheme"], c["position_rms_larmor"]["median"],
             c["energy_median_2nd_half"]["median"]))
    print("  the energy channel ranks the five schemes with a median rank "
          "correlation of %+.2f against the trajectory, negative in %d of %d "
          "cells" % (out["summary"]["rank_agreement_median_over_all_cells"],
                     out["summary"]["rank_agreement_negative_cells"],
                     out["summary"]["n_cells"]))


if __name__ == "__main__":
    raise SystemExit(main())
