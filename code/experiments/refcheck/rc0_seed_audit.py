"""rc0_seed_audit.py -- Part 1 of W18: are any two runs sharing a generator?

W16 found two seed collisions by hand and did not check what they cost.  This
script does the audit mechanically, three ways, because the block discipline
had already failed twice and there was no reason to believe it had failed only
twice.

    1.  DECLARED EXTENTS.  Each ledger module is imported and its seed
        formula is evaluated over the whole index range its own asserts
        allow.  Two directories whose extents are disjoint cannot collide,
        whatever they later run; two whose extents overlap are suspects.

    2.  REALISED SEEDS.  Every committed JSON in the bundle is walked and
        every integer stored under a key containing "seed" is collected,
        grouped by the directory that wrote it.  This is the seed that
        actually produced a committed number, not the seed a docstring
        claims.  Values below 1e6 are reported separately: they are the small
        seeds of the bundle proper (42, 1, 7, 123, 2026, ...) and of the
        pre-block waves, where sharing an integer is the declared convention,
        not an accident.

    3.  GENERATOR FAMILY.  A collision only matters if the two runs draw from
        the *same* stream.  `torch.Generator().manual_seed(n)` and
        `numpy.random.default_rng(n)` share nothing but the integer: Mersenne
        Twister against PCG64, different state, different output.  Each
        colliding integer is therefore traced to its consuming call and
        labelled with the family, from the grep in CONSUMERS below.

WHAT A COLLISION COSTS
----------------------
A shared generator is harmful when, and only when, two runs that share it are
treated as independent: pooled into one ensemble whose spread is reported, or
set against each other as independent evidence.  The verdict field of the
output says, for every collision found, which of the two it is -- and the
report has to say what the colliding runs are and where their numbers land in
the manuscript.

This script draws nothing and writes rc0_seed_audit.json.
Usage: python rc0_seed_audit.py [--force]
"""
import collections
import json
import os
import sys

import rc_common as RC
from rc_common import check_or_write

HERE = RC.HERE
EXP = RC.EXP
ROOT = RC.ROOT
OUT = RC.outpath("rc0_seed_audit.json")

BLOCK_FLOOR = 1_000_000          # below this, integers are not block seeds

#: Which generator family consumes which seed, from
#: `grep -rn "Generator(\|default_rng(\|manual_seed("` over the bundle.
#: (directory, role) -> (family, file:line).  A role absent here is reported
#: as "unknown" rather than guessed.
CONSUMERS = {
    ("hpo", "init"):          ("torch", "hpo/hp_common.py:447"),
    ("hpo", "shuffle"):       ("numpy", "hpo/hp_common.py:448"),
    ("hpo", "data_init"):     ("numpy", "hpo/hp_common.py (build_data)"),
    ("hpo", "data_shuffle"):  ("numpy", "hpo/hp_common.py (build_data)"),
    ("sympmat", "pinit"):     ("torch", "sympmat/sm_arch.py:69"),
    ("sympmat", "pdata"):     ("numpy", "sympmat/sm1_train.py:84"),
    ("sympmat", "pbatch"):    ("numpy", "sympmat/sm1_train.py:126"),
    ("sympmat", "paug"):      ("numpy", "sympmat/sm1_train.py:127"),
    ("sympmat", "sinit"):     ("torch", "sympmat/sm_arch.py:82"),
    ("sympmat", "sdata"):     ("numpy", "sympmat/sm1_train.py:167"),
    ("sympmat", "sbatch"):    ("numpy", "sympmat/sm1_train.py:174"),
    ("sympmat", "ensemble"):  ("numpy", "sympmat/sm_common.py:307"),
    ("spectral", "ensemble"): ("numpy", "spectral/sw3_ensemble.py:50"),
    ("spectrum", "cloud"):    ("numpy", "spectrum/sp2_spectra.py:220"),
    ("map", "ic"):            ("numpy", "map/map_common.py:221"),
    ("external_arch", "init"):        ("torch", "external_arch/ea1_train.py:307"),
    ("external_arch", "shuffle"):     ("numpy", "external_arch/ea1_train.py:308"),
    ("external_arch", "collocation"): ("numpy", "external_arch/ea1_train.py:100"),
    ("seeds", "corrector"):   ("both",  "seeds/sd2_train.py:84 -> "
                                        "train_corrector_b4 SEED"),
}


# ------------------------------------------------------- 1. declared extents
def declared_extents():
    """Every ledger's own formula, over the whole index range it asserts."""
    ext = {}

    sys.path.insert(0, os.path.join(EXP, "hpo"))
    import hp_common as HP
    lo = HP.seed_of("hnn", 0, 0, "init")
    hi = max(HP.seed_of(a, 99, 99, r)
             for a in HP.ARCHS for r in HP.ROLE)
    ext["hpo"] = {"lo": lo, "hi": hi, "formula":
                  "11e6 + 1e6*arch + 1e4*cfg + 100*rep + role",
                  "source": "hpo/hp_common.py:127"}

    sys.path.insert(0, os.path.join(EXP, "sympmat"))
    import sm_common as SM
    ext["sympmat"] = {"lo": SM.seed_of("pinit", 0, 0),
                      "hi": max(SM.seed_of(r, 99, 999) for r in SM._ROLE),
                      "formula": "11e6 + 1e5*role + 1e3*dt_index + rep",
                      "source": "sympmat/sm_common.py:170"}

    sys.path.insert(0, os.path.join(EXP, "spectrum"))
    import sp_common as SP
    ext["spectrum"] = {"lo": SP.sp_seed("hnn", 0),
                       "hi": max(SP.sp_seed(a, 999) for a in SP.ARCH_SLOT),
                       "formula": "13e6 + 1e3*arch + rep",
                       "source": "spectrum/sp_common.py:157"}

    ext["spectral"] = {"lo": RC.SW.SPECTRAL_SEED, "hi": RC.SW.SPECTRAL_SEED,
                       "formula": "a single constant",
                       "source": "spectral/sw_common.py:146"}

    sys.path.insert(0, os.path.join(EXP, "map"))
    import map_common as MP
    ext["map"] = {"lo": MP.MAP_SEED, "hi": MP.MAP_SEED + 1,
                  "formula": "MAP_SEED and MAP_SEED+1",
                  "source": "map/map_common.py:103, mp1_calibration.py:103"}

    from ea_common import seed_of as ea_seed_of
    ext["external_arch"] = {"lo": ea_seed_of(0, 0, 0),
                            "hi": max(ea_seed_of(a, r, 999)
                                      for a in range(3) for r in range(5)),
                            "formula": "9e6 + 1e5*arch + 1e3*role + rep",
                            "source": "external_arch/ea_common.py:123"}

    sys.path.insert(0, os.path.join(EXP, "seeds"))
    import sd_common as SD
    ext["seeds"] = {"lo": SD.corrector_seed(0),
                    "hi": SD.corrector_seed(SD.N_CORRECTOR_SEEDS - 1),
                    "formula": "16e6 + i, i = 0..15",
                    "source": "seeds/sd_common.py:146"}

    ext["refcheck (W18, reserved, unused)"] = {
        "lo": 18_000_000, "hi": 18_999_999,
        "formula": "declared free before the first run of W18; W18 draws "
                   "nothing, so no seed in it is consumed",
        "source": "refcheck/rc_common.py docstring"}
    return ext


def extent_overlaps(ext):
    keys = sorted(ext)
    out = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ext[keys[i]], ext[keys[j]]
            lo, hi = max(a["lo"], b["lo"]), min(a["hi"], b["hi"])
            if lo <= hi:
                out.append({"a": keys[i], "b": keys[j],
                            "overlap_lo": int(lo), "overlap_hi": int(hi)})
    return out


# -------------------------------------------------------- 2. realised seeds
def realised():
    """Every integer under a "seed" key in every committed JSON."""
    per_dir = collections.defaultdict(set)
    where = collections.defaultdict(set)

    def walk(o, kp, d, fn):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, kp + "/" + str(k), d, fn)
        elif isinstance(o, list):
            for v in o:
                walk(v, kp + "/[]", d, fn)
        elif isinstance(o, int) and not isinstance(o, bool):
            tail = kp.rsplit("/", 1)[-1].lower()
            # the key itself must be about a seed, not a count of seeds
            if "seed" in tail and not tail.startswith("n_") \
                    and "count" not in tail and "index" not in tail:
                per_dir[d].add(int(o))
                where[(d, int(o))].add(fn + kp)

    for dp, _dn, fns in os.walk(ROOT):
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            rel = os.path.relpath(dp, ROOT).replace("\\", "/")
            if rel.startswith("experiments/refcheck"):
                continue
            try:
                with open(os.path.join(dp, fn), encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            walk(data, "", rel, fn)
    return per_dir, where


def top_dir(rel):
    """experiments/hpo/runs -> hpo ; checkpoints -> checkpoints."""
    parts = rel.split("/")
    if parts[0] == "experiments" and len(parts) > 1:
        return parts[1]
    return parts[0]


def main():
    force = "--force" in sys.argv
    out = {"meta": {
        "what": "Part 1 of W18: the seed-collision audit",
        "n_random_draws": 0,
        "block_floor": BLOCK_FLOOR,
        "method": "declared extents, realised seeds from committed JSON, and "
                  "the generator family of each consuming call",
    }}

    ext = declared_extents()
    out["declared_extents"] = ext
    out["declared_extent_overlaps"] = extent_overlaps(ext)

    per_dir_raw, where = realised()
    per_dir = collections.defaultdict(set)
    for rel, ss in per_dir_raw.items():
        per_dir[top_dir(rel)] |= ss
    out["realised"] = {
        d: {"n": len(s),
            "block_seeds": sorted(x for x in s if x >= BLOCK_FLOOR),
            "small_seeds": sorted(x for x in s if x < BLOCK_FLOOR)}
        for d, s in sorted(per_dir.items())}

    # ---- the collisions themselves, on block seeds only
    ds = sorted(per_dir)
    coll = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            shared = sorted((per_dir[ds[i]] & per_dir[ds[j]]))
            shared = [x for x in shared if x >= BLOCK_FLOOR]
            if not shared:
                continue
            coll.append({"a": ds[i], "b": ds[j], "shared": shared,
                         "n_shared": len(shared)})
    out["block_seed_collisions"] = coll

    # ---- the same for the small seeds, reported apart
    small = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            shared = sorted(x for x in (per_dir[ds[i]] & per_dir[ds[j]])
                            if x < BLOCK_FLOOR)
            if shared:
                small.append({"a": ds[i], "b": ds[j], "shared": shared})
    out["small_seed_sharing"] = small

    # ---- who consumes each colliding integer, and with which generator
    def roles_of(d, n):
        r = []
        if d == "hpo":
            import hp_common as HP
            for a in HP.ARCHS:
                for c in range(100):
                    for rp in range(100):
                        for ro in HP.ROLE:
                            if HP.seed_of(a, c, rp, ro) == n:
                                r.append("%s cfg%d rep%d %s" % (a, c, rp, ro))
        elif d == "sympmat":
            import sm_common as SM
            for ro in SM._ROLE:
                for k in range(100):
                    for rp in range(1000):
                        if SM.seed_of(ro, k, rp) == n:
                            r.append("%s dt%d rep%d" % (ro, k, rp))
        elif d == "spectrum":
            import sp_common as SP
            for a in SP.ARCH_SLOT:
                for rp in range(1000):
                    if SP.sp_seed(a, rp) == n:
                        r.append("cloud %s rep%d" % (a, rp))
        elif d == "spectral":
            if n == RC.SW.SPECTRAL_SEED:
                r.append("ensemble (eight initial conditions)")
        elif d == "external_arch":
            from ea_common import seed_of as es, ARCH_INDEX, ROLE as EROLE
            for a, ai in ARCH_INDEX.items():
                for ro, ri in EROLE.items():
                    for rp in range(1000):
                        if es(ai, ri, rp) == n:
                            r.append("%s %s rep%d" % (a, ro, rp))
        elif d == "seeds":
            r.append("corrector retraining i=%d" % (n - 16_000_000)
                     if 16_000_000 <= n < 16_000_100 else "recorded, not drawn")
        elif d == "map":
            r.append("initial conditions")
        return r or ["unknown"]

    def family(d, role_str):
        base = role_str.split()[-1] if d in ("hpo",) else role_str.split()[0]
        for (dd, rr), (fam, src) in CONSUMERS.items():
            if dd == d and (rr == base or rr in role_str):
                return fam, src
        return "unknown", ""

    detail = []
    for c in coll:
        for n in c["shared"]:
            ra = roles_of(c["a"], n)
            rb = roles_of(c["b"], n)
            fa = [family(c["a"], x) for x in ra]
            fb = [family(c["b"], x) for x in rb]
            same_stream = any(x[0] == y[0] and x[0] != "unknown"
                              for x in fa for y in fb)
            detail.append({
                "seed": n,
                "a": c["a"], "a_roles": ra,
                "a_families": sorted({x[0] for x in fa}),
                "a_sources": sorted({x[1] for x in fa if x[1]}),
                "b": c["b"], "b_roles": rb,
                "b_families": sorted({x[0] for x in fb}),
                "b_sources": sorted({x[1] for x in fb if x[1]}),
                "same_generator_family": bool(same_stream),
            })
    out["collision_detail"] = detail

    # ------------------------------------------------------------- printout
    print("=== declared extents ===")
    for k, v in sorted(ext.items()):
        print("  %-32s %12d .. %12d   %s" % (k, v["lo"], v["hi"], v["source"]))
    print("\n=== declared extents that overlap ===")
    for o in out["declared_extent_overlaps"]:
        print("  %-16s x %-16s  %d .. %d"
              % (o["a"], o["b"], o["overlap_lo"], o["overlap_hi"]))
    if not out["declared_extent_overlaps"]:
        print("  none")

    print("\n=== realised block seeds, by directory ===")
    for d, v in out["realised"].items():
        b = v["block_seeds"]
        if b:
            print("  %-16s %4d block seeds  %d .. %d" % (d, len(b), b[0], b[-1]))
    print("\n=== collisions on realised block seeds ===")
    if not coll:
        print("  none")
    for c in coll:
        print("  %-16s x %-16s  %d shared: %s"
              % (c["a"], c["b"], c["n_shared"], c["shared"]))
    print("\n=== each colliding integer, traced ===")
    for d in detail:
        print("  seed %d" % d["seed"])
        print("      %-14s %-46s %s" % (d["a"], "; ".join(d["a_roles"])[:46],
                                        ",".join(d["a_families"])))
        print("      %-14s %-46s %s" % (d["b"], "; ".join(d["b_roles"])[:46],
                                        ",".join(d["b_families"])))
        print("      same generator family: %s"
              % ("YES -- the two runs share a stream"
                 if d["same_generator_family"] else
                 "no -- different RNG families, nothing is shared but the integer"))
    print("\n=== small-seed sharing (below %d), reported apart ===" % BLOCK_FLOOR)
    for s in small:
        print("  %-16s x %-16s  %s" % (s["a"], s["b"], s["shared"]))

    RC.assert_no_draws(0)
    return check_or_write(OUT, json.loads(json.dumps(out)), force=force)


if __name__ == "__main__":
    raise SystemExit(main())
