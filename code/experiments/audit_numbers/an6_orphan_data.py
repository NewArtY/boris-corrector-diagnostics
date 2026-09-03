"""AN6: does every committed data file have a script that writes it?

The paper states that every measured number is written by a script into a data
file.  The converse has to hold too, or the statement is empty: a JSON that no
script produces is a number typed in by hand as far as a reader can tell.

This walks `experiments/` and, for each .json and .npz, looks for its name in
the Python sources of its own directory.  Names that scripts build at run time
(`env_{config}_{base}.npz`, `f0_summary{TAG}.json`, ...) cannot be found that
way, so they are listed explicitly below together with the script that builds
them; each entry was checked by hand once and is re-checked here only for the
existence of that script.

Result as of wave W6.2: two orphans remain, both in `p_law/`, and both are
recorded as such in `plan/reports/P_LAW.md`, which says they were computed
inline and only their JSON was kept.

  p_law/pl_marginal.json    the marginal case a + H = 0.  Section 4.2 of the
                            paper quotes all three of its fields (0.051
                            analytic, 0.0398 measured, Var S / log N = 1.09).
                            The construction is the one of pl_limits.py case
                            (iv) at a = -1/2, but the ensemble, the horizon and
                            the estimator that give exactly these three numbers
                            were not recorded, and re-deriving them from the
                            surviving description does not reproduce them.  It
                            is therefore left standing and named, not guessed
                            at: a script that returned different numbers would
                            be worse than an acknowledged gap.
  p_law/pl_fgn_check.json   accuracy of the fGn generator.  No number from it
                            appears in the paper, and the same check, run over
                            a wider grid, is stored by pl_core.py in
                            pl_core.json ("A_fgn_generator_check").

Everything else in `experiments/` has a producing script.  Two files that did
not, until W6.2, now do: `classical/verdict.json` (see `classical/verdict.py`)
and the four numbers of `horizon/crossover.json`'s `gain_vs_horizon` (see
`horizon/crossover.py`).

Usage: python an6_orphan_data.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, os.pardir))

# data file -> the script that builds its name at run time
CONSTRUCTED = {
    "symproj/env_paper_shipped.npz": "symproj/main.py",
    "symproj/env_paper_staggered.npz": "symproj/main.py",
    "symproj/env_quasistatic_shipped.npz": "symproj/main.py",
    "symproj/env_quasistatic_staggered.npz": "symproj/main.py",
    "symproj/envelopes.npz": "symproj/short.py",
    "f0_variational/f0_runs_small.npz": "f0_variational/run_f0.py",
    "p_law_check/pc_stall_paper.npz": "p_law_check/pc_stall.py",
    "p_law_check/pc_stall_quasistatic.npz": "p_law_check/pc_stall.py",
}

# Orphans at block level: the file has a producing script, but one top-level
# key inside it does not.  Checked for presence, because a re-run of the
# producing script used to delete them.
ORPHAN_BLOCKS = {
    "ll_probe/results3.json": {
        "F6b_refit_clean_window":
            "refit of the energy-channel decay over a later, cleaner window; "
            "Section 6 quotes 1.7954 and 1.4286 from it.  fix_f3_f6.py writes "
            "F6b (1.4926 / 1.2573) but not this refit, which was done inline.  "
            "Reconstructing it from the surviving description gets 1.787 / "
            "1.407, close but not equal, so it is named rather than guessed.  "
            "Since W6.2 fix_f3_f6.py merges instead of overwriting, so a "
            "re-run no longer deletes it."},
}

# data file -> why it has no script, on the record
KNOWN_ORPHANS = {
    "p_law/pl_marginal.json":
        "computed inline; plan/reports/P_LAW.md records it as such.  Section "
        "4.2 quotes its three fields.",
    "p_law/pl_fgn_check.json":
        "computed inline; superseded by pl_core.json/A_fgn_generator_check.  "
        "No number from it appears in the paper.",
}

report = {"with_script": [], "constructed": [], "known_orphans": [],
          "NEW_ORPHANS": []}

for root, _, files in os.walk(EXP):
    if "__pycache__" in root:
        continue
    # the scripts of this directory and of its parent: `stats/` writes into
    # `stats/results/`, and that is a producing script like any other
    src_dirs = [root]
    parent = os.path.dirname(root)
    if os.path.normpath(parent).startswith(os.path.normpath(EXP)):
        src_dirs.append(parent)
    srcs = []
    for d in src_dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                srcs.append(open(os.path.join(d, f), encoding="utf-8",
                                 errors="replace").read())
    srcs = "\n".join(srcs)
    for f in sorted(files):
        if not f.endswith((".json", ".npz")):
            continue
        rel = os.path.relpath(os.path.join(root, f), EXP).replace("\\", "/")
        if f in srcs or f.rsplit(".", 1)[0] in srcs:
            report["with_script"].append(rel)
        elif rel in CONSTRUCTED:
            builder = os.path.join(EXP, CONSTRUCTED[rel])
            if not os.path.exists(builder):
                report["NEW_ORPHANS"].append(
                    {"file": rel, "why": "declared builder %s is missing"
                     % CONSTRUCTED[rel]})
            else:
                report["constructed"].append(
                    {"file": rel, "built_by": CONSTRUCTED[rel]})
        elif rel in KNOWN_ORPHANS:
            report["known_orphans"].append({"file": rel,
                                            "why": KNOWN_ORPHANS[rel]})
        else:
            report["NEW_ORPHANS"].append({"file": rel, "why": "no script "
                                          "in its directory names it"})

# ---- orphan blocks: present, and named as unreproduced --------------------
report["orphan_blocks"] = []
for rel, blocks in ORPHAN_BLOCKS.items():
    path = os.path.join(EXP, rel)
    have = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    for key, why in blocks.items():
        rec = {"file": rel, "key": key, "why": why, "present": key in have}
        report["orphan_blocks"].append(rec)
        if not rec["present"]:
            report["NEW_ORPHANS"].append(
                {"file": "%s#%s" % (rel, key),
                 "why": "block the manuscript cites has gone missing from the "
                        "file; restore it before submitting"})

report["_summary"] = {
    "data_files": (len(report["with_script"]) + len(report["constructed"])
                   + len(report["known_orphans"]) + len(report["NEW_ORPHANS"])),
    "with_a_producing_script": len(report["with_script"]),
    "name_built_at_run_time": len(report["constructed"]),
    "known_orphans": len(report["known_orphans"]),
    "orphan_blocks": len(report["orphan_blocks"]),
    "new_orphans": len(report["NEW_ORPHANS"]),
}

json.dump(report, open(os.path.join(HERE, "an6_orphan_data.json"), "w",
                       encoding="utf-8"), indent=1)
print(json.dumps(report["_summary"], indent=1))
for o in report["known_orphans"]:
    print("  known orphan: %-28s %s" % (o["file"], o["why"]))
for o in report["orphan_blocks"]:
    print("  orphan block: %-28s %s  [%s]"
          % (o["file"], o["key"], "present" if o["present"] else "MISSING"))
for o in report["NEW_ORPHANS"]:
    print("  NEW ORPHAN:   %-28s %s" % (o["file"], o["why"]))
raise SystemExit(1 if report["NEW_ORPHANS"] else 0)
