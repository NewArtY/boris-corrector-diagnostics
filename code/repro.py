#!/usr/bin/env python3
"""One-command reproduction of the manuscript's figures and numbers.

There is no `make` on the machine this bundle was assembled on, so the build
is this script rather than a Makefile.  It is a plain driver: it runs the
committed scripts in dependency order, times each one, and checks the result
against what is committed.

    python repro.py --list             what each stage does and what it costs
    python repro.py figures            redraw the four figures (default)
    python repro.py audit              regenerate the audit-of-numbers files
    python repro.py data               regenerate the experiment JSON the
                                       figures read, then figures, then audit
    python repro.py train              retrain the corrector from the datasets
    python repro.py all                train + data + figures + audit
    python repro.py --check-sync       figure scripts here == manuscript's

Stages are cumulative in the order train -> data -> figures -> audit, and a
stage name runs that stage together with everything downstream of it.  Each
stage is idempotent: running it twice gives the same files.

VERIFICATION.  `figures` does not merely redraw.  Every figure script writes a
`figNN_<slug>_values.json` holding every number that appears on the artwork,
and this driver diffs that file against the committed copy in
`../../../article/figures/` when the manuscript tree is present.  A silent
change in any figure therefore fails the run rather than passing unnoticed.
Two of the figure scripts additionally assert their own inputs: fig1 asserts
every scalar it draws equal to `verify_final/vf1_magnetic.json` at rtol 1e-9,
and fig2 refits the exponents and asserts them equal to `symproj/summary.json`.

WHAT IS NOT REGENERATED.  `data` covers the inputs of the four figures and the
files the audit reads.  It does not re-run the whole experiment suite: the
p-law grids, the reconciliation sweeps and the seed ensembles under
`experiments/` are hours of compute and are shipped as committed JSON.  Every
one of them is a single `python <script>.py` in its own directory, with the
output file named in the script's docstring, so any of them can be re-run on
its own.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
FIGS = os.path.join(HERE, "figures_paper")
ARTICLE_FIGS = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                             "article", "figures"))

FIGURES = ["fig1_two_channels", "fig2_readout", "fig3_work_precision",
           "fig4_dissipation"]

# (stage, working directory, argv after the interpreter, what it writes)
STEPS = [
    # ---- train -----------------------------------------------------------
    ("train", HERE, ["training/dataset_generation.py"],
     "training/data/*.npz  (SKIPS any that already exist; delete them to "
     "force a rebuild)"),
    ("train", HERE, ["training/train_corrector_b4.py"],
     "checkpoints/boris_corrector_b4.pt, corrector_b4_params.json"),
    ("train", os.path.join(EXP, "seeds"), ["sd2_train.py"],
     "seeds/ckpt/corrector_b4_s16*.pt, seeds/sd2_training.json   sixteen "
     "retrainings of the corrector at the declared seeds, calling "
     "training/train_corrector_b4.py's own build_dataset() and train() with "
     "SEED and CHECKPOINT_DIR redirected and nothing else.  About 75 min on "
     "one core; shard with --shard k --of n"),
    ("train", os.path.join(EXP, "seeds"), ["sd4_external.py"],
     "seeds/ckpt_ea/*.npz, seeds/sd4_external.json   repetitions 4..9 of the "
     "three external architectures, taking the four of W9.1 to ten.  About "
     "80 min; shard with --shard k --of n"),

    # ---- data: inputs of the four figures ---------------------------------
    ("data", os.path.join(EXP, "verify_final"), ["vf1_magnetic.py"],
     "verify_final/vf1_magnetic.json          -> figure 1"),
    ("data", os.path.join(EXP, "symproj"), ["main.py", "quasistatic", "shipped"],
     "symproj/env_quasistatic_shipped.npz     -> figure 2"),
    ("data", os.path.join(EXP, "symproj"), ["main.py", "quasistatic", "staggered"],
     "symproj/env_quasistatic_staggered.npz   -> figure 2"),
    ("data", os.path.join(EXP, "symproj"), ["main.py", "paper", "shipped"],
     "symproj/env_paper_shipped.npz"),
    ("data", os.path.join(EXP, "symproj"), ["main.py", "paper", "staggered"],
     "symproj/env_paper_staggered.npz"),
    ("data", os.path.join(EXP, "symproj"), ["collect.py"],
     "symproj/summary.json                    -> figure 2"),
    ("data", os.path.join(EXP, "classical"), ["run.py"],
     "classical/workprecision.json            -> figure 3"),
    ("data", os.path.join(EXP, "classical"), ["verdict.py"],
     "classical/verdict.json                  -> figure 3"),
    ("data", os.path.join(EXP, "classical"), ["timing.py"],
     "classical/timing.json  (wall clock; the paper's cost claims use flops)"),
    ("data", os.path.join(EXP, "cost"), ["bench.py"],
     "cost/work_precision.json                -> figure 3"),
    ("data", os.path.join(EXP, "cost"), ["analyze.py"],
     "cost/breakeven.json                     -> figure 3"),
    ("data", os.path.join(EXP, "ll_probe"), ["derive.py"],
     "ll_probe/prereg.json"),
    ("data", os.path.join(EXP, "ll_probe"), ["run_experiments.py"],
     "ll_probe/results.json"),
    ("data", os.path.join(EXP, "ll_probe"), ["followup.py"],
     "ll_probe/prereg2.json, results2.json    -> figure 4"),
    ("data", os.path.join(EXP, "ll_probe"), ["fix_f3_f6.py"],
     "ll_probe/prereg3.json, results3.json"),
    # verifies, rather than writes: recomputes the F6b_refit_clean_window
    # block of results3.json from scratch and exits non-zero if the two rates
    # Section 6 prints no longer come out bit for bit.
    ("data", os.path.join(EXP, "recover_numbers"), ["rn1_f6b_refit.py"],
     "verifies ll_probe/results3.json F6b_refit_clean_window (Sec. 6)"),
    # verifies, rather than writes: re-runs the two instrumented integrations
    # of Section 4.3, checks every field of pc_defect.json and pc_split.json
    # that Sections 4.3 and 6 quote, and runs the channel-selection rule of
    # Table 1 on them.  About 5 min.
    ("data", os.path.join(EXP, "probe4"), ["pb4_channel.py"],
     "probe4/pb4_channel.json   verifies pc_defect.json, pc_split.json (Secs. 4.3, 6)"),
    # the three external architectures of Section 7, the four probes run on
    # them and their cost against the classical family.  Each of the three
    # writes its JSON on the first run and, on every run after, recomputes and
    # exits non-zero if the committed file no longer reproduces.
    ("data", os.path.join(EXP, "external_arch"), ["ea1_train.py"],
     "external_arch/ea1_training.json, ckpt/*.npz  HNN, SympNet, PINN"),
    ("data", os.path.join(EXP, "external_arch"), ["ea2_probes.py"],
     "external_arch/ea2_probes.json    the four probes on the three (Sec. 6)"),
    ("data", os.path.join(EXP, "external_arch"), ["ea3_cost.py"],
     "external_arch/ea3_cost.json      flops against tab:family (Sec. 7)"),
    # the hyper-parameter search behind the undertraining paragraph of
    # Section 7: 96 training runs, about 10 h of single-core work.  Each run
    # writes one JSON under hpo/runs/ on the first pass and, on every pass
    # after, recomputes it and exits non-zero if it no longer reproduces.
    # Both training stages take --shard k --nshards N; sharding is scheduling
    # only and changes no number.
    ("data", os.path.join(EXP, "hpo"), ["hp1_grid.py"],
     "hpo/runs/*.json          17 configurations x 4 seeds at the base budget"),
    ("data", os.path.join(EXP, "hpo"), ["hp2_select.py"],
     "hpo/hp2_selection.json   validation selection, loss-vs-rollout ranks"),
    ("data", os.path.join(EXP, "hpo"), ["hp3_ladder.py", "--with-data"],
     "hpo/runs/*.json          the budget ladder and the data sweep"),
    ("data", os.path.join(EXP, "hpo"), ["hp4_report.py"],
     "hpo/hp4_analysis.json    trends, extrapolations, equal-cost vps4"),
    # the map of applicability of Section \ref{sec:map}: five schemes over
    # twelve step sizes, five field configurations and two horizons.  The
    # calibration stage reproduces tab:family on this stand and adjudicates
    # the reference in each configuration; the grid is sharded by
    # configuration, which is scheduling only and changes no number.  About
    # 50 min per configuration on one core, and the five are independent.
    ("data", os.path.join(EXP, "map"), ["mp1_calibration.py"],
     "map/mp1_calibration.json   the stand against tab:family, the bridge's "
     "bit-identity, and what the reference is worth per configuration"),
    ("data", os.path.join(EXP, "map"), ["mp4_saturation.py"],
     "map/mp4_saturation.json    the corrector's input standardisation off "
     "its one training point"),
    ("data", os.path.join(EXP, "map"), ["mp2_grid.py", "--field", "uniform"],
     "map/mp2_grid__uniform.json"),
    ("data", os.path.join(EXP, "map"), ["mp2_grid.py", "--field", "B1_radial"],
     "map/mp2_grid__B1_radial.json"),
    ("data", os.path.join(EXP, "map"), ["mp2_grid.py", "--field", "B2_wave"],
     "map/mp2_grid__B2_wave.json"),
    ("data", os.path.join(EXP, "map"), ["mp2_grid.py", "--field", "B3_tilted"],
     "map/mp2_grid__B3_tilted.json"),
    ("data", os.path.join(EXP, "map"), ["mp2_grid.py", "--field",
                                        "B4_decaying"],
     "map/mp2_grid__B4_decaying.json"),
    ("data", os.path.join(EXP, "map"), ["mp5_equalcost.py", "--field",
                                        "uniform"],
     "map/mp5_equalcost__uniform.json    the corrector's flop budget on "
     "Boris and vps2"),
    ("data", os.path.join(EXP, "map"), ["mp5_equalcost.py", "--field",
                                        "B1_radial"],
     "map/mp5_equalcost__B1_radial.json"),
    ("data", os.path.join(EXP, "map"), ["mp5_equalcost.py", "--field",
                                        "B2_wave"],
     "map/mp5_equalcost__B2_wave.json"),
    ("data", os.path.join(EXP, "map"), ["mp5_equalcost.py", "--field",
                                        "B3_tilted"],
     "map/mp5_equalcost__B3_tilted.json"),
    ("data", os.path.join(EXP, "map"), ["mp5_equalcost.py", "--field",
                                        "B4_decaying"],
     "map/mp5_equalcost__B4_decaying.json"),
    ("data", os.path.join(EXP, "map"), ["mp3_maps.py"],
     "map/mp3_maps.json          the three maps of Sec. \\ref{sec:map}"),
    # the table of orders of Table \ref{tab:gtable}: G = log10(E_Boris /
    # E_scheme) for five schemes in four channels -- trajectory, phase, total
    # energy and the spectral power below f/Omega_c = 0.2 -- over the five
    # field configurations at the working step Omega h = 0.3.  The
    # calibration stage reproduces tab:family, the committed in-band powers of
    # spectral/sw2_spectra.json and the one committed phase number of
    # theory_check/t1_boris_channels.json on this stand, declares the
    # reference gyrofrequency the band is taken against, and adjudicates the
    # reference per configuration in each of the four channels.  The channel
    # stage is sharded by configuration, which is scheduling only and changes
    # no number.  About 10 min for the calibration and 1 min per
    # configuration.
    ("data", os.path.join(EXP, "gtable"), ["gt1_calibration.py"],
     "gtable/gt1_calibration.json   the stand against tab:family, "
     "sw2_spectra.json and t1_boris_channels.json; the declared band; the "
     "reference floor per configuration and per channel"),
    ("data", os.path.join(EXP, "gtable"), ["gt2_channels.py", "--field",
                                           "uniform"],
     "gtable/gt2_channels__uniform.json"),
    ("data", os.path.join(EXP, "gtable"), ["gt2_channels.py", "--field",
                                           "B1_radial"],
     "gtable/gt2_channels__B1_radial.json"),
    ("data", os.path.join(EXP, "gtable"), ["gt2_channels.py", "--field",
                                           "B2_wave"],
     "gtable/gt2_channels__B2_wave.json"),
    ("data", os.path.join(EXP, "gtable"), ["gt2_channels.py", "--field",
                                           "B3_tilted"],
     "gtable/gt2_channels__B3_tilted.json"),
    ("data", os.path.join(EXP, "gtable"), ["gt2_channels.py", "--field",
                                           "B4_decaying"],
     "gtable/gt2_channels__B4_decaying.json"),
    ("data", os.path.join(EXP, "gtable"), ["gt3_gtable.py"],
     "gtable/gt3_gtable.json, gt3_report.txt   Table \\ref{tab:gtable}, the "
     "rank agreement between the four channels, and the decomposition of the "
     "map's own +0.00"),

    # the ensemble over training seeds.  Every corrector number in the paper
    # stands on one checkpoint; these steps retrain the same architecture at
    # the same hyper-parameters at sixteen declared seeds, put the committed
    # checkpoint and twenty independent retrainings through the same three
    # readouts, and report where the committed one sits.  The calibration
    # stage also prices the reference that Section 7's trajectory-advantage
    # numbers are scored against, which is the same Boris run at h/150 that
    # the corrector is trained on.  The training steps are hours and are
    # sharded (`--shard k --of n`, scheduling only); the measurement steps run
    # off the committed checkpoints in minutes.
    ("data", os.path.join(EXP, "seeds"), ["sd1_calibration.py"],
     "seeds/sd1_calibration.json   the stand against all five rows of "
     "tab:family and against gtable/gt2_channels__*.json; the measured error "
     "of the h/150 Boris reference itself.  About 6 min."),
    ("data", os.path.join(EXP, "seeds"), ["sd3_measure.py"],
     "seeds/sd3_ensemble.json      every checkpoint of the ensemble through "
     "the readout of Section 7, the row of tab:family and the four channels "
     "of W15 on five configurations.  About 4 min; needs seeds/ckpt/."),
    ("data", os.path.join(EXP, "seeds"), ["sd5_summary.py"],
     "seeds/sd5_summary.json, sd5_report.txt   medians, interquartile ranges, "
     "the percentile of the committed checkpoint in each channel, and the "
     "LaTeX of Tables \\ref{tab:seeds} and \\ref{tab:seed_channels}"),

    # inputs of the audit stage
    ("data", os.path.join(EXP, "horizon"), ["validate.py"],
     "horizon/validation.json"),
    ("data", os.path.join(EXP, "horizon"), ["long_runs.py"],
     "horizon/long_runs.npz, long_runs_summary.json"),
    ("data", os.path.join(EXP, "horizon"), ["traj.py"],
     "horizon/traj_summary.json"),
    ("data", os.path.join(EXP, "horizon"), ["crossover.py"],
     "horizon/crossover.json, crossover.npz"),

    # ---- figures ----------------------------------------------------------
    ("figures", FIGS, ["fig1_two_channels.py"], "fig1_two_channels.pdf"),
    ("figures", FIGS, ["fig2_readout.py"], "fig2_readout.pdf"),
    ("figures", FIGS, ["fig3_work_precision.py"], "fig3_work_precision.pdf"),
    ("figures", FIGS, ["fig4_dissipation.py"], "fig4_dissipation.pdf"),

    # ---- audit ------------------------------------------------------------
    ("audit", os.path.join(EXP, "audit_numbers"), ["an1_resonance_profile.py"],
     "an1_resonance_profile.json   Sec. 4.5 and Sec. 8"),
    ("audit", os.path.join(EXP, "audit_numbers"), ["an3_derived.py"],
     "an3_derived.json             Sec. 4.2 and Sec. 5"),
    ("audit", os.path.join(EXP, "audit_numbers"), ["an4_drift_reading.py"],
     "an4_drift_reading.json       Sec. 3.2"),
    ("audit", os.path.join(EXP, "audit_numbers"), ["an5_horizon_readout.py"],
     "an5_horizon_readout.json     Sec. 7"),
    ("audit", os.path.join(EXP, "audit_numbers"), ["an2_traceability.py"],
     "an2_traceability.json        sweep of Secs. 3-8 (needs article/main.tex)"),
    ("audit", os.path.join(EXP, "audit_numbers"), ["an6_orphan_data.py"],
     "an6_orphan_data.json         every data file has a producing script"),
]

ORDER = ["train", "data", "figures", "audit"]


def run_step(stage, cwd, argv, produces, dry):
    label = "%-8s %s" % (stage, " ".join(argv))
    if dry:
        print("  %-46s -> %s" % (label, produces))
        return 0.0, True
    print("[%s] %s" % (time.strftime("%H:%M:%S"), label), flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable] + argv, cwd=cwd,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    el = time.time() - t0
    ok = p.returncode == 0
    if not ok:
        sys.stdout.write(p.stdout.decode("utf-8", "replace"))
        print("  FAILED (exit %d) after %.1f s" % (p.returncode, el))
    else:
        print("  ok  %.1f s" % el)
    return el, ok


def check_figure_values():
    """Diff the numbers drawn on each figure against the committed copy."""
    if not os.path.isdir(ARTICLE_FIGS):
        print("  (manuscript tree not present, skipping the values diff)")
        return True
    good = True
    for f in FIGURES:
        a = os.path.join(ARTICLE_FIGS, f + "_values.json")
        b = os.path.join(FIGS, f + "_values.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            print("  %-22s MISSING" % f)
            good = False
            continue
        da = json.load(open(a, encoding="utf-8"))
        db = json.load(open(b, encoding="utf-8"))
        same = da == db
        print("  %-22s %s" % (f, "values reproduce" if same else "VALUES DIFFER"))
        good = good and same
    return good


def check_sync():
    """The bundled figure scripts must differ from the manuscript's only in
    the one line that locates the data."""
    import re
    if not os.path.isdir(ARTICLE_FIGS):
        print("manuscript tree not present; nothing to compare against")
        return True
    pat = re.compile(r'os\.pardir,\s*os\.pardir,\s*"code",\s*"bundle",\s*"code",')
    note = re.compile(r"\n#\n# -{5,}\n(?:# .*\n)*?# -{5,}\n")
    good = True
    for name in ["paper_style.py"] + [f + ".py" for f in FIGURES]:
        a = open(os.path.join(ARTICLE_FIGS, name), encoding="utf-8").read()
        b = open(os.path.join(FIGS, name), encoding="utf-8").read()
        a_norm = pat.sub("os.pardir,", a).replace("\r\n", "\n")
        b_norm = note.sub("", b).replace("\r\n", "\n")
        same = a_norm == b_norm
        print("  %-22s %s" % (name, "in sync" if same else "OUT OF SYNC"))
        good = good and same
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="figures",
                    choices=ORDER + ["all"])
    ap.add_argument("--list", action="store_true",
                    help="print the plan without running it")
    ap.add_argument("--check-sync", action="store_true",
                    help="compare the bundled figure scripts with the "
                         "manuscript's and exit")
    a = ap.parse_args()

    if a.check_sync:
        sys.exit(0 if check_sync() else 1)

    first = 0 if a.stage == "all" else ORDER.index(a.stage)
    wanted = ORDER[first:]
    steps = [s for s in STEPS if s[0] in wanted]

    if a.list:
        print("stages to run: %s\n" % ", ".join(wanted))
        for s in steps:
            run_step(*s, dry=True)
        return

    t0 = time.time()
    failed = []
    times = []
    for s in steps:
        el, ok = run_step(*s, dry=False)
        times.append((s[0], " ".join(s[2]), el))
        if not ok:
            failed.append(" ".join(s[2]))

    ok_vals = True
    if "figures" in wanted:
        print("\n---- figure values against the committed copies ----")
        ok_vals = check_figure_values()

    total = time.time() - t0
    print("\n---- timing ----")
    for stage, name, el in times:
        print("  %-8s %-42s %7.1f s" % (stage, name, el))
    by_stage = {}
    for stage, _, el in times:
        by_stage[stage] = by_stage.get(stage, 0.0) + el
    for stage in ORDER:
        if stage in by_stage:
            print("  %-8s %-42s %7.1f s" % (stage, "TOTAL", by_stage[stage]))
    print("  %-8s %-42s %7.1f s (%.1f min)" % ("", "WALL CLOCK", total, total / 60))

    if failed:
        print("\nFAILED: " + ", ".join(failed))
    if not ok_vals:
        print("\nFAILED: a figure no longer reproduces its committed values")
    sys.exit(1 if (failed or not ok_vals) else 0)


if __name__ == "__main__":
    main()
