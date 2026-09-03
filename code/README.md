# A readout floor can hide a secular growth of the energy error from an absolute-level diagnostic

Reproduction bundle for the manuscript of that title (N. S. Akintsov,
A. P. Nevecheria, S. N. Andreev, Qing-Hua Qin), submitted to the *Journal of
Computational Physics*.

> **Note on the previous version of this file.** Until this revision the README
> was headed *"Neural Integrators for Charged-Particle Motion"* and described a
> seven-figure paper about four learned integrators. That was the first version
> of the work, which was withdrawn. The present paper is a different paper: it
> is about what the energy-error history does and does not measure, it has four
> figures, and the learned corrector appears in it as one scheme among six
> rather than as the result. The directories that served the withdrawn version
> are still here, and are marked as such below, because several verification
> scripts read their output.

## What the paper claims and what this bundle has to support

The paper claims that every measured number in it is written by a script into a
data file, that no value was transcribed by hand, and that every figure is
produced by a script that reads those files (Appendix, *Reproducibility*). This
bundle is what that claim rests on. Concretely:

* every number printed in Sections 3 to 8 is held in a committed data file —
  the sweep that checks this is
  `experiments/audit_numbers/an2_traceability.py`, and its last run matched
  **433 of 438** printed literals to a stored value, the remaining five being
  publication years in prose;
* the converse is checked too, because it is the one that can fail silently:
  of the 107 committed data files, 105 have a script that writes them
  (`experiments/audit_numbers/an6_orphan_data.py`). Three places where a
  printed number has a file but no script are listed by name below, under
  *The three places where a number the paper prints has no script*, rather
  than papered over;
* the four figures are redrawn from committed data by four scripts in
  `figures_paper/`, and each writes a `figNN_<slug>_values.json` holding every
  number that appears on the artwork;
* the whole of it runs from one command, `python repro.py`.

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -r requirements-lock.txt
conda activate neural-integrators

python repro.py --list                   # the plan, with what each step writes
python repro.py                          # redraw the four figures and check them
python repro.py audit                    # regenerate the audit-of-numbers files
python repro.py data                     # regenerate the figures' inputs, then the above
python repro.py all                      # retrain the corrector first, then all of it
python repro.py --check-sync             # figure scripts here == the manuscript's
```

Stages are cumulative in the order `train -> data -> figures -> audit`, and
naming one runs it together with everything downstream. Every stage is
idempotent.

One deliberate exception to "from a clean state":
`training/dataset_generation.py` skips any dataset file that already exists, so
`repro.py train` retrains from the committed `training/data/*.npz` rather than
rebuilding them. Delete that directory first if you want the datasets rebuilt
too; the generator is seeded and rebuilds them identically.

`figures` does not merely redraw. The driver diffs each regenerated
`figNN_<slug>_values.json` against the copy committed with the manuscript, so a
figure that silently changes fails the run. Two of the scripts additionally
check their own inputs: `fig1` asserts every scalar it draws equal to
`experiments/verify_final/vf1_magnetic.json` at `rtol = 1e-9` and aborts
otherwise, and `fig2` refits its exponents and asserts them equal to
`experiments/symproj/summary.json`.

### Measured wall-clock

Windows 11, AMD64, Python 3.14.3, CPU only. Measured, not estimated: these are
the numbers `repro.py` printed on the end-to-end run of 2026-09-01, on a
machine that was not otherwise idle.

| stage | wall clock | what dominates it |
| :-- | --: | :-- |
| `figures` | 11 s | `fig4_dissipation.py`, 8 s |
| `audit` | 76 s | `an1_resonance_profile.py`, 74 s: a 901-point frequency scan at 10^4 gyro-orbits |
| `figures` + `audit` together | **1.4 min** | |
| `data`, its own steps | 45 min | four `experiments/symproj/main.py` configurations, 2.09 million steps each in a Python loop, 17 min between them; then `horizon/long_runs.py` 13 min, `horizon/traj.py` 5 min, `classical/run.py` 5 min |
| `data` end to end (it chains the two above) | **46 min** | |
| `train` | about 2 min | 400 epochs of `DefectNet` on 6000 pairs; the datasets are committed and are not rebuilt unless deleted |
| `all` | about 48 min | |

The point of the split is that a referee who wants to check the paper rather
than the campaign runs `figures` and `audit` and is done in a minute and a
half: that answers "do the committed data files really produce these figures
and these numbers". `data` answers the larger question, "do the experiments
really produce the committed data files", and on the run above the answer was
yes for every physical quantity in every file it touched — the four figures
came out with `_values.json` identical to the committed copies after the whole
chain beneath them had been recomputed from scratch. The only fields that
moved were wall-clock timings; see the caveat under *Reproducibility* below.

`figures` and `audit` need only NumPy, SciPy and matplotlib; they load no
checkpoint and import no PyTorch. PyTorch is needed for `data` and `train`.

## Layout

```
repro.py                    the build.  Run this.
common.py                   shared constants, global seed, paths
environment.yml             pinned environment (also requirements-lock.txt)
LICENSE                     MIT, four rightsholders.  Citation metadata is
                            CITATION.cff one level up, in the repository root,
                            which is where GitHub and Zenodo read it from

figures_paper/              THE FOUR FIGURES OF THE PAPER
  paper_style.py              shared style: text-width artwork, Okabe-Ito,
                              pdf.fonttype 42, dash and marker per series
  fig1_two_channels.py        Figure 1  (cited in Section 2)
  fig2_readout.py             Figure 2  (Section 3) -- the key figure
  fig3_work_precision.py      Figure 3  (Section 7)
  fig4_dissipation.py         Figure 4  (Section 5)
  figN_<slug>.pdf             the artwork
  figN_<slug>_values.json     every number drawn or annotated, generated

experiments/                ONE DIRECTORY PER EXPERIMENT, script + its output
  audit_numbers/              numbers that no other script wrote out (below)
  classical/                  five classical schemes on one work-precision grid
  cost/                       cost in flops and in seconds; break-even
  f0_settings/  f0_variational/   settings sweep and the variational integrator
  horizon/                    long-horizon trajectory and energy behaviour
  ll_probe/                   the dissipative rapidity system, Section 5
  p_law/  p_law_check/        the exponent law of Section 4 and its conditions
  reconcile_t2/               field-law and mechanism reconciliation, Section 3
  referee_check/              the two channels and the staggered drift
  stats/                      seed and initial-condition ensembles
  symproj/                    envelope growth under the two readouts, Section 3
  theory/  theory_check/  verify_f0/  verify_final/  verify_theory/
                              independent re-derivations and verifiers

training/                   dataset generation and training
  dataset_generation.py       Boris reference trajectories -> training/data/*.npz
  train_corrector_b4.py       the corrector of the paper (DefectNet)
  train.py                    the four integrators of the WITHDRAWN version
  scan_dt.py                  step scan for the degeneracy regime
checkpoints/                trained weights (*.pt) and training metadata
models/  fields/            integrators and field configurations
diagnostics/                energy drift, spectra, phase space, corrector eval

figures/                    LEGACY: the seven figures of the withdrawn version
output_figures/             LEGACY: their output, plus results_summary.json and
                            corrector_evaluation.json, which verify_i0/,
                            stats/ and cost/verify_model_dtype.py still read
verify_i0/                  LEGACY: the reproduction audit of the withdrawn
                            version, kept because it is the evidence behind the
                            reproducibility statement below
```

## Script -> number in the paper

The mapping below names, for each section, the numbers a referee is most likely
to check and the file each is read from. `experiments/` is elided from the
paths. Unless stated otherwise a file is written by the script of the same stem
in the same directory — `verify_final/vf1_magnetic.json` by
`verify_final/vf1_magnetic.py`, and so on. The exceptions are
`symproj/{summary.json, envelope_growth.json, env_*.npz}`, written by
`symproj/main.py` and `symproj/collect.py`, and `stats/results/*.json`, written
by the scripts one level up in `stats/`.

### Section 2, Two channels — Figure 1

| number in the paper | file | key |
| :-- | :-- | :-- |
| velocity turned `50.80` deg | `verify_final/vf1_magnetic.json` | `T1_T7/theta_final_deg` |
| relative speed error `6.253e-7` | `referee_check/rf1_channels.json` | `speed_rel_err_median_2nd_half` |
| the six-decade gap between the channels | `verify_final/vf1_magnetic.json` | `T1_T7` |
| the same three quantities, measured against the closed form by an independent script | `theory_check/t1_boris_channels.json` | `theta_end_deg`, `speed_err_median`, `energy_err_median` |

`figures_paper/fig1_two_channels.py` reproduces the run itself (the per-step
series is not committed, only the summary) and asserts every scalar it draws
equal to `vf1_magnetic.json`.

### Section 3, The readout floor — Figure 2

| number | file | key |
| :-- | :-- | :-- |
| floor collapse by `35,760` | `verify_final/vf1_magnetic.json` | `P1_1_r0_shift/ratio_base/shifted` |
| median error `1.24985e-6` before and `3.49516e-11` after the half-step shift | `verify_final/vf1_magnetic.json` | `P1_1_r0_shift/*/floor_median_2nd_half`; independently in `reconcile_t2/rc2_decisive.json`, `M_knob2_initial_data/named/r0=(1,h/2,0)/dev0` |
| trajectories agree to `6.25e-7` | `verify_final/vf1_magnetic.json` | `P1_1_r0_shift/traj_err_series_max_rel_diff` |
| exponents `0.977` and `0.0000` | `symproj/summary.json` | `results/quasistatic/*/envelope_exponent` |
| staggered Boris floor `0.02200`, corrector `0.0223` | `symproj/summary.json`, `symproj/envelope_growth.json` | `.../staggered/{boris,proj}/E_err_*` |
| the growth crosses the floor at `1.9e7` | `symproj/summary.json` + `figures_paper/fig2_readout_values.json` | |
| **Section 3.2 drift reading**: `8.96e-6`, `4.55e-5`, `7.92e-4`, `3.40e-6`, `3.40e-5`, `3.38e-4`, ratio `2.34`, slopes `0.535 / 0.939 / 2.175 / 0.468` | `audit_numbers/an4_drift_reading.json` | all of them |
| `R_art = 600.2` and the range `0.034 ... 603` | `reconcile_t2/rc1_grid.json` | `magnetic[*]/R_art` |
| `1.0004`, `0.4996`, `0.4998` field-law factors | `reconcile_t2/rc5_fieldlaws.json` | `R_art_over_2th` |

### Section 4, What generates secular growth

| number | file | key |
| :-- | :-- | :-- |
| horizon `25,033` gyro-orbits, `N = 2^19`, `h = 0.3` | `audit_numbers/an3_derived.json`, `p_law/pl_core.json` | `S42_grid_horizon`, `grid_setup/gyros` |
| regression over the 42 cells, `0.972 +- 0.078` | `p_law/pl_core.json` + `p_law/pl_antipersistent.json` | `q_rms_ensemble` |
| `0.051` analytic against `0.0398` measured | `p_law/pl_marginal.json` | `predicted_slope_sqrt_logN`, `measured_rms_slope` |
| limit points `0.0377 / 0.0156 / 0.00119 / 4.05e-6` | `p_law/pl_limits.json` | `L2_limit_points` |
| `0.503 -> 1.012` across the linear regime | `p_law/pl_transfer.json` | `M4_linear_regime_breakdown` |
| AR(1) sweep `0.748 / 0.545 / 0.504 / 0.500` | `p_law/pl_transfer.json` | `M3_AR1_horizon_sweep` |
| 18 measurements, `0.14 ... 0.25`, four horizons | `p_law/pl_horizon.json` | `R3_horizon_convergence` |
| the short-run protocol, `1,565` gyro-orbits | `p_law/pl_protocol.json` | `short_run_gyros` |
| Section 4.3 conditions, both runs | `p_law_check/pc_defect.json`, `pc_split.json` | `runs/{paper,quasistatic}/proj` |
| Section 4.4 local slopes, freeze at `5.75e4` | `p_law_check/pc_stall.json` | |
| Section 4.1 identity, `1.9e-14` | `verify_theory/vt_t3_trichotomy.json` | `A_lemma31_max_abs_error` |
| **Section 4.5 resonance**: peak `3.84e-2` at detuning `-2.5e-4`, `1.92e-2` on `omega_h(0)`, full width at half maximum `4.6e-4` | `audit_numbers/an1_resonance_profile.json` | `T=1e+04gyr` |
| suppression by `4.6 / 46 / 112` | `verify_theory/vt_t3_trichotomy.json` | `B_measured_v_caveat2` |
| ablation `1.89 / 0.613 / 35.2` | `symproj/ablation.json` | |

### Section 5, Protection by contraction — Figure 4

| number | file | key |
| :-- | :-- | :-- |
| plateau `1.1618617e-5` predicted / measured | `ll_probe/prereg.json`, `ll_probe/results.json` | `P1_DC_plateau_dtheta_discrete` |
| the `eps` sweep, exponents including `Lambda = 0` | `ll_probe/results2.json`, `prereg2.json` | `F7` |
| `+4.57 % / +2.52 %`, `1.700 / 1.800 / -18.350 %` | `ll_probe/results3.json` | `F3c` |
| **erosion threshold `c/Lambda = 0.956`** | `audit_numbers/an3_derived.json` | `S5_erosion_threshold` |

### Section 6, A validation protocol

| number | file | key |
| :-- | :-- | :-- |
| probe 4, short run against full horizon | `p_law/pl_protocol.json` | |
| where probe 4 misses, and the energy channel that fixes it | `p_law_check/pc_defect.json` | `E_energy_increment` |

### Section 7, Application to a family of schemes — Figure 3

| number | file | key |
| :-- | :-- | :-- |
| `113` and `114,091` flops per step | `cost/breakeven.json` | `flops/{boris,hybrid}_step_total` |
| `43.9` and `142.1` microseconds per step | `cost/breakeven.json` | `us_per_step` |
| `417.9x` fewer flops, `64.8x` and `62.2x` | `classical/verdict.json` | `schemes/vps4` |
| the work-precision grid of five classical schemes | `classical/workprecision.json` | `runs` |
| `13.8` and `27.6` Larmor radii at `Omega h = 0.2, 0.1` | `cost/work_precision.json` | `hybrid[1], hybrid[2]` |
| **trajectory advantage `117.8 / 32.7 / unity at 101 / 0.07`**, one-Larmor horizon `22.1 -> 74.1`, factor `3.4` | `horizon/crossover.json` | `gain_vs_horizon`, `*_reaches_1_larmor_at_gyr` |
| **worse by `143` and `1575`; reference check `4.87e-5`; crossing at `3496`; saturation `0.417 / 1.462 / 1.632`** | `audit_numbers/an5_horizon_readout.json` | |
| Boris envelope `2.50e-6` through `1e5` | `horizon/long_runs_summary.json` | `energy_err_max` |

### Section 8, Discussion

| number | file | key |
| :-- | :-- | :-- |
| full width at half maximum `4.6e-4` against `5e-5` | `audit_numbers/an1_resonance_profile.json` | `T=1e+04gyr/fwhm_rel` |
| the invariant resonant defect, amplitude `3.5e-7` | `verify_theory/vt_t3_trichotomy.json` | `kappa_from_dLd_amplitude` |

### `experiments/audit_numbers/` — the gap-closing scripts

These five exist because a number was printed in the paper that no shipped file
held in the form printed. Each writes its own JSON.

| script | what it is the only source of |
| :-- | :-- |
| `an1_resonance_profile.py` | Section 4.5 and Section 8: peak `3.84e-2`, its location `-2.5e-4`, the value `1.92e-2` on `omega_h(0)`, and the full width at half maximum `4.6e-4`. The shipped scan is a 37-point logarithmic grid with no node near the true peak; this is the same model on a 901-point linear grid, and it reproduces every node of the coarse scan to seven digits. |
| `an3_derived.py` | Section 5: the closed-form erosion threshold `c/Lambda = (1-rho)/(Lambda h) = 0.956`, which is a threshold and not a measurement, so no sweep contained it. Section 4.2: the grid horizon `25,033` gyro-orbits. |
| `an4_drift_reading.py` | Section 3.2: the six drift levels, the ratio `2.34` and the four local half-decade slopes of the drift envelope. |
| `an5_horizon_readout.py` | Section 7: the factors `143` and `1575` (reciprocals of a stored gain), the reference-refinement check `4.87e-5`, the signal crossing at `3496` gyro-orbits (read off `horizon/long_runs.npz` by the rule of `horizon/ablate.py`), and the saturation triple. |
| `an6_orphan_data.py` | not a source of any number either: it checks the converse claim, that every committed data file has a script that writes it. 107 files, 97 named directly in a script of their directory, 8 with names the script builds at run time and which are listed explicitly, and **2 orphans plus 1 orphan block** (see the next paragraph). The script exits non-zero if a new orphan appears or if a named orphan block goes missing. |
| `an2_traceability.py` | not a source of any number: it is the sweep that checks the others. It reads `article/main.tex`, finds Sections 3 to 8 by their `\section` commands, and looks for every printed literal among the 16,600 numeric leaves of the bundle's JSON files. Ranks candidates by relative distance and reports the three nearest with that distance, so a coincidence is visible as one. |

`an2` is a screen, not a proof: at two significant digits a printed value has
many coincidental neighbours, and a hit only means a candidate exists. The
authoritative, hand-checked mapping is the one in the tables above.

### The three places where a number the paper prints has no script

Stated plainly, because the alternative is a reader finding them.

1. **`p_law/pl_marginal.json`** — Section 4.2's marginal case `a + H = 0`:
   `0.051` analytic, `0.0398` measured, `Var S / log N = 1.09`. Computed inline
   during the campaign and only its JSON kept; `plan/reports/P_LAW.md` records
   it as one of two "inline" checks. The construction is that of
   `p_law/pl_limits.py` case (iv) at `a = -1/2` — that script does reproduce the
   `0.0377` of Section 4.2's *other* sentence exactly — but the ensemble and
   estimator behind these three particular numbers were not written down, and
   re-deriving them from the surviving description gives `0.0265`, not
   `0.0398`. No script is offered rather than one that returns something else.
2. **`ll_probe/results3.json`, key `F6b_refit_clean_window`** — Section 6's
   measured decay rates `1.7954` and `1.4286`. `ll_probe/fix_f3_f6.py` writes
   the `F6b` block next to it (`1.4926` / `1.2573`, a wider window) but not
   this refit. Reconstruction over the obvious window gives `1.787` / `1.407`.
   Since this revision `fix_f3_f6.py` merges into the existing file instead of
   overwriting it, so re-running the `data` stage no longer deletes the block —
   which it did, once, on the first end-to-end run.
3. **`p_law/pl_fgn_check.json`** — the fGn generator check. Same inline origin,
   but nothing in the paper cites it, and `p_law/pl_core.py` performs the same
   check on a wider grid and stores it in `pl_core.json`.

Everything else printed in Sections 3 to 8 has a script and a file.

## Reproducibility, stated honestly

Every script calls `common.set_global_seed(42)` before any stochastic
operation. Reproducibility was measured rather than assumed, and it is **not
uniform across the training scripts**:

* `training/train_corrector_b4.py` — the corrector of this paper (`DefectNet`)
  — retrains to within **`2.4e-14`** in the weights, and the figures and JSON
  it feeds reproduce to within `1.4e-8`. The residue is dominated by
  catastrophic cancellation in the Boris energy error, a difference of nearly
  equal quantities, and is sensitive to the BLAS build.
* `training/train.py` — the four integrators of the withdrawn version
  (PINN-symplectic, HNN, SympNet, `BorisCorrectorNet`) — **does not reproduce
  across PyTorch versions**: weight differences of up to **`0.93`** have been
  observed, and 24 of the 54 fields of `output_figures/results_summary.json`
  then move by more than 1 %. Nothing in the present paper depends on it. Its
  outputs are kept only because the legacy verification scripts read them, and
  they must be read as a single training run rather than as a property of the
  architectures.

The evidence for both statements is in `verify_i0/`: three independent runs,
their checkpoint and dataset comparisons, and the md5 record of the original
files.

The environment that reproduces the shipped numbers is pinned in
`environment.yml` and `requirements-lock.txt`: Python 3.14.3, NumPy 2.4.2,
SciPy 1.18.1, matplotlib 3.11.1, PyTorch 2.10.0, CPU only, Windows 11 on AMD64.

Two further caveats a reader of the data files should know:

* wall-clock fields (`wall_s`, `us_per_step`, `seconds`) are properties of the
  machine **and of the moment**, and running `repro.py data` overwrites them.
  On the end-to-end run above, which shared the machine with other work, the
  corrector's measured cost came back at 778 microseconds per step against the
  142.1 committed and quoted in Section 7 — a factor of five, with every
  physical quantity in the same files bit-identical. The committed timings are
  the ones taken on a quiet machine, and the values in the files here are those.
  This is exactly why the paper's cost comparison is in flops: `113` and
  `114,091` per step, and the `417.9x` ratio, came back unchanged on the
  loaded machine, because a flop count is not a measurement of a machine.
* `ll_probe` stores `Lambda = 2(alpha - eps) = 2` for the run at `eps = 0`,
  computed from a formula that presumes an attractor. At `eps = 0` the flow has
  no fixed point and the contraction rate is zero; zero is the value used in
  Section 5 and in Figure 4. This is stated in the paper's appendix as well.

## Checkpoints and data

`checkpoints/*.pt` and `training/data/*.npz` are committed here because the
experiments load them and would otherwise need a retraining run before anything
could be reproduced at all. (The four figure scripts do not: they read the
committed JSON and NPZ, which is why the `figures` stage needs no PyTorch.)
**The checkpoints and datasets stay in this repository and are also archived in
the Zenodo deposit.** An earlier revision of this file planned to move them out
on publication; that was reversed, because `checkpoints/` is 1.1 MB over eight
files and a reproduction bundle whose weights live somewhere else stops
reproducing on the day that somewhere else moves. Zenodo is the archival copy,
not a replacement. The DOI goes into `CITATION.cff` and into the paper's data
availability statement, both of which carry a placeholder until the deposit is
made.

## License and citation

MIT, four rightsholders — see `LICENSE`. Citation metadata is `CITATION.cff` in
the repository root, and the Zenodo deposit is described by `.zenodo.json` next
to it.

**Cite the software, not a paper.** The manuscript has not been submitted and no
journal has been chosen, so `CITATION.cff` deliberately carries no
`preferred-citation` block: an earlier revision named a journal and a year, which
asserted something untrue about the record. The block is restored on acceptance,
together with the real journal, year and article DOI. Until the deposit is made,
the repository URL, the Zenodo concept DOI and the release date are placeholders.

## If a figure script is edited

The four figure scripts exist twice: in `article/figures/` next to the
manuscript, and here in `figures_paper/` so that this bundle stands on its own
after it is zipped. The two copies differ in exactly one line, the one that
locates `experiments/`. `python repro.py --check-sync` verifies that this is
still the only difference, and fails if the copies have drifted apart.
