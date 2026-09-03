# spectrum — does the eigenvalue departure generalize?

**It does not, and nothing in this directory reaches the manuscript.** These
scripts exist so that a negative result has evidence behind it rather than an
assertion, and so that the next person who has the same idea can see it was
tested.

The files were swept into commit `a12bf02` by a `git add -A` whose message
describes only the SympMat sort fix. This README is the description they
should have had.

## The question

Wave В10 measured that the trained SympMat matrix of Drimalas et al. (2025)
has eigenvalues off the unit circle while its symplecticity defect sits at
`7e-16`. The tempting reading was general: structure preservation is a
certificate that does not bound the error. Before building anything on that,
two things had to be true — the phenomenon had to appear in more than one
architecture, and the departure had to predict the observed error growth.

## What was measured

The one-step map of each architecture of `../external_arch/` (HNN, G-SympNet,
PINN-symplectic), plus two controls, without retraining anything: the
checkpoints are taken as committed.

- `sp1_calibration.py` — the procedure against cases with a known answer. The
  Boris scheme is *not* symplectic (defect `0.13154`) yet its spectrum lies on
  the unit circle exactly, argument `2 arctan(omega h / 2)` to sixteen digits.
  Fourth-order Runge–Kutta over the exact field leaves the circle *inward* by
  `5.006e-6`, matching its closed form. A procedure that failed either of
  these could not be trusted on a network.
- `sp2_spectra.py` — symplecticity defect and spectrum of each trained map,
  linearized at many points of phase space rather than one.
- `sp3_horizon.py` — the error actually observed over `1e5` steps, against
  what the spectrum predicts.
- `sp4_summarize.py` — tables; writes nothing.

## The answers

**The phenomenon reproduces.** G-SympNet is symplectic to `3.6e-10` and leaves
the unit circle in 24–35 of 64 points, up to `|lambda| = 1.0509`.

**The consequence does not.** SympMat's map is *linear*, so a modulus above
one means the orbit grows like `rho^n`. G-SympNet is nonlinear and every run
stayed bounded over `1e5` steps: at seed 1, `rho = 1.00288` predicts a factor
`1e124.9` and `1.65` was observed. The spectrum of a one-step Jacobian does
not govern a nonlinear orbit.

**The general statement belongs to a weaker property.** What forbids
contraction is volume preservation, `det J = 1` implies `rho >= 1`, not
symplecticity; symplecticity adds only that departures come in reciprocal
pairs. The controls separate the two cleanly: Boris preserves volume and is
not symplectic, and its spectrum stays on the circle; the PINN is neither and
it contracts, `rho < 1` in 76 of 256 measurements, Lyapunov exponents summing
to `-0.17` per step.

**Quantitatively the spectrum predicts nothing.** Of four predictors none is
within three orders of the observed growth rate; `rho` overestimates by
`2.9e3` to `1.9e4`. Rank is another matter: the extrapolated Lyapunov exponent
puts both runs that actually grew at the top of twelve, Spearman `0.937`.

## The one finding worth keeping

G-SympNet's extrapolated leading Lyapunov exponent is `+3.3e-4` to `+4.7e-4`
where the true dynamics is integrable, against `-6.2e-5` for the Boris scheme
on the same procedure. The trained symplectic map is chaotic where the physics
is not. That is a different claim from the one this directory refutes, it has
not been checked against the literature, and it is not made anywhere in the
manuscript.

## Prior art

`../../../../plan/reports/W11_0_symplectic_prior_art.md` establishes that the
underlying statement is classical — Krein 1950, Gel'fand–Lidskii 1955, Moser
1958 — and that the case at hand is classified in Qin et al. 2015, whose first
author wrote `qin2013` in our own bibliography. Peng & Mohseni (2016) prove
boundedness under a definiteness hypothesis that is the same condition as a
definite Krein signature. The reading that survives is not "the certificate
does not certify" but "the certificate is conditional, and the target dynamics
violates the condition on exactly one mode — the guiding centre".

## Reproducing

    python sp1_calibration.py && python sp2_spectra.py && python sp3_horizon.py

Each recomputes and exits non-zero if the committed JSON stops reproducing
(`516`, `27363` and `856` leaves). Deliberately not added to `repro.py`: these
support no manuscript number.
