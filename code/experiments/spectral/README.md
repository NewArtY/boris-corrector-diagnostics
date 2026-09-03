# spectral — where the residual sits in frequency, and what it costs

The first author observes that in the band of the slow physical signal,
`f/Omega_c < 0.2`, the residual power of the classical scheme stands about six
orders of magnitude above the corrected ones, and that the corrector pushes
what is left out of that band into narrow lines at `Omega_c` and above. **The
observation is his.** What this directory adds is vps4, a flop budget, a
reference that is not the floor of the measurement, and the arithmetic that
separates an amplitude gain from a redistribution.

Nothing here is retrained and nothing here writes outside this directory. The
checkpoint of the corrector, the flop model of `../classical/schemes.py` and
the JSON of waves W8, W9 and W12 are read, never touched.

## The reference has a closed form

The motion of Section 2 is planar and linear. In the Larmor frame it reduces
to `zeta'' + (B_z^2/4) zeta = 0`, and the substitution
`s = (B_0 tau/2) e^{-t/tau}` turns that into Bessel's equation of order zero.
Every sample of the reference is therefore an independent function evaluation
with no accumulated error at all. The derivation is in the module docstring of
`sw_common.py`.

This matters because it decides the wave. `sw1_reference.py` measures the
error of DOP853 at the manuscript's own tolerance: `6.20e-12` Larmor radii
root mean square over `t = 120`, and `1.26e-10` over `t = 2457.6`. The
appendix reports vps4 at equal cost reaching `6.2e-12` and calls that the
floor of double precision. Re-measured against the closed form it is
`7.19e-14`: **the figure in the appendix is the reference, inflated by a
factor of `86`,** and the equal-cost factor of Section 7 is `1.00e11` rather
than `1.16e9`.

## What is measured

- `sw1_reference.py` — calibration against Table 4 (five schemes, exact to the
  last digit), the error of DOP853 itself, the appendix's equal-cost figure
  re-measured, and whether each in-band power moves when the reference is
  refined. Four of eleven runs are reference-limited, all of them equal-cost.
- `sw2_spectra.py` — the spectra. Five schemes, two residual channels, three
  power figures each, two comparisons, three record lengths, a sweep of the
  band edge and a leakage control under a second window.
- `sw3_ensemble.py` — the same at fixed step over eight initial conditions.
- `sw4_report.py` — tables; writes nothing. Every number in
  `../../../../plan/reports/W13_1_spectral.md` is printed by it.

## Reproducing

    python sw1_reference.py && python sw2_spectra.py && python sw3_ensemble.py
    python sw4_report.py

Each recomputes and exits non-zero if the committed JSON stops reproducing.
`--force` overwrites deliberately. Total runtime is about six minutes on one
core; the closed-form basis at `8193` samples is about half a minute of it.
Requires `mpmath`, which the environment already carries.

## Resolution is the constraint nobody costed

A record of `N` samples at spacing `h` resolves bins spaced `2 pi/(N h)`, so
the number of independent bins strictly inside `f/Omega_c < 0.2` is one fifth
of the number of gyro-orbits in the record. The window of Table 4 is `19.1`
gyro-orbits and carries **four** bins in the band the claim is about. Twenty
bins need `100` gyro-orbits — which is, to the resolution of Section 7, the
horizon at which the corrector's trajectory advantage reaches unity. That is
why three record lengths are run and reported side by side.
