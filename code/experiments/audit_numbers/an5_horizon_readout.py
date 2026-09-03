"""AN5: the Section 7 horizon numbers that no shipped JSON held as a literal.

Four numbers of the "Lengthening the run reverses what is left" paragraph were
measured but never written to a data file in the form the manuscript prints
them, which is exactly what Appendix A.7 of the paper forbids.  Two of them
were reciprocals of a stored gain, one was a crossing that only a time series
holds, and one was a convergence check that lived in a script's stdout.  This
script derives all of them from committed data and writes them out.

  (1) "worse than Boris by a factor of 143 at 1e3 and 1575 at 1e4"
      = 1 / traj_gain_projected of horizon/traj_summary.json.  The stored
        field is the gain (0.00702 and 0.000635); the paper prints its
        reciprocal.

  (2) "the energy error of the corrector crosses the physical signal at 3496
      gyro-orbits"
      = first sample of horizon/long_runs.npz at which the relative energy
        error of paper_tau1.2e5/proj exceeds 1 - exp(-t/tau).  Same rule as
        horizon/ablate.py, which stores the same quantity for the ablation
        cases but not for the shipped run.

  (3) "refining that reference from 150 to 1500 steps per Boris step changes
      the measured Boris error by 4.87e-5 in relative terms"
      = |a - b| / b over the two Boris pos_err_rms entries at 1e3 gyro-orbits
        in horizon/traj_summary.json.  horizon/crossover.py printed it; it is
        recomputed here so that it exists as data.

  (4) "the Boris trajectory error saturates at 0.417, 1.462 and 1.632 Larmor
      radii" -- collected from horizon/validation.json (19.1 gyro-orbits) and
      horizon/traj_summary.json (1e3, 1e4), so the triple sits in one place.

The remaining numbers of that paragraph -- 117.8, 32.7, unity at 101, 0.07 and
the 22.1 -> 74.1 horizon shift -- are since W6.2 stored directly by
horizon/crossover.py in horizon/crossover.json ("gain_vs_horizon"), and are
re-read here only to assert that the two files agree.

Reads only committed files; runs in well under a second.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HOR = os.path.join(HERE, os.pardir, "horizon")
TWO_PI = 2.0 * np.pi

traj = json.load(open(os.path.join(HOR, "traj_summary.json"), encoding="utf-8"))
cross = json.load(open(os.path.join(HOR, "crossover.json"), encoding="utf-8"))
valid = json.load(open(os.path.join(HOR, "validation.json"), encoding="utf-8"))

out = {}

# ------------------------------------------------------------------ (1) ----
out["S7_disadvantage_factors"] = {
    "note": "reciprocal of horizon/traj_summary.json traj_gain_projected",
    "at_1e3_gyro_orbits": 1.0 / traj["H1e+03_ref150x"]["traj_gain_projected"],
    "at_1e4_gyro_orbits": 1.0 / traj["H1e+04_ref150x"]["traj_gain_projected"],
    "manuscript_prints": [143, 1575],
}

# ------------------------------------------------------------------ (2) ----
z = np.load(os.path.join(HOR, "long_runs.npz"))
TAUS = {"paper_tau1.2e5": 1.2e5, "quasistatic_tau1.2e8": 1.2e8}
crossings = {}
for cname, tau in TAUS.items():
    for mode in ("boris", "raw", "proj"):
        t = z["%s/%s/t" % (cname, mode)]
        e = z["%s/%s/e_err" % (cname, mode)]
        idx = np.where(e > 1.0 - np.exp(-t / tau))[0]
        crossings["%s/%s" % (cname, mode)] = (float(t[idx[0]] / TWO_PI)
                                              if len(idx) else None)
out["S7_energy_crosses_physical_signal_gyro_orbits"] = {
    "note": "first sample with |dE|/E0 > 1 - exp(-t/tau); rule of "
            "horizon/ablate.py applied to horizon/long_runs.npz",
    "crossings": crossings,
    "manuscript_prints": 3496,
    "value_used": crossings["paper_tau1.2e5/proj"],
}

# ------------------------------------------------------------------ (3) ----
a = traj["H1e+03_ref150x"]["boris"]["pos_err_rms"]
b = traj["H1e+03_ref1500x"]["boris"]["pos_err_rms"]
out["S7_reference_convergence"] = {
    "boris_pos_err_rms_ref150x": a,
    "boris_pos_err_rms_ref1500x": b,
    "relative_change": abs(a - b) / b,
    "manuscript_prints": 4.87e-5,
}

# ------------------------------------------------------------------ (4) ----
out["S7_boris_trajectory_saturation_larmor"] = {
    "at_19.1_gyro_orbits": valid["published"]["boris"]["pos_err_rms"],
    "at_1e3_gyro_orbits": traj["H1e+03_ref150x"]["boris"]["pos_err_rms"],
    "at_1e4_gyro_orbits": traj["H1e+04_ref150x"]["boris"]["pos_err_rms"],
    "manuscript_prints": [0.417, 1.462, 1.632],
}

# ------------------------------------------------------- consistency -------
gains = {row["gyro_orbits_requested"]: row["traj_gain_projected"]
         for row in cross["gain_vs_horizon"]}
checks = {
    # same horizon, same reference: must agree to round-off
    "crossover_gain_at_19.1_vs_validation_traj_gain":
        [gains[19.1], valid["published"]["traj_gain_projected"], 1e-8],
    # crossover.py samples the last step below 1000 gyro-orbits (999.955),
    # traj.py integrates to 1000 exactly, so a part-per-ten-thousand offset
    # is expected and is what the tolerance allows
    "crossover_gain_at_1000_vs_traj_summary_1e3":
        [gains[1000.0], traj["H1e+03_ref150x"]["traj_gain_projected"], 1e-3],
    "one_larmor_horizon_gain": cross["one_larmor_horizon_gain"],
}
for key, val in list(checks.items()):
    if isinstance(val, list):
        assert abs(val[0] - val[1]) <= val[2] * abs(val[1]), (key, val)
out["consistency_checks"] = checks

json.dump(out, open(os.path.join(HERE, "an5_horizon_readout.json"), "w",
                    encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
