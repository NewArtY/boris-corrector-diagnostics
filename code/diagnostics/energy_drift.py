"""
energy_drift.py
=================
Energy-conservation and RMS-error diagnostics, implementing the metrics of
Eq. (13) of the Methods section:

    delta_r  = |r_num - r_ref|                  (position error)
    delta_v  = |v_num - v_ref|                  (velocity error)
    delta_E  = |E_num - E0| / |E0|               (relative energy drift)

For purely magnetic configurations, kinetic energy E = 0.5*m*|v|^2, so the
relative energy drift reduces to (|v|^2 - |v0|^2) / |v0|^2.
"""

import numpy as np


def kinetic_energy(v, m=1.0):
    v = np.atleast_2d(v)
    return 0.5 * m * np.sum(v ** 2, axis=-1)


def relative_energy_drift(v, m=1.0):
    """Return delta_E(t) = (E(t) - E0) / E0 for a velocity trajectory."""
    E = kinetic_energy(v, m=m)
    E0 = E[0]
    if E0 == 0:
        E0 = 1e-12
    return (E - E0) / E0


def mean_abs_energy_drift(v, m=1.0):
    """<|dE/E0|> time-averaged over the whole trajectory."""
    drift = relative_energy_drift(v, m=m)
    return np.mean(np.abs(drift))


def rms_energy_error(v, m=1.0):
    """RMS of the relative energy drift over the trajectory."""
    drift = relative_energy_drift(v, m=m)
    return np.sqrt(np.mean(drift ** 2))


def rms_position_error(r_num, r_ref):
    return np.sqrt(np.mean(np.sum((r_num - r_ref) ** 2, axis=-1)))


def rms_velocity_error(v_num, v_ref):
    return np.sqrt(np.mean(np.sum((v_num - v_ref) ** 2, axis=-1)))


def position_error_series(r_num, r_ref):
    return np.linalg.norm(r_num - r_ref, axis=-1)


def velocity_error_series(v_num, v_ref):
    return np.linalg.norm(v_num - v_ref, axis=-1)


def full_error_report(r_num, v_num, r_ref=None, v_ref=None, m=1.0):
    """Compute a full dictionary of diagnostic metrics for one trajectory."""
    report = {
        "mean_abs_dE_E0": float(mean_abs_energy_drift(v_num, m=m)),
        "rms_energy_error": float(rms_energy_error(v_num, m=m)),
    }
    if r_ref is not None and v_ref is not None:
        report["rms_position_error"] = float(rms_position_error(r_num, r_ref))
        report["rms_velocity_error"] = float(rms_velocity_error(v_num, v_ref))
    return report
