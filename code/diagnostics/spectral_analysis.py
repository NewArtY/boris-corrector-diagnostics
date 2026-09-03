"""
spectral_analysis.py
======================
Spectral (FFT / power spectral density) diagnostics of velocity errors,
used to characterize whether numerical error growth of a given integrator
is dominated by secular (low-frequency) drift or by broadband
high-frequency noise -- complementing the energy-drift metrics with a
frequency-domain view, as mentioned in Section 2.3 ("спектры ошибок").
"""

import numpy as np


def compute_psd(signal, dt):
    """Compute the one-sided power spectral density of a real signal via FFT.

    Returns (freqs, psd) with freqs in cycles per unit time (Hz-equivalent).
    """
    n = len(signal)
    signal = signal - np.mean(signal)
    window = np.hanning(n)
    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    psd = (np.abs(spectrum) ** 2) / (np.sum(window ** 2) / dt)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, psd


def velocity_error_spectrum(v_num, v_ref, dt):
    """PSD of the velocity-error magnitude time series."""
    err = np.linalg.norm(v_num - v_ref, axis=-1)
    return compute_psd(err, dt)


def dominant_frequency(freqs, psd, f_min=1e-6):
    mask = freqs > f_min
    if not np.any(mask):
        return 0.0
    idx = np.argmax(psd[mask])
    return freqs[mask][idx]
