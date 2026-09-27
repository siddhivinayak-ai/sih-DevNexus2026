"""Detection models.

fixed
    P(detect | signal present) = Pd,  P(detect | no signal) = Pfa  (constants).
snr
    Neyman-Pearson detector for a known signal in white Gaussian noise:
        tau = Q^-1(Pfa)
        Pd  = Q(tau - sqrt(2 * SNR))          (SNR linear, per dwell)
    Idealised textbook model; parameters are simulation values, not hardware claims.

Reported amplitude (receiver measurement, dB):
    signal present and detected : SNR_dB + N(0, sigma^2)
    false alarm                 : 10 log10(E) with E the square-law noise energy
                                  conditioned on exceeding the threshold
                                  -ln(Pfa):  E = -ln(Pfa) + Exp(1)
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

_N = NormalDist()


def q_function(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 - np.vectorize(_N.cdf, otypes=[float])(x)


def q_inverse(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return _N.inv_cdf(1.0 - p)


def pd_from_snr(snr_db: np.ndarray | float, pfa: float) -> np.ndarray:
    snr_lin = 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)
    return q_function(q_inverse(pfa) - np.sqrt(2.0 * snr_lin))


def detection_probability(present: np.ndarray, snr_db: np.ndarray, model: str, pd: float, pfa: float) -> np.ndarray:
    """Per-band P(detect) given presence and strongest SNR in the band."""
    present = np.asarray(present, dtype=bool)
    if model == "fixed":
        p_sig = np.full(present.shape, float(pd))
    elif model == "snr":
        p_sig = pd_from_snr(np.where(present, snr_db, 0.0), pfa)
    else:
        raise ValueError(f"Unknown detection model '{model}'")
    return np.where(present, p_sig, float(pfa))


def measured_amplitude(
    detected: np.ndarray,
    present: np.ndarray,
    snr_db: np.ndarray,
    pfa: float,
    sigma_db: float,
    gauss: np.ndarray,
    uniform: np.ndarray,
) -> np.ndarray:
    """Amplitude (dB) reported for each detection; NaN where nothing was detected."""
    tau = -np.log(max(float(pfa), 1e-12))
    noise_energy = tau - np.log(np.clip(uniform, 1e-12, 1.0))
    fa_db = 10.0 * np.log10(noise_energy)
    sig_db = np.where(present, snr_db, 0.0) + sigma_db * gauss
    amp = np.where(present, sig_db, fa_db)
    return np.where(detected, amp, np.nan)
