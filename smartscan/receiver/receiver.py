"""Limited-bandwidth scanning receiver.

The receiver observes only ``instantaneous_bands`` contiguous bands per dwell,
starting at the commanded band (clipped so the window fits in the spectrum).
A dwell integrates ``dwell_steps`` timesteps; changing band first costs
``retune_steps`` blind timesteps. A band counts as *signal present* during the
dwell if any emitter transmits in it at any integrated timestep.

Randomness uses *common random numbers*: the uniform/Gaussian draws are fixed
per (timestep, band) for a scenario seed, so two schedulers that scan the same
band at the same time get the same detector outcome. Differences between
schedulers are therefore caused by their decisions, not by noise luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartscan.config import ReceiverConfig
from smartscan.receiver.detector import detection_probability, measured_amplitude


@dataclass(eq=False)
class Observation:
    """What the receiver reports after one dwell. Contains no ground truth."""

    t_start: int  # first integrated timestep
    t_end: int  # last integrated timestep (inclusive)
    action: int  # commanded band
    bands: np.ndarray  # observed band indices
    detected: np.ndarray  # bool per observed band
    amplitude_db: np.ndarray  # measured amplitude per observed band (NaN if no detection)
    retuned: bool = False

    @property
    def any_detection(self) -> bool:
        return bool(self.detected.any())

    def __eq__(self, other) -> bool:
        return isinstance(other, Observation) and self.as_dict() == other.as_dict()

    def as_dict(self) -> dict:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "action": self.action,
            "bands": [int(b) for b in self.bands],
            "detected": [bool(d) for d in self.detected],
            "amplitude_db": [None if np.isnan(a) else round(float(a), 2) for a in self.amplitude_db],
            "retuned": self.retuned,
        }


@dataclass
class ReceiverNoise:
    """Pre-drawn random numbers per (timestep, band) for one scenario seed."""

    u_detect: np.ndarray
    gauss: np.ndarray
    u_amp: np.ndarray

    @classmethod
    def draw(cls, seed: int, total_steps: int, n_bands: int) -> "ReceiverNoise":
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5EC7]))
        shape = (total_steps, n_bands)
        return cls(rng.random(shape), rng.standard_normal(shape), rng.random(shape))


@dataclass
class DwellTruth:
    """Ground-truth side of a dwell, used only by the reward engine / evaluator."""

    present: np.ndarray  # bool per observed band
    snr_db: np.ndarray
    p_detect: np.ndarray
    extra: dict = field(default_factory=dict)


class Receiver:
    def __init__(self, cfg: ReceiverConfig, n_bands: int):
        if not 1 <= cfg.instantaneous_bands <= n_bands:
            raise ValueError("instantaneous_bands must be in 1..n_bands")
        if cfg.dwell_steps < 1 or cfg.retune_steps < 0:
            raise ValueError("dwell_steps >= 1 and retune_steps >= 0 required")
        self.cfg = cfg
        self.n_bands = n_bands
        self.current_band: int | None = None

    def reset(self) -> None:
        self.current_band = None

    def window(self, action: int) -> np.ndarray:
        w = self.cfg.instantaneous_bands
        start = int(min(max(0, action), self.n_bands - w))
        return np.arange(start, start + w)

    def dwell(self, action: int, t: int, scenario, noise: ReceiverNoise) -> tuple[Observation, DwellTruth]:
        """Tune to ``action`` at time ``t`` and dwell. Returns (observation, truth)."""
        if not 0 <= action < self.n_bands:
            raise ValueError(f"action {action} outside 0..{self.n_bands - 1}")
        retuned = self.current_band is not None and action != self.current_band
        t_start = t + (self.cfg.retune_steps if retuned else 0)
        t_start = min(t_start, scenario.total_steps - 1)
        t_end = min(t_start + self.cfg.dwell_steps, scenario.total_steps) - 1
        bands = self.window(action)

        snr_all = scenario.band_snr(t_start, t_end + 1)
        snr = snr_all[bands]
        present = np.isfinite(snr)
        p_det = detection_probability(present, snr, self.cfg.detection_model, self.cfg.pd, self.cfg.pfa)
        detected = noise.u_detect[t_start, bands] < p_det
        amp = measured_amplitude(
            detected, present, np.where(present, snr, 0.0), self.cfg.pfa,
            self.cfg.amplitude_sigma_db, noise.gauss[t_start, bands], noise.u_amp[t_start, bands],
        )
        self.current_band = int(action)
        obs = Observation(t_start, t_end, int(action), bands, detected, amp, retuned)
        return obs, DwellTruth(present=present, snr_db=snr, p_detect=p_det)
