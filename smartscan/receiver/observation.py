"""Receiver-side observation history and the RL state vector.

:class:`ReceiverHistory` accumulates only what the receiver itself has seen
(scan times, detections, non-detections, measured amplitudes) and derives the
features the schedulers may use. It never touches the scenario ground truth.

Bayesian band-activity belief (per band, independent two-state HMM)
------------------------------------------------------------------
prediction over d timesteps with assumed switching rates a = P(OFF->ON), b = P(ON->OFF):
    pi = a / (a + b),  lam = 1 - a - b,  p <- pi + (p - pi) * lam**d
update after a dwell with the receiver's own nominal Pd / Pfa:
    detection     p <- Pd p / (Pd p + Pfa (1 - p))
    no detection  p <- (1-Pd) p / ((1-Pd) p + (1-Pfa)(1 - p))

State vector (length 11 N + 3), all features in [0, 1]:
    per band (10 x N, band-major blocks):
        belief, uncertainty 4p(1-p), staleness, detection ratio, detection EMA,
        scan share, period estimate, period phase ("due") score,
        time since last detection, last measured amplitude
    one-hot previous action (N)
    globals: episode progress, spectrum coverage, previous dwell detected
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from smartscan.config import ObservationConfig, ReceiverConfig
from smartscan.receiver.receiver import Observation

PER_BAND_FEATURES = [
    "belief",
    "uncertainty",
    "staleness",
    "detection_ratio",
    "detection_ema",
    "scan_share",
    "period_estimate",
    "period_due",
    "time_since_detection",
    "last_amplitude",
]
GLOBAL_FEATURES = ["episode_progress", "coverage", "last_detected"]
AMPLITUDE_SCALE_DB = 30.0


def observation_dim(n_bands: int) -> int:
    return len(PER_BAND_FEATURES) * n_bands + n_bands + len(GLOBAL_FEATURES)


def feature_names(n_bands: int) -> list[str]:
    names = [f"{f}[F{b + 1}]" for f in PER_BAND_FEATURES for b in range(n_bands)]
    names += [f"prev_action[F{b + 1}]" for b in range(n_bands)]
    return names + GLOBAL_FEATURES


class ReceiverHistory:
    def __init__(self, n_bands: int, horizon: int, rx: ReceiverConfig, cfg: ObservationConfig):
        self.n = n_bands
        self.horizon = max(1, int(horizon))
        self.rx = rx
        self.cfg = cfg
        self.stale_norm = float(cfg.staleness_horizon or 2 * n_bands)
        self.reset()

    def reset(self) -> None:
        n = self.n
        self.t = 0
        self.belief = np.full(n, self.cfg.belief_prior)
        self.last_scan = np.full(n, -1, dtype=np.int64)
        self.last_detect = np.full(n, -1, dtype=np.int64)
        self.last_result = np.zeros(n, dtype=bool)
        self.n_scans = np.zeros(n, dtype=np.int64)
        self.n_detects = np.zeros(n, dtype=np.int64)
        self.detect_ema = np.zeros(n)
        self.last_amp = np.full(n, np.nan)
        self.onsets = [deque(maxlen=16) for _ in range(n)]
        self.period = np.zeros(n)
        self.prev_action = -1
        self.prev_detected = False
        self.total_dwells = 0

    # ----------------------------------------------------------- belief filter
    def _predict(self, steps: int) -> None:
        if steps <= 0:
            return
        a, b = self.cfg.belief_p_on, self.cfg.belief_p_off
        pi = a / (a + b)
        lam = 1.0 - a - b
        self.belief = pi + (self.belief - pi) * lam**steps

    def advance_to(self, t: int) -> None:
        self._predict(int(t) - self.t)
        self.t = max(self.t, int(t))

    def record(self, obs: Observation) -> None:
        """Incorporate one receiver observation."""
        self.advance_to(obs.t_end)
        pd, pfa = self.rx.pd, self.rx.pfa
        alpha = self.cfg.ema_alpha
        for band, det, amp in zip(obs.bands, obs.detected, obs.amplitude_db):
            p = self.belief[band]
            if det:
                p = pd * p / (pd * p + pfa * (1 - p))
            else:
                p = (1 - pd) * p / ((1 - pd) * p + (1 - pfa) * (1 - p))
            self.belief[band] = float(np.clip(p, 1e-4, 1 - 1e-4))
            if det and (self.n_scans[band] == 0 or not self.last_result[band]):
                self._add_onset(band, obs.t_start)
            self.n_scans[band] += 1
            self.n_detects[band] += int(det)
            self.detect_ema[band] = (1 - alpha) * self.detect_ema[band] + alpha * float(det)
            self.last_scan[band] = obs.t_end
            self.last_result[band] = bool(det)
            if det:
                self.last_detect[band] = obs.t_start
                self.last_amp[band] = amp
        self.prev_action = int(obs.action)
        self.prev_detected = obs.any_detection
        self.total_dwells += 1
        self.advance_to(obs.t_end + 1)

    def _add_onset(self, band: int, t: int) -> None:
        q = self.onsets[band]
        q.append(int(t))
        if len(q) < 3:
            return
        times = list(q)
        gaps = [b - a for a, b in zip(times, times[1:]) if b - a >= 2]
        if not gaps:
            return
        counts = Counter(gaps)
        best_count = max(counts.values())
        if best_count >= 2:
            self.period[band] = float(min(g for g, c in counts.items() if c == best_count))

    # ----------------------------------------------------------- features
    def staleness(self, t: int | None = None) -> np.ndarray:
        t = self.t if t is None else t
        since = np.where(self.last_scan >= 0, t - self.last_scan, self.stale_norm)
        return np.clip(since / self.stale_norm, 0.0, 1.0)

    def period_due(self, t: int | None = None) -> np.ndarray:
        """1 when a band's next periodic onset is predicted now, decaying to 0 half a period away."""
        t = self.t if t is None else t
        due = np.zeros(self.n)
        for b in range(self.n):
            P = self.period[b]
            if P <= 0 or not self.onsets[b]:
                continue
            d = (t - self.onsets[b][-1]) % P
            due[b] = 1.0 - min(d, P - d) / (P / 2.0)
        return np.clip(due, 0.0, 1.0)

    def coverage(self) -> float:
        return float((self.n_scans > 0).mean())

    def vector(self) -> np.ndarray:
        t = self.t
        p = self.belief
        ratio = np.where(self.n_scans > 0, self.n_detects / np.maximum(1, self.n_scans), 0.0)
        share = self.n_scans / max(1, self.n_scans.sum())
        period = np.clip(self.period / self.cfg.max_period, 0.0, 1.0)
        since_det = np.where(self.last_detect >= 0, (t - self.last_detect) / self.horizon, 1.0)
        amp = np.where(np.isnan(self.last_amp), 0.0, self.last_amp / AMPLITUDE_SCALE_DB)
        prev = np.zeros(self.n)
        if self.prev_action >= 0:
            prev[self.prev_action] = 1.0
        parts = [
            p,
            4.0 * p * (1.0 - p),
            self.staleness(t),
            ratio,
            self.detect_ema,
            share,
            period,
            self.period_due(t),
            np.clip(since_det, 0.0, 1.0),
            np.clip(amp, 0.0, 1.0),
            prev,
            np.array([min(1.0, t / self.horizon), self.coverage(), float(self.prev_detected)]),
        ]
        return np.concatenate(parts).astype(np.float32)
