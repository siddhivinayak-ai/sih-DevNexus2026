"""POMDP / belief-state scheduler (baseline carried over from the original app.py).

The algorithm, weights and update rules are unchanged from the original
implementation; it was only wrapped in the common scheduler interface.

Belief update (per decision, heuristic):
    b <- 0.9 b + 0.05 (1 - b)                     (prediction)
    scanned band: detection  b <- min(0.99, b + 0.3)
                  no detect  b <- max(0.01, b - 0.3)
Periodicity tracker: statistical mode of inter-detection intervals (trusted
after the same interval was seen twice).
Policy:
    score = 1.0 b + 0.5 b(1-b) + 0.1 freshness + 1.5 periodicity_hit,   a = argmax(score)
where freshness = clip(time_since_scan / (2N), 0, 1) and periodicity_hit = 1 when the
time since the last detection is within +-1 of a multiple of the estimated period.
"""

from __future__ import annotations

import numpy as np

from smartscan.receiver.receiver import Observation
from smartscan.schedulers.base import BaseScheduler, DecisionContext


class POMDPScheduler(BaseScheduler):
    key = "pomdp"
    label = "POMDP"

    ALPHA, BETA, GAMMA, DELTA = 1.0, 0.5, 0.1, 1.5
    PERSISTENCE, ACTIVATION = 0.9, 0.05

    def reset(self, n_bands: int, seed: int | None = None) -> None:
        super().reset(n_bands, seed)
        self.beliefs = np.ones(n_bands) / n_bands
        self.time_since_last_scan = np.zeros(n_bands)
        self.current_time = 0
        self.hit_history: list[list[int]] = [[] for _ in range(n_bands)]
        self.period_estimates = np.zeros(n_bands)
        self.delta_counts: list[dict[int, int]] = [{} for _ in range(n_bands)]
        self._last_scores: dict = {}

    def _scores(self) -> dict:
        uncertainty = self.beliefs * (1 - self.beliefs)
        freshness = np.clip(self.time_since_last_scan / (self.n_bands * 2), 0, 1.0)
        periodicity = np.zeros(self.n_bands)
        for b in range(self.n_bands):
            if self.period_estimates[b] > 0 and self.hit_history[b]:
                since_hit = self.current_time - self.hit_history[b][-1]
                if since_hit > 0:
                    rem = since_hit % self.period_estimates[b]
                    if rem <= 1 or rem >= self.period_estimates[b] - 1:
                        periodicity[b] = 1.0
        score = self.ALPHA * self.beliefs + self.BETA * uncertainty + self.GAMMA * freshness + self.DELTA * periodicity
        return {
            "belief": self.beliefs.copy(),
            "uncertainty": uncertainty,
            "freshness": freshness,
            "periodicity": periodicity,
            "score": score,
        }

    def select_action(self, ctx: DecisionContext) -> int:
        self._last_scores = self._scores()
        return int(np.argmax(self._last_scores["score"]))

    def update(self, action: int, observation: Observation) -> None:
        detected = observation.any_detection
        self.current_time += 1
        self.time_since_last_scan += 1
        self.time_since_last_scan[action] = 0

        predicted = self.beliefs * self.PERSISTENCE + (1 - self.beliefs) * self.ACTIVATION
        if detected:
            predicted[action] = min(0.99, predicted[action] + 0.3)
            hist = self.hit_history[action]
            hist.append(self.current_time)
            if len(hist) > 10:
                hist.pop(0)
            if len(hist) >= 2:
                delta = hist[-1] - hist[-2]
                counts = self.delta_counts[action]
                counts[delta] = counts.get(delta, 0) + 1
                best = max(counts, key=lambda k: counts[k])
                if counts[best] >= 2:
                    self.period_estimates[action] = best
        else:
            predicted[action] = max(0.01, predicted[action] - 0.3)
        self.beliefs = predicted

    def explain(self) -> dict:
        return {k: np.round(v, 4).tolist() for k, v in self._last_scores.items()}
