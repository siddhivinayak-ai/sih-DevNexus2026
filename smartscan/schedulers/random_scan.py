"""Uniform random band selection (seeded, independent of observations)."""

from __future__ import annotations

import numpy as np

from smartscan.schedulers.base import BaseScheduler, DecisionContext


class RandomScheduler(BaseScheduler):
    key = "random"
    label = "Random"

    def reset(self, n_bands: int, seed: int | None = None) -> None:
        super().reset(n_bands, seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([0 if seed is None else int(seed), 0xA11]))

    def select_action(self, ctx: DecisionContext) -> int:
        return int(self.rng.integers(0, self.n_bands))

    def explain(self) -> dict:
        return {"rule": "uniform random", "probabilities": [1.0 / self.n_bands] * self.n_bands}
