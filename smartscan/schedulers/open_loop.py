"""Open-loop sequential sweep: F1 -> F2 -> ... -> FN -> F1 ..., independent of observations."""

from __future__ import annotations

from smartscan.schedulers.base import BaseScheduler, DecisionContext


class OpenLoopScheduler(BaseScheduler):
    key = "open_loop"
    label = "Open Loop"

    def __init__(self, stride: int = 1, start_band: int = 0):
        self.stride = max(1, int(stride))
        self.start_band = int(start_band)

    def reset(self, n_bands: int, seed: int | None = None) -> None:
        super().reset(n_bands, seed)
        self.idx = self.start_band % n_bands

    def select_action(self, ctx: DecisionContext) -> int:
        a = self.idx
        self.idx = (self.idx + self.stride) % self.n_bands
        return int(a)

    def explain(self) -> dict:
        return {"rule": "fixed cyclic sweep", "next_band": int(self.idx)}
