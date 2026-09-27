"""Common scheduler interface.

Every scheduler is driven by the same loop (``smartscan.evaluation.experiment``):

    ctx    = DecisionContext(state vector, receiver history, time)
    action = scheduler.select_action(ctx)
    obs    = receiver dwell on ``action``          (inside SmartScanEnv.step)
    scheduler.update(action, obs)

A scheduler receives only receiver-side information: the state vector and the
``ReceiverHistory`` built from its own past observations, and each new
``Observation``. It never sees the scenario ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from smartscan.receiver.observation import ReceiverHistory
from smartscan.receiver.receiver import Observation


@dataclass
class DecisionContext:
    state: np.ndarray  # RL state vector (receiver-observable features)
    history: ReceiverHistory  # read-only view of the receiver-side history
    t: int  # current timestep


class BaseScheduler(ABC):
    key: str = "base"
    label: str = "Base"

    def reset(self, n_bands: int, seed: int | None = None) -> None:
        self.n_bands = int(n_bands)

    @abstractmethod
    def select_action(self, ctx: DecisionContext) -> int:
        """Return the band to scan next (0 .. n_bands-1)."""

    def update(self, action: int, observation: Observation) -> None:
        """Receive the receiver observation that resulted from ``action``."""

    def explain(self) -> dict:
        """Per-band decision quantities of the last ``select_action`` (for step-by-step display)."""
        return {}
