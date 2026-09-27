"""Scan schedulers behind the common :class:`BaseScheduler` interface."""

from __future__ import annotations

from pathlib import Path

from smartscan.schedulers.base import BaseScheduler, DecisionContext
from smartscan.schedulers.open_loop import OpenLoopScheduler
from smartscan.schedulers.pomdp import POMDPScheduler
from smartscan.schedulers.random_scan import RandomScheduler

SCHEDULER_LABELS = {"open_loop": "Open Loop", "random": "Random", "pomdp": "POMDP", "ppo": "PPO"}


def make_scheduler(key: str, model_path: str | Path | None = None) -> BaseScheduler:
    if key == "open_loop":
        return OpenLoopScheduler()
    if key == "random":
        return RandomScheduler()
    if key == "pomdp":
        return POMDPScheduler()
    if key == "ppo":
        from smartscan.schedulers.ppo import PPONotAvailable, PPOScheduler

        if model_path is None:
            raise PPONotAvailable("PPO model not trained.")
        return PPOScheduler(model_path)
    raise ValueError(f"Unknown scheduler '{key}'")


__all__ = [
    "BaseScheduler",
    "DecisionContext",
    "OpenLoopScheduler",
    "RandomScheduler",
    "POMDPScheduler",
    "make_scheduler",
    "SCHEDULER_LABELS",
]
