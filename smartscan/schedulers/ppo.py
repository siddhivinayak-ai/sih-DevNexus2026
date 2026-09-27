"""PPO scheduler: a trained Stable-Baselines3 policy pi(a | s) used as a scan scheduler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smartscan.receiver.observation import observation_dim
from smartscan.schedulers.base import BaseScheduler, DecisionContext


class PPONotAvailable(RuntimeError):
    pass


def read_model_metadata(model_path: str | Path) -> dict:
    meta = Path(model_path).parent / "metadata.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


class PPOScheduler(BaseScheduler):
    key = "ppo"
    label = "PPO"

    def __init__(self, model_path: str | Path, deterministic: bool = True, device: str = "cpu"):
        from stable_baselines3 import PPO

        path = Path(model_path)
        if not path.exists():
            raise PPONotAvailable(f"PPO model not trained (no file at {path})")
        self.model_path = path
        self.model = PPO.load(str(path), device=device)
        self.deterministic = deterministic
        self.metadata = read_model_metadata(path)
        self.obs_dim = int(self.model.observation_space.shape[0])
        self._probs: np.ndarray | None = None

    def reset(self, n_bands: int, seed: int | None = None) -> None:
        super().reset(n_bands, seed)
        if observation_dim(n_bands) != self.obs_dim or self.model.action_space.n != n_bands:
            raise PPONotAvailable(
                f"PPO model was trained for {self.model.action_space.n} bands; scenario has {n_bands}. "
                "Train a model for this scenario."
            )

    def action_probabilities(self, state: np.ndarray) -> np.ndarray:
        import torch

        obs_t, _ = self.model.policy.obs_to_tensor(state)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            return dist.distribution.probs.cpu().numpy()[0]

    def select_action(self, ctx: DecisionContext) -> int:
        self._probs = self.action_probabilities(ctx.state)
        action, _ = self.model.predict(ctx.state, deterministic=self.deterministic)
        return int(action)

    def explain(self) -> dict:
        if self._probs is None:
            return {}
        return {"probabilities": np.round(self._probs, 4).tolist(), "deterministic": self.deterministic}
