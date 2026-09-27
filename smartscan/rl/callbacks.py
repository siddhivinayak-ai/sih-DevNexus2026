"""Training callbacks: live progress (UI/CLI), cooperative stop, validation-based model selection.

All numbers exposed here are read from the running Stable-Baselines3 model
(Monitor episode statistics, SB3 logger values, validation roll-outs). Nothing
is estimated or synthesised.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from smartscan.config import ExperimentConfig


@dataclass
class TrainingProgress:
    """Shared, thread-safe-enough snapshot of a training run (written by callbacks, read by UI)."""

    total_timesteps: int = 0
    timesteps: int = 0
    episodes: int = 0
    status: str = "idle"  # idle | running | stopping | stopped | finished | error
    message: str = ""
    out_dir: str = ""
    started: float = 0.0
    fps: float = 0.0
    last_episode_reward: float | None = None
    last_episode_length: int | None = None
    episode_rewards: list[tuple[int, float]] = field(default_factory=list)  # (timesteps, reward)
    episode_lengths: list[int] = field(default_factory=list)
    train_logs: list[dict] = field(default_factory=list)  # SB3 logger values per update
    validation: list[dict] = field(default_factory=list)  # (timesteps, mean reward, interception)
    best_validation: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def mean_reward(self, window: int = 20) -> float | None:
        if not self.episode_rewards:
            return None
        return float(np.mean([r for _, r in self.episode_rewards[-window:]]))

    def mean_length(self, window: int = 20) -> float | None:
        if not self.episode_lengths:
            return None
        return float(np.mean(self.episode_lengths[-window:]))

    def request_stop(self) -> None:
        self.stop_event.set()
        if self.status == "running":
            self.status = "stopping"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "message": self.message,
            "timesteps": self.timesteps,
            "total_timesteps": self.total_timesteps,
            "episodes": self.episodes,
            "fps": round(self.fps, 1),
            "last_episode_reward": self.last_episode_reward,
            "mean_episode_reward_20": self.mean_reward(),
            "mean_episode_length_20": self.mean_length(),
            "best_validation_reward": self.best_validation,
            "validation": self.validation,
            "train_logs": self.train_logs[-200:],
            "episode_rewards": self.episode_rewards[-2000:],
        }


class ProgressCallback(BaseCallback):
    """Collects Monitor episode stats and SB3 train/* losses; honours stop requests."""

    def __init__(self, progress: TrainingProgress, write_every_s: float = 5.0):
        super().__init__()
        self.progress = progress
        self.write_every_s = write_every_s
        self._last_write = 0.0

    def _on_training_start(self) -> None:
        p = self.progress
        p.status = "running"
        p.started = time.time()

    def _on_step(self) -> bool:
        p = self.progress
        p.timesteps = int(self.num_timesteps)
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                p.episodes += 1
                p.last_episode_reward = float(ep["r"])
                p.last_episode_length = int(ep["l"])
                p.episode_rewards.append((p.timesteps, float(ep["r"])))
                p.episode_lengths.append(int(ep["l"]))
        elapsed = time.time() - p.started
        p.fps = p.timesteps / elapsed if elapsed > 0 else 0.0
        if p.out_dir and time.time() - self._last_write > self.write_every_s:
            self._write()
        return not p.stop_event.is_set()

    def _on_rollout_end(self) -> None:
        # values recorded by the previous PPO update (train/*), before SB3 dumps them
        vals = {k: float(v) for k, v in self.model.logger.name_to_value.items() if k.startswith("train/")}
        if vals:
            vals["timesteps"] = int(self.num_timesteps)
            self.progress.train_logs.append(vals)

    def _on_training_end(self) -> None:
        self._write()

    def _write(self) -> None:
        self._last_write = time.time()
        try:
            path = Path(self.progress.out_dir) / "progress.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.progress.as_dict(), indent=1), encoding="utf-8")
            tmp.replace(path)  # atomic: readers never see a half-written file
        except OSError:
            pass


class ValidationCallback(BaseCallback):
    """Every ``eval_freq`` timesteps, run the deterministic policy on ALL validation seeds
    (never the test seeds) and keep the best model by mean validation episode reward."""

    def __init__(self, config: ExperimentConfig, progress: TrainingProgress, out_dir: Path, eval_freq: int):
        super().__init__()
        self.config = config
        self.progress = progress
        self.out_dir = Path(out_dir)
        self.eval_freq = max(1, int(eval_freq))
        self._next = self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self.run_validation()
        return True

    def run_validation(self) -> dict:
        from smartscan.environment.smartscan_env import SmartScanEnv
        from smartscan.evaluation.metrics import episode_metrics

        env = SmartScanEnv(self.config, split="validation")
        rewards, rates = [], []
        for seed in self.config.seeds.seeds("validation"):
            obs, _ = env.reset(options={"scenario_seed": seed})
            total = 0.0
            while not env.done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, _, _, _ = env.step(int(action))
                total += r
            rewards.append(total)
            rate = episode_metrics(env)["interception_rate"]
            if rate is not None:
                rates.append(rate)
        row = {
            "timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "interception_rate": float(np.mean(rates)) if rates else None,
            "episodes": len(rewards),
        }
        self.progress.validation.append(row)
        if self.progress.best_validation is None or row["mean_reward"] > self.progress.best_validation:
            self.progress.best_validation = row["mean_reward"]
            self.model.save(str(self.out_dir / "best_model.zip"))
            row["saved_best"] = True
        self.logger.record("validation/mean_reward", row["mean_reward"])
        if row["interception_rate"] is not None:
            self.logger.record("validation/interception_rate", row["interception_rate"])
        return row
