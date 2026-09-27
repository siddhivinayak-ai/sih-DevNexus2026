"""Gymnasium environment: one decision = choose the band the receiver scans next.

    state s_t  : ReceiverHistory.vector()  (receiver-observable history only)
    action a_t : band index in {0 .. N-1}; the receiver really tunes there
    obs   o_t  : detection / no detection (+ amplitude) in the scanned band(s)
    reward r_t : smartscan.environment.reward (from ground-truth events)
    episode    : until the scenario timeline (warm-up + episode_length) is exhausted

Ground truth (``self.scenario``) is used only to (1) produce the receiver
observation through the detector model and (2) score reward/metrics. It is not
included in the observation vector and schedulers receive only ``Observation``.

The same class serves PPO training (seed drawn at random from the training
split) and the evaluation of *all* schedulers (explicit ``scenario_seed``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartscan.config import ExperimentConfig
from smartscan.environment.reward import RewardEvents, compute_reward
from smartscan.environment.scenario import Scenario, build_scenario
from smartscan.receiver.observation import ReceiverHistory, observation_dim
from smartscan.receiver.receiver import Observation, Receiver, ReceiverNoise


@dataclass
class StepRecord:
    """Evaluator-side log of one decision (contains ground truth; never fed to schedulers)."""

    step: int
    t_start: int
    t_end: int
    action: int
    bands: list[int]
    detected: bool
    signal_present: bool
    outcome: str  # DETECTION | FALSE_ALARM | MISS | NO_SIGNAL
    detections: int
    false_alarms: int
    misses: int
    correct_rejections: int
    new_intercepts: int
    retuned: bool
    reward: float
    terms: dict[str, float]
    warmup: bool
    amplitude_db: float | None
    belief: np.ndarray = field(repr=False)


class SmartScanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ExperimentConfig,
        split: str = "train",
        seeds: list[int] | None = None,
        sequential: bool = False,
        keep_records: bool = True,
    ):
        super().__init__()
        self.config = config
        self.n_bands = int(config.scenario.n_bands)
        self.split = split
        self.seeds = list(seeds) if seeds is not None else config.seeds.seeds(split)
        self._scenario_cfg = config.scenario
        if config.scenario.source == "turing":
            # Turing mode: train/validation/test episodes come from the dataset's own disjoint splits
            self._scenario_cfg = copy.deepcopy(config.scenario)
            self._scenario_cfg.turing.split = split
        if not self.seeds:
            raise ValueError("SmartScanEnv needs at least one scenario seed")
        self.sequential = sequential
        self.keep_records = keep_records
        self._seed_cursor = 0
        self._scenario_cache: dict[int, Scenario] = {}
        self.receiver = Receiver(config.receiver, self.n_bands)
        self.action_space = spaces.Discrete(self.n_bands)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(observation_dim(self.n_bands),), dtype=np.float32)
        self.scenario: Scenario | None = None

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        scenario_seed = options.get("scenario_seed")
        if scenario_seed is None:
            if self.sequential:
                scenario_seed = self.seeds[self._seed_cursor % len(self.seeds)]
                self._seed_cursor += 1
            else:
                scenario_seed = int(self.np_random.choice(self.seeds))
        self.scenario_seed = int(scenario_seed)
        self.scenario = self._get_scenario(self.scenario_seed)
        sc = self.scenario
        self.noise = ReceiverNoise.draw(self.scenario_seed, sc.total_steps, sc.n_bands)
        self.receiver.reset()
        self.history = ReceiverHistory(sc.n_bands, sc.total_steps, self.config.receiver, self.config.observation)
        self.t = 0
        self.step_count = 0
        n_tx = len(sc.transmissions)
        self.intercepted = np.zeros(n_tx, dtype=bool)
        self.intercept_time = np.full(n_tx, -1, dtype=np.int64)
        self.cumulative_reward = 0.0
        self.records: list[StepRecord] = []
        self.last_observation: Observation | None = None
        return self.history.vector(), self._info()

    def step(self, action):
        if self.scenario is None:
            raise RuntimeError("call reset() before step()")
        if self.t >= self.scenario.total_steps:
            raise RuntimeError("episode finished; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        sc, w = self.scenario, self.config.reward

        # empty-revisit check uses the receiver history *before* this dwell
        prev_scan = self.history.last_scan.copy()

        obs, truth = self.receiver.dwell(action, self.t, sc, self.noise)
        det, present = obs.detected, truth.present
        true_det = det & present
        ev = RewardEvents(
            detections=int(true_det.sum()),
            false_alarms=int((det & ~present).sum()),
            misses=int((~det & present).sum()),
            switched=int(obs.retuned),
        )
        # interception: every not-yet-intercepted transmission active in a truly detected band
        for band in obs.bands[true_det]:
            ids = sc.burst_id[:, obs.t_start : obs.t_end + 1, band]
            for tid in np.unique(ids[ids >= 0]):
                if not self.intercepted[tid]:
                    self.intercepted[tid] = True
                    self.intercept_time[tid] = max(obs.t_start, sc.transmissions[tid].start)
                    ev.new_intercepts += 1
        # latency: emitters with an active, un-intercepted transmission at the end of the dwell
        if sc.n_emitters and sc.transmissions:
            ids = sc.burst_id[:, obs.t_end, :]
            waiting = (ids >= 0) & ~self.intercepted[np.where(ids >= 0, ids, 0)]
            ev.latency = float(waiting.any(axis=1).sum()) / sc.n_emitters
        recent = (prev_scan[obs.bands] >= 0) & (obs.t_start - prev_scan[obs.bands] <= w.revisit_window)
        ev.empty_revisits = int((recent & ~det).sum())

        reward, terms = compute_reward(ev, w)
        self.history.record(obs)
        self.t = obs.t_end + 1
        self.cumulative_reward += reward
        self.last_observation = obs
        truncated = self.t >= sc.total_steps

        if self.keep_records:
            if ev.detections:
                outcome = "DETECTION"
            elif ev.false_alarms:
                outcome = "FALSE_ALARM"
            elif ev.misses:
                outcome = "MISS"
            else:
                outcome = "NO_SIGNAL"
            amps = obs.amplitude_db[~np.isnan(obs.amplitude_db)]
            self.records.append(
                StepRecord(
                    step=self.step_count,
                    t_start=obs.t_start,
                    t_end=obs.t_end,
                    action=action,
                    bands=[int(b) for b in obs.bands],
                    detected=obs.any_detection,
                    signal_present=bool(present.any()),
                    outcome=outcome,
                    detections=ev.detections,
                    false_alarms=ev.false_alarms,
                    misses=ev.misses,
                    correct_rejections=int((~det & ~present).sum()),
                    new_intercepts=ev.new_intercepts,
                    retuned=obs.retuned,
                    reward=reward,
                    terms=terms,
                    warmup=obs.t_start < sc.warmup_steps,
                    amplitude_db=float(amps.max()) if amps.size else None,
                    belief=self.history.belief.copy(),
                )
            )
        self.step_count += 1
        info = self._info()
        info["observation"] = obs
        info["reward_terms"] = terms
        return self.history.vector(), reward, False, truncated, info

    # ------------------------------------------------------------ helpers
    def _get_scenario(self, seed: int) -> Scenario:
        if seed not in self._scenario_cache:
            if len(self._scenario_cache) > 256:
                self._scenario_cache.clear()
            self._scenario_cache[seed] = build_scenario(self._scenario_cfg, seed)
        return self._scenario_cache[seed]

    def _info(self) -> dict:
        return {"t": self.t, "scenario_seed": self.scenario_seed, "step": self.step_count}

    @property
    def done(self) -> bool:
        return self.scenario is not None and self.t >= self.scenario.total_steps
