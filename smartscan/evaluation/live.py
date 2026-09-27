"""Step-by-step simulation engine used by the Live Simulation page.

Decision cycle exposed to the user (one STEP):
    1. state      receiver-side features at time t        (``pending_state``)
    2. action     band chosen by the scheduler             (``pending_action`` + ``pending_explain``)
    3. scan       the receiver dwells on that band         (SmartScanEnv.step)
    4. observation detection / no detection (+ amplitude)  (``last.observation``)
    5. reward     from ground-truth events                 (``last.reward``, ``last.terms``)
    6. update     scheduler + receiver history updated
    7. next decision computed for time t'                  (new ``pending_*``)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartscan.config import ExperimentConfig
from smartscan.environment.smartscan_env import SmartScanEnv
from smartscan.evaluation.metrics import episode_metrics, records_frame
from smartscan.schedulers import DecisionContext, make_scheduler


@dataclass
class StepView:
    decision: int
    t: int
    action: int
    state: pd.DataFrame
    explain: dict
    observation: dict
    outcome: str
    reward: float
    terms: dict
    new_intercepts: int


class LiveSimulation:
    def __init__(self, config: ExperimentConfig, scheduler_key: str, seed: int,
                 model_path: str | None = None, split: str = "test"):
        self.config = config
        self.scheduler_key = scheduler_key
        self.seed = int(seed)
        self.env = SmartScanEnv(config, split=split, seeds=[self.seed])
        self.scheduler = make_scheduler(scheduler_key, model_path)
        self.reset()

    # ------------------------------------------------------------ control
    def reset(self) -> None:
        self.state, _ = self.env.reset(options={"scenario_seed": self.seed})
        self.scheduler.reset(self.env.n_bands, seed=self.seed)
        self.last: StepView | None = None
        self.history: list[StepView] = []
        self._decide()

    def step(self) -> bool:
        if self.pending_action is None:
            return False
        a = self.pending_action
        state_df, explain, t = self.pending_state, self.pending_explain, self.env.t
        self.state, reward, _, _, info = self.env.step(a)
        obs = info["observation"]
        self.scheduler.update(a, obs)
        rec = self.env.records[-1]
        self.last = StepView(
            decision=rec.step, t=t, action=a, state=state_df, explain=explain,
            observation=obs.as_dict(), outcome=rec.outcome, reward=reward,
            terms=dict(info["reward_terms"]), new_intercepts=rec.new_intercepts,
        )
        self.history.append(self.last)
        if len(self.history) > 200:
            self.history.pop(0)
        self._decide()
        return True

    def run(self, n: int) -> int:
        done = 0
        for _ in range(int(n)):
            if not self.step():
                break
            done += 1
        return done

    def _decide(self) -> None:
        if self.env.done:
            self.pending_action, self.pending_explain, self.pending_state = None, {}, self.state_table()
            return
        ctx = DecisionContext(state=self.state, history=self.env.history, t=self.env.t)
        self.pending_state = self.state_table()
        self.pending_action = int(self.scheduler.select_action(ctx))
        self.pending_explain = self.scheduler.explain()

    # ------------------------------------------------------------ views
    @property
    def done(self) -> bool:
        return self.env.done

    @property
    def scenario(self):
        return self.env.scenario

    def state_table(self) -> pd.DataFrame:
        """Receiver-side state per band (what the scheduler may use)."""
        h = self.env.history
        t = h.t
        return pd.DataFrame({
            "band": [f"F{b + 1}" for b in range(h.n)],
            "belief": np.round(h.belief, 3),
            "steps_since_scan": [int(t - s) if s >= 0 else None for s in h.last_scan],
            "scans": h.n_scans.astype(int),
            "detections": h.n_detects.astype(int),
            "det_ratio": np.round(np.where(h.n_scans > 0, h.n_detects / np.maximum(1, h.n_scans), 0.0), 3),
            "period_est": [int(p) if p > 0 else None for p in h.period],
            "period_due": np.round(h.period_due(t), 2),
        })

    def trace(self) -> pd.DataFrame:
        return records_frame(self.env.records)

    def beliefs(self) -> tuple[np.ndarray, list[int]]:
        recs = self.env.records
        if not recs:
            return np.zeros((0, self.env.n_bands)), []
        return np.stack([r.belief for r in recs]), [r.t_end for r in recs]

    def metrics(self) -> dict | None:
        return episode_metrics(self.env) if self.env.records else None
