"""Scenario realisation: config + seed -> ground-truth electromagnetic environment.

The :class:`Scenario` object is the *ground truth*. It is owned by the RF
environment and the evaluator/reward engine only. Schedulers and the PPO policy
never receive it; they only see receiver observations.

Transmissions
-------------
A *transmission* is a maximal run of consecutive timesteps during which one
emitter is active in one band. A periodic emitter produces one transmission per
ON window, an agile emitter one per hop, a static emitter a single long one.
Interception is scored per transmission (see ``smartscan.evaluation.metrics``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from smartscan.config import EmitterSpec, ScenarioConfig
from smartscan.environment.emitter import RealizedEmitter, realize_emitter


@dataclass
class Transmission:
    id: int
    emitter: int
    band: int
    start: int  # first active timestep
    end: int  # last active timestep (inclusive)

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


@dataclass
class Scenario:
    name: str
    seed: int
    source: str
    n_bands: int
    total_steps: int  # warm-up + episode
    warmup_steps: int
    band_labels: list[str]
    band_centres: list[float]  # GHz (synthetic) or data units (Turing)
    band_unit: str
    emitters: list[RealizedEmitter]
    activity: np.ndarray  # (E, T, N) bool, emitter e active in band n at t
    snr_db: np.ndarray  # (E,)
    notes: list[str] = field(default_factory=list)
    transmissions: list[Transmission] = field(init=False)
    burst_id: np.ndarray = field(init=False)  # (E, T, N) int32, transmission id or -1

    def __post_init__(self) -> None:
        self.transmissions, self.burst_id = _segment_transmissions(self.activity)

    @property
    def n_emitters(self) -> int:
        return int(self.activity.shape[0])

    @property
    def ground_truth(self) -> np.ndarray:
        """(T, N) bool: any emitter active in band n at time t."""
        if self.n_emitters == 0:
            return np.zeros((self.total_steps, self.n_bands), dtype=bool)
        return self.activity.any(axis=0)

    def band_snr(self, t0: int, t1: int) -> np.ndarray:
        """(N,) strongest SNR (dB) of any emitter active in [t0, t1); -inf if none."""
        out = np.full(self.n_bands, -np.inf)
        if self.n_emitters == 0:
            return out
        act = self.activity[:, t0:t1, :].any(axis=1)  # (E, N)
        snr = np.where(act, self.snr_db[:, None], -np.inf)
        return snr.max(axis=0)


def _segment_transmissions(activity: np.ndarray) -> tuple[list[Transmission], np.ndarray]:
    E, T, N = activity.shape
    burst_id = np.full((E, T, N), -1, dtype=np.int32)
    txs: list[Transmission] = []
    for e in range(E):
        for n in range(N):
            col = activity[e, :, n]
            if not col.any():
                continue
            padded = np.concatenate([[False], col, [False]]).astype(np.int8)
            diff = np.diff(padded)
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1) - 1
            for s, en in zip(starts, ends):
                tid = len(txs)
                txs.append(Transmission(tid, e, n, int(s), int(en)))
                burst_id[e, s : en + 1, n] = tid
    return txs, burst_id


def expand_emitter_specs(cfg: ScenarioConfig) -> list[EmitterSpec]:
    """Explicit emitters followed by ``random_emitters`` counts (fully random specs)."""
    specs = list(cfg.emitters)
    for kind, count in (cfg.random_emitters or {}).items():
        for _ in range(int(count)):
            specs.append(EmitterSpec(kind=kind))
    return specs


def build_synthetic_scenario(cfg: ScenarioConfig, seed: int) -> Scenario:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xE417]))
    total = int(cfg.episode_length) + int(cfg.warmup_steps)
    emitters = [
        realize_emitter(spec, i, cfg.n_bands, total, rng, cfg.ranges)
        for i, spec in enumerate(expand_emitter_specs(cfg))
    ]
    activity = np.zeros((len(emitters), total, cfg.n_bands), dtype=bool)
    for e, em in enumerate(emitters):
        t = np.flatnonzero(em.trace >= 0)
        activity[e, t, em.trace[t]] = True
    return Scenario(
        name=cfg.name,
        seed=int(seed),
        source="synthetic",
        n_bands=cfg.n_bands,
        total_steps=total,
        warmup_steps=int(cfg.warmup_steps),
        band_labels=cfg.band_labels(),
        band_centres=cfg.band_centres_ghz(),
        band_unit="GHz",
        emitters=emitters,
        activity=activity,
        snr_db=np.array([em.snr_db for em in emitters], dtype=float),
    )


def build_scenario(cfg: ScenarioConfig, seed: int) -> Scenario:
    """Realise the scenario for one seed (synthetic generator or Turing PDW source)."""
    if cfg.source == "turing":
        from smartscan.environment.turing import build_turing_scenario

        return build_turing_scenario(cfg, seed)
    if cfg.source != "synthetic":
        raise ValueError(f"Unknown scenario source '{cfg.source}'")
    return build_synthetic_scenario(cfg, seed)
