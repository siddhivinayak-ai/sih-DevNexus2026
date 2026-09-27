"""Typed configuration for SmartScan experiments.

A single :class:`ExperimentConfig` fully describes an experiment: the RF scenario,
the receiver, the reward weights, the observation normalisation, the PPO
hyper-parameters and the seed splits. It round-trips through YAML/JSON so every
saved result can be reproduced from its stored ``config.json``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DATA_DIR = PROJECT_ROOT / "data"

EMITTER_KINDS = ("static", "periodic", "intermittent", "agile")


@dataclass
class EmitterSpec:
    """One emitter. Fields left as ``None`` are drawn at random per scenario seed."""

    kind: str = "periodic"  # static | periodic | intermittent | agile
    name: str = ""
    band: int | None = None  # home band (static / periodic / intermittent)
    snr_db: float | None = None  # received SNR at the receiver
    start: int = 0  # first timestep the emitter may be active
    stop: int | None = None  # last timestep (exclusive); None = end of episode
    # periodic
    period: int | None = None  # timesteps between burst onsets
    duty: float | None = None  # fraction of the period the emitter is ON
    phase: int | None = None  # onset offset within the period
    # intermittent (two-state Markov chain)
    p_on: float | None = None  # P(OFF -> ON) per timestep
    p_off: float | None = None  # P(ON -> OFF) per timestep
    # frequency agile
    hop_bands: list[int] | None = None  # cyclic hop sequence; None = random hop set
    hop_interval: int | None = None  # timesteps spent on each hop
    hop_random: bool = False  # True = pick next hop at random from hop set


@dataclass
class RandomEmitterRanges:
    """Parameter ranges used when an EmitterSpec field is ``None``."""

    snr_db: tuple[float, float] = (6.0, 15.0)
    period: tuple[int, int] = (6, 24)
    duty: tuple[float, float] = (0.15, 0.35)
    p_on: tuple[float, float] = (0.02, 0.06)
    p_off: tuple[float, float] = (0.15, 0.40)
    hop_interval: tuple[int, int] = (4, 12)
    hop_set_size: tuple[int, int] = (3, 5)


@dataclass
class TuringSourceConfig:
    """Mode B: build the ground-truth EM environment from Turing PDW files."""

    local_dir: str = "data/turing"  # root that contains <mode>/<split>_<mode>/*.h5
    mode: str = "stare"  # stare = oracle view of the EME (recommended as ground truth)
    split: str = "train"
    max_pulses: int = 200_000  # pulses read per episode window (bounded memory)
    cf_min: float | None = None  # band grid lower edge (None = from data)
    cf_max: float | None = None  # band grid upper edge (None = from data)


@dataclass
class ScenarioConfig:
    name: str = "mixed_8band"
    description: str = ""
    source: str = "synthetic"  # synthetic | turing
    n_bands: int = 8
    f_start_ghz: float = 1.0
    band_spacing_ghz: float = 0.1
    episode_length: int = 256  # timesteps per episode
    warmup_steps: int = 0  # timesteps run before metrics start (all schedulers)
    emitters: list[EmitterSpec] = field(default_factory=list)
    random_emitters: dict[str, int] = field(default_factory=dict)  # kind -> count
    ranges: RandomEmitterRanges = field(default_factory=RandomEmitterRanges)
    turing: TuringSourceConfig = field(default_factory=TuringSourceConfig)

    def band_labels(self) -> list[str]:
        return [f"F{i + 1}" for i in range(self.n_bands)]

    def band_centres_ghz(self) -> list[float]:
        return [round(self.f_start_ghz + i * self.band_spacing_ghz, 6) for i in range(self.n_bands)]


@dataclass
class ReceiverConfig:
    detection_model: str = "fixed"  # fixed (Pd/Pfa constants) | snr (Pd from SNR)
    pd: float = 0.9  # fixed model: P(detect | signal present)
    pfa: float = 0.05  # P(detect | no signal) per dwell
    instantaneous_bands: int = 1  # contiguous bands observed per dwell
    dwell_steps: int = 1  # timesteps integrated per dwell
    retune_steps: int = 0  # blind timesteps when changing band
    amplitude_sigma_db: float = 1.0  # measurement noise on reported amplitude


@dataclass
class RewardConfig:
    """Weights of the per-decision reward (see docs/06_reward_function.md)."""

    detection: float = 0.1  # w_d  per true detection
    new_intercept: float = 1.0  # w_i  per transmission intercepted for the first time
    false_alarm: float = 0.2  # w_f  per false alarm
    miss: float = 0.0  # w_m  per missed detection (signal present, not detected)
    switch_cost: float = 0.0  # w_s  per retune to a different band
    latency: float = 0.1  # w_l  x fraction of emitters with an un-intercepted active transmission
    revisit: float = 0.05  # w_v  per empty re-scan inside the revisit window
    revisit_window: int = 2  # timesteps


@dataclass
class ObservationConfig:
    staleness_horizon: float | None = None  # None = 2 * n_bands
    ema_alpha: float = 0.3
    belief_p_on: float = 0.05  # receiver's assumed P(OFF->ON) in its Bayesian filter
    belief_p_off: float = 0.10  # receiver's assumed P(ON->OFF)
    belief_prior: float = 0.2
    max_period: int = 64


@dataclass
class PPOConfig:
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    ent_coef: float = 0.01
    clip_range: float = 0.2
    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    n_envs: int = 8
    seed: int = 42
    checkpoint_freq: int = 50_000  # timesteps
    eval_freq: int = 25_000  # timesteps between validation runs (all validation seeds)


@dataclass
class SeedSplits:
    """Disjoint scenario-seed ranges (inclusive). Test seeds are never used in training."""

    train: tuple[int, int] = (1, 100)
    validation: tuple[int, int] = (501, 520)
    test: tuple[int, int] = (1001, 1100)

    def seeds(self, split: str) -> list[int]:
        lo, hi = getattr(self, split)
        return list(range(int(lo), int(hi) + 1))


@dataclass
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seeds: SeedSplits = field(default_factory=SeedSplits)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy(self) -> "ExperimentConfig":
        return copy.deepcopy(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _plain(self.to_dict())
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExperimentConfig":
        return _build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        return cls.from_dict(data)


def _plain(obj: Any) -> Any:
    """Convert tuples to lists recursively so YAML stays human readable."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a (possibly partial) dict; unknown keys are rejected."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    defaults = cls()
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        current = getattr(defaults, name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[name] = _build(type(current), value)
        elif name == "emitters":
            kwargs[name] = [e if isinstance(e, EmitterSpec) else _build(EmitterSpec, e) for e in (value or [])]
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def list_config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load a config file; ``None`` returns the built-in defaults (configs/default.yaml if present)."""
    if path is None:
        default = CONFIG_DIR / "default.yaml"
        return ExperimentConfig.load(default) if default.exists() else ExperimentConfig()
    return ExperimentConfig.load(path)
