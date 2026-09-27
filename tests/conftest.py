import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartscan.config import EmitterSpec, ExperimentConfig, ScenarioConfig  # noqa: E402


@pytest.fixture
def small_config() -> ExperimentConfig:
    """4 bands, short episodes, fully specified emitters."""
    cfg = ExperimentConfig()
    cfg.scenario = ScenarioConfig(
        name="test_small",
        n_bands=4,
        episode_length=60,
        emitters=[
            EmitterSpec(kind="static", band=1, snr_db=12.0),
            EmitterSpec(kind="periodic", band=3, period=6, duty=0.5, phase=0, snr_db=12.0),
        ],
    )
    cfg.seeds.train = (1, 5)
    cfg.seeds.validation = (11, 12)
    cfg.seeds.test = (21, 23)
    return cfg


@pytest.fixture
def default_config() -> ExperimentConfig:
    return ExperimentConfig.load(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
