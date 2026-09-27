"""Turing integration against a tiny HDF5 file written in the dataset's schema.

The fixture is a TEST FILE with made-up pulses; it only checks that the reader
and the binning follow the published layout (data (n,5) + labels + metadata).
"""

import h5py
import numpy as np
import pytest

from smartscan.config import ExperimentConfig, ScenarioConfig, TuringSourceConfig
from smartscan.environment import turing
from smartscan.environment.smartscan_env import SmartScanEnv


@pytest.fixture
def turing_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        d = tmp_path / "stare" / f"{split}_stare"
        d.mkdir(parents=True)
        for i in range(2):
            n = 3000
            labels = rng.integers(0, 3, n).astype(np.int8)
            toa = np.sort(rng.uniform(0, 1e4, n))
            cf = 1000.0 + 100.0 * labels + rng.normal(0, 5, n)
            data = np.stack([toa, cf, rng.uniform(1, 5, n), rng.uniform(0, 360, n), rng.normal(-40, 3, n)], axis=1)
            with h5py.File(d / f"{split}_stare_{i}.h5", "w") as f:
                f.create_dataset("data", data=data.astype(np.float32))
                f.create_dataset("labels", data=labels)
                g = f.create_group("metadata")
                g.attrs["type"] = "synthetic"
    monkeypatch.setenv("SMARTSCAN_TURING_DIR", str(tmp_path))
    return tmp_path


def test_status_without_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTSCAN_TURING_DIR", str(tmp_path / "missing"))
    ok, msg = turing.status()
    assert not ok and "not configured" in msg


def test_discovery_and_reading(turing_root):
    files = turing.find_files()
    assert len(files) == 4
    assert {f.split for f in files} == {"train", "test"} and {f.mode for f in files} == {"stare"}
    info = turing.file_info(files[0].path)
    assert info["n_pulses"] == 3000 and info["has_labels"]
    df = turing.read_pdws(files[0].path, 0, 500, stride=2)
    assert len(df) == 500 and list(df.columns[:5]) == turing.PDW_COLUMNS


def test_turing_scenario_and_env(turing_root):
    cfg = ExperimentConfig(scenario=ScenarioConfig(name="t", source="turing", n_bands=6, episode_length=50,
                                                   turing=TuringSourceConfig(mode="stare", split="train")))
    cfg.seeds.test = (0, 1)
    env = SmartScanEnv(cfg, split="test")
    obs, info = env.reset(options={"scenario_seed": 0})
    sc = env.scenario
    assert "test" in sc.name  # evaluation split draws from the dataset's test files
    assert sc.n_emitters == 3 and sc.activity.shape == (3, 50, 6)
    assert sc.ground_truth.any()
    while not env.done:
        env.step(0)
