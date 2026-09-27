import numpy as np
import pytest

from smartscan.config import EmitterSpec, RandomEmitterRanges, ScenarioConfig
from smartscan.environment.emitter import realize_emitter
from smartscan.environment.scenario import build_scenario


def _rng(s=0):
    return np.random.default_rng(s)


def test_static_emitter_always_on():
    em = realize_emitter(EmitterSpec(kind="static", band=2), 0, 8, 50, _rng(), RandomEmitterRanges())
    assert (em.trace == 2).all()


def test_periodic_emitter_duty_and_period():
    em = realize_emitter(EmitterSpec(kind="periodic", band=0, period=10, duty=0.3, phase=0), 0, 8, 100,
                         _rng(), RandomEmitterRanges())
    on = em.trace >= 0
    assert on.sum() == 30
    assert on[:3].all() and not on[3:10].any()
    assert on[10:13].all()


def test_agile_emitter_hops_cyclically():
    em = realize_emitter(EmitterSpec(kind="agile", hop_bands=[1, 5, 3, 6], hop_interval=4), 0, 8, 64,
                         _rng(), RandomEmitterRanges())
    hops = list(em.trace[::4])
    assert set(hops) == {1, 5, 3, 6}
    order = [1, 5, 3, 6]
    i = order.index(hops[0])
    assert hops[:4] == [order[(i + k) % 4] for k in range(4)]


def test_intermittent_emitter_bursts():
    em = realize_emitter(EmitterSpec(kind="intermittent", band=4, p_on=0.1, p_off=0.3), 0, 8, 2000,
                         _rng(3), RandomEmitterRanges())
    on = em.trace >= 0
    assert 0.15 < on.mean() < 0.35  # stationary ON share = 0.1 / 0.4 = 0.25
    assert set(em.trace[on]) == {4}


def test_invalid_band_rejected():
    with pytest.raises(ValueError):
        realize_emitter(EmitterSpec(kind="static", band=9), 0, 8, 10, _rng(), RandomEmitterRanges())


def test_scenario_reproducible_and_seed_dependent(default_config):
    a = build_scenario(default_config.scenario, 7)
    b = build_scenario(default_config.scenario, 7)
    c = build_scenario(default_config.scenario, 8)
    assert np.array_equal(a.activity, b.activity)
    assert not np.array_equal(a.activity, c.activity)


def test_transmissions_segmentation(small_config):
    sc = build_scenario(small_config.scenario, 1)
    static = [t for t in sc.transmissions if t.emitter == 0]
    assert len(static) == 1 and static[0].duration == 60
    periodic = [t for t in sc.transmissions if t.emitter == 1]
    assert len(periodic) == 10 and all(t.duration == 3 for t in periodic)
    for t in sc.transmissions:
        assert (sc.burst_id[t.emitter, t.start : t.end + 1, t.band] == t.id).all()


def test_multiple_emitters_ground_truth_union():
    cfg = ScenarioConfig(n_bands=4, episode_length=20, emitters=[
        EmitterSpec(kind="static", band=0), EmitterSpec(kind="static", band=0), EmitterSpec(kind="static", band=2)])
    sc = build_scenario(cfg, 0)
    assert sc.ground_truth[:, 0].all() and sc.ground_truth[:, 2].all()
    assert not sc.ground_truth[:, 1].any()


def test_config_roundtrip(tmp_path, default_config):
    from smartscan.config import ExperimentConfig

    p = tmp_path / "c.yaml"
    default_config.save(p)
    again = ExperimentConfig.load(p)
    assert again.to_dict() == default_config.to_dict()
    j = tmp_path / "c.json"
    default_config.save(j)
    assert ExperimentConfig.load(j).to_dict() == default_config.to_dict()


def test_config_rejects_unknown_keys():
    from smartscan.config import ExperimentConfig

    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"scenario": {"n_bandz": 3}})
