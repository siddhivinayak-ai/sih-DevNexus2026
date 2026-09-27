import numpy as np
import pytest

from smartscan.environment.smartscan_env import SmartScanEnv
from smartscan.evaluation.experiment import run_episode
from smartscan.schedulers import DecisionContext, OpenLoopScheduler, POMDPScheduler, RandomScheduler, make_scheduler


def _ctx(env, state):
    return DecisionContext(state=state, history=env.history, t=env.t)


def test_open_loop_sweeps_in_order(small_config):
    env = SmartScanEnv(small_config)
    res = run_episode(env, OpenLoopScheduler(), 21)
    assert res.trace["action"].tolist()[:8] == [0, 1, 2, 3, 0, 1, 2, 3]


def test_random_in_bounds_and_seeded(small_config):
    env = SmartScanEnv(small_config)
    a = run_episode(env, RandomScheduler(), 21).trace["action"].tolist()
    b = run_episode(env, RandomScheduler(), 21).trace["action"].tolist()
    assert a == b
    assert min(a) >= 0 and max(a) < 4
    assert len(set(a)) == 4


@pytest.mark.parametrize("key", ["open_loop", "random", "pomdp"])
def test_scheduler_outputs_valid_actions(small_config, key):
    env = SmartScanEnv(small_config)
    sch = make_scheduler(key)
    state, _ = env.reset(options={"scenario_seed": 21})
    sch.reset(env.n_bands, seed=21)
    while not env.done:
        a = sch.select_action(_ctx(env, state))
        assert isinstance(a, int) and 0 <= a < env.n_bands
        state, _, _, _, info = env.step(a)
        sch.update(a, info["observation"])


def test_pomdp_is_adaptive(small_config):
    """POMDP concentrates on bands where it detected activity (it uses observations)."""
    small_config.receiver.pfa = 0.0
    env = SmartScanEnv(small_config)
    tr = run_episode(env, POMDPScheduler(), 21).trace
    share = tr["action"].value_counts(normalize=True)
    # band 1 (static emitter) and band 3 (periodic) dominate over empty bands 0 and 2
    assert share.get(1, 0) + share.get(3, 0) > 0.6


def test_pomdp_legacy_update_rules():
    s = POMDPScheduler()
    s.reset(4)
    from smartscan.receiver.receiver import Observation

    obs = Observation(0, 0, 2, np.array([2]), np.array([True]), np.array([10.0]))
    s.update(2, obs)
    # prediction 0.25*0.9 + 0.75*0.05 = 0.2625, + 0.3 on detection
    assert np.isclose(s.beliefs[2], 0.5625)
    assert np.isclose(s.beliefs[0], 0.2625)


def test_different_actions_give_different_observations(small_config):
    """Scan strategy matters: the observation depends on the band actually scanned."""
    small_config.receiver.pd, small_config.receiver.pfa = 1.0, 0.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 21})
    _, _, _, _, info_a = env.step(1)
    env.reset(options={"scenario_seed": 21})
    _, _, _, _, info_b = env.step(0)
    assert info_a["observation"].any_detection and not info_b["observation"].any_detection
