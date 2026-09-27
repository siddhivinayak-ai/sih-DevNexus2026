import numpy as np

from smartscan.config import RewardConfig
from smartscan.environment.reward import RewardEvents, compute_reward
from smartscan.environment.smartscan_env import SmartScanEnv


def test_reward_equation():
    w = RewardConfig(detection=0.1, new_intercept=1.0, false_alarm=0.2, miss=0.3, switch_cost=0.05,
                     latency=0.1, revisit=0.05)
    ev = RewardEvents(detections=1, new_intercepts=2, false_alarms=1, misses=1, switched=1, latency=0.5,
                      empty_revisits=1)
    r, terms = compute_reward(ev, w)
    expected = 0.1 + 2.0 - 0.2 - 0.3 - 0.05 - 0.05 - 0.05
    assert abs(r - expected) < 1e-12
    assert abs(sum(terms.values()) - r) < 1e-12


def test_env_reward_is_sum_of_terms_and_cumulative(small_config):
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    total = 0.0
    while not env.done:
        _, r, _, _, info = env.step(env.step_count % 4)
        assert abs(r - sum(info["reward_terms"].values())) < 1e-12
        total += r
    assert abs(total - env.cumulative_reward) < 1e-9


def test_first_detection_is_new_intercept(small_config):
    small_config.receiver.pd = 1.0
    small_config.receiver.pfa = 0.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    env.step(1)
    rec = env.records[-1]
    assert rec.outcome == "DETECTION" and rec.new_intercepts == 1
    assert np.isclose(rec.terms["new_intercept"], small_config.reward.new_intercept)
    env.step(1)
    assert env.records[-1].new_intercepts == 0  # same static transmission
    assert env.records[-1].terms["revisit"] == 0.0  # a detection is not an empty revisit


def test_false_alarm_penalised(small_config):
    small_config.receiver.pfa = 1.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    env.step(0)  # band 0 has no emitter
    rec = env.records[-1]
    assert rec.outcome == "FALSE_ALARM"
    assert np.isclose(rec.terms["false_alarm"], -small_config.reward.false_alarm)


def test_miss_recorded(small_config):
    small_config.receiver.pd = 0.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    env.step(1)
    assert env.records[-1].outcome == "MISS"
    assert env.records[-1].new_intercepts == 0


def test_empty_revisit_penalised(small_config):
    small_config.receiver.pfa = 0.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    env.step(0)
    env.step(0)
    assert np.isclose(env.records[0].terms["revisit"], 0.0)
    assert np.isclose(env.records[-1].terms["revisit"], -small_config.reward.revisit)


def test_latency_counts_uninterceped_emitters(small_config):
    small_config.receiver.pfa = 0.0
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    env.step(0)  # t=0: static (band 1) and periodic (band 3, phase 0) both active, neither intercepted
    assert np.isclose(env.records[-1].terms["latency"], -small_config.reward.latency * 1.0)
