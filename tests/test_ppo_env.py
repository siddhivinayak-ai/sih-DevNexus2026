import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from smartscan.environment.smartscan_env import SmartScanEnv
from smartscan.receiver.observation import observation_dim


def test_gymnasium_api_compliance(small_config):
    check_env(SmartScanEnv(small_config), skip_render_check=True)


def test_reset_and_step_shapes(small_config):
    env = SmartScanEnv(small_config)
    obs, info = env.reset(seed=0)
    assert obs.shape == (observation_dim(4),) and env.observation_space.contains(obs)
    assert info["scenario_seed"] in small_config.seeds.seeds("train")
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and isinstance(r, float) and term is False


def test_episode_length(small_config):
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    n = 0
    trunc = False
    while not trunc:
        _, _, _, trunc, _ = env.step(0)
        n += 1
    assert n == small_config.scenario.episode_length
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_action_rejected(small_config):
    env = SmartScanEnv(small_config)
    env.reset(options={"scenario_seed": 1})
    with pytest.raises(ValueError):
        env.step(4)


def test_training_env_uses_only_training_seeds(small_config):
    env = SmartScanEnv(small_config, split="train")
    seen = set()
    for i in range(40):
        _, info = env.reset(seed=i)
        seen.add(info["scenario_seed"])
    assert seen <= set(small_config.seeds.seeds("train"))
    assert not seen & set(small_config.seeds.seeds("test"))


def test_observation_does_not_contain_ground_truth(small_config):
    """Two scenarios with different ground truth but identical receiver outcomes give identical states."""
    small_config.receiver.pd, small_config.receiver.pfa = 1.0, 0.0
    env = SmartScanEnv(small_config)
    obs_a, _ = env.reset(options={"scenario_seed": 1})
    other = small_config.copy()
    other.scenario.emitters[1].band = 2  # move the periodic emitter; band 0 stays empty in both
    env_b = SmartScanEnv(other)
    obs_b, _ = env_b.reset(options={"scenario_seed": 1})
    assert np.array_equal(obs_a, obs_b)
    for _ in range(10):  # only ever scan band 0 (empty in both) -> identical observation streams
        obs_a = env.step(0)[0]
        obs_b = env_b.step(0)[0]
        assert np.array_equal(obs_a, obs_b)


def test_ppo_trains_saves_loads_and_acts(tmp_path, small_config):
    """End-to-end: PPO learns for a few updates, is saved, reloaded and drives the receiver."""
    from smartscan.evaluation.experiment import run_episode
    from smartscan.rl.train import train_ppo
    from smartscan.schedulers.ppo import PPOScheduler

    small_config.ppo.n_envs = 2
    small_config.ppo.n_steps = 64
    small_config.ppo.batch_size = 64
    small_config.ppo.eval_freq = 256
    small_config.ppo.checkpoint_freq = 256
    out = train_ppo(small_config, tmp_path / "ppo", total_timesteps=512)
    for f in ("best_model.zip", "final_model.zip", "config.json", "metadata.json"):
        assert (out / f).exists()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["timesteps_trained"] >= 512 and meta["obs_dim"] == observation_dim(4)

    sch = PPOScheduler(out / "best_model.zip")
    env = SmartScanEnv(small_config, split="test")
    res = run_episode(env, sch, 21)
    assert len(res.trace) == small_config.scenario.episode_length
    # the receiver scanned exactly what the policy selected
    probs = sch.explain()["probabilities"]
    assert len(probs) == 4 and np.isclose(sum(probs), 1.0, atol=1e-4)

    # deterministic inference is reproducible
    res2 = run_episode(env, PPOScheduler(out / "best_model.zip"), 21)
    assert res.trace["action"].tolist() == res2.trace["action"].tolist()


def test_ppo_band_mismatch_reported(tmp_path, small_config):
    from smartscan.rl.train import train_ppo
    from smartscan.schedulers.ppo import PPONotAvailable, PPOScheduler

    small_config.ppo.n_envs = 1
    small_config.ppo.n_steps = 64
    small_config.ppo.batch_size = 64
    out = train_ppo(small_config, tmp_path / "ppo", total_timesteps=64)
    sch = PPOScheduler(out / "final_model.zip")
    with pytest.raises(PPONotAvailable):
        sch.reset(8)
