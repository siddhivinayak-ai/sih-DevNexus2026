import numpy as np

from smartscan.config import EmitterSpec, ScenarioConfig
from smartscan.environment.smartscan_env import SmartScanEnv
from smartscan.evaluation.experiment import run_experiment
from smartscan.evaluation.metrics import aggregate, episode_metrics


def _run(cfg, actions, seed=1):
    env = SmartScanEnv(cfg)
    env.reset(options={"scenario_seed": seed})
    for a in actions:
        if env.done:
            break
        env.step(a)
    return env


def test_perfect_receiver_counts(small_config):
    small_config.receiver.pd = 1.0
    small_config.receiver.pfa = 0.0
    env = _run(small_config, [1] * 60)  # stare at the static emitter
    m = episode_metrics(env)
    assert m["decisions"] == 60
    assert m["detections"] == 60 and m["misses"] == 0 and m["false_alarms"] == 0
    assert m["pd"] == 1.0
    assert m["pfa"] is None  # never scanned an empty band -> insufficient data
    assert m["transmissions"] == 11 and m["intercepted"] == 1  # 1 static + 10 periodic bursts
    assert np.isclose(m["interception_rate"], 1 / 11)
    assert m["intercept_time_error"] == 0.0
    assert m["coverage"] == 0.25
    assert m["immediate_revisit_rate"] == 1.0
    assert m["emitter_discovery_rate"] == 0.5


def test_false_alarm_rate(small_config):
    small_config.receiver.pfa = 1.0
    env = _run(small_config, [0] * 60)
    m = episode_metrics(env)
    assert m["false_alarms"] == 60 and m["pfa"] == 1.0 and m["pd"] is None
    assert m["interception_rate"] == 0.0


def test_intercept_delay_measured_from_onset():
    cfg = ScenarioConfig(n_bands=2, episode_length=20, emitters=[
        EmitterSpec(kind="periodic", band=1, period=10, duty=0.5, phase=0, snr_db=12.0)])
    from smartscan.config import ExperimentConfig

    ec = ExperimentConfig(scenario=cfg)
    ec.receiver.pd, ec.receiver.pfa = 1.0, 0.0
    # scan band 0 for 3 steps, then band 1: first burst [0,4] intercepted at t=3 (delay 3)
    env = _run(ec, [0, 0, 0] + [1] * 17)
    m = episode_metrics(env)
    assert m["intercepted"] == 2
    # burst 1 delay 3, burst 2 (onset 10) delay 0
    assert np.isclose(m["intercept_time_error"], 1.5)
    assert np.isclose(m["avg_intercept_time"], 3.0)


def test_aggregate_ignores_none():
    s = aggregate([1.0, None, 3.0])
    assert s.n == 2 and s.mean == 2.0 and np.isclose(s.std, np.sqrt(2))
    assert aggregate([None]).format() == "Insufficient data"


def test_experiment_reproducible(small_config):
    a = run_experiment(small_config, ["open_loop", "random", "pomdp"], [21, 22], split="test")
    b = run_experiment(small_config, ["open_loop", "random", "pomdp"], [21, 22], split="test")
    cols = ["detections", "false_alarms", "cumulative_reward"]
    assert a.per_episode[cols].equals(b.per_episode[cols])
    assert len(a.per_episode) == 6


def test_experiment_skips_untrained_ppo(small_config):
    res = run_experiment(small_config, ["open_loop", "ppo"], [21], split="test", model_path=None)
    assert "ppo" in res.skipped and res.schedulers == ["open_loop"]


def test_result_save_and_load(tmp_path, small_config):
    from smartscan.evaluation.experiment import load_result

    res = run_experiment(small_config, ["open_loop", "pomdp"], [21, 22], split="test")
    out = res.save(tmp_path / "exp")
    for f in ("config.json", "metrics.json", "per_episode.csv", "summary.csv", "decisions.csv"):
        assert (out / f).exists()
    again = load_result(out)
    assert again.schedulers == ["open_loop", "pomdp"]
    assert len(again.per_episode) == 4
