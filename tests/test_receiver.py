import numpy as np

from smartscan.config import EmitterSpec, ObservationConfig, ReceiverConfig, ScenarioConfig
from smartscan.environment.scenario import build_scenario
from smartscan.receiver.detector import pd_from_snr
from smartscan.receiver.observation import ReceiverHistory, observation_dim
from smartscan.receiver.receiver import Observation, Receiver, ReceiverNoise


def _scenario(T=2000):
    cfg = ScenarioConfig(n_bands=4, episode_length=T, emitters=[EmitterSpec(kind="static", band=1, snr_db=10.0)])
    return build_scenario(cfg, 0)


def _dwell_many(rx_cfg, band, T=2000):
    sc = _scenario(T)
    noise = ReceiverNoise.draw(0, sc.total_steps, sc.n_bands)
    rx = Receiver(rx_cfg, sc.n_bands)
    return np.array([rx.dwell(band, t, sc, noise)[0].detected[0] for t in range(T)])


def test_receiver_sees_only_selected_band():
    sc = _scenario(10)
    noise = ReceiverNoise.draw(0, sc.total_steps, sc.n_bands)
    obs, truth = Receiver(ReceiverConfig(), sc.n_bands).dwell(2, 0, sc, noise)
    assert list(obs.bands) == [2]
    assert not truth.present[0]


def test_detection_rate_matches_pd():
    assert abs(_dwell_many(ReceiverConfig(pd=0.8, pfa=0.05), band=1).mean() - 0.8) < 0.04


def test_false_alarm_rate_matches_pfa():
    assert abs(_dwell_many(ReceiverConfig(pd=0.8, pfa=0.1), band=0).mean() - 0.1) < 0.03


def test_amplitude_reported_only_on_detection():
    sc = _scenario(200)
    noise = ReceiverNoise.draw(0, sc.total_steps, sc.n_bands)
    rx = Receiver(ReceiverConfig(pd=0.9), sc.n_bands)
    for t in range(200):
        obs, _ = rx.dwell(1, t, sc, noise)
        assert np.isnan(obs.amplitude_db[0]) != bool(obs.detected[0])


def test_snr_model_monotonic():
    p = pd_from_snr(np.array([-5.0, 0.0, 5.0, 10.0]), 0.05)
    assert np.all(np.diff(p) > 0) and p[-1] > 0.99 and p[0] > 0.05


def test_instantaneous_bandwidth_window_clipped():
    rx = Receiver(ReceiverConfig(instantaneous_bands=3), 8)
    assert list(rx.window(0)) == [0, 1, 2]
    assert list(rx.window(7)) == [5, 6, 7]


def test_retune_costs_time():
    sc = _scenario(50)
    noise = ReceiverNoise.draw(0, sc.total_steps, sc.n_bands)
    rx = Receiver(ReceiverConfig(retune_steps=2, dwell_steps=3), sc.n_bands)
    o1, _ = rx.dwell(0, 0, sc, noise)
    assert (o1.t_start, o1.t_end, o1.retuned) == (0, 2, False)
    o2, _ = rx.dwell(1, 3, sc, noise)
    assert (o2.t_start, o2.t_end, o2.retuned) == (5, 7, True)


def test_common_random_numbers():
    n1 = ReceiverNoise.draw(5, 100, 4)
    n2 = ReceiverNoise.draw(5, 100, 4)
    assert np.array_equal(n1.u_detect, n2.u_detect)


def _obs(t, band, det):
    return Observation(t, t, band, np.array([band]), np.array([det]), np.array([10.0 if det else np.nan]))


def test_belief_update_direction():
    h = ReceiverHistory(4, 100, ReceiverConfig(pd=0.9, pfa=0.05), ObservationConfig())
    p0 = h.belief[2]
    h.record(_obs(0, 2, True))
    assert h.belief[2] > p0
    p1 = h.belief[1]
    h.record(_obs(1, 1, False))
    assert h.belief[1] < p1


def test_state_vector_shape_and_range():
    h = ReceiverHistory(8, 100, ReceiverConfig(), ObservationConfig())
    for t in range(30):
        h.record(_obs(t, t % 8, t % 3 == 0))
    v = h.vector()
    assert v.shape == (observation_dim(8),)
    assert v.dtype == np.float32
    assert np.all(v >= 0) and np.all(v <= 1)


def test_period_estimator_finds_period():
    h = ReceiverHistory(2, 500, ReceiverConfig(), ObservationConfig())
    # band 0 detected at onsets every 7 steps, silent scan in between
    for t in range(0, 70):
        h.record(_obs(t, 0, t % 7 == 0))
    assert h.period[0] == 7
