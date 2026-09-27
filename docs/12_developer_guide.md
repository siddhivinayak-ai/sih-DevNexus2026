# 12 — Developer Guide

**Add an emitter behaviour.** Add a branch in `environment/emitter.py::realize_emitter` that fills `trace` (band or −1
per step), add the kind to `config.EMITTER_KINDS`, and add a test in `tests/test_environment.py`.

**Add a scheduler.** Subclass `schedulers.base.BaseScheduler`, implement `select_action(ctx)` (and `update` if it
learns online), register it in `schedulers/__init__.py::make_scheduler` and `SCHEDULER_LABELS`, and add a colour in
`visualization/style.py::SCHEDULER_COLORS` (validate the palette). It then appears in every experiment and page.
Use only `ctx.state`, `ctx.history` and the `Observation` objects. Never read `env.scenario`.

**Change the state vector.** Edit `receiver/observation.py` (`PER_BAND_FEATURES`, `vector()`, `observation_dim`).
Existing PPO models become incompatible (obs size), and the loader reports this.

**Change the reward.** Edit `environment/reward.py` and the event extraction in `SmartScanEnv.step`, and update
`docs/06_reward_function.md` and `tests/test_reward.py` together.

**Conventions.** All randomness goes through `numpy.random.SeedSequence` derived from the scenario seed. Every
number shown in the UI comes from a simulation run or a training log. Missing values display
"Insufficient data" or "PPO model not trained."

**Tests.** `python -m pytest -q`. `test_ppo_env.py` includes a short end-to-end PPO train/save/load/act run and
Gymnasium's `check_env`.
