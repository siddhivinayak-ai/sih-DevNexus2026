# 07 — PPO Training

1. **Environment**: `SmartScanEnv(config, split="train")`; each `reset()` draws a scenario seed from the training range.
   `n_envs` copies run in a `DummyVecEnv`, each wrapped in an SB3 `Monitor`.
2. **Observation space**: `Box(0, 1, (11N+3,), float32)` ([05](05_rl_formulation.md)).
3. **Action space**: `Discrete(N)`.
4. **Configuration** (`ppo:` in YAML; defaults): `MlpPolicy`, net_arch [128,128] for policy and value,
   lr 3e-4, γ 0.95, GAE λ 0.95, n_steps 1024 per env, batch 256, 10 epochs, entropy 0.01, clip 0.2, 8 envs, seed 42,
   300 000 timesteps.
5. **Training**
   ```
   python -m smartscan.rl.train --config configs/default.yaml
   python -m smartscan.rl.train --config configs/default.yaml --timesteps 1000000 --seed 7 --out models/ppo_long
   ```
   Or use the **RL Training** page (START / STOP; STOP ends cleanly and still saves).
6. **Checkpointing**: `models/ppo_<scenario>/checkpoints/ppo_<steps>_steps.zip` every `checkpoint_freq` timesteps.
   Every `eval_freq` timesteps the deterministic policy runs on **all validation seeds**; the best mean validation
   reward is saved as `best_model.zip`. `final_model.zip` is the last policy.
   Also written: `config.json`, `metadata.json` (dims, seeds, versions, wall time), `progress.json`,
   `monitor/*.csv`, `tb/` (TensorBoard: `tensorboard --logdir models/ppo_mixed_8band/tb`).
7. **Evaluation on unseen seeds**
   ```
   python -m smartscan.rl.evaluate --model models/ppo_mixed_8band/best_model.zip                  # test split
   python -m smartscan.rl.evaluate --model models/ppo_mixed_8band/best_model.zip --split train --episodes 100
   python -m smartscan.rl.evaluate --model models/ppo_mixed_8band/best_model.zip --with-baselines
   ```
8. **Inference**: `PPOScheduler` loads the model with `PPO.load`, calls `model.predict(obs, deterministic=True)`,
   and reports the action probabilities `π(·∣s)`, which are shown on the Live Simulation page.
9. **Reproducibility**: `set_random_seed(ppo.seed)`, a seeded vector env, seeded scenario and receiver draws, and the stored
   `config.json`. PyTorch CPU training is reproducible for a given machine and library version; results can differ
   slightly across platforms.

A model is tied to the number of bands N (observation size). The UI and CLI refuse an incompatible model with an
explicit message.
