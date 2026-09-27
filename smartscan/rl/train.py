"""PPO training for the SmartScan environment (Stable-Baselines3).

    python -m smartscan.rl.train --config configs/default.yaml
    python -m smartscan.rl.train --config configs/default.yaml --timesteps 50000 --out models/ppo_quick

Outputs (``models/ppo_<scenario>/`` by default):
    best_model.zip        best policy on the validation seeds (model selection)
    final_model.zip       policy at the end of training
    checkpoints/          periodic checkpoints
    config.json           exact experiment configuration
    metadata.json         obs/action dims, seeds, timesteps, library versions, wall time
    progress.json         episode rewards, SB3 losses, validation history
    monitor/              SB3 Monitor CSVs (per training env)
    tb/                   TensorBoard logs (if tensorboard is installed)
Training uses only the *train* seeds; model selection only the *validation* seeds.
The *test* seeds are reserved for ``smartscan.rl.evaluate`` / the comparison.
"""

from __future__ import annotations

import argparse
import sys
import importlib.util
import json
import platform
import shutil
import time
from pathlib import Path

from smartscan.config import MODELS_DIR, ExperimentConfig, load_config
from smartscan.rl.callbacks import ProgressCallback, TrainingProgress, ValidationCallback


def default_out_dir(cfg: ExperimentConfig) -> Path:
    return MODELS_DIR / f"ppo_{cfg.scenario.name}"


def make_vec_env(cfg: ExperimentConfig, out_dir: Path):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from smartscan.environment.smartscan_env import SmartScanEnv

    mon_dir = out_dir / "monitor"
    mon_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def _make():
            env = SmartScanEnv(cfg, split="train", keep_records=False)
            return Monitor(env, filename=str(mon_dir / f"env_{rank}"))

        return _make

    vec = DummyVecEnv([factory(i) for i in range(max(1, cfg.ppo.n_envs))])
    vec.seed(cfg.ppo.seed)
    return vec


def train_ppo(
    cfg: ExperimentConfig,
    out_dir: str | Path | None = None,
    progress: TrainingProgress | None = None,
    total_timesteps: int | None = None,
) -> Path:
    """Train PPO; returns the output directory. Safe to call from a background thread."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed

    from smartscan.receiver.observation import observation_dim

    out = Path(out_dir) if out_dir else default_out_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    p = cfg.ppo
    total = int(total_timesteps or p.total_timesteps)
    progress = progress or TrainingProgress()
    progress.total_timesteps = total
    progress.out_dir = str(out)
    progress.status = "running"
    cfg.save(out / "config.json")

    try:
        set_random_seed(p.seed)
        vec = make_vec_env(cfg, out)
        tb = str(out / "tb") if importlib.util.find_spec("tensorboard") else None
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=p.learning_rate,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            clip_range=p.clip_range,
            ent_coef=p.ent_coef,
            policy_kwargs={"net_arch": {"pi": list(p.net_arch), "vf": list(p.net_arch)}},
            tensorboard_log=tb,
            seed=p.seed,
            device="cpu",
            verbose=0,
        )
        n_envs = vec.num_envs
        callbacks = CallbackList(
            [
                ProgressCallback(progress),
                CheckpointCallback(
                    save_freq=max(1, p.checkpoint_freq // n_envs),
                    save_path=str(out / "checkpoints"),
                    name_prefix="ppo",
                ),
                ValidationCallback(cfg, progress, out, eval_freq=p.eval_freq),
            ]
        )
        t0 = time.time()
        model.learn(total_timesteps=total, callback=callbacks, tb_log_name="ppo", progress_bar=False)
        wall = time.time() - t0
        stopped = progress.stop_event.is_set()
        model.save(str(out / "final_model.zip"))
        if not (out / "best_model.zip").exists():
            shutil.copy(out / "final_model.zip", out / "best_model.zip")
        meta = {
            "scenario": cfg.scenario.name,
            "n_bands": cfg.scenario.n_bands,
            "obs_dim": observation_dim(cfg.scenario.n_bands),
            "timesteps_trained": int(model.num_timesteps),
            "timesteps_requested": total,
            "stopped_early": stopped,
            "episodes": progress.episodes,
            "wall_time_s": round(wall, 1),
            "train_seeds": list(cfg.seeds.train),
            "validation_seeds": list(cfg.seeds.validation),
            "test_seeds": list(cfg.seeds.test),
            "best_validation_reward": progress.best_validation,
            "selection": "best mean episode reward on validation seeds (deterministic policy)",
            "versions": _versions(),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        progress.status = "stopped" if stopped else "finished"
        progress.message = f"{'Stopped' if stopped else 'Finished'} after {model.num_timesteps} timesteps ({wall:.0f} s)"
        ProgressCallback(progress)._write()
    except Exception as exc:
        progress.status = "error"
        progress.message = f"{type(exc).__name__}: {exc}"
        raise
    return out


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "gymnasium", "stable_baselines3"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Train a PPO scan scheduler on SmartScanEnv.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--timesteps", type=int, default=None, help="override ppo.total_timesteps")
    ap.add_argument("--seed", type=int, default=None, help="override ppo.seed")
    ap.add_argument("--out", default=None, help="output directory (default models/ppo_<scenario>)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.ppo.seed = args.seed
    progress = TrainingProgress()
    total = args.timesteps or cfg.ppo.total_timesteps
    print(f"Training PPO on '{cfg.scenario.name}' for {total} timesteps "
          f"(train seeds {cfg.seeds.train}, validation seeds {cfg.seeds.validation})")

    import threading

    result: dict = {}

    def run():
        try:
            result["out"] = train_ppo(cfg, args.out, progress, total)
        except Exception as exc:  # reported below
            result["error"] = exc

    th = threading.Thread(target=run, daemon=True)
    th.start()
    last_val = 0
    try:
        while th.is_alive():
            th.join(timeout=5.0)
            mr = progress.mean_reward()
            print(
                f"  t={progress.timesteps:>8d}/{total}  episodes={progress.episodes:>5d}  "
                f"mean_ep_reward(20)={'n/a' if mr is None else f'{mr:8.2f}'}  fps={progress.fps:6.0f}",
                flush=True,
            )
            while last_val < len(progress.validation):
                v = progress.validation[last_val]
                rate = v["interception_rate"]
                print(f"  [validation] t={v['timesteps']}  mean_reward={v['mean_reward']:.2f}  "
                      f"interception={'n/a' if rate is None else f'{rate:.3f}'}"
                      f"{'  * best' if v.get('saved_best') else ''}")
                last_val += 1
    except KeyboardInterrupt:
        print("Stop requested; finishing current step and saving...")
        progress.request_stop()
        th.join()
    if "error" in result:
        raise result["error"]
    print(progress.message)
    print(f"Model saved in {result['out']}")


if __name__ == "__main__":
    main()
