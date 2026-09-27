"""Evaluate a trained PPO scheduler on a seed split (default: unseen test seeds).

    python -m smartscan.rl.evaluate --model models/ppo_mixed_8band/best_model.zip
    python -m smartscan.rl.evaluate --model models/ppo_mixed_8band/best_model.zip --split train --episodes 50
    python -m smartscan.rl.evaluate --model ... --with-baselines

The configuration stored next to the model (config.json) is used unless --config is given,
so the evaluation scenario/receiver/reward match what the policy was trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from smartscan.config import ExperimentConfig, load_config
from smartscan.evaluation.experiment import run_experiment
from smartscan.evaluation.metrics import HEADLINE


def config_for_model(model_path: str | Path, config_path: str | None = None) -> ExperimentConfig:
    if config_path:
        return load_config(config_path)
    stored = Path(model_path).parent / "config.json"
    if stored.exists():
        return ExperimentConfig.load(stored)
    return load_config(None)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Evaluate a PPO scan scheduler.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--with-baselines", action="store_true", help="also run Open Loop, Random, POMDP")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = config_for_model(args.model, args.config)
    seeds = cfg.seeds.seeds(args.split)
    if args.episodes:
        seeds = seeds[: args.episodes]
    keys = (["open_loop", "random", "pomdp"] if args.with_baselines else []) + ["ppo"]
    res = run_experiment(cfg, keys, seeds, split=args.split, model_path=args.model)
    for key, why in res.skipped.items():
        print(f"SKIPPED {key}: {why}")
    pd.set_option("display.width", 200)
    print(f"\nSplit '{args.split}' ({len(seeds)} seeds {seeds[0]}..{seeds[-1]}), mean ± std:\n")
    print(res.table(HEADLINE + ["detections", "misses", "false_alarms", "cumulative_reward"]).to_string())
    out = res.save(args.out)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
