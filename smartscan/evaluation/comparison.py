"""Algorithm comparison: Open Loop vs Random vs POMDP vs PPO on identical conditions.

Paired statistics: every scheduler runs on the same seeds with common random
numbers, so per-seed differences (scheduler - reference) are paired samples.
The 95 % interval uses the Student-t quantile.

CLI:
    python -m smartscan.evaluation.comparison --config configs/default.yaml \
        --model models/ppo_default/best_model.zip --split test --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan.config import MODELS_DIR, load_config
from smartscan.evaluation.experiment import ExperimentResult, run_experiment
from smartscan.evaluation.metrics import HEADLINE, METRICS
from smartscan.schedulers import SCHEDULER_LABELS

ALL_SCHEDULERS = ["open_loop", "random", "pomdp", "ppo"]


def _t95(dof: int) -> float:
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, dof))
    except ImportError:  # normal approximation
        return 1.96


def paired_differences(result: ExperimentResult, reference: str = "open_loop", metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean paired difference (scheduler - reference) with 95 % CI, per metric."""
    metrics = metrics or HEADLINE
    df = result.per_episode
    if reference not in result.schedulers:
        return pd.DataFrame()
    ref = df[df["scheduler"] == reference].set_index("seed")
    rows = []
    for key in result.schedulers:
        if key == reference:
            continue
        cur = df[df["scheduler"] == key].set_index("seed")
        for m in metrics:
            a = pd.to_numeric(cur[m], errors="coerce")
            b = pd.to_numeric(ref[m], errors="coerce")
            d = (a - b).dropna()
            if len(d) < 2:
                rows.append({"scheduler": key, "metric": m, "mean_diff": None, "ci95": None, "n": len(d)})
                continue
            half = _t95(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"scheduler": key, "metric": m, "mean_diff": float(d.mean()), "ci95": float(half), "n": len(d)})
    return pd.DataFrame(rows)


def default_model_path(scenario_name: str) -> Path | None:
    for cand in (MODELS_DIR / f"ppo_{scenario_name}" / "best_model.zip", MODELS_DIR / f"ppo_{scenario_name}" / "final_model.zip"):
        if cand.exists():
            return cand
    return None


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Compare scan schedulers on identical scenarios and seeds.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="PPO model .zip (default: models/ppo_<scenario>/best_model.zip)")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--episodes", type=int, default=None, help="use the first N seeds of the split")
    ap.add_argument("--schedulers", nargs="+", default=ALL_SCHEDULERS)
    ap.add_argument("--out", default=None, help="output directory (default experiments/results/<timestamp>_...)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seeds = cfg.seeds.seeds(args.split)
    if args.episodes:
        seeds = seeds[: args.episodes]
    model = args.model or default_model_path(cfg.scenario.name)

    def progress(frac: float, msg: str) -> None:
        print(f"\r[{frac:6.1%}] {msg:40s}", end="", flush=True)

    res = run_experiment(cfg, args.schedulers, seeds, split=args.split, model_path=model, progress=progress)
    print()
    for key, why in res.skipped.items():
        print(f"SKIPPED {SCHEDULER_LABELS.get(key, key)}: {why}")
    pd.set_option("display.width", 200)
    print(f"\nScenario '{cfg.scenario.name}', split '{args.split}', {len(seeds)} seeds (mean ± std):\n")
    print(res.table(HEADLINE + ["detections", "misses", "false_alarms", "cumulative_reward"]).to_string())
    diffs = paired_differences(res)
    if not diffs.empty:
        diffs["metric"] = diffs["metric"].map(lambda m: METRICS[m]["label"])
        diffs["scheduler"] = diffs["scheduler"].map(SCHEDULER_LABELS)
        print("\nPaired difference vs Open Loop (mean ± 95% CI half-width):\n")
        print(diffs.to_string(index=False))
    out = res.save(args.out)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
