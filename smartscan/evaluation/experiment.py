"""Multi-seed experiment runner shared by every scheduler.

For each scenario seed, every scheduler runs one episode on an identically
configured :class:`SmartScanEnv` (same scenario realisation, same receiver,
same common random numbers). Only the scan decisions differ.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from smartscan.config import RESULTS_DIR, ExperimentConfig
from smartscan.environment.smartscan_env import SmartScanEnv
from smartscan.evaluation.metrics import METRICS, aggregate, episode_metrics, records_frame
from smartscan.schedulers import SCHEDULER_LABELS, BaseScheduler, DecisionContext, make_scheduler


@dataclass
class EpisodeResult:
    scheduler: str
    seed: int
    metrics: dict
    trace: pd.DataFrame


def run_episode(
    env: SmartScanEnv,
    scheduler: BaseScheduler,
    scenario_seed: int,
    on_step: Callable[[SmartScanEnv], None] | None = None,
) -> EpisodeResult:
    state, _ = env.reset(options={"scenario_seed": int(scenario_seed)})
    scheduler.reset(env.n_bands, seed=int(scenario_seed))
    while not env.done:
        ctx = DecisionContext(state=state, history=env.history, t=env.t)
        action = scheduler.select_action(ctx)
        state, _, _, _, info = env.step(action)
        scheduler.update(action, info["observation"])
        if on_step is not None:
            on_step(env)
    return EpisodeResult(scheduler.key, int(scenario_seed), episode_metrics(env), records_frame(env.records))


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    schedulers: list[str]
    seeds: list[int]
    split: str
    per_episode: pd.DataFrame
    traces: dict[tuple[str, int], pd.DataFrame] = field(default_factory=dict)
    model_path: str | None = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    elapsed_s: float = 0.0
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Long table: scheduler x metric -> mean, std, n."""
        rows = []
        for key in self.schedulers:
            df = self.per_episode[self.per_episode["scheduler"] == key]
            for m in METRICS:
                if m not in df:
                    continue
                s = aggregate(df[m].tolist())
                rows.append({"scheduler": key, "metric": m, "mean": s.mean, "std": s.std, "n": s.n})
        return pd.DataFrame(rows)

    def table(self, metrics: list[str] | None = None) -> pd.DataFrame:
        """Wide display table: rows = metrics, columns = schedulers, cells 'mean ± std'."""
        metrics = metrics or list(METRICS)
        out = {}
        for key in self.schedulers:
            df = self.per_episode[self.per_episode["scheduler"] == key]
            out[SCHEDULER_LABELS.get(key, key)] = [
                aggregate(df[m].tolist()).format(METRICS[m]["fmt"]) if m in df else "Insufficient data"
                for m in metrics
            ]
        return pd.DataFrame(out, index=[METRICS[m]["label"] for m in metrics])

    def save(self, out_dir: str | Path | None = None, max_traces: int = 5) -> Path:
        if out_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = RESULTS_DIR / f"{stamp}_{self.config.scenario.name}_{self.split}"
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.config.save(out / "config.json")
        self.per_episode.to_csv(out / "per_episode.csv", index=False)
        summ = self.summary()
        summ.to_csv(out / "summary.csv", index=False)
        self.table().to_csv(out / "comparison_table.csv")
        meta = {
            "created": self.created,
            "split": self.split,
            "seeds": self.seeds,
            "schedulers": self.schedulers,
            "skipped": self.skipped,
            "model_path": self.model_path,
            "elapsed_s": round(self.elapsed_s, 2),
            "summary": {
                k: {r["metric"]: {"mean": r["mean"], "std": r["std"], "n": r["n"]} for _, r in g.iterrows()}
                for k, g in summ.groupby("scheduler")
            },
        }
        (out / "metrics.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        frames, kept = [], {}
        for (key, seed), tr in self.traces.items():
            kept[key] = kept.get(key, 0) + 1
            if kept[key] <= max_traces:
                frames.append(tr.assign(scheduler=key, seed=seed))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / "decisions.csv", index=False)
        return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_result(path: str | Path) -> ExperimentResult:
    path = Path(path)
    meta = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig.load(path / "config.json")
    per = pd.read_csv(path / "per_episode.csv")
    per = per.astype(object).where(per.notna(), None)
    traces = {}
    if (path / "decisions.csv").exists():
        dec = pd.read_csv(path / "decisions.csv")
        for (k, s), g in dec.groupby(["scheduler", "seed"]):
            traces[(k, int(s))] = g.drop(columns=["scheduler", "seed"]).reset_index(drop=True)
    return ExperimentResult(
        config=cfg,
        schedulers=meta["schedulers"],
        seeds=meta["seeds"],
        split=meta["split"],
        per_episode=per,
        traces=traces,
        model_path=meta.get("model_path"),
        created=meta.get("created", ""),
        elapsed_s=meta.get("elapsed_s", 0.0),
        skipped=meta.get("skipped", {}),
    )


def list_results() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted((p for p in RESULTS_DIR.iterdir() if (p / "metrics.json").exists()), reverse=True)


def run_experiment(
    config: ExperimentConfig,
    schedulers: list[str],
    seeds: list[int] | None = None,
    split: str = "test",
    model_path: str | Path | None = None,
    keep_traces: int = 5,
    progress: Callable[[float, str], None] | None = None,
) -> ExperimentResult:
    """Run every scheduler on every seed. PPO is skipped (and reported) if no model is usable."""
    seeds = list(seeds) if seeds is not None else config.seeds.seeds(split)
    t0 = time.perf_counter()
    rows, traces, skipped, active = [], {}, {}, []
    instances: dict[str, BaseScheduler] = {}
    for key in schedulers:
        try:
            sch = make_scheduler(key, model_path)
            sch.reset(config.scenario.n_bands, seed=0)  # validates PPO / scenario compatibility
            instances[key] = sch
            active.append(key)
        except Exception as exc:  # PPO not trained / incompatible
            skipped[key] = str(exc)
    env = SmartScanEnv(config, split=split, seeds=seeds)
    total = max(1, len(active) * len(seeds))
    done = 0
    for seed in seeds:
        for key in active:
            res = run_episode(env, instances[key], seed)
            row = {"scheduler": key, "seed": seed}
            row.update({k: v for k, v in res.metrics.items() if k != "band_scan_share"})
            row["band_scan_share"] = json.dumps(res.metrics["band_scan_share"])
            rows.append(row)
            if sum(1 for (k, _) in traces if k == key) < keep_traces:
                traces[(key, seed)] = res.trace
            done += 1
            if progress:
                progress(done / total, f"{SCHEDULER_LABELS.get(key, key)} seed {seed}")
    return ExperimentResult(
        config=config,
        schedulers=active,
        seeds=seeds,
        split=split,
        per_episode=pd.DataFrame(rows),
        traces=traces,
        model_path=str(model_path) if model_path else None,
        elapsed_s=time.perf_counter() - t0,
        skipped=skipped,
    )
