"""Episode metrics. One implementation, applied identically to every scheduler.

Counts are over non-warm-up dwells, per observed band ("band-dwell"):
    D  detections          signal present, detector fired
    M  missed detections   signal present, detector silent
    F  false alarms        no signal, detector fired
    C  correct rejections  no signal, detector silent

    Pd  (empirical)        D / (D + M)
    Pfa (empirical)        F / (F + C)
    hit rate               (D + M) / (D + M + F + C)   share of dwells that landed on an active band

Transmission k (maximal run of one emitter in one band) with onset s_k is
*intercepted* if some dwell scans its band while it is active and the detector
fires; t_k is the first such timestep inside the transmission.
    interception rate      #intercepted / #transmissions (onset after warm-up)
    intercept time error   mean_k (t_k - s_k) over intercepted transmissions; the reference
                           s_k is what an ideal wideband (stare) receiver would achieve
    normalised delay       mean_k (t_k - s_k) / duration_k
Emitter e with first activation a_e (after warm-up):
    average intercept time mean_e (first intercept time of e - a_e) over discovered emitters
    emitter discovery rate #emitters intercepted at least once / #emitters active

Scan behaviour:
    coverage               #distinct bands scanned / N
    revisit interval       mean over bands of the mean gap (timesteps) between successive scans
    immediate revisit rate share of decisions that repeat the previous band
    band scan share        scans of band n / total band-dwells
Reward:
    cumulative reward      sum of per-decision rewards (non-warm-up)
    mean reward            cumulative reward / #decisions

A metric whose denominator is zero is reported as None ("Insufficient data").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS: dict[str, dict] = {
    "pd": {"label": "Probability of Detection (Pd)", "fmt": "{:.3f}", "higher": True},
    "pfa": {"label": "Probability of False Alarm (Pfa)", "fmt": "{:.3f}", "higher": False},
    "interception_rate": {"label": "Interception Rate", "fmt": "{:.3f}", "higher": True},
    "avg_intercept_time": {"label": "Average Intercept Time (steps)", "fmt": "{:.2f}", "higher": False},
    "intercept_time_error": {"label": "Intercept Time Error (steps)", "fmt": "{:.2f}", "higher": False},
    "normalized_delay": {"label": "Normalised Intercept Delay", "fmt": "{:.3f}", "higher": False},
    "emitter_discovery_rate": {"label": "Emitter Discovery Rate", "fmt": "{:.3f}", "higher": True},
    "detections": {"label": "Detection Count", "fmt": "{:.1f}", "higher": True},
    "misses": {"label": "Missed Detection Count", "fmt": "{:.1f}", "higher": False},
    "false_alarms": {"label": "False Alarm Count", "fmt": "{:.1f}", "higher": False},
    "hit_rate": {"label": "Hit Rate (dwells on active band)", "fmt": "{:.3f}", "higher": True},
    "coverage": {"label": "Scan Coverage", "fmt": "{:.3f}", "higher": True},
    "revisit_interval": {"label": "Mean Revisit Interval (steps)", "fmt": "{:.2f}", "higher": None},
    "immediate_revisit_rate": {"label": "Immediate Revisit Rate", "fmt": "{:.3f}", "higher": None},
    "cumulative_reward": {"label": "Cumulative Reward", "fmt": "{:.2f}", "higher": True},
    "mean_reward": {"label": "Average Reward per Decision", "fmt": "{:.4f}", "higher": True},
    "transmissions": {"label": "Transmissions (ground truth)", "fmt": "{:.1f}", "higher": None},
    "intercepted": {"label": "Transmissions Intercepted", "fmt": "{:.1f}", "higher": True},
    "decisions": {"label": "Decisions", "fmt": "{:.0f}", "higher": None},
}

HEADLINE = [
    "pd",
    "pfa",
    "interception_rate",
    "avg_intercept_time",
    "intercept_time_error",
    "emitter_discovery_rate",
    "coverage",
    "mean_reward",
]


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den > 0 else None


def records_frame(records) -> pd.DataFrame:
    """StepRecords -> DataFrame (one row per decision)."""
    rows = []
    for r in records:
        row = {
            "step": r.step,
            "t_start": r.t_start,
            "t_end": r.t_end,
            "action": r.action,
            "band": f"F{r.action + 1}",
            "detected": r.detected,
            "signal_present": r.signal_present,
            "outcome": r.outcome,
            "detections": r.detections,
            "false_alarms": r.false_alarms,
            "misses": r.misses,
            "correct_rejections": r.correct_rejections,
            "new_intercepts": r.new_intercepts,
            "retuned": r.retuned,
            "amplitude_db": r.amplitude_db,
            "reward": r.reward,
            "warmup": r.warmup,
        }
        row.update({f"r_{k}": v for k, v in r.terms.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def episode_metrics(env) -> dict:
    """Compute all metrics for the episode currently held by a finished SmartScanEnv."""
    sc = env.scenario
    recs = [r for r in env.records if not r.warmup]
    D = sum(r.detections for r in recs)
    M = sum(r.misses for r in recs)
    F = sum(r.false_alarms for r in recs)
    C = sum(r.correct_rejections for r in recs)
    dwells = D + M + F + C

    w = sc.warmup_steps
    tx_idx = [k for k, tx in enumerate(sc.transmissions) if tx.start >= w]
    got = [k for k in tx_idx if env.intercepted[k]]
    delays = [env.intercept_time[k] - sc.transmissions[k].start for k in got]
    ndelays = [(env.intercept_time[k] - sc.transmissions[k].start) / sc.transmissions[k].duration for k in got]

    # emitter-level first intercept
    first_active: dict[int, int] = {}
    first_hit: dict[int, int] = {}
    for k, tx in enumerate(sc.transmissions):
        end = tx.end
        if end < w:
            continue
        onset = max(tx.start, w)
        first_active[tx.emitter] = min(first_active.get(tx.emitter, onset), onset)
        if env.intercepted[k] and env.intercept_time[k] >= w:
            first_hit[tx.emitter] = min(first_hit.get(tx.emitter, env.intercept_time[k]), env.intercept_time[k])
    ttfi = [first_hit[e] - first_active[e] for e in first_hit]

    scans: dict[int, list[int]] = {}
    for r in recs:
        for b in r.bands:
            scans.setdefault(b, []).append(r.t_start)
    gaps = [np.mean(np.diff(v)) for v in scans.values() if len(v) >= 2]
    actions = [r.action for r in recs]
    repeats = sum(1 for a, b in zip(actions, actions[1:]) if a == b)
    share = np.zeros(sc.n_bands)
    for b, v in scans.items():
        share[b] = len(v)
    share = share / max(1, share.sum())
    cum = float(sum(r.reward for r in recs))

    return {
        "decisions": len(recs),
        "detections": D,
        "misses": M,
        "false_alarms": F,
        "correct_rejections": C,
        "pd": _ratio(D, D + M),
        "pfa": _ratio(F, F + C),
        "hit_rate": _ratio(D + M, dwells),
        "transmissions": len(tx_idx),
        "intercepted": len(got),
        "interception_rate": _ratio(len(got), len(tx_idx)),
        "intercept_time_error": float(np.mean(delays)) if delays else None,
        "normalized_delay": float(np.mean(ndelays)) if ndelays else None,
        "avg_intercept_time": float(np.mean(ttfi)) if ttfi else None,
        "emitter_discovery_rate": _ratio(len(first_hit), len(first_active)),
        "coverage": len(scans) / sc.n_bands,
        "revisit_interval": float(np.mean(gaps)) if gaps else None,
        "immediate_revisit_rate": _ratio(repeats, len(actions) - 1),
        "band_scan_share": share.round(4).tolist(),
        "cumulative_reward": cum,
        "mean_reward": _ratio(cum, len(recs)),
    }


@dataclass
class Stat:
    mean: float | None
    std: float | None
    n: int

    def format(self, fmt: str = "{:.3f}") -> str:
        if self.mean is None:
            return "Insufficient data"
        return f"{fmt.format(self.mean)} ± {fmt.format(self.std or 0.0)}"


def aggregate(values) -> Stat:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if v.size == 0:
        return Stat(None, None, 0)
    return Stat(float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))
