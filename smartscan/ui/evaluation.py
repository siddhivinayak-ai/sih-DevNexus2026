"""Policy Evaluation: one scheduler, a seed range, per-episode metrics and raw event traces."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from smartscan.environment.scenario import build_scenario
from smartscan.evaluation.experiment import run_experiment
from smartscan.evaluation.metrics import METRICS, aggregate
from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.ui.common import active_model, get_config, kpis, note, panel, plot, seed_picker
from smartscan.visualization.comparison import metric_distribution
from smartscan.visualization.spectrum import spectrum_activity_map
from smartscan.visualization.trajectory import event_timeline

KPI_METRICS = ["pd", "pfa", "interception_rate", "avg_intercept_time", "intercept_time_error", "mean_reward", "coverage"]


def render() -> None:
    cfg = get_config()
    n = cfg.scenario.n_bands
    model = active_model(n)
    with panel("Evaluation setup"):
        c1, c2 = st.columns([1, 3])
        keys = ["ppo", "pomdp", "open_loop", "random"]
        key = c1.selectbox("Scheduler", keys, format_func=lambda k: SCHEDULER_LABELS[k])
        with c2:
            split, seeds = seed_picker(cfg, "eval")
        st.caption(f"Scenario '{cfg.scenario.name}' · receiver Pd={cfg.receiver.pd}, Pfa={cfg.receiver.pfa}"
                   + (f" · PPO model {model.path.parent.name}/{model.path.name}" if key == "ppo" and model else ""))
        if key == "ppo" and model is None:
            note("PPO model not trained.", "warn")
            return
        if st.button("EVALUATE", type="primary"):
            bar = st.progress(0.0)
            res = run_experiment(cfg, [key], seeds, split=split, model_path=model.path if model else None,
                                 keep_traces=len(seeds), progress=lambda f, m: bar.progress(f, text=m))
            bar.empty()
            if res.skipped:
                note(f"{SCHEDULER_LABELS[key]} skipped: {res.skipped[key]}", "err")
                return
            st.session_state.eval_result = res
            st.session_state.run_history.append((f"{SCHEDULER_LABELS[key]} eval {split} {datetime.now():%H:%M:%S}", res))

    res = st.session_state.get("eval_result")
    if res is None:
        note("No evaluation run yet in this session.")
        return
    key = res.schedulers[0]
    df = res.per_episode
    with panel(f"Results — {SCHEDULER_LABELS[key]} · split '{res.split}' · n = {len(df)} episodes (mean ± std)"):
        items = []
        for m in KPI_METRICS:
            s = aggregate(df[m].tolist())
            items.append((METRICS[m]["label"], "Insufficient data" if s.mean is None else METRICS[m]["fmt"].format(s.mean),
                          f"± {METRICS[m]['fmt'].format(s.std)} (n={s.n})" if s.mean is not None else ""))
        kpis(items)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            cols = ["seed", "decisions", "detections", "misses", "false_alarms", "pd", "pfa", "interception_rate",
                    "avg_intercept_time", "intercept_time_error", "coverage", "cumulative_reward"]
            st.dataframe(df[cols], hide_index=True, width="stretch", height=300)
            st.download_button("EXPORT CSV", df.to_csv(index=False), file_name=f"eval_{key}_{res.split}.csv")
        with c2:
            metric = st.selectbox("Distribution of", KPI_METRICS, format_func=lambda m: METRICS[m]["label"])
            plot(metric_distribution(df, [key], metric, height=300), key="eval_dist")
        if st.button("SAVE RESULTS TO experiments/results"):
            out = res.save()
            st.success(f"Saved to {out}")

    with panel("Raw event trace"):
        seed = st.selectbox("Episode (scenario seed)", [s for (_, s) in res.traces])
        tr = res.traces[(key, seed)]
        scn = build_scenario(res.config.scenario if res.config.scenario.source != "turing" else _split_cfg(res), seed)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            plot(spectrum_activity_map(scn, tr, height=440, title=f"Seed {seed} — spectrum activity map"), key="eval_spec")
        with c2:
            show = ["step", "t_start", "band", "detected", "signal_present", "outcome", "new_intercepts", "reward"]
            st.dataframe(tr[show], hide_index=True, width="stretch", height=440)
        plot(event_timeline(tr, height=220), key="eval_events")


def _split_cfg(res):
    sc = res.config.copy().scenario
    sc.turing.split = res.split
    return sc


__all__ = ["render", "pd"]
