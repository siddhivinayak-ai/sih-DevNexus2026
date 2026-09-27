"""Algorithm Comparison: Open Loop / Random / POMDP / PPO on identical scenarios, seeds and receiver."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from smartscan.environment.scenario import build_scenario
from smartscan.evaluation.comparison import ALL_SCHEDULERS, paired_differences
from smartscan.evaluation.experiment import run_experiment
from smartscan.evaluation.metrics import HEADLINE, METRICS
from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.ui.common import active_model, get_config, note, panel, plot, seed_picker
from smartscan.visualization.comparison import band_share_chart, metric_distribution, metric_panels, paired_difference_chart
from smartscan.visualization.trajectory import compare_trajectories

TABLE_METRICS = HEADLINE + ["normalized_delay", "detections", "misses", "false_alarms", "hit_rate",
                            "revisit_interval", "immediate_revisit_rate", "cumulative_reward"]


def render() -> None:
    cfg = get_config()
    n = cfg.scenario.n_bands
    model = active_model(n)
    with panel("Experiment setup"):
        c1, c2 = st.columns([1.2, 3])
        options = [k for k in ALL_SCHEDULERS if k != "ppo" or model is not None]
        chosen = c1.multiselect("Schedulers", options, default=options, format_func=lambda k: SCHEDULER_LABELS[k])
        with c2:
            split, seeds = seed_picker(cfg, "cmp")
        if model is None:
            note("PPO model not trained — PPO is excluded from the comparison.", "warn")
        st.caption("Every scheduler runs on the same scenario realisations (same seeds), the same receiver and "
                   "the same detector random numbers; only the scan decisions differ.")
        if st.button("RUN COMPARISON", type="primary", disabled=not chosen):
            bar = st.progress(0.0)
            res = run_experiment(cfg, chosen, seeds, split=split, model_path=model.path if model else None,
                                 keep_traces=3, progress=lambda f, m: bar.progress(f, text=m))
            bar.empty()
            st.session_state.cmp_result = res
            st.session_state.run_history.append((f"Comparison {split} n={len(seeds)} {datetime.now():%H:%M:%S}", res))
    res = st.session_state.get("cmp_result")
    if res is None:
        note("No comparison run in this session. Saved results can be opened on the Analytics page.")
        return
    show_result(res, key="cmp")


def show_result(res, key: str) -> None:
    for k, why in res.skipped.items():
        note(f"{SCHEDULER_LABELS.get(k, k)} skipped: {why}", "warn")
    n_eps = len(res.seeds)
    with panel(f"Measured results — split '{res.split}', seeds {res.seeds[0]}–{res.seeds[-1]}, "
               f"n = {n_eps} episodes per scheduler (mean ± std)"):
        st.dataframe(res.table(TABLE_METRICS), width="stretch", height=600)
        c1, c2, c3 = st.columns(3)
        c1.download_button("EXPORT TABLE (CSV)", res.table(TABLE_METRICS).to_csv(), file_name="comparison_table.csv",
                           key=f"{key}_dl1")
        c2.download_button("EXPORT PER-EPISODE (CSV)", res.per_episode.to_csv(index=False), file_name="per_episode.csv",
                           key=f"{key}_dl2")
        if c3.button("SAVE TO experiments/results", key=f"{key}_save"):
            st.success(f"Saved to {res.save()}")
    diffs = paired_differences(res)
    if not diffs.empty:
        with panel("Paired differences vs Open Loop (same seeds) — mean and 95% confidence interval"):
            t = diffs.copy()
            t["metric"] = t["metric"].map(lambda m: METRICS[m]["label"])
            t["scheduler"] = t["scheduler"].map(SCHEDULER_LABELS)
            t["difference"] = [("Insufficient data" if d is None or pd.isna(d) else f"{d:+.4g} ± {c:.3g}")
                               for d, c in zip(diffs["mean_diff"], diffs["ci95"])]
            c1, c2 = st.columns([1.2, 1])
            with c1:
                wide = t.pivot(index="metric", columns="scheduler", values="difference")
                st.dataframe(wide, width="stretch")
                st.caption("An interval that excludes 0 indicates a difference unlikely to be due to seed-to-seed variation "
                           "alone (paired t-interval). Direction of 'better' depends on the metric.")
            with c2:
                m = st.selectbox("Metric", HEADLINE, format_func=lambda x: METRICS[x]["label"], key=f"{key}_pdm")
                plot(paired_difference_chart(diffs, m), key=f"{key}_pd")
    with panel("Metric panels"):
        plot(metric_panels(res.per_episode, res.schedulers, HEADLINE), key=f"{key}_panels")
    c1, c2 = st.columns(2)
    with c1:
        with panel("Per-episode distribution"):
            m = st.selectbox("Metric", HEADLINE, index=2, format_func=lambda x: METRICS[x]["label"], key=f"{key}_dm")
            plot(metric_distribution(res.per_episode, res.schedulers, m), key=f"{key}_dist")
    with c2:
        with panel("Band revisit frequency"):
            plot(band_share_chart(res.per_episode, res.schedulers, res.config.scenario.n_bands), key=f"{key}_share")
    seeds_with_traces = sorted({s for (_, s) in res.traces})
    if seeds_with_traces:
        with panel("Scan trajectories on one seed"):
            seed = st.selectbox("Seed", seeds_with_traces, key=f"{key}_trseed")
            traces = {k: res.traces[(k, seed)] for k in res.schedulers if (k, seed) in res.traces}
            plot(compare_trajectories(traces, res.config.scenario.n_bands), key=f"{key}_traj")
            if res.config.scenario.source == "synthetic":
                scn = build_scenario(res.config.scenario, seed)
                st.caption("Emitters: " + "; ".join(f"{e.name}: {e.describe()}" for e in scn.emitters))
