"""Analytics: saved experiment results, session run history, training-vs-unseen generalisation check."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from smartscan.evaluation.experiment import list_results, load_result, run_experiment
from smartscan.evaluation.metrics import HEADLINE, METRICS, aggregate
from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.ui.common import active_model, get_config, note, panel
from smartscan.ui.comparison import show_result


def render() -> None:
    tab_saved, tab_session, tab_gen = st.tabs(["Saved results", "Session history", "Training vs unseen seeds"])
    with tab_saved:
        results = list_results()
        if not results:
            note("No saved results in experiments/results. Save one from Algorithm Comparison or run "
                 "<code>python -m smartscan.evaluation.comparison</code>.")
        else:
            path = st.selectbox("Result folder", results, format_func=lambda p: p.name)
            show_result(load_result(path), key="saved")
    with tab_session:
        hist = st.session_state.get("run_history", [])
        if not hist:
            note("No runs in this session yet.")
        else:
            labels = [h[0] for h in hist]
            i = st.selectbox("Run", range(len(hist)), index=len(hist) - 1, format_func=lambda j: labels[j])
            show_result(hist[i][1], key="session")
    with tab_gen:
        _generalisation()


def _generalisation() -> None:
    cfg = get_config()
    model = active_model(cfg.scenario.n_bands)
    with panel("Generalisation check"):
        st.caption("Runs the same schedulers on training seeds and on unseen test seeds. A large gap for PPO "
                   "indicates over-fitting to the training scenarios.")
        n = st.number_input("Episodes per split", 5, 100, 20)
        keys = ["open_loop", "pomdp"] + (["ppo"] if model else [])
        if model is None:
            note("PPO model not trained — showing baselines only.", "warn")
        if st.button("RUN CHECK", type="primary"):
            rows = []
            for split in ("train", "test"):
                seeds = cfg.seeds.seeds(split)[: int(n)]
                res = run_experiment(cfg, keys, seeds, split=split, model_path=model.path if model else None, keep_traces=0)
                for k in res.schedulers:
                    df = res.per_episode[res.per_episode["scheduler"] == k]
                    row = {"scheduler": SCHEDULER_LABELS[k], "split": f"{split} ({seeds[0]}–{seeds[-1]})"}
                    for m in HEADLINE:
                        row[METRICS[m]["label"]] = aggregate(df[m].tolist()).format(METRICS[m]["fmt"])
                    rows.append(row)
            st.session_state.gen_table = pd.DataFrame(rows)
        if "gen_table" in st.session_state:
            st.dataframe(st.session_state.gen_table.sort_values(["scheduler", "split"]), hide_index=True, width="stretch")
