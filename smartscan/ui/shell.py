"""Application shell: navigation, toolstrip, workspace browser (sidebar)."""

from __future__ import annotations

import html

import streamlit as st

from smartscan.environment import turing
from smartscan.ui import (
    analytics,
    comparison,
    dataset,
    docs,
    evaluation,
    overview,
    scenario,
    simulation,
    training,
)
from smartscan.ui.common import active_model, get_config, init_state, inject_css, toolstrip

PAGES = {
    "Project": [
        (overview.render, "Overview", ":material/home:"),
        (scenario.render, "Scenario Builder", ":material/tune:"),
    ],
    "Simulation": [
        (simulation.render, "Live Simulation", ":material/play_circle:"),
    ],
    "Learning": [
        (training.render, "RL Training", ":material/model_training:"),
        (evaluation.render, "Policy Evaluation", ":material/fact_check:"),
    ],
    "Analysis": [
        (comparison.render, "Algorithm Comparison", ":material/compare_arrows:"),
        (analytics.render, "Analytics", ":material/query_stats:"),
        (dataset.render, "Dataset Explorer", ":material/database:"),
    ],
    "Reference": [
        (docs.render, "Methodology", ":material/menu_book:"),
    ],
}


def _page(fn, title, icon):
    def wrapped():
        toolstrip(title)
        fn()

    wrapped.__name__ = f"page_{fn.__module__.rsplit('.', 1)[-1]}"
    return st.Page(wrapped, title=title, icon=icon, url_path=wrapped.__name__[5:])


def workspace() -> None:
    cfg = get_config()
    sc, rx = cfg.scenario, cfg.receiver
    model = active_model(sc.n_bands)
    ok, _ = turing.status(sc.turing.local_dir)
    rows = [
        ("config", st.session_state.get("config_path", "(edited)")),
        ("scenario", sc.name),
        ("source", sc.source),
        ("bands", f"{sc.n_bands}  ({sc.f_start_ghz:.2f}–{sc.f_start_ghz + (sc.n_bands - 1) * sc.band_spacing_ghz:.2f} GHz)"
         if sc.source == "synthetic" else str(sc.n_bands)),
        ("episode", f"{sc.episode_length} steps"),
        ("receiver", f"{rx.detection_model}, Pd={rx.pd}, Pfa={rx.pfa}"),
        ("inst. BW", f"{rx.instantaneous_bands} band(s)"),
        ("seeds", f"train {cfg.seeds.train[0]}–{cfg.seeds.train[1]}, test {cfg.seeds.test[0]}–{cfg.seeds.test[1]}"),
        ("PPO model", model.path.parent.name + "/" + model.path.name if model else "not trained"),
        ("Turing data", "available" if ok else "not configured"),
    ]
    body = "<br>".join(
        f'<span class="k">{html.escape(k):<10}</span> <span class="v">{html.escape(str(v))}</span>' for k, v in rows
    )
    st.sidebar.markdown('<div class="ss-sidehdr">Workspace</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="ss-ws">{body}</div>', unsafe_allow_html=True)
    tp = st.session_state.get("training")
    if tp is not None and tp.status in ("running", "stopping"):
        st.sidebar.markdown('<div class="ss-sidehdr">Background job</div>', unsafe_allow_html=True)
        st.sidebar.progress(min(1.0, tp.timesteps / max(1, tp.total_timesteps)),
                            text=f"PPO training {tp.timesteps}/{tp.total_timesteps}")
    st.sidebar.caption("Research simulation prototype. Not an operational EW system.")


def run() -> None:
    init_state()
    inject_css()
    nav = st.navigation({section: [_page(*p) for p in pages] for section, pages in PAGES.items()},
                        position="sidebar", expanded=True)
    workspace()
    nav.run()
