"""Algorithm comparison figures: mean +/- std per metric (small multiples) and per-episode spread."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartscan.evaluation.metrics import METRICS, aggregate
from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.visualization.style import SCHEDULER_COLORS, TEMPLATE, style_subplots


def metric_panels(per_episode: pd.DataFrame, schedulers: list[str], metrics: list[str],
                  cols: int = 4, height_per_row: int = 230) -> go.Figure:
    """One panel per metric; bar = mean, whisker = +/- 1 std, over episodes."""
    rows = int(np.ceil(len(metrics) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[METRICS[m]["label"] for m in metrics],
                        horizontal_spacing=0.06, vertical_spacing=0.16)
    labels = [SCHEDULER_LABELS.get(k, k) for k in schedulers]
    for i, m in enumerate(metrics):
        r, c = i // cols + 1, i % cols + 1
        for key, label in zip(schedulers, labels):
            vals = per_episode.loc[per_episode["scheduler"] == key, m].tolist() if m in per_episode else []
            s = aggregate(vals)
            fig.add_trace(go.Bar(
                x=[label], y=[s.mean if s.mean is not None else 0],
                error_y=dict(type="data", array=[s.std or 0], thickness=1.2, width=4, color="#262626"),
                marker=dict(color=SCHEDULER_COLORS.get(key), line=dict(width=0)),
                name=label, legendgroup=key, showlegend=(i == 0),
                hovertemplate=f"{label}<br>{METRICS[m]['label']}<br>"
                + ("mean %{y:.4g} ± " + f"{(s.std or 0):.4g} (n={s.n})" if s.mean is not None else "Insufficient data")
                + "<extra></extra>",
            ), row=r, col=c)
    fig.update_layout(template=TEMPLATE, height=rows * height_per_row + 40, bargap=0.35,
                      title_text="Mean ± 1 std across episodes",
                      legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"),
                      margin=dict(t=70))
    fig.update_xaxes(showticklabels=False)
    return style_subplots(fig)


def metric_distribution(per_episode: pd.DataFrame, schedulers: list[str], metric: str, height: int = 320) -> go.Figure:
    """Per-episode values (box + points) for one metric."""
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height, title_text=f"{METRICS[metric]['label']} — per-episode distribution",
                      showlegend=False)
    for key in schedulers:
        vals = pd.to_numeric(per_episode.loc[per_episode["scheduler"] == key, metric], errors="coerce").dropna()
        fig.add_trace(go.Box(y=vals, name=SCHEDULER_LABELS.get(key, key), boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(size=4, color=SCHEDULER_COLORS.get(key), opacity=0.6),
                             line=dict(color=SCHEDULER_COLORS.get(key), width=1.5), fillcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title_text=METRICS[metric]["label"])
    return fig


def paired_difference_chart(diffs: pd.DataFrame, metric: str, height: int = 240) -> go.Figure:
    d = diffs[diffs["metric"] == metric].dropna(subset=["mean_diff"])
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height, showlegend=False,
                      title_text=f"Paired difference vs Open Loop — {METRICS[metric]['label']} (mean, 95% CI)")
    for _, row in d.iterrows():
        fig.add_trace(go.Scatter(x=[row["mean_diff"]], y=[SCHEDULER_LABELS.get(row["scheduler"], row["scheduler"])],
                                 mode="markers", marker=dict(size=10, color=SCHEDULER_COLORS.get(row["scheduler"])),
                                 error_x=dict(type="data", array=[row["ci95"]], thickness=1.5, width=5, color="#262626"),
                                 hovertemplate="Δ = %{x:.4g} ± " + f"{row['ci95']:.4g} (n={row['n']})<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#262626", width=1, dash="dash"))
    fig.update_xaxes(title_text="Scheduler − Open Loop")
    fig.update_layout(margin=dict(l=90))
    return fig


def band_share_chart(per_episode: pd.DataFrame, schedulers: list[str], n_bands: int, height: int = 300) -> go.Figure:
    """Mean share of dwells per band (band revisit frequency)."""
    import json

    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height, title_text="Band revisit frequency (share of dwells)",
                      barmode="group", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    for key in schedulers:
        rows = per_episode.loc[per_episode["scheduler"] == key, "band_scan_share"]
        arr = np.array([json.loads(r) if isinstance(r, str) else r for r in rows], dtype=float)
        if arr.size == 0:
            continue
        fig.add_trace(go.Bar(x=[f"F{i + 1}" for i in range(n_bands)], y=arr.mean(axis=0),
                             name=SCHEDULER_LABELS.get(key, key), marker=dict(color=SCHEDULER_COLORS.get(key), line=dict(width=0))))
    fig.update_yaxes(title_text="Share of dwells")
    fig.update_xaxes(title_text="Band")
    return fig
