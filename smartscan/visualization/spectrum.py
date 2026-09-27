"""Spectrum activity map: frequency band (x) vs time (y, downward like a waterfall).

Background = ground-truth occupancy (evaluator view, hidden from the scheduler).
Overlay    = the receiver's dwells, marker symbol/colour by outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from smartscan.visualization.style import OUTCOME_STYLE, TRUTH, band_ticks, figure


def _band_axis_labels(scenario) -> list[str]:
    if scenario.band_unit == "GHz":
        return [f"{l}<br>{c:.2f} GHz" for l, c in zip(scenario.band_labels, scenario.band_centres)]
    return list(scenario.band_labels)


def outcome_markers(fig: go.Figure, df: pd.DataFrame, x: str, y: str, size: int = 8, **kw) -> None:
    for outcome, st in OUTCOME_STYLE.items():
        sub = df[df["outcome"] == outcome]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub[x],
                y=sub[y],
                mode="markers",
                name=st["label"],
                marker=dict(symbol=st["symbol"], color=st["color"], size=size,
                            line=dict(width=1.5 if "open" in st["symbol"] or st["symbol"] == "x" else 0.5,
                                      color=st["color"] if st["symbol"] != "circle" else "#ffffff")),
                customdata=np.stack([sub["step"], sub["band"], sub["reward"]], axis=1),
                hovertemplate="decision %{customdata[0]}<br>t=%{y}<br>band %{customdata[1]}<br>"
                + st["label"] + "<br>reward %{customdata[2]:.3f}<extra></extra>",
                **kw,
            )
        )


def spectrum_activity_map(scenario, trace: pd.DataFrame | None = None, t_max: int | None = None,
                          height: int = 460, title: str = "Spectrum activity map") -> go.Figure:
    T, N = scenario.total_steps, scenario.n_bands
    t_max = T if t_max is None else int(min(T, max(1, t_max)))
    gt = scenario.ground_truth[:t_max].astype(float)
    names = np.full((t_max, N), "", dtype=object)
    for e, em in enumerate(scenario.emitters):
        tt, bb = np.nonzero(scenario.activity[e, :t_max])
        for t, b in zip(tt, bb):
            names[t, b] = (names[t, b] + ", " if names[t, b] else "") + em.name
    fig = figure(title, height=height)
    fig.add_trace(
        go.Heatmap(
            z=gt,
            x=list(range(N)),
            y=list(range(t_max)),
            colorscale=[[0, "#ffffff"], [1, TRUTH]],
            zmin=0,
            zmax=1,
            showscale=False,
            customdata=names,
            hovertemplate="t=%{y}<br>band %{x}<br>active: %{customdata}<extra>ground truth</extra>",
            name="Ground truth occupancy",
            xgap=1,
        )
    )
    # legend proxy for the ground-truth fill
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Emitter active (ground truth)",
                             marker=dict(symbol="square", size=10, color=TRUTH)))
    if trace is not None and not trace.empty:
        tr = trace[trace["t_start"] < t_max]
        outcome_markers(fig, tr, "action", "t_start", size=7)
    labels = _band_axis_labels(scenario)
    fig.update_xaxes(title_text="Frequency band", range=[-0.5, N - 0.5], showgrid=False, **band_ticks(N, labels))
    fig.update_yaxes(title_text="Time (steps)", autorange="reversed", showgrid=False)
    fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0, yanchor="top"), margin=dict(b=70))
    if scenario.warmup_steps and t_max > scenario.warmup_steps:
        fig.add_hline(y=scenario.warmup_steps - 0.5, line=dict(color="#8c8c8c", dash="dash", width=1),
                      annotation_text="end of warm-up", annotation_position="bottom right")
    return fig
