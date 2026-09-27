"""Scan trajectory (band vs time) and the detection / miss / false-alarm event timeline."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.visualization.spectrum import outcome_markers
from smartscan.visualization.style import OUTCOME_STYLE, SCHEDULER_COLORS, band_ticks, figure


def scan_trajectory(trace: pd.DataFrame, n_bands: int, scheduler: str = "", height: int = 300,
                    title: str = "Scan trajectory", show_outcomes: bool = True) -> go.Figure:
    fig = figure(title, height=height)
    if trace is None or trace.empty:
        _empty(fig, "No decisions yet")
    else:
        fig.add_trace(
            go.Scatter(
                x=trace["t_start"],
                y=trace["action"],
                mode="lines",
                line=dict(shape="hv", width=1.5, color=SCHEDULER_COLORS.get(scheduler, "#0072BD")),
                name=f"{SCHEDULER_LABELS.get(scheduler, scheduler) or 'Receiver'} tuned band",
                hoverinfo="skip",
            )
        )
        if show_outcomes:
            outcome_markers(fig, trace, "t_start", "action", size=7)
    fig.update_xaxes(title_text="Time (steps)")
    fig.update_yaxes(title_text="Band", range=[-0.5, n_bands - 0.5], **band_ticks(n_bands))
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


def compare_trajectories(traces: dict[str, pd.DataFrame], n_bands: int, height: int = 320,
                         title: str = "Scan trajectories (same scenario seed)") -> go.Figure:
    fig = figure(title, height=height)
    for key, tr in traces.items():
        if tr is None or tr.empty:
            continue
        fig.add_trace(go.Scatter(x=tr["t_start"], y=tr["action"], mode="lines",
                                 line=dict(shape="hv", width=1.5, color=SCHEDULER_COLORS.get(key)),
                                 name=SCHEDULER_LABELS.get(key, key)))
    fig.update_xaxes(title_text="Time (steps)")
    fig.update_yaxes(title_text="Band", range=[-0.5, n_bands - 0.5], **band_ticks(n_bands))
    return fig


def event_timeline(trace: pd.DataFrame, height: int = 220, title: str = "Event timeline") -> go.Figure:
    """One row per event type; markers at the dwell time. Intercepts of new transmissions on top."""
    fig = figure(title, height=height)
    rows = ["FALSE_ALARM", "MISS", "DETECTION"]
    labels = [OUTCOME_STYLE[r]["label"] for r in rows] + ["New intercept"]
    if trace is None or trace.empty:
        _empty(fig, "No events yet")
    else:
        for i, outcome in enumerate(rows):
            st = OUTCOME_STYLE[outcome]
            sub = trace[trace["outcome"] == outcome]
            fig.add_trace(go.Scatter(
                x=sub["t_start"], y=[i] * len(sub), mode="markers", name=st["label"],
                marker=dict(symbol=st["symbol"], color=st["color"], size=9, line=dict(width=1.5, color=st["color"])),
                customdata=sub[["step", "band"]], showlegend=False,
                hovertemplate="t=%{x}<br>%{customdata[1]}<br>" + st["label"] + "<extra></extra>",
            ))
        new = trace[trace["new_intercepts"] > 0]
        fig.add_trace(go.Scatter(
            x=new["t_start"], y=[3] * len(new), mode="markers", name="New intercept",
            marker=dict(symbol="triangle-up", color="#1a1a1a", size=9), showlegend=False,
            customdata=new[["band", "new_intercepts"]],
            hovertemplate="t=%{x}<br>%{customdata[0]}: %{customdata[1]} new transmission(s)<extra></extra>",
        ))
    fig.update_yaxes(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=labels, range=[-0.6, 3.6], showgrid=False)
    fig.update_xaxes(title_text="Time (steps)")
    fig.update_layout(margin=dict(l=110))
    return fig


def _empty(fig: go.Figure, text: str) -> None:
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(color="#7f7f7f"))
