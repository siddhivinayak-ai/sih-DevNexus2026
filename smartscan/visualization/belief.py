"""Belief / estimated band-activity views (receiver-side Bayesian belief)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from smartscan.visualization.style import INK_2, SEQ_BLUE, band_ticks, figure


def belief_map(beliefs: np.ndarray, times: list[int], n_bands: int, height: int = 300,
               title: str = "Belief map  P(band active | observations)") -> go.Figure:
    fig = figure(title, height=height)
    if beliefs is None or len(beliefs) == 0:
        fig.add_annotation(text="No observations yet", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#7f7f7f"))
    else:
        fig.add_trace(go.Heatmap(
            z=np.asarray(beliefs).T, x=list(times), y=list(range(n_bands)), zmin=0, zmax=1,
            colorscale=SEQ_BLUE,
            colorbar=dict(title=dict(text="P(active)", side="right", font=dict(size=10)), thickness=10,
                          outlinecolor="#262626", outlinewidth=1, tickfont=dict(size=9)),
            hovertemplate="t=%{x}<br>band %{y}<br>belief %{z:.3f}<extra></extra>",
        ))
    fig.update_xaxes(title_text="Time (steps)", showgrid=False)
    fig.update_yaxes(title_text="Band", showgrid=False, **band_ticks(n_bands))
    return fig


def belief_bars(belief: np.ndarray, n_bands: int, highlight: int | None = None, extra: dict | None = None,
                height: int = 260, title: str = "Current belief by band") -> go.Figure:
    fig = figure(title, height=height)
    colors = ["#86b6ef"] * n_bands
    if highlight is not None and 0 <= highlight < n_bands:
        colors[highlight] = "#1c5cab"
    fig.add_trace(go.Bar(x=list(range(n_bands)), y=belief, marker=dict(color=colors, line=dict(width=0)),
                         name="belief", hovertemplate="band %{x}<br>belief %{y:.3f}<extra></extra>",
                         text=[f"{b:.2f}" for b in belief] if n_bands <= 16 else None,
                         textposition="outside", textfont=dict(size=9, color=INK_2), cliponaxis=False))
    if extra:
        for name, (vals, color, dash) in extra.items():
            fig.add_trace(go.Scatter(x=list(range(n_bands)), y=vals, mode="lines+markers", name=name,
                                     line=dict(color=color, width=1.5, dash=dash), marker=dict(size=6)))
    fig.update_yaxes(title_text="Probability", range=[0, 1.12])
    fig.update_xaxes(title_text="Band (dark bar = next scan)" if highlight is not None else "Band",
                     **band_ticks(n_bands))
    fig.update_layout(showlegend=bool(extra), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig
