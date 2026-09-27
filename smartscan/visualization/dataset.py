"""Turing PDW plots. Up to three emitter labels are coloured; the rest fold into 'Other'."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartscan.environment.turing import PDW_LABELS
from smartscan.visualization.style import CATEGORICAL, OTHER, TEMPLATE, figure, style_subplots


def _groups(df: pd.DataFrame, highlight: list[int]) -> list[tuple[str, pd.DataFrame, str]]:
    out = []
    rest = df[~df["label"].isin(highlight)]
    if not rest.empty:
        out.append(("Other emitters", rest, OTHER))
    for lab, color in zip(highlight, CATEGORICAL):
        out.append((f"Emitter {lab}", df[df["label"] == lab], color))
    return out


def top_labels(df: pd.DataFrame, k: int = 3) -> list[int]:
    return df["label"].value_counts().head(k).index.astype(int).tolist()


def pdw_scatter(df: pd.DataFrame, x: str, y: str, highlight: list[int], height: int = 380) -> go.Figure:
    fig = figure(f"{PDW_LABELS.get(y, y)} vs {PDW_LABELS.get(x, x)}", height=height)
    for name, sub, color in _groups(df, highlight):
        fig.add_trace(go.Scattergl(x=sub[x], y=sub[y], mode="markers", name=name,
                                   marker=dict(size=3 if name.startswith("Other") else 4, color=color,
                                               opacity=0.5 if name.startswith("Other") else 0.85)))
    fig.update_xaxes(title_text=PDW_LABELS.get(x, x))
    fig.update_yaxes(title_text=PDW_LABELS.get(y, y))
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


def pdw_histograms(df: pd.DataFrame, cols: list[str], bins: int = 60, height: int = 300) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=[PDW_LABELS.get(c, c) for c in cols],
                        horizontal_spacing=0.06)
    for i, c in enumerate(cols):
        fig.add_trace(go.Histogram(x=df[c], nbinsx=bins, marker=dict(color="#0072BD", line=dict(width=0)),
                                   name=c, showlegend=False), row=1, col=i + 1)
    fig.update_layout(template=TEMPLATE, height=height, title_text="Feature distributions (sampled pulses)",
                      bargap=0.05, margin=dict(t=60))
    return style_subplots(fig)


def pulse_density(df: pd.DataFrame, n_time: int = 100, n_freq: int = 40, height: int = 340) -> go.Figure:
    """Pulse count per (ToA bin, CF bin): temporal x frequency occupancy of the sample."""
    fig = figure("Pulse density — time x centre frequency", height=height)
    h, xe, ye = np.histogram2d(df["toa"], df["cf"], bins=[n_time, n_freq])
    fig.add_trace(go.Heatmap(z=h.T, x=0.5 * (xe[1:] + xe[:-1]), y=0.5 * (ye[1:] + ye[:-1]),
                             colorscale=[[0, "#ffffff"], [0.2, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
                             colorbar=dict(title=dict(text="pulses", font=dict(size=10)), thickness=10),
                             hovertemplate="ToA %{x:.4g}<br>CF %{y:.4g}<br>%{z} pulses<extra></extra>"))
    fig.update_xaxes(title_text=PDW_LABELS["toa"], showgrid=False)
    fig.update_yaxes(title_text=PDW_LABELS["cf"], showgrid=False)
    return fig


def label_counts(df: pd.DataFrame, height: int = 280) -> go.Figure:
    vc = df["label"].value_counts().sort_index()
    fig = figure("Pulses per emitter label (sample)", height=height)
    fig.add_trace(go.Bar(x=[str(i) for i in vc.index], y=vc.values, marker=dict(color="#0072BD", line=dict(width=0)),
                         hovertemplate="label %{x}<br>%{y} pulses<extra></extra>"))
    fig.update_xaxes(title_text="Emitter label", type="category")
    fig.update_yaxes(title_text="Pulses")
    return fig
