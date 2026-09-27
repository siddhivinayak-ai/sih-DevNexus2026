"""Reward curves (episode) and PPO training curves (real logged values only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from smartscan.visualization.style import SCHEDULER_COLORS, TEMPLATE, figure, style_subplots


def reward_curve(trace: pd.DataFrame, height: int = 320, color: str = "#0072BD",
                 title: str = "Reward") -> go.Figure:
    """Per-decision reward (top) and cumulative reward (bottom): two panels, one scale each."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Reward per decision", "Cumulative reward"))
    fig.update_layout(template=TEMPLATE, height=height, title_text=title, showlegend=False)
    if trace is not None and not trace.empty:
        fig.add_trace(go.Bar(x=trace["t_start"], y=trace["reward"], marker=dict(color=color, line=dict(width=0)),
                             hovertemplate="t=%{x}<br>r=%{y:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trace["t_start"], y=trace["reward"].cumsum(), mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate="t=%{x}<br>sum r=%{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Time (steps)", row=2, col=1)
    fig.update_layout(margin=dict(t=48))
    return style_subplots(fig)


def reward_terms_bar(terms: dict[str, float], height: int = 220, title: str = "Reward decomposition (last decision)") -> go.Figure:
    fig = figure(title, height=height)
    keys = list(terms)
    vals = [terms[k] for k in keys]
    fig.add_trace(go.Bar(y=keys, x=vals, orientation="h",
                         marker=dict(color=["#2a78d6" if v >= 0 else "#d03b3b" for v in vals], line=dict(width=0)),
                         text=[f"{v:+.3f}" for v in vals], textposition="outside", cliponaxis=False,
                         hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Contribution", zeroline=True, zerolinecolor="#262626")
    fig.update_layout(margin=dict(l=100, r=40))
    return fig


def training_reward_curve(episode_rewards: list[tuple[int, float]], validation: list[dict] | None = None,
                          height: int = 320, window: int = 20) -> go.Figure:
    fig = figure("Episode reward during training", height=height)
    if not episode_rewards:
        fig.add_annotation(text="No completed training episodes yet", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#7f7f7f"))
        return fig
    ts, rs = zip(*episode_rewards)
    fig.add_trace(go.Scattergl(x=ts, y=rs, mode="markers", name="Training episode",
                               marker=dict(size=4, color="#86b6ef")))
    if len(rs) >= window:
        roll = pd.Series(rs).rolling(window).mean()
        fig.add_trace(go.Scatter(x=ts, y=roll, mode="lines", name=f"Rolling mean ({window})",
                                 line=dict(color=SCHEDULER_COLORS["open_loop"], width=2)))
    if validation:
        v = pd.DataFrame(validation)
        fig.add_trace(go.Scatter(x=v["timesteps"], y=v["mean_reward"], mode="lines+markers",
                                 name="Validation mean (unseen seeds)",
                                 error_y=dict(type="data", array=v["std_reward"], thickness=1, width=3),
                                 line=dict(color=SCHEDULER_COLORS["ppo"], width=2), marker=dict(size=7)))
    fig.update_xaxes(title_text="Timesteps")
    fig.update_yaxes(title_text="Episode reward")
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    return fig


LOSS_KEYS = [
    ("train/policy_gradient_loss", "Policy-gradient loss"),
    ("train/value_loss", "Value loss"),
    ("train/entropy_loss", "Entropy loss"),
    ("train/approx_kl", "Approx. KL"),
    ("train/clip_fraction", "Clip fraction"),
    ("train/explained_variance", "Explained variance"),
]


def training_losses(train_logs: list[dict], height: int = 380) -> go.Figure:
    """Small multiples of the SB3 PPO update statistics (one scale per panel)."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=[l for _, l in LOSS_KEYS],
                        horizontal_spacing=0.08, vertical_spacing=0.2)
    fig.update_layout(template=TEMPLATE, height=height, showlegend=False,
                      title_text="PPO update statistics (Stable-Baselines3 logger)")
    if train_logs:
        df = pd.DataFrame(train_logs)
        for i, (key, label) in enumerate(LOSS_KEYS):
            if key in df:
                fig.add_trace(go.Scatter(x=df["timesteps"], y=df[key], mode="lines",
                                         line=dict(color="#0072BD", width=1.5), name=label,
                                         hovertemplate="t=%{x}<br>%{y:.4g}<extra>" + label + "</extra>"),
                              row=i // 3 + 1, col=i % 3 + 1)
    else:
        fig.add_annotation(text="No PPO updates logged yet", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#7f7f7f"))
    fig.update_layout(margin=dict(t=60))
    return style_subplots(fig)


def monitor_curve(monitor_df: pd.DataFrame, window: int = 20, height: int = 300) -> go.Figure:
    """Episode reward from SB3 Monitor CSVs of a finished run."""
    df = monitor_df.sort_values("t")
    ts = np.cumsum(df["l"].to_numpy())
    return training_reward_curve(list(zip(ts.tolist(), df["r"].tolist())), None, height, window)
