"""MATLAB-like Plotly styling: boxed axes, inward ticks, light grid, boxed legend.

Colour roles (validated with the dataviz palette checker, light surface):
    schedulers  MATLAB 'lines' order with the too-light yellow replaced by a sea green
    outcomes    status colours, always paired with a distinct marker symbol + label
    magnitude   single-hue blue sequential ramp (belief, occupancy)
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT = "Segoe UI, Helvetica, Arial, sans-serif"
INK = "#1a1a1a"
INK_2 = "#4d4d4d"
AXIS = "#262626"
GRID = "#e3e3e3"
SURFACE = "#ffffff"
TRUTH = "#9aa4ae"  # ground-truth occupancy (neutral gray, recessive)

SCHEDULER_COLORS = {
    "open_loop": "#0072BD",
    "random": "#D95319",
    "pomdp": "#7E2F8E",
    "ppo": "#2E8B57",
}

OUTCOME_STYLE = {
    "DETECTION": {"color": "#0ca30c", "symbol": "circle", "label": "Detection"},
    "FALSE_ALARM": {"color": "#d03b3b", "symbol": "x", "label": "False alarm"},
    "MISS": {"color": "#ec835a", "symbol": "diamond-open", "label": "Missed detection"},
    "NO_SIGNAL": {"color": "#5a6570", "symbol": "line-ew-open", "label": "No signal"},
}

SEQ_BLUE = [
    [0.0, "#ffffff"],
    [0.15, "#cde2fb"],
    [0.35, "#86b6ef"],
    [0.55, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # all-pairs-safe first three slots
OTHER = "#b3b3b3"


def _axis() -> dict:
    return dict(
        showline=True,
        linecolor=AXIS,
        linewidth=1,
        mirror=True,
        ticks="inside",
        ticklen=4,
        tickcolor=AXIS,
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        title=dict(font=dict(size=11, color=INK)),
        tickfont=dict(size=10, color=INK_2),
    )


TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=11, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        title=dict(font=dict(size=12, color=INK), x=0.0, xanchor="left", y=0.98, yanchor="top", pad=dict(l=4)),
        xaxis=_axis(),
        yaxis=_axis(),
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#bfbfbf",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=56, r=16, t=36, b=40),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#8c8c8c", font=dict(family=FONT, size=11, color=INK)),
        colorway=list(SCHEDULER_COLORS.values()),
        bargap=0.25,
    )
)
pio.templates["smartscan"] = TEMPLATE


def figure(title: str = "", height: int = 320, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="smartscan", title_text=title, height=height, **layout)
    return fig


def style_subplots(fig: go.Figure) -> go.Figure:
    """Apply the boxed-axis look to every subplot axis."""
    fig.update_xaxes(**{k: v for k, v in _axis().items() if k != "title"})
    fig.update_yaxes(**{k: v for k, v in _axis().items() if k != "title"})
    fig.update_annotations(font=dict(size=11, color=INK))
    return fig


def band_ticks(n_bands: int, labels: list[str] | None = None) -> dict:
    labels = labels or [f"F{i + 1}" for i in range(n_bands)]
    step = max(1, n_bands // 16)
    idx = list(range(0, n_bands, step))
    return dict(tickmode="array", tickvals=idx, ticktext=[labels[i] for i in idx])
