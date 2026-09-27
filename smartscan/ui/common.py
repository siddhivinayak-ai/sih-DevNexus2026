"""Shared UI infrastructure: styling, session state, model registry, small widgets."""

from __future__ import annotations

import html
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from smartscan.config import MODELS_DIR, PROJECT_ROOT, ExperimentConfig, load_config
from smartscan.receiver.observation import observation_dim

CSS = """
<style>
:root {
  --ss-ink: #1a1a1a; --ss-ink2: #4d4d4d; --ss-muted: #6b6b6b;
  --ss-line: #c8c8c8; --ss-panel: #f0f0f0; --ss-strip: #e6e6e6;
  --ss-blue: #0072BD; --ss-navy: #1f3a5f;
}
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
  font-family: "Segoe UI", Helvetica, Arial, sans-serif;
}
.block-container { padding-top: 0.6rem; padding-bottom: 2rem; max-width: 100%; }
header[data-testid="stHeader"] { height: 0; background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { font-weight: 600; letter-spacing: 0; color: var(--ss-ink); }
h2 { font-size: 1.15rem; margin: 0.2rem 0 0.4rem 0; }
h3 { font-size: 1.0rem; margin: 0.2rem 0 0.3rem 0; }
p, li { font-size: 0.9rem; }

/* toolstrip header */
.ss-toolstrip { border: 1px solid var(--ss-line); border-top: 3px solid var(--ss-navy);
  background: #f7f7f7; padding: 6px 12px; display: flex; align-items: baseline; gap: 14px;
  margin-bottom: 8px; flex-wrap: wrap; }
.ss-toolstrip .ss-app { font-weight: 700; font-size: 1.05rem; color: var(--ss-navy); letter-spacing: 0.06em; }
.ss-toolstrip .ss-sub { font-size: 0.82rem; color: var(--ss-ink2); }
.ss-toolstrip .ss-page { font-size: 0.82rem; color: var(--ss-ink); margin-left: auto; font-weight: 600; }

/* panels: bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 2px !important; border-color: var(--ss-line) !important; }
.ss-ptitle { background: var(--ss-strip); border-bottom: 1px solid var(--ss-line);
  margin: -0.5rem -0.5rem 0.5rem -0.5rem; padding: 3px 8px; font-size: 0.78rem;
  font-weight: 600; color: var(--ss-ink); text-transform: uppercase; letter-spacing: 0.04em; }

/* buttons: flat, rectangular */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 2px; border: 1px solid #adadad; background: #f3f3f3; color: var(--ss-ink);
  font-size: 0.82rem; padding: 0.2rem 0.7rem; min-height: 2rem; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--ss-blue); background: #e5f1fb; color: var(--ss-ink); }
.stButton > button[kind="primary"] { background: var(--ss-blue); border-color: #005a96; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #005a96; color: #fff; }
.stButton > button:disabled { color: #9a9a9a; background: #f7f7f7; border-color: #d6d6d6; }

/* inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] { border-radius: 2px !important; }
label, .stSlider label, .stNumberInput label { font-size: 0.8rem !important; color: var(--ss-ink2) !important; }

/* KPI strip */
.ss-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 0;
  border: 1px solid var(--ss-line); background: #fff; margin-bottom: 8px; }
.ss-kpi { padding: 4px 8px; border-right: 1px solid #e3e3e3; border-bottom: 1px solid #e3e3e3; min-width: 0; }
.ss-kpi .l { font-size: 0.68rem; color: var(--ss-muted); text-transform: uppercase; letter-spacing: 0.04em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ss-kpi .v { font-family: Consolas, "Courier New", monospace; font-size: 1.02rem; color: var(--ss-ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ss-kpi .s { font-size: 0.68rem; color: var(--ss-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ss-kpi .v.good { color: #0a7a0a; } .ss-kpi .v.bad { color: #b3261e; } .ss-kpi .v.warn { color: #9a5b00; }

/* command window */
.ss-cmd { font-family: Consolas, "Courier New", monospace; font-size: 0.78rem; background: #fff;
  border: 1px solid var(--ss-line); padding: 6px 8px; white-space: pre; overflow-x: auto;
  max-height: 260px; overflow-y: auto; color: var(--ss-ink); line-height: 1.35; }
.ss-cmd .p { color: var(--ss-blue); } .ss-cmd .ok { color: #0a7a0a; } .ss-cmd .bad { color: #b3261e; }
.ss-cmd .warn { color: #9a5b00; } .ss-cmd .dim { color: #7f7f7f; }

/* notices */
.ss-note { border: 1px solid var(--ss-line); border-left: 3px solid var(--ss-blue); background: #f7fafd;
  padding: 5px 10px; font-size: 0.84rem; margin: 4px 0 8px 0; }
.ss-note.warn { border-left-color: #c77c02; background: #fdf8ef; }
.ss-note.err { border-left-color: #b3261e; background: #fdf3f2; }

/* sidebar = workspace browser */
section[data-testid="stSidebar"] { background: #f0f0f0; border-right: 1px solid var(--ss-line); }
section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
.ss-ws { font-family: Consolas, "Courier New", monospace; font-size: 0.74rem; background: #fff;
  border: 1px solid var(--ss-line); padding: 4px 6px; line-height: 1.45; }
.ss-ws .k { color: var(--ss-muted); } .ss-ws .v { color: var(--ss-ink); }
.ss-sidehdr { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--ss-ink2); margin: 8px 0 3px 0; }

/* dataframes */
div[data-testid="stDataFrame"] { border: 1px solid var(--ss-line); }
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--ss-line); }
.stTabs [data-baseweb="tab"] { border: 1px solid transparent; border-bottom: none; padding: 4px 14px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { border-color: var(--ss-line); background: #fff; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- session state
def init_state() -> None:
    ss = st.session_state
    if "config" not in ss:
        ss.config = load_config(None)
        ss.config_path = "configs/default.yaml"
    ss.setdefault("model_path", None)
    ss.setdefault("run_history", [])  # list of (label, ExperimentResult)


def get_config() -> ExperimentConfig:
    return st.session_state.config


def set_config(cfg: ExperimentConfig, path: str | None = None) -> None:
    st.session_state.config = cfg
    st.session_state.config_version = st.session_state.get("config_version", 0) + 1
    if path is not None:
        st.session_state.config_path = path
    # a new configuration invalidates the live simulation
    st.session_state.pop("live", None)


# ---------------------------------------------------------------- model registry
@dataclass
class ModelInfo:
    path: Path
    meta: dict

    @property
    def label(self) -> str:
        rel = self.path.relative_to(PROJECT_ROOT) if self.path.is_relative_to(PROJECT_ROOT) else self.path
        m = self.meta
        extra = f"{m.get('n_bands', '?')} bands, {m.get('timesteps_trained', '?')} steps" if m else "no metadata"
        return f"{rel.as_posix()}  ({extra})"

    def compatible(self, n_bands: int) -> bool:
        if "obs_dim" in self.meta:
            return int(self.meta["obs_dim"]) == observation_dim(n_bands)
        return True


def list_models() -> list[ModelInfo]:
    out = []
    if not MODELS_DIR.exists():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = {}
        if (d / "metadata.json").exists():
            try:
                meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        for name in ("best_model.zip", "final_model.zip"):
            if (d / name).exists():
                out.append(ModelInfo(d / name, meta))
    return out


def active_model(n_bands: int) -> ModelInfo | None:
    """Selected model if compatible with the scenario, else the first compatible one."""
    models = [m for m in list_models() if m.compatible(n_bands)]
    sel = st.session_state.get("model_path")
    for m in models:
        if sel and str(m.path) == str(sel):
            return m
    scen = get_config().scenario.name
    for m in models:
        if m.meta.get("scenario") == scen and m.path.name == "best_model.zip":
            return m
    return models[0] if models else None


# ---------------------------------------------------------------- widgets
def toolstrip(page: str) -> None:
    st.markdown(
        f"""<div class="ss-toolstrip"><span class="ss-app">SMARTSCAN</span>
        <span class="ss-sub">Adaptive RF Spectrum Scanning &amp; Interception — research simulation platform</span>
        <span class="ss-page">{html.escape(page)}</span></div>""",
        unsafe_allow_html=True,
    )


@contextmanager
def panel(title: str, border: bool = True):
    with st.container(border=border):
        st.markdown(f'<div class="ss-ptitle">{html.escape(title)}</div>', unsafe_allow_html=True)
        yield


def kpis(items: list[tuple], ) -> None:
    """items: (label, value, sub=None, tone=None) with tone in {good, bad, warn}."""
    cells = []
    for it in items:
        label, value = it[0], it[1]
        sub = it[2] if len(it) > 2 and it[2] is not None else ""
        tone = it[3] if len(it) > 3 and it[3] else ""
        cells.append(
            f'<div class="ss-kpi"><div class="l" title="{html.escape(str(label))}">{html.escape(str(label))}</div>'
            f'<div class="v {tone}">{html.escape(str(value))}</div><div class="s">{html.escape(str(sub))}</div></div>'
        )
    st.markdown(f'<div class="ss-kpis">{"".join(cells)}</div>', unsafe_allow_html=True)


def note(text: str, kind: str = "") -> None:
    st.markdown(f'<div class="ss-note {kind}">{text}</div>', unsafe_allow_html=True)


def command_window(lines: list[str]) -> None:
    """Lines may contain the span classes p/ok/bad/warn/dim (already escaped by the caller)."""
    st.markdown(f'<div class="ss-cmd">{"<br>".join(lines) if lines else "<span class=dim>&gt;&gt; </span>"}</div>',
                unsafe_allow_html=True)


def fmt(v, spec: str = "{:.3f}", none: str = "Insufficient data") -> str:
    if v is None:
        return none
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return str(v)


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False, "responsive": True})


def seed_picker(cfg: ExperimentConfig, key: str, label: str = "Seed split") -> tuple[str, list[int]]:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    split = c1.selectbox(label, ["test", "validation", "train"], key=f"{key}_split",
                         help="test = unseen evaluation seeds; train = seeds PPO was trained on")
    seeds = cfg.seeds.seeds(split)
    n = c2.number_input("Episodes", 1, len(seeds), min(len(seeds), 20), key=f"{key}_n")
    offset = c3.number_input("First seed index", 0, max(0, len(seeds) - 1), 0, key=f"{key}_off")
    chosen = seeds[int(offset): int(offset) + int(n)]
    st.caption(f"Seeds {chosen[0]}–{chosen[-1]} of split '{split}' ({len(chosen)} episodes)")
    return split, chosen
