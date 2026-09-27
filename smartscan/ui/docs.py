"""Methodology: renders the project documentation from docs/."""

from __future__ import annotations

import streamlit as st

from smartscan.config import PROJECT_ROOT
from smartscan.receiver.observation import GLOBAL_FEATURES, PER_BAND_FEATURES
from smartscan.ui.common import panel


def render() -> None:
    docs = sorted((PROJECT_ROOT / "docs").glob("*.md"))
    c1, c2 = st.columns([1, 3])
    with c1:
        with panel("Documents"):
            doc = st.radio("Document", docs, format_func=lambda p: p.stem.replace("_", " "), label_visibility="collapsed")
        with panel("State vector"):
            st.caption("Per band (×N): " + ", ".join(PER_BAND_FEATURES) + "; one-hot previous action (×N); "
                       "globals: " + ", ".join(GLOBAL_FEATURES) + ".")
    with c2:
        with panel(doc.stem.replace("_", " ") if doc else "Documentation"):
            if doc:
                st.markdown(doc.read_text(encoding="utf-8"))
