"""SmartScan engineering console (Streamlit entry point).

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="SmartScan", layout="wide", page_icon=":material/radar:",
                   initial_sidebar_state="expanded")

from smartscan.ui import shell  # noqa: E402  (set_page_config must run first)

shell.run()
