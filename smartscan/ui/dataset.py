"""Dataset Explorer: Turing Synthetic Radar Dataset PDWs (sampled, bounded reads)."""

from __future__ import annotations

import streamlit as st

from smartscan.environment import turing
from smartscan.ui.common import get_config, kpis, note, panel, plot, set_config
from smartscan.visualization.dataset import label_counts, pdw_histograms, pdw_scatter, pulse_density, top_labels


def render() -> None:
    cfg = get_config()
    ok, msg = turing.status(cfg.scenario.turing.local_dir)
    if not ok:
        note(msg, "warn")
        with panel("Configure the Turing dataset"):
            st.markdown(
                "Place pulse-train files as `data/turing/<mode>/<split>_<mode>/*.h5` (or set the environment "
                "variable `SMARTSCAN_TURING_DIR`). The dataset is gated on Hugging Face "
                f"(`{turing.HF_DATASET_ID}`); request access, then download a sample here or with "
                "`huggingface-cli download` / the challenge's `download_dataset()`."
            )
            c = st.columns([1, 1, 1, 2])
            mode = c[0].selectbox("Mode", ["stare", "scan"])
            split = c[1].selectbox("Split", ["train", "validation", "test"])
            nfiles = c[2].number_input("Files", 1, 20, 2)
            token = c[3].text_input("Hugging Face token (not stored)", type="password")
            if st.button("DOWNLOAD SAMPLE"):
                try:
                    with st.spinner("Downloading..."):
                        paths = turing.download_sample(mode, split, int(nfiles), token or None)
                    st.success(f"Downloaded {len(paths)} file(s).")
                    st.rerun()
                except Exception as exc:
                    note(f"Download failed: {type(exc).__name__}: {exc}", "err")
        return

    files = turing.find_files(turing.dataset_root(cfg.scenario.turing.local_dir))
    with panel("File"):
        c = st.columns([1, 1, 3])
        modes = sorted({f.mode for f in files})
        mode = c[0].selectbox("Receiver mode", modes)
        splits = sorted({f.split for f in files if f.mode == mode})
        split = c[1].selectbox("Split", splits)
        sel = [f for f in files if f.mode == mode and f.split == split]
        tf = c[2].selectbox("Pulse train", sel, format_func=lambda f: f.label)
        info = turing.file_info(tf.path)
        c = st.columns(3)
        start = int(c[0].number_input("Start pulse", 0, max(0, info["n_pulses"] - 1), 0))
        count = int(c[1].number_input("Pulses to read", 1000, 500_000, min(100_000, info["n_pulses"]), step=10_000))
        stride = int(c[2].number_input("Stride (sampling)", 1, 1000, 1))
        if mode == "scan":
            note("Scan-mode pulse trains already contain another receiver's sweep; SmartScan uses stare mode as ground truth.", "warn")

    df = _read(str(tf.path), start, count, stride)
    kpis([
        ("PDW count (file)", f"{info['n_pulses']:,}"),
        ("Sampled pulses", f"{len(df):,}", f"stride {stride}"),
        ("Features", info["n_features"]),
        ("Emitter labels", df["label"].nunique() if info["has_labels"] else "n/a", "in sample"),
        ("ToA span", f"{df['toa'].max() - df['toa'].min():.4g}", "us"),
        ("CF range", f"{df['cf'].min():.4g}–{df['cf'].max():.4g}"),
    ])
    hl = top_labels(df)
    c1, c2 = st.columns(2)
    with c1:
        with panel("Pulse train"):
            plot(pdw_scatter(df, "toa", "cf", hl), key="ds_toa_cf")
    with c2:
        with panel("Temporal / frequency occupancy"):
            plot(pulse_density(df), key="ds_density")
    c1, c2 = st.columns(2)
    with c1:
        with panel("Pulse width vs centre frequency"):
            plot(pdw_scatter(df, "cf", "pw", hl), key="ds_pw")
    with c2:
        with panel("Angle of arrival vs time"):
            plot(pdw_scatter(df, "toa", "aoa", hl), key="ds_aoa")
    with panel("Distributions"):
        plot(pdw_histograms(df, ["cf", "pw", "aoa", "amplitude"]), key="ds_hist")
        plot(label_counts(df), key="ds_labels")
    with panel("PDW table (first 500 sampled rows)"):
        st.dataframe(df.head(500), width="stretch", height=260)
        if info["metadata"]:
            st.json(info["metadata"], expanded=False)
    if st.button("USE TURING (stare) AS SCENARIO SOURCE"):
        new = cfg.copy()
        new.scenario.source = "turing"
        new.scenario.turing.mode = "stare"
        set_config(new, "(edited)")
        st.success("Scenario source set to Turing. A PPO model must be trained for this scenario.")


@st.cache_data(max_entries=8, show_spinner="Reading PDWs...")
def _read(path: str, start: int, count: int, stride: int):
    from pathlib import Path

    return turing.read_pdws(Path(path), start, count, stride)
