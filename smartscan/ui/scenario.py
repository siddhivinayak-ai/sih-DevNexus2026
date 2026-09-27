"""Scenario Builder: presets, emitter table, receiver, reward weights, seeds, ground-truth preview."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from smartscan.config import CONFIG_DIR, EMITTER_KINDS, EmitterSpec, ExperimentConfig, list_config_files
from smartscan.environment import turing
from smartscan.environment.scenario import build_scenario
from smartscan.ui.common import get_config, kpis, note, panel, plot, set_config
from smartscan.visualization.spectrum import spectrum_activity_map

EMITTER_COLS = ["kind", "name", "band", "snr_db", "period", "duty", "phase", "p_on", "p_off",
                "hop_bands", "hop_interval", "hop_random", "start", "stop"]


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def emitters_to_frame(specs: list[EmitterSpec]) -> pd.DataFrame:
    rows = []
    for s in specs:
        rows.append({
            "kind": s.kind, "name": s.name,
            "band": None if s.band is None else s.band + 1,
            "snr_db": s.snr_db, "period": s.period, "duty": s.duty, "phase": s.phase,
            "p_on": s.p_on, "p_off": s.p_off,
            "hop_bands": ",".join(str(b + 1) for b in s.hop_bands) if s.hop_bands else "",
            "hop_interval": s.hop_interval, "hop_random": bool(s.hop_random),
            "start": s.start, "stop": s.stop,
        })
    df = pd.DataFrame(rows, columns=EMITTER_COLS)
    for c in ("band", "period", "phase", "hop_interval", "start", "stop"):
        df[c] = df[c].astype("Int64")
    for c in ("snr_db", "duty", "p_on", "p_off"):
        df[c] = df[c].astype("Float64")
    return df


def frame_to_emitters(df: pd.DataFrame, n_bands: int) -> list[EmitterSpec]:
    specs = []
    for i, r in df.iterrows():
        if _blank(r.get("kind")):
            continue
        kind = str(r["kind"])
        if kind not in EMITTER_KINDS:
            raise ValueError(f"row {i + 1}: unknown kind '{kind}'")

        def num(key, cast=float):
            v = r.get(key)
            return None if _blank(v) or pd.isna(v) else cast(v)

        band = num("band", int)
        if band is not None and not 1 <= band <= n_bands:
            raise ValueError(f"row {i + 1}: band must be 1..{n_bands} (F1..F{n_bands})")
        hops = None
        if not _blank(r.get("hop_bands")):
            hops = [int(x) - 1 for x in str(r["hop_bands"]).replace(" ", "").split(",") if x]
            if any(not 0 <= h < n_bands for h in hops):
                raise ValueError(f"row {i + 1}: hop bands must be 1..{n_bands}")
        specs.append(EmitterSpec(
            kind=kind, name="" if _blank(r.get("name")) else str(r["name"]),
            band=None if band is None else band - 1, snr_db=num("snr_db"),
            start=num("start", int) or 0, stop=num("stop", int),
            period=num("period", int), duty=num("duty"), phase=num("phase", int),
            p_on=num("p_on"), p_off=num("p_off"), hop_bands=hops, hop_interval=num("hop_interval", int),
            hop_random=bool(r.get("hop_random")) if not _blank(r.get("hop_random")) else False,
        ))
    return specs


def render() -> None:
    cfg: ExperimentConfig = get_config().copy()

    with panel("Configuration file"):
        c1, c2, c3 = st.columns([2, 1, 1])
        files = list_config_files()
        names = [f.name for f in files]
        current = st.session_state.get("config_path", "configs/default.yaml").split("/")[-1]
        choice = c1.selectbox("Preset", names, index=names.index(current) if current in names else 0)
        if c2.button("LOAD PRESET", width="stretch"):
            set_config(ExperimentConfig.load(CONFIG_DIR / choice), f"configs/{choice}")
            st.rerun()
        c3.download_button("DOWNLOAD YAML", yaml.safe_dump(_plain(cfg.to_dict()), sort_keys=False),
                           file_name=f"{cfg.scenario.name}.yaml", width="stretch")
        st.caption(cfg.scenario.description or "")

    tab_s, tab_e, tab_r, tab_w = st.tabs(["Scenario", "Emitters", "Receiver", "Reward & seeds"])
    sc = cfg.scenario
    with tab_s:
        c = st.columns(4)
        sc.name = c[0].text_input("Scenario name", sc.name)
        ok, msg = turing.status(sc.turing.local_dir)
        sources = ["synthetic", "turing"]
        sc.source = c[1].selectbox("Environment source", sources, index=sources.index(sc.source),
                                   help="turing = ground truth binned from Turing pulse trains (Mode B)")
        if sc.source == "turing" and not ok:
            note(msg + " Switch back to <b>synthetic</b> or configure the dataset in the Dataset Explorer.", "warn")
        sc.n_bands = int(c[2].number_input("Number of bands N", 2, 64, sc.n_bands))
        sc.episode_length = int(c[3].number_input("Episode length (steps)", 16, 4096, sc.episode_length, step=16))
        c = st.columns(4)
        sc.f_start_ghz = float(c[0].number_input("F1 centre (GHz)", 0.01, 100.0, sc.f_start_ghz, step=0.1, format="%.3f"))
        sc.band_spacing_ghz = float(c[1].number_input("Band spacing (GHz)", 0.001, 10.0, sc.band_spacing_ghz, step=0.01, format="%.3f"))
        sc.warmup_steps = int(c[2].number_input("Warm-up steps (excluded from metrics)", 0, 20000, sc.warmup_steps, step=50,
                                                help="Applied equally to every scheduler (the original app warmed up only POMDP)."))
        sc.description = c[3].text_input("Description", sc.description)
        if sc.source == "turing":
            c = st.columns(4)
            sc.turing.mode = c[0].selectbox("Turing mode", ["stare", "scan"], index=["stare", "scan"].index(sc.turing.mode))
            sc.turing.max_pulses = int(c[1].number_input("Pulses per episode window", 1000, 2_000_000, sc.turing.max_pulses, step=10000))
            sc.turing.local_dir = c[2].text_input("Local dataset dir", sc.turing.local_dir)
            st.caption("Train / validation / test episodes use the dataset's own train / validation / test splits.")

    with tab_e:
        st.caption("Band numbers are 1-based (1 = F1). Blank cells are drawn at random for each scenario seed "
                   "from the configured ranges. hop_bands: comma-separated list, e.g. 2,6,4,7.")
        df = st.data_editor(
            emitters_to_frame(sc.emitters), num_rows="dynamic", width="stretch",
            key=f"emitter_editor_{st.session_state.get('config_version', 0)}",
            column_config={
                "kind": st.column_config.SelectboxColumn("kind", options=list(EMITTER_KINDS), required=True),
                "band": st.column_config.NumberColumn("band (F#)", min_value=1, max_value=sc.n_bands, step=1),
                "snr_db": st.column_config.NumberColumn("SNR dB", format="%.1f"),
                "duty": st.column_config.NumberColumn("duty", min_value=0.01, max_value=0.99, format="%.2f"),
                "p_on": st.column_config.NumberColumn("p_on", min_value=0.0, max_value=1.0, format="%.3f"),
                "p_off": st.column_config.NumberColumn("p_off", min_value=0.0, max_value=1.0, format="%.3f"),
                "hop_random": st.column_config.CheckboxColumn("hop_random"),
            },
        )
        try:
            sc.emitters = frame_to_emitters(df, sc.n_bands)
        except ValueError as exc:
            note(f"Emitter table error: {exc}", "err")
        st.markdown("**Additional fully random emitters per seed**")
        c = st.columns(4)
        for i, kind in enumerate(EMITTER_KINDS):
            n = int(c[i].number_input(kind, 0, 32, int(sc.random_emitters.get(kind, 0)), key=f"rand_{kind}_{st.session_state.get('config_version', 0)}"))
            if n:
                sc.random_emitters[kind] = n
            else:
                sc.random_emitters.pop(kind, None)

    rx = cfg.receiver
    with tab_r:
        c = st.columns(4)
        rx.detection_model = c[0].selectbox("Detection model", ["fixed", "snr"], index=["fixed", "snr"].index(rx.detection_model),
                                            help="fixed: constant Pd; snr: Pd = Q(Q⁻¹(Pfa) − √(2·SNR))")
        rx.pd = float(c[1].number_input("Pd (fixed model / belief filter)", 0.01, 1.0, rx.pd, step=0.01))
        rx.pfa = float(c[2].number_input("Pfa per dwell", 0.0, 0.5, rx.pfa, step=0.01, format="%.3f"))
        rx.amplitude_sigma_db = float(c[3].number_input("Amplitude σ (dB)", 0.0, 10.0, rx.amplitude_sigma_db, step=0.5))
        c = st.columns(4)
        rx.instantaneous_bands = int(c[0].number_input("Instantaneous bandwidth (bands)", 1, sc.n_bands,
                                                        min(rx.instantaneous_bands, sc.n_bands)))
        rx.dwell_steps = int(c[1].number_input("Dwell (steps)", 1, 32, rx.dwell_steps))
        rx.retune_steps = int(c[2].number_input("Retune latency (steps)", 0, 32, rx.retune_steps))
        st.caption("A PPO model is tied to N (observation size). Changing the number of bands requires a new model.")

    w = cfg.reward
    with tab_w:
        st.markdown("**Reward weights** — R = w_d·D + w_i·I − w_f·F − w_m·M − w_s·S − w_l·L − w_v·V")
        c = st.columns(4)
        w.detection = float(c[0].number_input("w_d detection", 0.0, 10.0, w.detection, step=0.05))
        w.new_intercept = float(c[1].number_input("w_i new intercept", 0.0, 10.0, w.new_intercept, step=0.1))
        w.false_alarm = float(c[2].number_input("w_f false alarm", 0.0, 10.0, w.false_alarm, step=0.05))
        w.miss = float(c[3].number_input("w_m missed detection", 0.0, 10.0, w.miss, step=0.05))
        c = st.columns(4)
        w.switch_cost = float(c[0].number_input("w_s retune", 0.0, 10.0, w.switch_cost, step=0.01))
        w.latency = float(c[1].number_input("w_l latency", 0.0, 10.0, w.latency, step=0.05))
        w.revisit = float(c[2].number_input("w_v empty revisit", 0.0, 10.0, w.revisit, step=0.01))
        w.revisit_window = int(c[3].number_input("revisit window (steps)", 1, 64, w.revisit_window))
        st.markdown("**Scenario seed splits** (inclusive ranges; must not overlap)")
        c = st.columns(3)
        for col, split in zip(c, ("train", "validation", "test")):
            lo, hi = getattr(cfg.seeds, split)
            a = col.number_input(f"{split} from", 0, 10**6, int(lo), key=f"seed_{split}_lo_{st.session_state.get('config_version', 0)}")
            b = col.number_input(f"{split} to", 0, 10**6, int(hi), key=f"seed_{split}_hi_{st.session_state.get('config_version', 0)}")
            setattr(cfg.seeds, split, (int(a), int(max(a, b))))
        sets = [set(cfg.seeds.seeds(s)) for s in ("train", "validation", "test")]
        if sets[0] & sets[2] or sets[1] & sets[2] or sets[0] & sets[1]:
            note("Seed ranges overlap — test results would not be on unseen scenarios.", "err")

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("APPLY TO WORKSPACE", type="primary", width="stretch"):
        set_config(cfg, "(edited)")
        st.rerun()
    fname = c3.text_input("Save as", f"{cfg.scenario.name}.yaml", label_visibility="collapsed")
    if c2.button("SAVE TO configs/", width="stretch"):
        cfg.save(CONFIG_DIR / fname)
        set_config(cfg, f"configs/{fname}")
        st.success(f"Saved configs/{fname}")

    _preview(cfg)


def _preview(cfg: ExperimentConfig) -> None:
    with panel("Ground-truth preview (evaluator view)"):
        c1, c2 = st.columns([1, 3])
        seed = int(c1.number_input("Scenario seed", 0, 10**6, cfg.seeds.test[0]))
        try:
            scn = build_scenario(cfg.scenario, seed)
        except Exception as exc:
            note(f"Cannot build scenario: {exc}", "err")
            return
        occ = scn.ground_truth.mean() if scn.total_steps else 0
        kpis([
            ("Emitters", scn.n_emitters),
            ("Transmissions", len(scn.transmissions)),
            ("Mean duration", f"{np.mean([t.duration for t in scn.transmissions]):.1f}" if scn.transmissions else "—", "steps"),
            ("Occupancy", f"{occ:.1%}", "band-steps active"),
            ("Timeline", scn.total_steps, "steps"),
        ])
        with c1:
            st.dataframe(pd.DataFrame({"emitter": [e.name for e in scn.emitters],
                                       "behaviour": [e.describe() for e in scn.emitters],
                                       "SNR dB": [round(e.snr_db, 1) for e in scn.emitters]}),
                         hide_index=True, width="stretch")
            for n in scn.notes:
                st.caption(n)
        with c2:
            plot(spectrum_activity_map(scn, None, height=420, title=f"Ground truth — seed {seed}"))


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj
