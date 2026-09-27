"""Live Simulation: step-by-step decision cycle with spectrum, trajectory, belief, events, reward."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from smartscan.evaluation.live import LiveSimulation
from smartscan.schedulers import SCHEDULER_LABELS
from smartscan.ui.common import active_model, command_window, fmt, get_config, kpis, note, panel, plot
from smartscan.visualization.belief import belief_bars, belief_map
from smartscan.visualization.reward import reward_curve, reward_terms_bar
from smartscan.visualization.spectrum import spectrum_activity_map
from smartscan.visualization.style import SCHEDULER_COLORS
from smartscan.visualization.trajectory import event_timeline, scan_trajectory

OUTCOME_CLASS = {"DETECTION": "ok", "FALSE_ALARM": "bad", "MISS": "warn", "NO_SIGNAL": "dim"}


def _sim() -> LiveSimulation | None:
    return st.session_state.get("live")


def _build(key: str, seed: int) -> None:
    cfg = get_config()
    model = active_model(cfg.scenario.n_bands) if key == "ppo" else None
    st.session_state.live = LiveSimulation(cfg, key, seed, model_path=str(model.path) if model else None)
    st.session_state.live_running = False


def render() -> None:
    cfg = get_config()
    n = cfg.scenario.n_bands

    with panel("Simulation control"):
        c = st.columns([1.3, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.1])
        keys = ["open_loop", "random", "pomdp", "ppo"]
        model = active_model(n)
        key = c[0].selectbox("Scheduler", keys, format_func=lambda k: SCHEDULER_LABELS[k] + ("" if k != "ppo" or model else " (not trained)"),
                             index=keys.index(st.session_state.get("live_key", "pomdp")))
        seed = int(c[1].number_input("Scenario seed", 0, 10**6, int(st.session_state.get("live_seed", cfg.seeds.test[0])),
                                     help="Default: first unseen test seed"))
        speed = c[8].select_slider("Steps / tick", [1, 2, 5, 10, 25], value=st.session_state.get("live_speed", 2))
        st.session_state.live_speed = speed

        if key == "ppo" and model is None:
            note("PPO model not trained. Train one on the RL Training page (or choose another scheduler).", "warn")
            return
        sim = _sim()
        stale = sim is None or sim.scheduler_key != key or sim.seed != seed or sim.config is not cfg
        if stale:
            st.session_state.live_key, st.session_state.live_seed = key, seed
            try:
                _build(key, seed)
            except Exception as exc:
                note(f"Cannot start simulation: {html.escape(str(exc))}", "err")
                return
            sim = _sim()

        running = st.session_state.get("live_running", False)
        if c[2].button("START", type="primary", disabled=running or sim.done, width="stretch"):
            st.session_state.live_running = True
            st.rerun()
        if c[3].button("PAUSE", disabled=not running, width="stretch"):
            st.session_state.live_running = False
            st.rerun()
        if c[4].button("STEP", disabled=running or sim.done, width="stretch"):
            sim.step()
        if c[5].button("RUN 10", disabled=running or sim.done, width="stretch"):
            sim.run(10)
        if c[6].button("RUN 100", disabled=running or sim.done, width="stretch"):
            sim.run(100)
        if c[7].button("RESET", width="stretch"):
            sim.reset()
            st.session_state.live_running = False
            st.rerun()
        show_truth = st.toggle("Show ground truth on spectrum map (evaluator view — the scheduler never sees it)", value=True)

    run_every = 0.6 if st.session_state.get("live_running") else None

    @st.fragment(run_every=run_every)
    def live_view():
        s = _sim()
        if s is None:
            return
        if st.session_state.get("live_running"):
            s.run(st.session_state.get("live_speed", 2))
            if s.done:
                st.session_state.live_running = False
                st.rerun()
        _render_view(s, show_truth)

    live_view()


def _render_view(sim: LiveSimulation, show_truth: bool) -> None:
    sc = sim.scenario
    n = sc.n_bands
    tr = sim.trace()
    last = sim.last
    m = sim.metrics()
    intercepted = int(sim.env.intercepted.sum())
    color = SCHEDULER_COLORS.get(sim.scheduler_key)
    status = "finished" if sim.done else ("running" if st.session_state.get("live_running") else "paused")

    kpis([
        ("Time", f"{sim.env.t} / {sc.total_steps}", "steps"),
        ("Decision", sim.env.step_count, status),
        ("Current band", sc.band_labels[last.action] if last else "—", "last scan"),
        ("Next action", sc.band_labels[sim.pending_action] if sim.pending_action is not None else "—",
         SCHEDULER_LABELS[sim.scheduler_key]),
        ("Observation", ("DETECT" if last.observation["detected"][0] else "no detect") if last else "—",
         (f"{last.observation['amplitude_db'][0]} dB" if last and last.observation["amplitude_db"][0] is not None else "")),
        ("Outcome", last.outcome.replace("_", " ").lower() if last else "—", "evaluator",
         {"DETECTION": "good", "FALSE_ALARM": "bad", "MISS": "warn"}.get(last.outcome if last else "", "")),
        ("Reward", fmt(last.reward, "{:+.3f}", "—") if last else "—", "this decision"),
        ("Cumulative", f"{sim.env.cumulative_reward:.2f}", "reward"),
        ("Intercepted", f"{intercepted} / {len(sc.transmissions)}", "transmissions"),
        ("Interception", fmt(m["interception_rate"] if m else None, "{:.3f}", "—"), "so far"),
    ])

    left, right = st.columns([1.5, 1])
    with left:
        with panel("A — Spectrum activity map"):
            if show_truth:
                fig = spectrum_activity_map(sc, tr, t_max=max(1, sim.env.t), height=470,
                                            title=f"{sc.name} · seed {sim.seed} · {SCHEDULER_LABELS[sim.scheduler_key]}")
            else:
                from smartscan.visualization.trajectory import scan_trajectory as _st

                fig = _st(tr, n, sim.scheduler_key, height=470, title="Receiver dwells (ground truth hidden)")
            plot(fig, key="live_spectrum")
    with right:
        with panel("Decision cycle — step by step"):
            _decision_panel(sim)

    c1, c2 = st.columns(2)
    with c1:
        with panel("B — Scan trajectory"):
            plot(scan_trajectory(tr, n, sim.scheduler_key, height=280, title=""), key="live_traj")
    with c2:
        with panel("C — Belief / activity map"):
            b, times = sim.beliefs()
            plot(belief_map(b, times, n, height=280, title=""), key="live_belief")
    c1, c2 = st.columns(2)
    with c1:
        with panel("D — Event timeline"):
            plot(event_timeline(tr, height=250, title=""), key="live_events")
    with c2:
        with panel("E — Reward"):
            plot(reward_curve(tr, height=250, color=color, title=""), key="live_reward")

    with panel("Command window — decision log"):
        lines = []
        for v in sim.history[-14:]:
            obs = v.observation
            det = "DETECT" if obs["detected"][0] else "------"
            cls = OUTCOME_CLASS.get(v.outcome, "dim")
            lines.append(
                f'<span class="p">&gt;&gt;</span> k={v.decision:<4d} t={v.t:<4d} a={sc.band_labels[v.action]:<4s} '
                f'obs={det} <span class="{cls}">{v.outcome:<11s}</span> r={v.reward:+.3f}'
                + (f'  <span class="ok">+{v.new_intercepts} intercept</span>' if v.new_intercepts else "")
            )
        if sim.pending_action is not None:
            lines.append(f'<span class="dim">&gt;&gt; next: t={sim.env.t} a={sc.band_labels[sim.pending_action]}</span>')
        else:
            lines.append('<span class="dim">&gt;&gt; episode finished — RESET or change seed</span>')
        command_window(lines)
    if sim.done and m:
        note(f"Episode complete: Pd {fmt(m['pd'])}, Pfa {fmt(m['pfa'])}, interception rate {fmt(m['interception_rate'])}, "
             f"intercept time error {fmt(m['intercept_time_error'], '{:.2f}')} steps, coverage {fmt(m['coverage'])}.")


def _decision_panel(sim: LiveSimulation) -> None:
    sc = sim.scenario
    n = sc.n_bands
    last = sim.last
    tabs = st.tabs(["1 State", "2 Action", "3–5 Observe & reward", "6–7 Next decision"])
    with tabs[0]:
        st.caption("Receiver-side state before the last decision (what the scheduler could use).")
        st.dataframe(last.state if last else sim.pending_state, hide_index=True, width="stretch", height=250)
    with tabs[1]:
        if last is None:
            st.caption("No decision executed yet — press STEP.")
        else:
            st.markdown(f"Selected **{sc.band_labels[last.action]}** at t = {last.t} by **{SCHEDULER_LABELS[sim.scheduler_key]}**.")
            _explain(last.explain, n, last.action)
    with tabs[2]:
        if last is None:
            st.caption("No observation yet.")
        else:
            obs = last.observation
            kpis([
                ("Scanned", ", ".join(sc.band_labels[b] for b in obs["bands"]), f"t {obs['t_start']}–{obs['t_end']}"),
                ("Detector", "DETECT" if obs["detected"][0] else "no detect"),
                ("Amplitude", "—" if obs["amplitude_db"][0] is None else f"{obs['amplitude_db'][0]} dB"),
                ("Outcome", last.outcome.replace("_", " ").lower(), "ground truth"),
                ("New intercepts", last.new_intercepts),
            ])
            plot(reward_terms_bar(last.terms, height=230, title=f"Reward r = {last.reward:+.3f}"), key="live_terms")
    with tabs[3]:
        if sim.pending_action is None:
            st.caption("Episode finished.")
        else:
            st.markdown(f"Next decision at t = {sim.env.t}: **{sc.band_labels[sim.pending_action]}**")
            _explain(sim.pending_explain, n, sim.pending_action)


def _explain(explain: dict, n: int, action: int) -> None:
    if "score" in explain:  # POMDP
        extra = {
            "periodicity (×1.5)": (np.array(explain["periodicity"]) * 1.5, "#7E2F8E", "dot"),
            "freshness (×0.1)": (np.array(explain["freshness"]) * 0.1, "#D95319", "dash"),
        }
        fig = belief_bars(np.array(explain["belief"]), n, highlight=action, extra=extra, height=240,
                          title="POMDP: belief (bars) + score components; action = argmax score")
        plot(fig, key=f"explain_pomdp_{action}_{id(explain)}")
        st.dataframe(pd.DataFrame({k: v for k, v in explain.items()}, index=[f"F{i + 1}" for i in range(n)]).T.round(3),
                     width="stretch")
    elif "probabilities" in explain:  # PPO or random
        probs = np.array(explain["probabilities"])
        fig = belief_bars(probs, n, highlight=action, height=240,
                          title="Policy π(a|s): action probabilities" + (" (deterministic = argmax)" if explain.get("deterministic") else ""))
        fig.update_yaxes(title_text="π(a|s)")
        plot(fig, key=f"explain_probs_{action}_{id(explain)}")
    else:
        st.caption(f"Rule: {explain.get('rule', 'n/a')} — independent of observations.")
