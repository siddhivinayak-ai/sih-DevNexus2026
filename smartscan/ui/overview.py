"""Overview: problem, closed-loop architecture, system status."""

from __future__ import annotations

import streamlit as st

from smartscan.environment import turing
from smartscan.evaluation.experiment import list_results
from smartscan.ui.common import active_model, get_config, kpis, note, panel

ARCH_DOT = """
digraph G {
  graph [rankdir=LR, splines=ortho, nodesep=0.35, ranksep=0.45, bgcolor="white", fontname="Segoe UI", pad=0.2];
  node  [shape=box, style="filled", fillcolor="#f3f3f3", color="#5a5a5a", fontname="Segoe UI", fontsize=10,
         height=0.45, penwidth=1];
  edge  [color="#3a3a3a", arrowsize=0.6, fontname="Segoe UI", fontsize=8, fontcolor="#4d4d4d"];

  data  [label="Synthetic emitters\\n| Turing PDW data", fillcolor="#ffffff", style="dashed"];
  env   [label="RF / EM environment\\n(ground truth)", fillcolor="#e8eef5"];
  rx    [label="Receiver\\nlimited inst. BW\\nPd / Pfa", fillcolor="#e8eef5"];
  obs   [label="Observation\\nhit / no hit"];
  st    [label="State / belief\\n(receiver history)"];
  pol   [label="Scheduler\\nPPO | POMDP |\\nOpen Loop | Random", fillcolor="#e6f2ea", penwidth=1.5];
  rew   [label="Reward engine\\n(simulator only)", fillcolor="#fbeee6"];
  ev    [label="Evaluation engine\\nPd, Pfa, interception,\\nintercept time", fillcolor="#fbeee6"];

  data -> env;
  env -> rx [label="signals"];
  rx -> obs;
  obs -> st;
  st -> pol [label="s_t"];
  pol -> rx [label="a_t: next band", constraint=false, color="#0072BD", fontcolor="#0072BD", penwidth=1.5];
  env -> rew [style=dashed, label="ground truth"];
  obs -> rew;
  rew -> pol [label="r_t (PPO training)", style=dashed, constraint=false];
  env -> ev [style=dashed];
  obs -> ev;
}
"""


def render() -> None:
    cfg = get_config()
    left, right = st.columns([1.35, 1])
    with left:
        with panel("Problem"):
            st.markdown(
                "A receiver must monitor a wide spectrum but has **limited instantaneous bandwidth**: it can inspect "
                "only one band per dwell. Every scan decision therefore changes *what information becomes available "
                "next*. SmartScan formulates band selection as a **sequential decision problem** and learns which band "
                "to inspect next from receiver observations and feedback."
            )
            st.code(
                "Spectrum   F1  F2  F3  F4  F5  F6  F7  F8   (ONE band per dwell)\n"
                "Open loop  F1 → F2 → F3 → ... → F8 → F1     fixed, ignores observations\n"
                "Threat     short burst on F6 while sweep is at F2\n"
                "           → may be over before F6 is visited\n"
                "SmartScan  observation → state/belief → π(a|s) → next band\n"
                "           → observe → reward → learn",
                language=None,
            )
        with panel("Closed-loop architecture"):
            st.graphviz_chart(ARCH_DOT, width="stretch")
            st.caption("Ground truth (dashed) reaches only the reward and evaluation engines — never the scheduler.")
    with right:
        with panel("System status"):
            model = active_model(cfg.scenario.n_bands)
            ok, msg = turing.status(cfg.scenario.turing.local_dir)
            results = list_results()
            kpis([
                ("Scenario", cfg.scenario.name, f"{cfg.scenario.n_bands} bands"),
                ("Receiver", f"Pd {cfg.receiver.pd:.2f}", f"Pfa {cfg.receiver.pfa:.2f}"),
                ("PPO model", "trained" if model else "not trained",
                 f"{model.meta.get('timesteps_trained', '?')} steps" if model else "see RL Training",
                 "good" if model else "warn"),
                ("Dataset", "Turing" if ok else "synthetic", "configured" if ok else "not configured"),
                ("Saved results", len(results), "experiments/results"),
            ])
            if not model:
                note("PPO model not trained. Train one on the <b>RL Training</b> page or run "
                     "<code>python -m smartscan.rl.train --config configs/default.yaml</code>.", "warn")
            if not ok:
                note(turing.NOT_CONFIGURED)
        with panel("Schedulers under test"):
            st.markdown(
                "| Scheduler | Uses observations | Decision rule |\n|---|---|---|\n"
                "| Open Loop | no | fixed cyclic sweep F1…FN |\n"
                "| Random | no | uniform random band |\n"
                "| POMDP | yes | heuristic belief + periodicity score (original project baseline) |\n"
                "| PPO | yes | learned policy π(a\\|s) (Stable-Baselines3) |"
            )
        with panel("Suggested walkthrough"):
            st.markdown(
                "1. **Scenario Builder** — pick or edit the emitter scenario and receiver.\n"
                "2. **Live Simulation** — STEP through Open Loop, then POMDP, then PPO decisions.\n"
                "3. **RL Training** — train PPO (train seeds) with validation-based model selection.\n"
                "4. **Algorithm Comparison** — all schedulers on the same unseen test seeds, mean ± std.\n"
                "5. **Methodology** — formulation, reward, metrics, limitations."
            )
    note("Research prototype operating on a <b>synthetic RF environment</b> (optionally seeded from the Turing "
         "synthetic radar dataset). All numbers shown are produced by the simulation on this machine; "
         "no field validation is claimed.")
