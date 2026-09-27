# 02 — System Architecture

```mermaid
flowchart TD
    DS["Turing Synthetic Radar Dataset<br/>(stare-mode PDWs)"] -->|Mode B| ENV
    SYN["Synthetic emitter generator<br/>static / periodic / intermittent / agile"] -->|Mode A| ENV
    ENV["RF Environment<br/>ground truth (E x T x N activity)"] --> RX
    RX["Receiver<br/>limited instantaneous BW, Pd / Pfa, dwell, retune"] --> OBS["Observation<br/>detect / no detect, amplitude"]
    OBS --> HIST["Receiver history / belief<br/>(state vector s_t)"]
    HIST --> GYM["SmartScanEnv (Gymnasium)"]
    GYM --> PPO["PPO agent  pi(a|s)"]
    PPO -->|"a_t = next band"| RX
    ENV -. ground truth .-> REW["Reward engine"]
    OBS --> REW
    REW -->|"r_t (training only)"| PPO

    subgraph Baselines["Parallel baselines (same env, same seeds)"]
        OL["Open Loop"]
        RND["Random"]
        POMDP["POMDP belief-state"]
    end
    HIST --> OL & RND & POMDP
    OL & RND & POMDP -->|a_t| RX

    ENV -. ground truth .-> EVAL["Evaluation engine<br/>Pd, Pfa, interception, intercept time, coverage"]
    OBS --> EVAL
```

Dashed arrows carry ground truth. They end at the reward and evaluation engines and never at a scheduler.

## Package layout

| Module | Responsibility |
|---|---|
| `smartscan/config.py` | Typed `ExperimentConfig` (scenario, receiver, reward, observation, PPO, seed splits), YAML/JSON I/O |
| `smartscan/environment/emitter.py` | Emitter behaviour models → per-seed band traces |
| `smartscan/environment/scenario.py` | Scenario realisation, ground-truth activity, transmission segmentation |
| `smartscan/environment/turing.py` | Turing dataset discovery, bounded/strided PDW reads, Mode-B scenario builder |
| `smartscan/environment/reward.py` | Reward equation |
| `smartscan/environment/smartscan_env.py` | Gymnasium env used by PPO **and** by every baseline |
| `smartscan/receiver/` | Detector models, receiver dwell, receiver-side history and state vector |
| `smartscan/schedulers/` | `BaseScheduler` + Open Loop, Random, POMDP, PPO |
| `smartscan/rl/` | Training, evaluation CLI, callbacks |
| `smartscan/evaluation/` | Metrics, multi-seed runner, comparison CLI, step-by-step engine |
| `smartscan/visualization/` | Plotly figures (MATLAB-like style) |
| `smartscan/ui/` + `app.py` | Streamlit engineering console |

## One decision (identical for every scheduler)

```
ctx    = DecisionContext(state vector, receiver history, t)
a_t    = scheduler.select_action(ctx)
obs, r = SmartScanEnv.step(a_t)        # receiver dwells on a_t; reward/metrics from ground truth
scheduler.update(a_t, obs)
```
