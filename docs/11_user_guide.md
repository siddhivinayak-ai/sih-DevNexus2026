# 11 — User Guide

## Install
```
python -m venv .venv
.venv\Scripts\activate            (Windows)      source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```
On CPU-only machines a smaller PyTorch can be installed first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Run
```
streamlit run app.py
```
Open http://localhost:8501. The left sidebar holds the page list and the **Workspace** (current configuration).

## Create a scenario
**Scenario Builder** → choose a preset → *LOAD PRESET* → edit the emitter table (blank cells are random per seed),
receiver, reward and seeds → *APPLY TO WORKSPACE* (and optionally *SAVE TO configs/*). The preview shows the ground
truth for any seed.

## Run Open Loop / POMDP step by step
**Live Simulation** → Scheduler *Open Loop* or *POMDP* → *STEP* (one decision), *RUN 10*, *RUN 100*, *START* /
*PAUSE* (continuous), *RESET*. The decision-cycle tabs show 1 state → 2 action → 3–5 observation and reward →
6–7 next decision.

## Train PPO
**RL Training** → *START TRAINING* (live timesteps, episodes, rewards, PPO losses, validation) or
```
python -m smartscan.rl.train --config configs/default.yaml
```

## Load PPO
**RL Training → Models** → select → *LOAD MODEL*. Models are stored in `models/ppo_<scenario>/`.

## Run the live PPO simulation
**Live Simulation** → Scheduler *PPO*. The action tab shows the policy probabilities π(a|s).

## Compare algorithms
**Algorithm Comparison** → schedulers, split *test*, number of episodes → *RUN COMPARISON*, or
```
python -m smartscan.evaluation.comparison --config configs/default.yaml --split test --episodes 100
```

## Export results
*EXPORT TABLE / EXPORT PER-EPISODE* (CSV) or *SAVE TO experiments/results*. Saved folders can be reopened in
**Analytics → Saved results**.

## Tests
```
python -m pytest -q
```
