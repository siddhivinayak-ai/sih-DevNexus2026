# 14 — Demo Script (≈ 4½ minutes)

Before the demo: `streamlit run app.py`, a trained model in `models/ppo_mixed_8band/`, and a comparison run on the
test split saved in `experiments/results/`. Quote only the numbers the app shows.

| Time | Page | Say / do |
|---|---|---|
| 00:00 | Overview | "A receiver can look at one band at a time. Every scan decision decides what we learn next, so this is a sequential decision problem." Point at the block diagram: ground truth reaches only the reward and evaluation engines. |
| 00:30 | Scenario Builder | Load `sih_example.yaml`: periodic F3, short-burst threat on F6, agile hopper, static F1. Show the ground-truth preview. |
| 01:00 | Live Simulation, Open Loop | *RUN 100*. Show the fixed staircase trajectory and the F6 bursts it misses on the spectrum map (grey = truth, markers = dwells). |
| 01:30 | Live Simulation, POMDP | *RESET*, *STEP* a few times. Show belief + score components, then *RUN 100*: it adapts, and also concentrates on confirmed bands. |
| 02:00 | RL Training | Show the saved run: episode reward, validation curve on unseen seeds, PPO losses. These are all logged values. Mention train / validation / test seeds. |
| 02:30 | Live Simulation, PPO | *STEP*: state table → π(a\|s) bars → observation → reward decomposition → next decision. |
| 03:00 | Algorithm Comparison | Test split, all four schedulers, same seeds. Walk through the mean ± std table and the paired-difference CIs. No winner badge: read the measured numbers. |
| 04:00 | Analytics | Training vs unseen seeds gap. Band revisit frequency. |
| 04:30 | Methodology | Architecture, reward equation, limitations: research simulation, not an operational system. |
