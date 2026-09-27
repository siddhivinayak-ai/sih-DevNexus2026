# 13 — Limitations

- **Simulation only.** Emitters, receiver and detector are simplified models with simulation parameters. There is no
  hardware, no RF front-end and no field validation.
- **Binary detection per band.** No pulse-level processing, deinterleaving, or frequency/AoA estimation inside a band.
  Several emitters in one band are detected together.
- **Discrete time and bands.** One decision per dwell. Interception is scored per transmission at timestep resolution.
- **State is a hand-built summary** of the history, which is only approximately Markov. A recurrent policy or
  frame-stacking might capture longer patterns; this was not evaluated.
- **PPO generalisation** is limited to the scenario family it was trained on (emitter mix, N, receiver). A model is
  tied to N. Results on other scenario families require retraining.
- **Reward shaping needs ground truth**, so PPO can only be trained in simulation (or with labelled recordings).
- **POMDP baseline** is the original heuristic, not an optimal POMDP solver.
- **Turing Mode B** uses a nominal SNR (dataset amplitude is not calibrated) and coarse time/frequency binning.
- **Transmissions starting near the end of an episode** have little time to be intercepted. This censoring affects
  all schedulers equally.
- **Statistics** assume independent seeds; the CIs are paired t-intervals and are not corrected for multiple comparisons.
