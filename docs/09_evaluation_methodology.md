# 09 — Evaluation Methodology

## Fair comparison

`smartscan/evaluation/experiment.py::run_experiment` runs, for **each seed**, **each scheduler** on the same
`SmartScanEnv`:

- same scenario realisation (seed → identical ground truth)
- same receiver parameters and **the same detector random numbers** (common random numbers)
- same simulation duration and warm-up
- same metric code (`episode_metrics`) for every scheduler

Results are reported as **mean ± sample std over episodes, with n**. Paired differences against Open Loop
(scheduler − Open Loop on the same seed) come with a 95 % Student-t confidence interval.

## Training / test separation

| Split | Default seeds | Used for |
|---|---|---|
| train | 1–100 | PPO training episodes |
| validation | 501–520 | model selection (`best_model.zip`) |
| test | 1001–1100 | reported results; never seen during training or selection |

In Turing mode the splits map onto the dataset's own train/validation/test files. The **Analytics → Training vs
unseen seeds** tab reports train-seed and test-seed results side by side.

## Metrics (`smartscan/evaluation/metrics.py`)

Counts are per band-dwell and exclude warm-up. D = detections, M = missed detections, F = false alarms,
C = correct rejections.

| Metric | Definition |
|---|---|
| Probability of Detection | Pd = D / (D + M) |
| Probability of False Alarm | Pfa = F / (F + C) |
| Interception Rate | intercepted transmissions / transmissions with onset after warm-up |
| Average Intercept Time | mean over discovered emitters of (first intercept time − first activation) |
| Intercept Time Error | mean over intercepted transmissions of (t_intercept − onset). Reference: onset, i.e. what an ideal wideband (stare) receiver achieves |
| Normalised delay | mean of (t_intercept − onset) / duration |
| Emitter discovery rate | emitters intercepted at least once / active emitters |
| Detection / Miss / False Alarm Count | D, M, F |
| Hit rate | (D + M) / all band-dwells |
| Scan Coverage | distinct bands scanned / N |
| Band revisit frequency | per-band share of dwells; mean revisit interval; immediate revisit rate |
| Cumulative / Average Reward | Σ r_t, Σ r_t / decisions |

A transmission is **intercepted** when the receiver scans its band while it is active and the detector fires. The
definition is the same for all schedulers. A metric whose denominator is zero is reported as
**"Insufficient data"**, never as 0.

Pd and Pfa measure the receiver, conditioned on where it looked. They stay close to the configured values for
every scheduler. Strategy quality shows in interception rate, intercept time, discovery rate and coverage.

## Commands

```
python -m smartscan.evaluation.comparison --config configs/default.yaml --split test --episodes 100
```
Outputs `experiments/results/<timestamp>_<scenario>_<split>/`: `config.json`, `metrics.json`,
`per_episode.csv`, `summary.csv`, `comparison_table.csv`, `decisions.csv`.
