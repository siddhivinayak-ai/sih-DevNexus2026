# 08 — Baseline Algorithms

All schedulers implement `BaseScheduler` (`smartscan/schedulers/base.py`):
`reset(n_bands, seed)`, `select_action(ctx) -> band`, `update(action, observation)`, `explain()`.

| Scheduler | File | Rule |
|---|---|---|
| Open Loop | `open_loop.py` | cyclic sweep F1→…→FN (configurable stride/start); ignores observations |
| Random | `random_scan.py` | uniform random band, seeded per scenario seed |
| POMDP | `pomdp.py` | original project scheduler, algorithm unchanged (below) |
| PPO | `ppo.py` | trained Stable-Baselines3 policy |

## POMDP (belief-state) baseline, carried over unchanged

```
prediction : b ← 0.9·b + 0.05·(1 − b)
update     : scanned band  detection → min(0.99, b + 0.3),  no detection → max(0.01, b − 0.3)
periodicity: mode of intervals between detections on a band, trusted after seen twice
score      : 1.0·b + 0.5·b(1−b) + 0.1·freshness + 1.5·periodicity_hit
action     : argmax(score)
```

The only change is the interface: the per-band state now lives in the scheduler object, and the warm-up applies to all
schedulers, not only POMDP. Because belief dominates the score, the rule tends to exploit bands with confirmed
activity. The Algorithm Comparison page shows the measured effect on coverage and interception.
