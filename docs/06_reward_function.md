# 06 — Reward Function

Implemented in `smartscan/environment/reward.py` and computed per decision in `SmartScanEnv.step`:

```
R_t = w_d·D_t + w_i·I_t − w_f·F_t − w_m·M_t − w_s·S_t − w_l·L_t − w_v·V_t
```

| Term | Meaning | Default weight | Reason |
|---|---|---|---|
| D_t | true detections in the dwell | `detection` 0.1 | small reward for confirming activity; kept small so that camping on a static emitter does not pay |
| I_t | transmissions intercepted **for the first time** | `new_intercept` 1.0 | the primary objective: catch transmissions, including new ones |
| F_t | false alarms | `false_alarm` 0.2 | false reports waste downstream analysis effort |
| M_t | missed detections (signal present, detector silent) | `miss` 0.0 | off by default: caused by receiver noise, not by the decision; configurable |
| S_t | 1 if the receiver retuned | `switch_cost` 0.0 | models retuning cost when `retune_steps > 0` |
| L_t | fraction of emitters with an active, not-yet-intercepted transmission at the end of the dwell | `latency` 0.1 | pressure to intercept quickly, directly related to intercept time |
| V_t | empty re-scans: band scanned again within `revisit_window` steps and nothing detected | `revisit` 0.05 | discourages redundant empty dwells |

All weights are in `reward:` of the YAML config and editable in the Scenario Builder.

**Ground truth.** D, I, F, M and L require it. The reward is therefore a *training signal produced by the simulator*
(like labelled training data) and is never part of the agent's observation. Every term is computed from actual
simulated events. `tests/test_reward.py` checks the equation and each event type.

**Original app.** The original `app.py` used `10·detections − 2·false_alarms − 5·missed_opportunities` at the end of
the run. It rewarded staring at a static emitter and penalised every step on which any emitter was active elsewhere.
The per-transmission terms above replace it.
