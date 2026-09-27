# 01 — Problem Statement

**SIH 26055 — Smart Scan Strategy for Electronic Warfare in the absence of prior reliable intelligence of emitters.**

A receiver has to monitor a wide spectrum, but its *instantaneous bandwidth* is limited: it can inspect only one
band per dwell.

```
Wide spectrum:   F1   F2   F3   F4   F5   F6   F7   F8     (1.0 … 1.7 GHz)
Receiver:        can inspect ONE band at a time
```

**Traditional open-loop scan**

```
F1 → F2 → F3 → F4 → F5 → F6 → F7 → F8 → F1 → …
```

Suppose a threat emitter transmits on F6 for a short time while the sweep is at F2:

```
                 F6 active
                    ↓
sweep:  F2 → F3 → F4 → F5 → F6        the transmission may be over before F6 is visited
```

The fixed sweep ignores what it has already observed: if F3 was detected three times in a row, the next visit to F3
is still N dwells away.

**SmartScan**

```
observations → state / belief → RL policy π(a|s) → select next band → observe → reward → learn
```

## Why this is a sequential decision problem

The question is not *"which frequency contains a signal?"* but

> Given everything the receiver has observed so far, **which band should it scan next?**

Each decision determines what the receiver gets to observe next, and therefore what it will know when it makes the
following decision. Decisions are coupled through time: scanning F3 now means F6 is not observed now, which changes
the information available for every later choice. This is the defining property of a sequential decision problem
(MDP/POMDP), and the reason reinforcement learning and belief-state planning are appropriate tools.

## What this repository delivers

A research **simulation** platform: synthetic (optionally Turing-dataset-driven) RF environment, a limited-bandwidth
receiver, four scan schedulers (Open Loop, Random, POMDP, PPO), a fair multi-seed evaluation engine and an
engineering console. It is not an operational EW system and makes no field-performance claims.
