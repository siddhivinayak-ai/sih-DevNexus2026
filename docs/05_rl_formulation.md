# 05 — RL Formulation

| Element | Definition in SmartScan |
|---|---|
| **Ground truth** | `Scenario.activity` (which emitter is on which band at every step). Hidden from the agent. |
| **Observation** | Receiver report after a dwell: scanned band(s), detect/no-detect, amplitude. |
| **State** `s_t` | Vector built only from receiver-observable history (`ReceiverHistory.vector()`), below. |
| **Action** `a_t` | Band index in `{0,…,N−1}` (`Discrete(N)`); the receiver really tunes there. |
| **Reward** `r_t` | Interception-oriented reward from simulation events ([06](06_reward_function.md)). |
| **Transition** | Environment evolves (emitters follow their models); the receiver dwells on `a_t`; the detector samples an outcome; the history and belief update. |
| **Policy** | `π(a∣s)`, an MLP trained with PPO. |
| **Episode** | One scenario seed, `warmup_steps + episode_length` timesteps (256 decisions by default). |

## State vector (length 11·N + 3, every feature in [0, 1])

Per band (10 blocks of N):

1. `belief`: Bayesian P(band active), per-band two-state HMM filter using the receiver's own nominal Pd/Pfa and assumed switching rates (`observation.belief_p_on/p_off`)
2. `uncertainty = 4p(1−p)`
3. `staleness`: steps since last scan / (2N), clipped
4. `detection_ratio`: detections / scans
5. `detection_ema`: exponential average of recent scan results
6. `scan_share`: share of all scans
7. `period_estimate`: mode of intervals between detection onsets / `max_period`
8. `period_due`: 1 when a periodic onset is predicted now
9. `time_since_detection` / episode length
10. `last_amplitude` / 30 dB

Plus a one-hot of the previous action (N) and 3 globals: episode progress, spectrum coverage, and whether the last dwell detected.

**Excluded on purpose.** Emitter identities, true activity, and the *reward*: the reward separates true detections
from false alarms using ground truth, so feeding it back would leak hidden state.

## Why an MDP approximation is reasonable

The true problem is a POMDP: emitter states are hidden. The state vector is a hand-built sufficient-statistic
approximation of the observation history (belief, recency, periodicity, counts). With it the process is
approximately Markov, and a memoryless policy on `s_t` can act well. Behaviour that depends on longer history than
these statistics capture is a known limitation ([13](13_limitations.md)).

## Relation to the POMDP baseline

- **POMDP baseline**: explicit belief state with hand-designed update and scoring rules (the original project's scheduler, unchanged).
- **PPO**: learns the mapping from a belief-like state to actions from interaction and reward, without hand-set weights.
