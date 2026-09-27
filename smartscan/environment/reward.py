"""Reward engine. Computed from ground-truth simulation events of one dwell.

    R_t = w_d D_t + w_i I_t - w_f F_t - w_m M_t - w_s S_t - w_l L_t - w_v V_t

D_t  true detections (observed band has a signal and the detector fired)
I_t  transmissions intercepted for the first time during this dwell
F_t  false alarms (detector fired, no signal in the band)
M_t  missed detections (signal in the band, detector did not fire)
S_t  1 if the receiver retuned to a different band
L_t  fraction of emitters that, at the end of the dwell, have an active
     transmission that has not been intercepted yet (anywhere in the spectrum)
V_t  empty re-scans: observed bands scanned again within ``revisit_window``
     timesteps and no detection this time

The reward is a *training signal produced by the simulator*: it needs ground truth
(D vs F, I, L) and is therefore never part of the agent's observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from smartscan.config import RewardConfig

TERM_ORDER = ["detection", "new_intercept", "false_alarm", "miss", "switch", "latency", "revisit"]


@dataclass
class RewardEvents:
    detections: int = 0
    new_intercepts: int = 0
    false_alarms: int = 0
    misses: int = 0
    switched: int = 0
    latency: float = 0.0
    empty_revisits: int = 0


def compute_reward(ev: RewardEvents, w: RewardConfig) -> tuple[float, dict[str, float]]:
    terms = {
        "detection": w.detection * ev.detections,
        "new_intercept": w.new_intercept * ev.new_intercepts,
        "false_alarm": -w.false_alarm * ev.false_alarms,
        "miss": -w.miss * ev.misses,
        "switch": -w.switch_cost * ev.switched,
        "latency": -w.latency * ev.latency,
        "revisit": -w.revisit * ev.empty_revisits,
    }
    return float(sum(terms.values())), terms
