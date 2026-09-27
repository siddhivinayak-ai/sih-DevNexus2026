# 04 — Receiver Model

`smartscan/receiver/receiver.py`, `detector.py`

| Parameter | Config key | Meaning |
|---|---|---|
| instantaneous bandwidth | `receiver.instantaneous_bands` | contiguous bands observed per dwell (window starts at the commanded band, clipped to the spectrum) |
| selected frequency | action `a_t` | commanded band |
| dwell time | `receiver.dwell_steps` | timesteps integrated per dwell; a band counts as *signal present* if active at any of them |
| scan latency | `receiver.retune_steps` | blind timesteps before a dwell on a *different* band |
| Pd | `receiver.pd` | P(detect ∣ signal) for the `fixed` model |
| Pfa | `receiver.pfa` | P(detect ∣ no signal) per dwell |
| detection model | `receiver.detection_model` | `fixed` or `snr` |
| measurement uncertainty | `receiver.amplitude_sigma_db` | Gaussian noise on the reported amplitude |

**SNR model.** Neyman–Pearson detection of a known signal in white Gaussian noise:
`τ = Q⁻¹(Pfa)`, `Pd = Q(τ − √(2·SNR))`, using the strongest emitter's SNR in the band. This is an idealised textbook
model and the parameters are simulation values.

**Reported amplitude.** On a true detection: `SNR_dB + N(0, σ²)`. On a false alarm: the square-law noise energy
conditioned on exceeding the threshold `−ln Pfa`, i.e. `10·log10(−ln Pfa + Exp(1))`. No amplitude is reported
without a detection.

**Outcome per observed band.** Detection (signal present, fired), Miss (present, silent), False alarm (absent, fired),
Correct rejection (absent, silent).

**Common random numbers.** The detector's uniform and Gaussian draws are pre-drawn per (timestep, band) from the
scenario seed. Two schedulers that scan the same band at the same time get the same detector outcome, so differences
between schedulers come from their decisions and not from noise luck.

The receiver never observes bands outside its window. Scan strategy matters because the action decides which band is
observed (see `tests/test_schedulers.py::test_different_actions_give_different_observations`).
