# 03 — RF Environment

Time is discrete (timesteps). The spectrum is divided into N bands (`F1…FN`, centre `f_start_ghz + i·band_spacing_ghz`).
A **scenario seed** fixes every random choice, so a seed reproduces its ground truth exactly
(`numpy.random.SeedSequence([seed, …])`).

## Emitter models (`smartscan/environment/emitter.py`)

| Kind | Parameters | Behaviour |
|---|---|---|
| static | band | on for the whole activity window |
| periodic | band, period P, duty d, phase φ | on when `(t − φ) mod P < round(d·P)` |
| intermittent | band, p_on, p_off | two-state Markov chain, stationary start |
| agile | hop_bands, hop_interval, hop_random | always on, changes band every `hop_interval` steps, cyclic or random within the hop set |

Every field left empty (`null`) is drawn per seed from `scenario.ranges` (SNR, period, duty, p_on, p_off, hop interval,
hop-set size). `random_emitters: {kind: count}` adds fully random emitters. Several emitters may share a band.
`start`/`stop` restrict an emitter to a time window (e.g. a *new* emitter appearing mid-episode).

## Ground truth (`smartscan/environment/scenario.py`)

- `activity[e, t, n]`: emitter e active in band n at time t
- `ground_truth[t, n]`: any emitter active
- **transmission**: maximal run of consecutive active steps of one emitter in one band. Periodic → one per ON
  window; agile → one per hop; static → one. Interception is scored per transmission.

Ground truth is used only by the receiver's detector model, the reward engine and the evaluator.

## Presets (`configs/`)

| File | Content |
|---|---|
| `default.yaml` | 8 bands, 1 static + 2 periodic + 1 intermittent + 1 agile emitter re-drawn per seed |
| `sih_example.yaml` | The problem-statement example: periodic F3 (P=5), short-burst threat on F6, agile F2→F6→F4→F7, static F1 |
| `wideband_32.yaml` | 32 bands, 14 emitters, SNR-dependent detection |
| `turing.yaml` | Mode B, ground truth from Turing stare-mode pulse trains |
