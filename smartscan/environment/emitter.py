"""Emitter behaviour models.

Every emitter is realised, for one scenario seed, into a *band trace*: an integer
array of length T holding the band it transmits on at each timestep, or -1 when
it is silent. Randomness comes only from the ``numpy.random.Generator`` passed in,
so a scenario seed fully determines the ground truth.

Supported behaviours
--------------------
static        always on, one band
periodic      ON for ``duty * period`` timesteps at the start of every period
intermittent  two-state Markov chain (P(OFF->ON) = p_on, P(ON->OFF) = p_off)
agile         continuously on, hopping band every ``hop_interval`` timesteps,
              cyclically through ``hop_bands`` (or at random within the hop set)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from smartscan.config import EmitterSpec, RandomEmitterRanges


@dataclass
class RealizedEmitter:
    """An emitter whose random parameters have been fixed for one scenario seed."""

    name: str
    kind: str
    snr_db: float
    params: dict
    trace: np.ndarray  # (T,) int, band index or -1

    def describe(self) -> str:
        p = self.params
        if self.kind == "static":
            return f"static on F{p['band'] + 1}"
        if self.kind == "periodic":
            return f"periodic on F{p['band'] + 1}, T={p['period']}, duty={p['duty']:.2f}, phase={p['phase']}"
        if self.kind == "intermittent":
            return f"intermittent on F{p['band'] + 1}, p_on={p['p_on']:.3f}, p_off={p['p_off']:.3f}"
        if self.kind == "agile":
            hops = "->".join(f"F{b + 1}" for b in p["hop_bands"])
            mode = "random" if p["hop_random"] else "cyclic"
            return f"agile {mode} {hops}, hop every {p['hop_interval']}"
        if self.kind == "dataset":
            return f"Turing emitter label {p.get('label')}"
        return self.kind


def _uniform(rng: np.random.Generator, lo_hi, integer: bool = False):
    lo, hi = lo_hi
    if integer:
        return int(rng.integers(int(lo), int(hi) + 1))
    return float(rng.uniform(lo, hi))


def realize_emitter(
    spec: EmitterSpec,
    index: int,
    n_bands: int,
    total_steps: int,
    rng: np.random.Generator,
    ranges: RandomEmitterRanges,
) -> RealizedEmitter:
    """Fix all random parameters of ``spec`` and generate its band trace."""
    kind = spec.kind
    if kind not in ("static", "periodic", "intermittent", "agile"):
        raise ValueError(f"Unknown emitter kind '{kind}'")

    def pick_band(b):
        if b is None:
            return int(rng.integers(0, n_bands))
        if not 0 <= int(b) < n_bands:
            raise ValueError(f"Emitter band {b} outside 0..{n_bands - 1}")
        return int(b)

    snr = float(spec.snr_db) if spec.snr_db is not None else _uniform(rng, ranges.snr_db)
    start = max(0, int(spec.start))
    stop = total_steps if spec.stop is None else min(total_steps, int(spec.stop))
    trace = np.full(total_steps, -1, dtype=np.int32)
    params: dict = {"start": start, "stop": stop}

    if kind == "static":
        band = pick_band(spec.band)
        params["band"] = band
        trace[start:stop] = band

    elif kind == "periodic":
        band = pick_band(spec.band)
        period = int(spec.period) if spec.period is not None else _uniform(rng, ranges.period, integer=True)
        period = max(2, period)
        duty = float(spec.duty) if spec.duty is not None else _uniform(rng, ranges.duty)
        on_len = int(np.clip(round(duty * period), 1, period - 1))
        phase = int(spec.phase) % period if spec.phase is not None else int(rng.integers(0, period))
        t = np.arange(total_steps)
        on = ((t - phase) % period) < on_len
        on[:start] = False
        on[stop:] = False
        trace[on] = band
        params.update(band=band, period=period, duty=on_len / period, phase=phase)

    elif kind == "intermittent":
        band = pick_band(spec.band)
        p_on = float(spec.p_on) if spec.p_on is not None else _uniform(rng, ranges.p_on)
        p_off = float(spec.p_off) if spec.p_off is not None else _uniform(rng, ranges.p_off)
        u = rng.random(total_steps)
        state = bool(rng.random() < p_on / max(1e-9, p_on + p_off))  # stationary start
        for t in range(total_steps):
            state = (u[t] >= p_off) if state else (u[t] < p_on)
            if start <= t < stop and state:
                trace[t] = band
        params.update(band=band, p_on=p_on, p_off=p_off)

    else:  # agile
        interval = (
            int(spec.hop_interval) if spec.hop_interval is not None
            else _uniform(rng, ranges.hop_interval, integer=True)
        )
        interval = max(1, interval)
        if spec.hop_bands:
            hop_bands = [pick_band(b) for b in spec.hop_bands]
        else:
            size = min(n_bands, _uniform(rng, ranges.hop_set_size, integer=True))
            hop_bands = [int(b) for b in rng.choice(n_bands, size=size, replace=False)]
        offset = int(rng.integers(0, len(hop_bands)))
        n_hops = (total_steps + interval - 1) // interval
        if spec.hop_random:
            seq = rng.choice(hop_bands, size=n_hops)
        else:
            seq = np.array([hop_bands[(offset + k) % len(hop_bands)] for k in range(n_hops)])
        full = np.repeat(seq, interval)[:total_steps].astype(np.int32)
        trace[start:stop] = full[start:stop]
        params.update(hop_bands=hop_bands, hop_interval=interval, hop_random=bool(spec.hop_random))

    name = spec.name or f"E{index + 1}-{kind}"
    return RealizedEmitter(name=name, kind=kind, snr_db=snr, params=params, trace=trace)


def spec_summary(spec: EmitterSpec) -> dict:
    """Flat dict of an EmitterSpec for tables (None = randomised per seed)."""
    return {k: v for k, v in asdict(spec).items()}
