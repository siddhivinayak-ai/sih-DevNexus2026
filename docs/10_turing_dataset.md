# 10 — Turing Dataset

Source: Alan Turing Institute, *Turing Deinterleaving Challenge* / Turing Synthetic Radar Dataset
(`alan-turing-institute/turing-synthetic-radar-dataset` on Hugging Face, gated).

**Layout** `<root>/<mode>/<split>_<mode>/*.h5`, mode ∈ {stare, scan}, split ∈ {train, validation, test}.
Each file has `data` float32 (n_pulses, 5) = [ToA (µs), centre frequency, pulse width (µs), AoA, amplitude],
`labels` int8 (emitter id per pulse) and a `metadata` group.

**What it is and is not.** The dataset simulates the *emitter environment* (PDWs). It is **not** a SmartScan
training environment and does not solve the scheduling problem. SmartScan uses it only as a ground-truth source
(Mode B) and builds its own limited-bandwidth receiver on top. *Stare* mode is an almost-oracle view of the
environment, so it is the appropriate ground truth. *Scan* mode already contains another receiver's sweep, so the
explorer shows it but it is not used as ground truth.

**Mode B binning** (`environment/turing.py::build_turing_scenario`): scenario seed → file `files[seed % n]` of the
split; the first `max_pulses` pulses; ToA span → `T` equal time bins; CF → `N` equal bins over `[cf_min, cf_max]`;
one emitter per label. Amplitude is not calibrated to SNR, so a nominal SNR is used and recorded in the scenario notes.

**Memory.** Reads are sliced (`max_pulses`, start offset, stride). The explorer caches only the sampled frame. The
full dataset is never loaded.

**Setup.** Put files under `data/turing/` (or set `SMARTSCAN_TURING_DIR`), or use *Dataset Explorer → DOWNLOAD
SAMPLE* with a Hugging Face token. The token is used for the request only and is not stored. Without files the app
shows *"Turing dataset not configured — using synthetic RF environment."* and keeps working.
