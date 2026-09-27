"""Turing Synthetic Radar Dataset integration (Mode B).

The Alan Turing Institute *Turing Deinterleaving Challenge* publishes simulated
pulse trains (HDF5) with one PDW per pulse and an emitter label per pulse:

    <root>/<mode>/<split>_<mode>/*.h5        mode in {stare, scan}, split in {train, validation, test}
    data   : float32 (n_pulses, 5) = [ToA (us), centre frequency, pulse width (us), AoA, amplitude]
    labels : int8    (n_pulses,)   = emitter id
    metadata (group)

This module never loads a whole file into memory for display: reads are sliced
(``max_pulses``) and optionally strided. The dataset describes the *emitter
environment*; SmartScan builds its own limited-bandwidth receiver on top. The
*stare* mode is an (almost) oracle view of the environment and is therefore the
appropriate ground truth. The *scan* mode already contains the effect of another
scanning receiver and is shown in the explorer only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from smartscan.config import PROJECT_ROOT, ScenarioConfig

try:
    import h5py

    HAS_H5PY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_H5PY = False

HF_DATASET_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
PDW_COLUMNS = ["toa", "cf", "pw", "aoa", "amplitude"]
PDW_LABELS = {
    "toa": "Time of Arrival (us)",
    "cf": "Centre Frequency",
    "pw": "Pulse Width (us)",
    "aoa": "Angle of Arrival",
    "amplitude": "Amplitude",
}
NOT_CONFIGURED = "Turing dataset not configured — using synthetic RF environment."


class TuringUnavailable(RuntimeError):
    pass


@dataclass
class TuringFile:
    path: Path
    mode: str
    split: str
    index: int

    @property
    def label(self) -> str:
        return f"{self.mode}/{self.split}/{self.path.name}"


def dataset_root(local_dir: str | None = None) -> Path:
    """Environment variable SMARTSCAN_TURING_DIR overrides the configured directory."""
    env = os.environ.get("SMARTSCAN_TURING_DIR")
    root = Path(env or local_dir or "data/turing")
    return root if root.is_absolute() else PROJECT_ROOT / root


def _parse(path: Path) -> TuringFile:
    mode, split = "unknown", "unknown"
    parent = path.parent.name
    if "_" in parent:
        split, mode = parent.rsplit("_", 1)
    else:
        for part in path.parts:
            if part in ("stare", "scan"):
                mode = part
            if part in ("train", "validation", "test"):
                split = part
    digits = "".join(ch for ch in path.stem.split("_")[-1] if ch.isdigit())
    return TuringFile(path=path, mode=mode, split=split, index=int(digits) if digits else 0)


def find_files(root: Path | None = None, mode: str | None = None, split: str | None = None) -> list[TuringFile]:
    root = root or dataset_root()
    if not root.exists():
        return []
    files = [_parse(p) for p in root.rglob("*.h5")]
    if mode:
        files = [f for f in files if f.mode == mode]
    if split:
        files = [f for f in files if f.split == split]
    return sorted(files, key=lambda f: (f.mode, f.split, f.index, f.path.name))


def status(local_dir: str | None = None) -> tuple[bool, str]:
    if not HAS_H5PY:
        return False, "h5py is not installed. " + NOT_CONFIGURED
    root = dataset_root(local_dir)
    files = find_files(root)
    if not files:
        return False, f"{NOT_CONFIGURED} (no .h5 files under {root})"
    return True, f"{len(files)} Turing pulse-train file(s) found under {root}"


def file_info(path: Path) -> dict:
    """Shape/labels/metadata attributes without reading the pulse array."""
    with h5py.File(path, "r") as f:
        data = f["data"]
        info = {
            "n_pulses": int(data.shape[0]),
            "n_features": int(data.shape[1]) if data.ndim > 1 else 1,
            "dtype": str(data.dtype),
            "has_labels": "labels" in f,
            "metadata": {},
        }
        if "metadata" in f:
            meta = f["metadata"]
            for k, v in meta.attrs.items():
                info["metadata"][k] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
            if "feature_names" in meta:
                names = meta["feature_names"][()]
                info["metadata"]["feature_names"] = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return info


def read_pdws(path: Path, start: int = 0, count: int = 100_000, stride: int = 1) -> pd.DataFrame:
    """Read a bounded, optionally strided slice of PDWs as a DataFrame."""
    with h5py.File(path, "r") as f:
        n = f["data"].shape[0]
        start = int(np.clip(start, 0, max(0, n - 1)))
        stop = int(min(n, start + count * max(1, stride)))
        sl = slice(start, stop, max(1, int(stride)))
        data = np.asarray(f["data"][sl], dtype=np.float64)
        labels = np.asarray(f["labels"][sl]).reshape(-1) if "labels" in f else None
    cols = PDW_COLUMNS[: data.shape[1]] + [f"f{i}" for i in range(len(PDW_COLUMNS), data.shape[1])]
    df = pd.DataFrame(data, columns=cols)
    df["label"] = labels.astype(int) if labels is not None else -1
    return df


def download_sample(mode: str = "stare", split: str = "train", n_files: int = 1, token: str | None = None) -> list[Path]:
    """Download a few pulse-train files from Hugging Face (gated: requires an access token)."""
    from huggingface_hub import HfApi, hf_hub_download

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    prefix = f"{mode}/{split}_{mode}/"
    names = [p for p in HfApi().list_repo_files(HF_DATASET_ID, repo_type="dataset", token=token) if p.startswith(prefix)]
    names = sorted(names, key=lambda p: _parse(Path(p)).index)[: max(1, n_files)]
    if not names:
        raise TuringUnavailable(f"No files matching {prefix} in {HF_DATASET_ID}")
    root = dataset_root()
    return [
        Path(hf_hub_download(HF_DATASET_ID, filename=name, repo_type="dataset", token=token, local_dir=root))
        for name in names
    ]


def build_turing_scenario(cfg: ScenarioConfig, seed: int):
    """Bin one Turing pulse train into a (emitter, time, band) activity grid.

    * file  : chosen as ``files[seed % n_files]`` from the configured mode/split
    * time  : the first ``max_pulses`` pulses, ToA span split into ``T`` equal bins
    * bands : centre frequency split into ``n_bands`` equal bins over [cf_min, cf_max]
    * emitters : one per emitter label; SNR is not calibrated in the dataset, so a
      nominal SNR is used (see scenario notes) and the fixed Pd/Pfa receiver is recommended.
    """
    from smartscan.environment.emitter import RealizedEmitter
    from smartscan.environment.scenario import Scenario

    if not HAS_H5PY:
        raise TuringUnavailable("h5py is not installed")
    tcfg = cfg.turing
    files = find_files(dataset_root(tcfg.local_dir), tcfg.mode, tcfg.split)
    if not files:
        raise TuringUnavailable(NOT_CONFIGURED)
    tf = files[int(seed) % len(files)]
    df = read_pdws(tf.path, 0, tcfg.max_pulses).sort_values("toa")
    if df.empty:
        raise TuringUnavailable(f"{tf.label} contains no pulses")

    total = int(cfg.episode_length) + int(cfg.warmup_steps)
    N = int(cfg.n_bands)
    toa = df["toa"].to_numpy()
    t0, t1 = toa.min(), toa.max()
    dt = (t1 - t0) / total if t1 > t0 else 1.0
    t_idx = np.clip(((toa - t0) / dt).astype(int), 0, total - 1)

    cf = df["cf"].to_numpy()
    lo = tcfg.cf_min if tcfg.cf_min is not None else float(cf.min())
    hi = tcfg.cf_max if tcfg.cf_max is not None else float(cf.max())
    width = (hi - lo) / N if hi > lo else 1.0
    in_range = (cf >= lo) & (cf <= hi)
    b_idx = np.clip(((cf - lo) / width).astype(int), 0, N - 1)

    labels = df["label"].to_numpy()
    uniq = np.unique(labels[in_range])
    activity = np.zeros((len(uniq), total, N), dtype=bool)
    emitters = []
    nominal_snr = float(np.mean(cfg.ranges.snr_db))
    for e, lab in enumerate(uniq):
        m = in_range & (labels == lab)
        activity[e, t_idx[m], b_idx[m]] = True
        trace = np.where(activity[e].any(axis=1), activity[e].argmax(axis=1), -1).astype(np.int32)
        emitters.append(
            RealizedEmitter(
                name=f"T{int(lab)}",
                kind="dataset",
                snr_db=nominal_snr,
                params={"label": int(lab), "pulses": int(m.sum())},
                trace=trace,
            )
        )
    centres = [lo + (i + 0.5) * width for i in range(N)]
    return Scenario(
        name=f"{cfg.name} [{tf.label}]",
        seed=int(seed),
        source="turing",
        n_bands=N,
        total_steps=total,
        warmup_steps=int(cfg.warmup_steps),
        band_labels=cfg.band_labels(),
        band_centres=centres,
        band_unit="dataset CF units",
        emitters=emitters,
        activity=activity,
        snr_db=np.full(len(emitters), nominal_snr),
        notes=[
            f"source file {tf.label}, {len(df)} pulses",
            f"timestep = {dt:.3f} ToA units, band width = {width:.4g} CF units",
            f"{int((~in_range).sum())} pulses outside [cf_min, cf_max] ignored",
            f"nominal SNR {nominal_snr:.1f} dB (dataset amplitude not calibrated)",
        ],
    )
