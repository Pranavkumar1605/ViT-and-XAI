"""Per-indicator standardization fitted on a fold's training dates only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fit_norm_stats(store: dict, start: str, end: str) -> dict[str, np.ndarray]:
    """store: ticker -> (dates DatetimeIndex, values ndarray days x n_ind).
    Pools every ticker's rows with start <= date < end."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    chunks = [vals[(dates >= lo) & (dates < hi)] for dates, vals in store.values()]
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        raise ValueError(f"no indicator rows between {start} and {end} to fit normalization")
    x = np.concatenate(chunks, axis=0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # constant rows stay at 0
    return {"mean": mean, "std": std, "fit_start": np.array(start), "fit_end": np.array(end), "n_rows": np.array(len(x))}


def apply_norm(values: np.ndarray, stats: dict, clip: float) -> np.ndarray:
    z = (values - stats["mean"]) / stats["std"]
    return np.clip(z, -clip, clip).astype(np.float32)


def save_norm_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)


def load_norm_stats(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
