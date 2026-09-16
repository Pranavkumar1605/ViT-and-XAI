"""Phase 4: indicator images built on the fly.

Image for (ticker, day t): rows = indicators in config order, columns = days t-window+1 .. t
(oldest on the left), shape (1, n_indicators, window). Nothing is written to disk except the
per-ticker indicator matrices, so window length / row subsets can change via config.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vitxai.config import data_dir
from vitxai.data import ticker_file
from vitxai.features.indicators import indicator_names, indicators_dir
from vitxai.images.normalize import apply_norm, fit_norm_stats
from vitxai.labels.folds import split_mask


def labels_path(cfg: dict) -> Path:
    """`labels.file` lets runs with different thresholds share one data folder."""
    return data_dir(cfg) / "processed" / cfg["labels"].get("file", "labels.parquet")


def load_indicator_store(cfg: dict, tickers: list[str]) -> dict:
    """ticker -> (dates, values days x n_ind float64), columns in config order."""
    names = indicator_names(cfg)
    store = {}
    for t in tickers:
        df = pd.read_parquet(indicators_dir(cfg) / f"{ticker_file(t)}.parquet")
        store[t] = (pd.DatetimeIndex(df.index), df[names].to_numpy(np.float64))
    return store


class IndicatorImageDataset(Dataset):
    def __init__(self, arrays: list[np.ndarray], samples: pd.DataFrame, window: int):
        """arrays[i]: normalized float32 matrix (days x n_ind) of ticker i.
        samples: columns ticker_idx, row, label (+ date, ticker for bookkeeping)."""
        self.arrays = arrays
        self.samples = samples.reset_index(drop=True)
        self.window = window
        self._ti = self.samples["ticker_idx"].to_numpy()
        self._row = self.samples["row"].to_numpy()
        self._y = self.samples["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.samples)

    def image(self, i: int) -> np.ndarray:
        r = self._row[i]
        win = self.arrays[self._ti[i]][r - self.window + 1: r + 1]  # (window, n_ind)
        return win.T[None]                                         # (1, n_ind, window)

    def __getitem__(self, i: int):
        return torch.from_numpy(np.ascontiguousarray(self.image(i))), int(self._y[i])


def build_fold_datasets(cfg: dict, fold: dict, store: dict, labels: pd.DataFrame):
    """Returns ({split: dataset}, norm_stats). Stats are fitted on the fold's train dates only."""
    window = cfg["image"]["window"]
    stats = fit_norm_stats(store, fold["train"][0], fold["train"][1])
    tickers = list(store.keys())
    arrays = [apply_norm(store[t][1], stats, cfg["image"]["clip"]) for t in tickers]
    t_idx = {t: i for i, t in enumerate(tickers)}

    labels = labels[labels["ticker"].isin(list(t_idx))].copy()
    labels["ticker_idx"] = labels["ticker"].map(t_idx)
    labels = labels[labels["row"] >= window - 1]

    datasets = {}
    end_col = "label_end" if "label_end" in labels else "next_date"   # older label files
    for split in ("train", "val", "test"):
        m = split_mask(labels["date"], labels[end_col], fold[split])
        datasets[split] = IndicatorImageDataset(arrays, labels[m], window)
    return datasets, stats
