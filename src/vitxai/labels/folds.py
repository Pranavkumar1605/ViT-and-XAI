"""Phase 3b: walk-forward folds.

Fold for test year Y:
    train : [Y-5-01-01, validation start)
    val   : last `val_months` of the 5-year window
    test  : [Y-01-01, Y+1-01-01)
A sample on day t belongs to a segment if t is inside it AND the last day its label uses
(`label_end`: t+1 for threshold labels, t+h for peak/valley labels) is also inside it, so no
label reaches into the next segment.
"""
from __future__ import annotations

import pandas as pd

SPLITS = ("train", "val", "test")


def make_folds(first_test_year: int, last_test_year: int, train_years: int, val_months: int) -> list[dict]:
    folds = []
    for k, year in enumerate(range(first_test_year, last_test_year + 1)):
        train_start = pd.Timestamp(year=year - train_years, month=1, day=1)
        test_start = pd.Timestamp(year=year, month=1, day=1)
        val_start = test_start - pd.DateOffset(months=val_months)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        folds.append({
            "fold": k,
            "test_year": year,
            "train": [train_start.strftime("%Y-%m-%d"), val_start.strftime("%Y-%m-%d")],
            "val": [val_start.strftime("%Y-%m-%d"), test_start.strftime("%Y-%m-%d")],
            "test": [test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")],
        })
    return folds


def folds_from_cfg(cfg: dict) -> list[dict]:
    f = cfg["folds"]
    return make_folds(f["first_test_year"], f["last_test_year"], f["train_years"], f["val_months"])


def split_mask(dates: pd.Series, next_dates: pd.Series, bounds: list[str]) -> pd.Series:
    """Half-open [start, end) for both the sample day and its label day."""
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    return (dates >= start) & (dates < end) & (next_dates < end)
