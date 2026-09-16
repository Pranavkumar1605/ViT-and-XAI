"""Phase 3a: Buy / Hold / Sell labels.

method "threshold"   (workflow PDF, base paper text): next-day return above +theta / below -theta.
method "peak_valley" (CNN-TA, Sezer & Ozbayoglu 2018): day t is Buy if its close is the lowest in the
    centred window [t-h, t+h] (h = window // 2), Sell if it is the highest, Hold otherwise. The label
    uses closes up to t+h, so `label_end` = day t+h and fold boundaries embargo h days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLD, BUY, SELL = 0, 1, 2


def label_returns(ret: np.ndarray | pd.Series, theta: float) -> np.ndarray:
    ret = np.asarray(ret, dtype=np.float64)
    y = np.full(ret.shape, HOLD, dtype=np.int64)
    y[ret > theta] = BUY
    y[ret < -theta] = SELL
    return y


def peak_valley_labels(close: np.ndarray, window: int) -> np.ndarray:
    """Labels for the centre of each full window; days without a full window get -1."""
    if window < 3 or window % 2 == 0:
        raise ValueError("peak_valley window must be odd and >= 3")
    close = np.asarray(close, dtype=np.float64)
    h = window // 2
    y = np.full(len(close), -1, dtype=np.int64)
    if len(close) < window:
        return y
    win = np.lib.stride_tricks.sliding_window_view(close, window)
    centre = win[:, h]
    lab = np.full(len(win), HOLD, dtype=np.int64)
    lab[centre == win.min(axis=1)] = BUY
    lab[(centre == win.max(axis=1)) & (lab == HOLD)] = SELL
    y[h:len(close) - h] = lab
    return y


def make_labels(close: pd.Series, theta: float, method: str = "threshold", window: int = 11) -> pd.DataFrame:
    """For each day t: r = Close(t+1) / Close(t) - 1 (kept for every method), the label, and
    `label_end`, the last day whose close the label uses. Days without a label are dropped.
    Fold boundaries use `label_end` so no label crosses into the next segment."""
    dates = pd.Series(close.index, index=close.index)
    df = pd.DataFrame({"ret_next": close.shift(-1) / close - 1.0, "next_date": dates.shift(-1)}, index=close.index)
    if method == "threshold":
        df["label_end"] = df["next_date"]
        df["label"] = label_returns(df["ret_next"].fillna(0.0), theta)
    elif method == "peak_valley":
        df["label_end"] = dates.shift(-(window // 2))
        df["label"] = peak_valley_labels(close.to_numpy(), window)
        df = df[df["label"] >= 0]
    else:
        raise ValueError(f"unknown label method {method!r}")
    df = df.dropna()
    df.index.name = "date"
    return df
