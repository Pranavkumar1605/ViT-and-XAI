"""Phase 2: compute the indicator matrix (days x n_indicators) from configs/indicators.yaml.

Every indicator uses data up to day t only (TA-Lib functions are causal), so the value on
day t is unchanged if prices after t change. tests/test_indicators.py checks this.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import talib

from vitxai.config import data_dir

CATEGORY_ORDER = ["overlap", "momentum", "volume", "volatility", "price_transform", "statistics"]


# ---------------------------------------------------------------- spec helpers
def indicator_specs(cfg: dict) -> list[dict]:
    """Ordered specs, restricted to `active_subset` when one is set."""
    ind_cfg = cfg["indicators"]
    specs = ind_cfg["indicators"]
    subset = ind_cfg.get("active_subset")
    if subset:
        keep = set(ind_cfg["subsets"][subset])
        specs = [s for s in specs if s["name"] in keep]
    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate indicator names in indicators.yaml")
    return specs


def indicator_names(cfg: dict) -> list[str]:
    return [s["name"] for s in indicator_specs(cfg)]


def indicator_categories(cfg: dict) -> list[str]:
    return [s["category"] for s in indicator_specs(cfg)]


# ---------------------------------------------------------------- custom functions
def _vol_ratio(volume: np.ndarray, timeperiod: int) -> np.ndarray:
    return volume / talib.SMA(volume, timeperiod=timeperiod)


def _ret_std(close: np.ndarray, timeperiod: int) -> np.ndarray:
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    return pd.Series(ret).rolling(timeperiod).std(ddof=0).to_numpy()


def _bb_width(close: np.ndarray, timeperiod: int) -> np.ndarray:
    upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=2, nbdevdn=2)
    return (upper - lower) / middle


CUSTOM_FUNCS = {"VOL_RATIO": _vol_ratio, "RET_STD": _ret_std, "BB_WIDTH": _bb_width}


# ---------------------------------------------------------------- transforms
def _apply_transform(x: np.ndarray, kind: str, series: dict[str, np.ndarray]) -> np.ndarray:
    close = series["close"]
    if kind == "none":
        return x
    if kind == "rel_close":
        return x / close - 1.0
    if kind == "div_close":
        return x / close
    if kind == "div_close_sq":
        return x / close**2
    if kind == "div_vol20":
        return x / series["vol_sma20"]
    if kind == "diff_div_vol20":
        d = np.full_like(x, np.nan)
        d[1:] = np.diff(x)
        return d / series["vol_sma20"]
    raise ValueError(f"unknown transform {kind!r}")


# ---------------------------------------------------------------- main
def compute_indicators(prices: pd.DataFrame, index_close: pd.Series, specs: list[dict]) -> pd.DataFrame:
    """prices: OHLCV indexed by date. index_close: benchmark close (aligned by date).
    Returns one float column per spec, same index as `prices`, NaN during warm-up."""
    series = {
        "open": prices["Open"].to_numpy(np.float64),
        "high": prices["High"].to_numpy(np.float64),
        "low": prices["Low"].to_numpy(np.float64),
        "close": prices["Close"].to_numpy(np.float64),
        "volume": prices["Volume"].to_numpy(np.float64),
        "index_close": index_close.reindex(prices.index).ffill().to_numpy(np.float64),
    }
    series["logclose100"] = 100.0 * np.log(series["close"])
    series["vol_sma20"] = talib.SMA(series["volume"], timeperiod=20)

    cache: dict[tuple, object] = {}  # multi-output functions are computed once
    out = {}
    for spec in specs:
        func, params = spec["func"], spec.get("params", {}) or {}
        key = (func, tuple(spec["inputs"]), tuple(sorted(params.items())))
        if key not in cache:
            args = [series[i] for i in spec["inputs"]]
            fn = CUSTOM_FUNCS.get(func) or getattr(talib, func)
            cache[key] = fn(*args, **params)
        res = cache[key]
        if isinstance(res, (tuple, list)):
            res = res[spec["output"]]
        x = np.asarray(res, dtype=np.float64)
        out[spec["name"]] = _apply_transform(x, spec.get("transform", "none"), series)

    df = pd.DataFrame(out, index=prices.index)
    return df.replace([np.inf, -np.inf], np.nan)


def drop_warmup(ind: pd.DataFrame) -> pd.DataFrame:
    """Drop leading rows until every indicator is available; later NaNs are forward-filled
    (e.g. a zero-range day), and any row still NaN is dropped."""
    valid = ind.notna().all(axis=1)
    if not valid.any():
        return ind.iloc[0:0]
    ind = ind.loc[valid.idxmax():]
    return ind.ffill().dropna()


def indicators_dir(cfg: dict) -> Path:
    return data_dir(cfg) / "processed" / "indicators"
