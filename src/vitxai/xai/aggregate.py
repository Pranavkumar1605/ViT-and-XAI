"""Aggregate per-sample attribution maps into class-level indicator rankings and recency profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def normalize_maps(maps: np.ndarray, signed: bool = False) -> np.ndarray:
    """Scale each sample so that sum(|map|) = 1, making samples comparable before averaging.
    signed=False returns |map| (magnitude); signed=True keeps the sign (+ supports the target class)."""
    m = maps.astype(np.float32)
    s = np.abs(m).sum(axis=(1, 2), keepdims=True)
    out = m if signed else np.abs(m)
    return np.divide(out, s, out=np.zeros_like(out), where=s > 0)


def class_mean_maps(maps_norm: np.ndarray, classes: np.ndarray, class_names: list[str],
                    mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    out = {}
    for c, name in enumerate(class_names):
        sel = classes == c
        if mask is not None:
            sel &= mask
        if sel.any():
            out[name] = maps_norm[sel].mean(axis=0)
    return out


def indicator_importance(mean_map: np.ndarray) -> np.ndarray:
    """Sum across columns (days) -> one score per indicator row."""
    return mean_map.sum(axis=1)


def recency_profile(mean_map: np.ndarray) -> np.ndarray:
    """Sum across rows (indicators) -> one score per day column (oldest ... newest)."""
    return mean_map.sum(axis=0)


def rankings_table(mean_maps: dict[str, np.ndarray], names: list[str], categories: list[str],
                   signed_mean_maps: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """importance_<cls>: share of |attribution| (ranking basis).
    signed_<cls>: net signed share; negative = the indicator pushed against the class on average.
    rank_<cls>: 1 = most important; tied scores share the best rank (patch-level methods such as
    Chefer give every row of a 10-row patch the same score)."""
    df = pd.DataFrame({"row": np.arange(len(names)), "indicator": names, "category": categories})
    for cls, m in mean_maps.items():
        imp = indicator_importance(m)
        df[f"importance_{cls}"] = imp
        if signed_mean_maps is not None and cls in signed_mean_maps:
            df[f"signed_{cls}"] = indicator_importance(signed_mean_maps[cls])
        df[f"rank_{cls}"] = rankdata(-np.round(imp, 12), method="min").astype(int)
    return df


def category_importance(rankings: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in rankings.columns if c.startswith(("importance_", "signed_"))]
    return rankings.groupby("category", sort=False)[cols].sum()


def method_agreement(rankings_by_method: dict[str, pd.DataFrame], class_names: list[str]) -> pd.DataFrame:
    """Spearman correlation of indicator importance between each pair of methods, per class."""
    rows = []
    methods = list(rankings_by_method)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            for cls in class_names:
                col = f"importance_{cls}"
                if col in rankings_by_method[a] and col in rankings_by_method[b]:
                    rho, p = spearmanr(rankings_by_method[a][col], rankings_by_method[b][col])
                    rows.append({"method_a": a, "method_b": b, "class": cls, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)
