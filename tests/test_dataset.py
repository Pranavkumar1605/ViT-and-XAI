import numpy as np
import pandas as pd
import torch

from vitxai.images.dataset import build_fold_datasets
from vitxai.images.normalize import apply_norm, fit_norm_stats
from vitxai.labels.folds import make_folds


def _toy_store(n_ind=90, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-06-01", "2014-03-31")
    store, frames = {}, []
    for t in ("AAA", "BBB"):
        vals = rng.normal(size=(len(dates), n_ind)).cumsum(axis=0)
        store[t] = (dates, vals)
        frames.append(pd.DataFrame({
            "date": dates[:-1], "ticker": t, "row": np.arange(len(dates) - 1),
            "ret_next": 0.0, "next_date": dates[1:], "label": rng.integers(0, 3, len(dates) - 1),
        }))
    return store, pd.concat(frames, ignore_index=True)


def _cfg(cfg):
    c = dict(cfg)
    c["folds"] = {"train_years": 2, "val_months": 6, "first_test_year": 2013, "last_test_year": 2013}
    return c


def test_image_shape_and_orientation(cfg):
    store, labels = _toy_store()
    fold = make_folds(2013, 2013, 2, 6)[0]
    datasets, stats = build_fold_datasets(_cfg(cfg), fold, store, labels)
    window = cfg["image"]["window"]
    for split, ds in datasets.items():
        assert len(ds) > 0, split
        x, y = ds[len(ds) // 2]
        assert tuple(x.shape) == (1, 90, window)
        assert x.dtype == torch.float32
        assert np.isfinite(x.numpy()).all()

    ds = datasets["test"]
    i = 10
    s = ds.samples.iloc[i]
    dates, vals = store[s["ticker"]]
    norm = apply_norm(vals, stats, cfg["image"]["clip"])
    day_pos = dates.get_loc(s["date"])
    img = ds.image(i)[0]
    np.testing.assert_array_equal(img[:, -1], norm[day_pos])                   # rightmost column = day t
    np.testing.assert_array_equal(img[:, 0], norm[day_pos - window + 1])        # leftmost = oldest day


def test_norm_stats_use_train_dates_only(cfg):
    store, labels = _toy_store()
    fold = make_folds(2013, 2013, 2, 6)[0]
    _, stats_a = build_fold_datasets(_cfg(cfg), fold, store, labels)

    # Change every value from validation start onward: training stats must not move.
    val_start = pd.Timestamp(fold["val"][0])
    changed = {t: (d, np.where((d >= val_start)[:, None], v * 100 + 1e6, v)) for t, (d, v) in store.items()}
    _, stats_b = build_fold_datasets(_cfg(cfg), fold, changed, labels)
    np.testing.assert_array_equal(stats_a["mean"], stats_b["mean"])
    np.testing.assert_array_equal(stats_a["std"], stats_b["std"])


def test_fit_norm_stats_values():
    dates = pd.bdate_range("2020-01-01", periods=10)
    vals = np.arange(20, dtype=float).reshape(10, 2)
    stats = fit_norm_stats({"X": (dates, vals)}, "2020-01-01", str(dates[5].date()))
    np.testing.assert_allclose(stats["mean"], vals[:5].mean(axis=0))
