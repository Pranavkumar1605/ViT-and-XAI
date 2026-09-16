import numpy as np
import pandas as pd

from vitxai.labels.folds import SPLITS, make_folds, split_mask
from vitxai.labels.labels import BUY, HOLD, SELL, label_returns, make_labels


def test_label_formula():
    close = pd.Series([100.0, 102.0, 100.0, 100.5, 99.0], index=pd.bdate_range("2020-01-01", periods=5))
    lab = make_labels(close, theta=0.01)
    assert len(lab) == 4                                   # last day has no next close
    np.testing.assert_allclose(lab["ret_next"], [0.02, 100 / 102 - 1, 0.005, 99 / 100.5 - 1])
    assert lab["label"].tolist() == [BUY, SELL, HOLD, SELL]
    assert (lab["next_date"] > lab.index).all()


def test_threshold_is_strict():
    assert label_returns([0.01, -0.01, 0.0100001, -0.0100001], 0.01).tolist() == [HOLD, HOLD, BUY, SELL]


def test_fold_layout():
    folds = make_folds(2013, 2025, train_years=5, val_months=6)
    assert len(folds) == 13
    f0 = folds[0]
    assert f0["train"] == ["2008-01-01", "2012-07-01"]
    assert f0["val"] == ["2012-07-01", "2013-01-01"]
    assert f0["test"] == ["2013-01-01", "2014-01-01"]
    assert folds[1]["train"][0] == "2009-01-01"


def test_splits_disjoint_and_labels_do_not_cross_boundaries():
    dates = pd.Series(pd.bdate_range("2008-01-01", "2014-12-31"))
    next_dates = dates.shift(-1)
    valid = next_dates.notna()
    dates, next_dates = dates[valid], next_dates[valid]
    fold = make_folds(2013, 2013, 5, 6)[0]
    masks = {s: split_mask(dates, next_dates, fold[s]) for s in SPLITS}
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    # every train label date is strictly before validation start, every val label before test start
    assert next_dates[masks["train"]].max() < pd.Timestamp(fold["val"][0])
    assert next_dates[masks["val"]].max() < pd.Timestamp(fold["test"][0])


def test_peak_valley_labels():
    close = pd.Series([5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0], index=pd.bdate_range("2020-01-01", periods=9))
    lab = make_labels(close, theta=0.01, method="peak_valley", window=3)
    # centres 1..7; valley at 2 (3.0), peak at 5 (6.0), valley at 7 (4.0)
    assert lab["label"].tolist() == [HOLD, BUY, HOLD, HOLD, SELL, HOLD, BUY]
    assert (lab["label_end"] == pd.Series(close.index, index=close.index).shift(-1).loc[lab.index]).all()
    lab5 = make_labels(close, theta=0.01, method="peak_valley", window=5)
    assert lab5.index[0] == close.index[2] and lab5.index[-1] == close.index[6]
    assert (lab5["label_end"].to_numpy() == close.index[4:9].to_numpy()).all()
