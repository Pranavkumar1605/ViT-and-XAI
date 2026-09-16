import numpy as np
import pandas as pd

from vitxai.features.indicators import CATEGORY_ORDER, compute_indicators, drop_warmup, indicator_specs


def test_ninety_rows_grouped_by_category(cfg):
    specs = indicator_specs(cfg)
    assert len(specs) == 90 == cfg["model"]["img_size"][0]
    cats = [s["category"] for s in specs]
    # categories appear as contiguous blocks in the fixed order
    blocks = [c for i, c in enumerate(cats) if i == 0 or c != cats[i - 1]]
    assert blocks == CATEGORY_ORDER
    counts = {c: cats.count(c) for c in CATEGORY_ORDER}
    assert counts == {"overlap": 18, "momentum": 38, "volume": 6, "volatility": 8,
                      "price_transform": 4, "statistics": 16}


def test_no_nan_after_warmup(cfg, prices):
    p, idx = prices
    ind = drop_warmup(compute_indicators(p, idx, indicator_specs(cfg)))
    assert ind.shape[1] == 90
    assert len(ind) > 200
    assert np.isfinite(ind.to_numpy()).all()


def test_indicators_are_causal(cfg, prices):
    """Values up to day t must not change when prices after t are removed or altered."""
    p, idx = prices
    specs = indicator_specs(cfg)
    full = compute_indicators(p, idx, specs)
    for t in (250, 330):
        cut = compute_indicators(p.iloc[: t + 1], idx, specs)
        np.testing.assert_allclose(full.iloc[: t + 1].to_numpy(), cut.to_numpy(), rtol=1e-9, atol=1e-12,
                                   equal_nan=True)
        altered = p.copy()
        altered.iloc[t + 1:] *= 3.0
        alt = compute_indicators(altered, idx, specs)
        np.testing.assert_allclose(full.iloc[: t + 1].to_numpy(), alt.iloc[: t + 1].to_numpy(),
                                   rtol=1e-9, atol=1e-12, equal_nan=True)


def test_beta_is_stock_on_index(cfg):
    """BETA_20 must be cov(stock, index) / var(index): a stock moving 2x the index has beta ~2."""
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.bdate_range("2015-01-01", periods=n)
    m = rng.normal(0, 0.01, n)
    s = 2.0 * m + rng.normal(0, 0.002, n)
    close = 100 * np.cumprod(1 + s)
    prices = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                           "Volume": 1e6}, index=dates)
    index_close = pd.Series(1000 * np.cumprod(1 + m), index=dates)
    beta_spec = [s for s in indicator_specs(cfg) if s["name"] == "BETA_20"]
    beta = compute_indicators(prices, index_close, beta_spec)["BETA_20"].dropna()
    assert abs(beta.median() - 2.0) < 0.1


def test_relative_form_is_scale_free(cfg, prices):
    """Scaling all prices by a constant leaves price-transformed indicators unchanged."""
    p, idx = prices
    specs = [s for s in indicator_specs(cfg) if s["transform"] in ("rel_close", "div_close", "div_close_sq")]
    a = compute_indicators(p, idx, specs)
    scaled = p.copy()
    scaled[["Open", "High", "Low", "Close"]] *= 7.0
    b = compute_indicators(scaled, idx, specs)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)
