import numpy as np
import pandas as pd
import pytest

from vitxai.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def synthetic_prices(n_days: int = 420, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    close = 500 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    volume = rng.integers(100_000, 1_000_000, n_days).astype(float)
    prices = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    index_close = pd.Series(10_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))), index=dates)
    return prices, index_close


@pytest.fixture
def prices():
    return synthetic_prices()
