import numpy as np
import pandas as pd

from vitxai.eval.backtest import backtest_ticker, run_strategy
from vitxai.labels.labels import BUY, HOLD, SELL


def test_strategy_follows_first_signal_and_charges_commission():
    close = np.array([10.0, 10.0, 20.0, 20.0, 5.0, 10.0])
    signal = np.array([BUY, BUY, SELL, SELL, BUY, HOLD])   # repeated Buy / Sell ignored; last day closes
    res = run_strategy(close, signal, start_money=101.0, commission=1.0)
    t1, t2 = res["trades"]
    assert (t1["entry"], t1["exit"], t2["entry"], t2["exit"]) == (0, 2, 4, 5)
    # 100 / 10 = 10 shares -> 10 * 20 - 1 = 199; 198 / 5 = 39.6 shares -> 39.6 * 10 - 1 = 395
    np.testing.assert_allclose(res["equity"], [100.0, 100.0, 199.0, 199.0, 198.0, 395.0])
    np.testing.assert_allclose([t1["profit_pct"], t2["profit_pct"]], [100 * (199 / 101 - 1), 100 * (395 / 199 - 1)])


def test_no_buy_means_no_trades():
    close = np.linspace(10, 20, 30)
    res = run_strategy(close, np.full(30, SELL), 1000.0, 1.0)
    assert res["trades"] == [] and np.all(res["equity"] == 1000.0)


def test_backtest_ticker_summary_and_buy_and_hold():
    dates = pd.bdate_range("2020-01-01", periods=253)
    close = pd.Series(np.linspace(100, 200, 253), index=dates)
    preds = pd.DataFrame({"date": dates, "y_pred": HOLD})
    model, bah, curve = backtest_ticker(close, preds, 10000.0, 0.0)
    assert model["transactions"] == 0 and model["final_capital"] == 10000.0 and model["idle_ratio_pct"] == 100.0
    assert bah["transactions"] == 1 and abs(bah["final_capital"] - 20000.0) < 1e-6
    assert list(curve.columns) == ["date", "model", "buy_and_hold"]
