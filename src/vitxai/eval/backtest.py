"""Long-only trading backtest of the base paper (Gezici & Sefer 2024, Eq. 19-22).

Per ticker, starting with `start_money` in cash:
  * Buy  and no position -> spend all cash on shares at the day's close (minus commission)
  * Sell and a position  -> sell all shares at the day's close (minus commission)
  * Hold, or a repeated Buy / Sell -> nothing
An open position is closed at the last close. The prediction for day t uses data up to the close
of day t, so trading at that close is the earliest possible execution (no look-ahead into t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vitxai.labels.labels import BUY, SELL

TRADING_DAYS = 252


def run_strategy(close: np.ndarray, signal: np.ndarray, start_money: float, commission: float) -> dict:
    """close, signal: aligned daily arrays. Returns equity curve and closed trades."""
    cash, shares = start_money, 0.0
    entry_value, entry_i = 0.0, -1
    equity = np.empty(len(close))
    trades = []
    for i, (price, s) in enumerate(zip(close, signal)):
        last = i == len(close) - 1
        if shares == 0.0 and s == BUY and not last and cash > commission:
            entry_value, entry_i = cash, i
            shares = (cash - commission) / price
            cash = 0.0
        elif shares > 0.0 and (s == SELL or last):
            cash = shares * price - commission
            trades.append({"entry": entry_i, "exit": i, "length": i - entry_i,
                           "profit_pct": 100.0 * (cash / entry_value - 1.0)})
            shares = 0.0
        equity[i] = cash + shares * price
    return {"equity": equity, "trades": trades}


def summarize(equity: np.ndarray, trades: list[dict], dates: pd.DatetimeIndex, start_money: float,
              risk_free_annual: float = 0.0) -> dict:
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    daily = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0.0])
    excess = daily - risk_free_annual / TRADING_DAYS
    sd = excess.std(ddof=1) if len(excess) > 1 else 0.0
    profits = np.array([t["profit_pct"] for t in trades])
    in_market = sum(t["length"] for t in trades)
    n = len(trades)
    return {
        "start_date": dates[0].date(), "end_date": dates[-1].date(), "years": years,
        "final_capital": float(equity[-1]),
        "annualized_return_pct": 100.0 * ((equity[-1] / start_money) ** (1.0 / years) - 1.0),
        "sharpe_daily_annualized": float(np.sqrt(TRADING_DAYS) * excess.mean() / sd) if sd > 0 else np.nan,
        "max_drawdown_pct": float(100.0 * (equity / np.maximum.accumulate(equity) - 1.0).min()),
        "transactions": n,
        "transactions_per_year": n / years,
        "success_pct": 100.0 * float((profits > 0).mean()) if n else np.nan,
        "avg_profit_per_transaction_pct": float(profits.mean()) if n else np.nan,
        "avg_transaction_length_days": in_market / n if n else np.nan,
        "max_profit_per_transaction_pct": float(profits.max()) if n else np.nan,
        "max_loss_per_transaction_pct": float(profits.min()) if n else np.nan,
        "idle_ratio_pct": 100.0 * (1.0 - in_market / len(equity)),
        "max_capital": float(equity.max()), "min_capital": float(equity.min()),
    }


def backtest_ticker(close: pd.Series, preds: pd.DataFrame, start_money: float, commission: float,
                    risk_free_annual: float = 0.0) -> tuple[dict, dict, pd.DataFrame]:
    """preds: date, y_pred for one ticker. Returns (strategy summary, buy-and-hold summary, daily equity)."""
    p = preds.assign(date=pd.to_datetime(preds["date"])).sort_values("date").drop_duplicates("date")
    dates = pd.DatetimeIndex(p["date"])
    px = close.reindex(dates).to_numpy(np.float64)
    if np.isnan(px).any():
        raise ValueError("missing close price on a prediction date")
    signal = p["y_pred"].to_numpy()
    model = run_strategy(px, signal, start_money, commission)
    bah_signal = np.zeros(len(px), dtype=int)
    bah_signal[0] = BUY
    bah = run_strategy(px, bah_signal, start_money, commission)
    curves = pd.DataFrame({"date": dates, "model": model["equity"], "buy_and_hold": bah["equity"]})
    return (summarize(model["equity"], model["trades"], dates, start_money, risk_free_annual),
            summarize(bah["equity"], bah["trades"], dates, start_money, risk_free_annual),
            curves)
