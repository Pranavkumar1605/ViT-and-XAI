"""Phase 8: trading backtest of the base paper's strategy on saved predictions (no retraining).

All test folds of a seed are joined into one continuous test period per ticker, as in the paper
(each test year once). Settings come from `backtest:` in the config (defaults: $10,000, $1 commission).

Outputs in results/<run_id>/backtest/:
  backtest_per_ticker_seed<S>.csv   model and buy-and-hold metrics per ticker
  backtest_summary_seed<S>.csv      mean over tickers
  equity_seed<S>.parquet            daily equity per ticker
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from vitxai.config import add_common_args, load_config, run_dir
from vitxai.data import load_prices
from vitxai.eval.backtest import backtest_ticker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    bt = {"start_money": 10000.0, "commission": 1.0, "risk_free_annual": 0.0, **(cfg.get("backtest") or {})}
    rdir = run_dir(cfg)
    out = rdir / "backtest"
    out.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(rdir.glob("fold_*/seed_*/predictions.parquet"))
    if not pred_files:
        raise SystemExit(f"no predictions found under {rdir}")
    by_seed = {}
    for pf in pred_files:
        by_seed.setdefault(int(pf.parent.name.split("_")[1]), []).append(pd.read_parquet(pf))

    for seed, frames in sorted(by_seed.items()):
        preds = pd.concat(frames, ignore_index=True)
        rows, curves = [], []
        for ticker, p in preds.groupby("ticker"):
            model, bah, curve = backtest_ticker(load_prices(cfg, ticker)["Close"], p, bt["start_money"],
                                                bt["commission"], bt["risk_free_annual"])
            rows.append({"ticker": ticker, "strategy": "model", **model})
            rows.append({"ticker": ticker, "strategy": "buy_and_hold", **bah})
            curves.append(curve.assign(ticker=ticker))
        per_ticker = pd.DataFrame(rows)
        per_ticker.to_csv(out / f"backtest_per_ticker_seed{seed}.csv", index=False)
        pd.concat(curves, ignore_index=True).to_parquet(out / f"equity_seed{seed}.parquet", index=False)
        num = per_ticker.select_dtypes("number").columns
        summary = per_ticker.groupby("strategy")[num].mean()
        summary.to_csv(out / f"backtest_summary_seed{seed}.csv")

        show = ["ticker", "strategy", "final_capital", "annualized_return_pct", "sharpe_daily_annualized",
                "max_drawdown_pct", "transactions", "success_pct", "idle_ratio_pct"]
        print(f"seed {seed}: {preds['date'].min()} .. {preds['date'].max()}")
        print(per_ticker[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\nMean over tickers:")
        print(summary[show[2:]].to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
