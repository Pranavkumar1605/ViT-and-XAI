"""Phase 9: no-model baseline on exactly the ViT's test samples.

Rule: Buy if today's close is the lowest of the last k closes, Sell if it is the highest, Hold
otherwise (k = labels.window // 2 + 1 = 6 for 11-day peak/valley labels, i.e. the half of the
label window that is already known on day t). Uses closes up to day t only.

Predictions are written in the same format as 05_train.py to results/<run_id>/fold_XX/seed_0/,
so 06_evaluate.py and 08_backtest.py work on it unchanged. Use a separate run_id, e.g.
  python scripts/09_rule_baseline.py --set labels.method=peak_valley labels.file=labels_pv11.parquet run_id=rule_pv11
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from vitxai.config import add_common_args, load_config, run_dir, save_config
from vitxai.data import load_prices, load_universe
from vitxai.images.dataset import labels_path
from vitxai.labels.folds import folds_from_cfg, split_mask
from vitxai.labels.labels import BUY, HOLD, SELL
from vitxai.train.train import fold_dir


def rule_signals(close: pd.Series, k: int) -> pd.Series:
    lo = close.eq(close.rolling(k).min())
    hi = close.eq(close.rolling(k).max())
    return pd.Series(np.where(lo, BUY, np.where(hi, SELL, HOLD)), index=close.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--k", type=int, default=None, help="look-back length (default labels.window // 2 + 1)")
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    k = args.k or cfg["labels"].get("window", 11) // 2 + 1
    window = cfg["image"]["window"]

    tickers = load_universe(cfg)
    signals = {t: rule_signals(load_prices(cfg, t)["Close"], k) for t in tickers}
    labels = pd.read_parquet(labels_path(cfg))
    labels = labels[labels["ticker"].isin(tickers) & (labels["row"] >= window - 1)]   # same samples as the ViT
    end_col = "label_end" if "label_end" in labels else "next_date"

    save_config(cfg, run_dir(cfg) / "config.yaml")
    for fold in folds_from_cfg(cfg):
        test = labels[split_mask(labels["date"], labels[end_col], fold["test"])]
        if test.empty:
            continue
        y_pred = np.array([signals[t].at[d] for t, d in zip(test["ticker"], test["date"])], dtype=np.int64)
        onehot = np.eye(3)[y_pred]
        preds = pd.DataFrame({"date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy(),
                              "y_true": test["label"].to_numpy(), "y_pred": y_pred,
                              "p_hold": onehot[:, 0], "p_buy": onehot[:, 1], "p_sell": onehot[:, 2]})
        out = fold_dir(run_dir(cfg), fold["fold"], 0)
        out.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / "predictions.parquet", index=False)
        print(f"fold {fold['fold']} ({fold['test_year']}): {len(preds)} samples, "
              f"acc={np.mean(preds['y_true'] == y_pred):.3f}")
    print(f"k={k}; saved to {run_dir(cfg)}")


if __name__ == "__main__":
    main()
