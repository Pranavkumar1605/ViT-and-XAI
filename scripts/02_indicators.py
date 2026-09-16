"""Phase 2: compute the indicator matrix for every kept ticker and drop warm-up rows.

Outputs: data/processed/indicators/<ticker>.parquet (date x indicators), data/processed/indicator_list.csv
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from vitxai.config import add_common_args, data_dir, load_config
from vitxai.data import load_prices, load_universe, ticker_file
from vitxai.features.indicators import compute_indicators, drop_warmup, indicator_specs, indicators_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)

    specs = indicator_specs(cfg)
    n_rows = cfg["model"]["img_size"][0]
    assert len(specs) == n_rows, f"{len(specs)} indicators but model img_size rows = {n_rows}"

    out_dir = indicators_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"row": i, "name": s["name"], "category": s["category"], "func": s["func"],
                   "params": s.get("params"), "transform": s.get("transform", "none")}
                  for i, s in enumerate(specs)]).to_csv(data_dir(cfg) / "processed" / "indicator_list.csv", index=False)

    index_close = load_prices(cfg, cfg["data"]["index_ticker"])["Close"]
    for t in load_universe(cfg):
        prices = load_prices(cfg, t)
        ind = drop_warmup(compute_indicators(prices, index_close, specs))
        assert ind.shape[1] == len(specs) and not ind.isna().any().any()
        ind.to_parquet(out_dir / f"{ticker_file(t)}.parquet")
        print(f"  {t:12s} {len(prices):5d} days -> {len(ind):5d} after warm-up "
              f"(first {ind.index.min().date()})")


if __name__ == "__main__":
    main()
