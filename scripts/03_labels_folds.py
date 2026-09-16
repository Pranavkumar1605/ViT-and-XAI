"""Phase 3: labels for every (ticker, day) and the walk-forward fold manifest.

Outputs: data/processed/labels.parquet, data/processed/folds.yaml, data/processed/class_distribution.csv
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import yaml

from vitxai.config import add_common_args, data_dir, load_config
from vitxai.data import load_calendar, load_prices, load_universe, ticker_file
from vitxai.features.indicators import indicators_dir
from vitxai.images.dataset import labels_path
from vitxai.labels.folds import SPLITS, folds_from_cfg, split_mask
from vitxai.labels.labels import make_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    window = cfg["image"]["window"]
    names = cfg["labels"]["names"]
    lcfg = cfg["labels"]

    cal = load_calendar(cfg)
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    frames, dropped_gaps, dropped_synthetic = [], 0, 0
    for t in load_universe(cfg):
        prices = load_prices(cfg, t)
        lab = make_labels(prices["Close"], lcfg["theta"], lcfg.get("method", "threshold"), lcfg.get("window", 11))
        # the label must span consecutive sessions: drop labels spanning a dropped data gap
        pos = pd.Series(np.arange(len(prices)), index=prices.index)
        rows_spanned = pos.reindex(lab["label_end"]).to_numpy() - pos.reindex(lab.index).to_numpy()
        one_session = (cal_pos.reindex(lab["next_date"]).to_numpy() - cal_pos.reindex(lab.index).to_numpy() == 1)             & (cal_pos.reindex(lab["label_end"]).to_numpy() - cal_pos.reindex(lab.index).to_numpy() == rows_spanned)
        # ... and use only real price rows (placeholder / filled days have no real price)
        syn = np.concatenate([[0], np.cumsum(prices["synthetic"].to_numpy())])
        start_row = pos.reindex(lab.index).to_numpy()
        if lcfg.get("method", "threshold") == "peak_valley":
            start_row = start_row - lcfg.get("window", 11) // 2
        end_row = np.nan_to_num(pos.reindex(lab["label_end"]).to_numpy(), nan=0).astype(int)
        real = syn[end_row + 1] - syn[np.clip(start_row, 0, None).astype(int)] == 0
        dropped_gaps += int((~one_session).sum())
        dropped_synthetic += int((one_session & ~real).sum())
        lab = lab[one_session & real]
        ind_dates = pd.read_parquet(indicators_dir(cfg) / f"{ticker_file(t)}.parquet", columns=[]).index
        pos = pd.Series(np.arange(len(ind_dates)), index=ind_dates)
        lab = lab[lab.index.isin(ind_dates)]
        lab = lab.assign(ticker=t, row=pos.loc[lab.index].to_numpy())
        lab = lab[lab["row"] >= window - 1]          # a full window of history must exist
        frames.append(lab.reset_index())
    labels = pd.concat(frames, ignore_index=True)[["date", "ticker", "row", "ret_next", "next_date", "label_end", "label"]]
    labels_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_path(cfg), index=False)
    print(f"labels: {len(labels)} samples, {labels['ticker'].nunique()} tickers, "
          f"{labels['date'].min().date()} .. {labels['date'].max().date()} "
          f"(removed: {dropped_gaps} spanning data gaps, {dropped_synthetic} touching placeholder/filled days)")

    folds = folds_from_cfg(cfg)
    with open(data_dir(cfg) / "processed" / "folds.yaml", "w") as f:
        yaml.safe_dump(folds, f, sort_keys=False)

    rows = []
    for fold in folds:
        for split in SPLITS:
            m = split_mask(labels["date"], labels["label_end"], fold[split])
            counts = np.bincount(labels.loc[m, "label"], minlength=len(names))
            total = int(counts.sum())
            rows.append({"fold": fold["fold"], "test_year": fold["test_year"], "split": split,
                         "start": fold[split][0], "end": fold[split][1], "n": total,
                         "tickers": int(labels.loc[m, "ticker"].nunique()),
                         **{names[c]: int(counts[c]) for c in range(len(names))},
                         **{f"{names[c]}_pct": round(100 * counts[c] / total, 2) if total else np.nan
                            for c in range(len(names))}})
    dist = pd.DataFrame(rows)
    stem = labels_path(cfg).stem
    suffix = "" if stem == "labels" else stem.removeprefix("labels")
    dist.to_csv(data_dir(cfg) / "processed" / f"class_distribution{suffix}.csv", index=False)
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
