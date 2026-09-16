"""Phase 1: download OHLCV for the configured tickers + index, keep full-history tickers, build the trading
calendar, clean and align every ticker to it.

Outputs: data/raw/*.parquet (unchanged), data/interim/*.parquet, data/universe.csv,
         data/calendar.csv, data/data_quality_report.csv, data/big_moves.csv
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from vitxai.config import add_common_args, data_dir, load_config
from vitxai.data import interim_dir, ticker_file
from vitxai.data.clean import big_moves, clean_rows, clean_ticker, has_full_history, trading_calendar
from vitxai.data.download import download_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--skip-download", action="store_true", help="reuse data/raw")
    args = ap.parse_args()
    cfg = load_config(args.set, args.config_dir)
    dcfg = cfg["data"]

    tickers = list(dcfg["tickers"])
    if dcfg.get("max_tickers"):
        tickers = tickers[: dcfg["max_tickers"]]
    index_t = dcfg["index_ticker"]
    raw_dir = data_dir(cfg) / "raw"

    if not args.skip_download:
        print(f"Downloading {len(tickers)} tickers + {index_t} ({dcfg['start']} .. {dcfg['end']})")
        log = download_all([index_t] + tickers, dcfg["start"], dcfg["end"], raw_dir, dcfg.get("yahoo_suffix", ".NS"))
        log.to_csv(data_dir(cfg) / "download_log.csv", index=False)

    # 1) full-history filter
    raws, universe = {}, []
    for t in tickers:
        path = raw_dir / f"{ticker_file(t)}.parquet"
        raw = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        ok, reason = has_full_history(raw, dcfg["start"], dcfg["end"], dcfg["history_tolerance_days"])
        if ok:
            raws[t] = raw
        universe.append({"ticker": t, "kept": ok, "reason": reason,
                         "first_date": raw.index.min() if len(raw) else None,
                         "last_date": raw.index.max() if len(raw) else None})
        print(f"  {t:12s} {'KEEP' if ok else 'drop'}  {reason}")
    pd.DataFrame(universe).to_csv(data_dir(cfg) / "universe.csv", index=False)
    if not raws:
        raise SystemExit("no ticker has full history; check the download log")

    # 2) trading calendar from the kept stocks + index
    idx_path = raw_dir / f"{ticker_file(index_t)}.parquet"
    if not idx_path.exists():
        raise SystemExit(f"{index_t} was not downloaded; it is needed for the calendar and BETA / CORREL")
    idx, _ = clean_rows(pd.read_parquet(idx_path))
    calendar = trading_calendar([clean_rows(r)[0] for r in raws.values()], idx, dcfg["start"], dcfg["end"],
                                dcfg.get("calendar_min_share", 0.5))
    pd.DataFrame({"date": calendar}).to_csv(data_dir(cfg) / "calendar.csv", index=False)
    print(f"Trading calendar: {len(calendar)} sessions, {calendar.min().date()} .. {calendar.max().date()}")

    # 3) clean + align
    out_dir = interim_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out_dir / f"{ticker_file(index_t)}.parquet")   # used for its close only (BETA / CORREL)
    missing_idx = calendar.difference(idx.index)
    print(f"{index_t}: {len(missing_idx)} calendar sessions missing (forward-filled for BETA / CORREL)")

    quality, moves = [], []
    for t, raw in raws.items():
        clean, report = clean_ticker(raw, calendar)
        clean.to_parquet(out_dir / f"{ticker_file(t)}.parquet")
        quality.append({"ticker": t, "rows": len(clean), **report})
        moves.append(big_moves(clean, t))
    pd.DataFrame(quality).to_csv(data_dir(cfg) / "data_quality_report.csv", index=False)
    pd.concat(moves, ignore_index=True).to_csv(data_dir(cfg) / "big_moves.csv", index=False)
    print(f"Kept {len(raws)} / {len(tickers)} tickers; review data/big_moves.csv for unadjusted corporate actions")


if __name__ == "__main__":
    main()
