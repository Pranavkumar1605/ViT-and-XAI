import numpy as np
import pandas as pd

from vitxai.data.clean import align_to_calendar, clean_rows, clean_ticker, trading_calendar


def _bars(dates, close, volume, flat=False):
    close = np.asarray(close, dtype=float)
    spread = 0.0 if flat else 0.01
    return pd.DataFrame({"Open": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Close": close,
                         "Volume": np.asarray(volume, dtype=float)}, index=pd.DatetimeIndex(dates))


def test_calendar_combines_stocks_and_index():
    days = pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
                           "2020-01-06", "2020-01-07"])
    # Tue; Wed = real session missing from the index; Thu = holiday (zero-volume stock rows, no index row);
    # Fri; Sat = special session; Mon; Tue = real session where every stock row is a zero-volume placeholder
    stock_vol = [100, 100, 0, 100, 50, 100, 0]
    stocks = [_bars(days, [10, 11, 11, 12, 12.1, 13, 13], stock_vol) for _ in range(3)]
    index = _bars(pd.to_datetime(["2019-12-31", "2020-01-03", "2020-01-04", "2020-01-06", "2020-01-07"]),
                  [100, 101, 101.5, 102, 104], [0, 0, 0, 0, 0])
    cal = trading_calendar(stocks, index, "2019-01-01", "2020-12-31", min_share=0.5)
    assert list(cal.strftime("%Y-%m-%d")) == ["2019-12-31", "2020-01-01", "2020-01-03", "2020-01-06", "2020-01-07"]


def test_align_flags_placeholders_and_filled_days():
    cal = pd.bdate_range("2021-01-04", periods=10)
    #        0    1    2 (vol glitch)  3    4 (placeholder)  5 missing  6    7-8 missing  9
    keep = [0, 1, 2, 3, 4, 6, 9]
    df = pd.concat([
        _bars(cal[[0, 1, 2, 3]], [100, 101, 105, 104], [10, 10, 0, 12]),
        _bars(cal[[4]], [104], [0], flat=True),
        _bars(cal[[6, 9]], [108, 110], [13, 14]),
    ])
    assert list(df.index) == list(cal[keep])
    out, report = align_to_calendar(clean_rows(df)[0], cal)
    assert report == {"filled_days": 1, "placeholder_days": 1, "zero_volume_days": 1, "dropped_gap_days": 2}
    assert out.loc[cal[2], "Close"] == 105 and out.loc[cal[2], "Volume"] == 10 and not out.loc[cal[2], "synthetic"]
    assert out.loc[cal[4], "synthetic"] and out.loc[cal[4], "Volume"] == 12
    assert out.loc[cal[5], "Close"] == 104 and out.loc[cal[5], "synthetic"]
    assert cal[7] not in out.index and cal[8] not in out.index and cal[9] in out.index
    assert out["synthetic"].sum() == 2


def test_clean_ticker_output_columns():
    cal = pd.bdate_range("2021-01-04", periods=5)
    out, _ = clean_ticker(_bars(cal, [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]), cal)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume", "synthetic"]
    assert out["synthetic"].dtype == bool and not out["synthetic"].any()


def test_clean_rows_drops_bad_prices_and_duplicates():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-05", "2021-01-06"])
    df = _bars(dates, [10, 11, 11.5, -1], [1, 1, 1, 1])
    out, removed = clean_rows(df)
    assert list(out["Close"]) == [10, 11.5] and removed == 2
