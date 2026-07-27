from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _normalise_yfinance_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        elif ticker in frame.columns.get_level_values(0):
            frame = frame.xs(ticker, axis=1, level=0)
        else:
            frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_column: "Date"})
    keep = [name for name in ["Date", "Open", "High", "Low", "Close", "Volume"] if name in frame]
    frame = frame[keep].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_convert(None)
    frame = frame.drop_duplicates("Date").sort_values("Date")
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{ticker} is missing columns: {sorted(missing)}")
    return frame


def download_market_data(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str | Path,
    pause_seconds: float = 0.5,
) -> None:
    import yfinance as yf

    raw = Path(raw_dir)
    market_dir = raw / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for ticker in tickers:
        output = market_dir / f"{ticker}.csv"
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=pd.Timestamp(end) + pd.Timedelta(days=1),
                auto_adjust=True,
                progress=False,
                actions=False,
                threads=False,
            )
            clean = _normalise_yfinance_frame(frame, ticker)
            clean.to_csv(output, index=False)
            print(
                f"saved {ticker}: {len(clean):,} rows "
                f"({clean['Date'].min().date()} to {clean['Date'].max().date()}) -> {output}"
            )
        except Exception as exc:  # keep collecting failures for a useful summary
            errors.append(f"{ticker}: {exc}")
        time.sleep(pause_seconds)
    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"Market downloads failed:\n{joined}")


def _validate_fred_coverage(
    frame: pd.DataFrame,
    alias: str,
    requested_start: str,
    requested_end: str,
    tolerance_days: int = 45,
) -> None:
    valid = frame.dropna(subset=[alias])
    if valid.empty:
        raise ValueError("series contains no numeric observations")
    first = pd.Timestamp(valid["Date"].min())
    last = pd.Timestamp(valid["Date"].max())
    start_limit = pd.Timestamp(requested_start) + pd.Timedelta(days=tolerance_days)
    end_limit = pd.Timestamp(requested_end) - pd.Timedelta(days=tolerance_days)
    if first > start_limit:
        raise ValueError(
            f"coverage starts at {first.date()}, later than required for the "
            f"{requested_start} experiment. Select a longer-history series."
        )
    if last < end_limit:
        raise ValueError(
            f"coverage ends at {last.date()}, earlier than required for the "
            f"{requested_end} experiment."
        )


def download_fred_data(
    series: dict[str, str],
    raw_dir: str | Path,
    start: str,
    end: str,
    timeout: int = 60,
) -> None:
    raw = Path(raw_dir)
    fred_dir = raw / "fred"
    fred_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "RiskGraphResearch/1.0.1"})
    errors: list[str] = []
    for series_id, alias in series.items():
        params = {"id": series_id, "cosd": start, "coed": end}
        try:
            response = session.get(FRED_CSV, params=params, timeout=timeout)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            if frame.shape[1] < 2:
                raise ValueError("unexpected FRED CSV format")
            frame = frame.iloc[:, :2].copy()
            frame.columns = ["Date", alias]
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
            frame = frame.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
            _validate_fred_coverage(frame, alias, start, end)
            output = fred_dir / f"{series_id}_{alias}.csv"
            frame.to_csv(output, index=False)
            valid = frame.dropna(subset=[alias])
            print(
                f"saved FRED {series_id}: {len(frame):,} rows, "
                f"{len(valid):,} numeric "
                f"({valid['Date'].min().date()} to {valid['Date'].max().date()}) -> {output}"
            )
        except Exception as exc:
            errors.append(f"{series_id}: {exc}")
    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"FRED downloads failed:\n{joined}")
