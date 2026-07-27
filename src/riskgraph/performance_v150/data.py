from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from riskgraph.config import Fold
from riskgraph.data.features import asset_features, build_panel

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass(frozen=True)
class LongHistoryPanel:
    dates: pd.DatetimeIndex
    values: np.ndarray
    masks: np.ndarray
    feature_names: list[str]
    target_returns: np.ndarray


def _fixed_width_unicode(values: Any) -> np.ndarray:
    """Return a pickle-free fixed-width Unicode array.

    Pandas ``Index.astype(str).to_numpy()`` commonly produces ``dtype=object``.
    NumPy then stores that array through pickle, which is intentionally rejected by
    the research loader.  Explicit fixed-width Unicode keeps the NPZ portable and
    safe to load with ``allow_pickle=False``.
    """

    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{max(1, width)}")


def load_long_history_panel(path: str | Path) -> LongHistoryPanel:
    try:
        with np.load(path, allow_pickle=False) as data:
            dates = data["dates"]
            feature_names = data["feature_names"]
            if dates.dtype.kind not in {"U", "S", "M"}:
                raise ValueError(f"Unsafe dates dtype in long-history panel: {dates.dtype}")
            if feature_names.dtype.kind not in {"U", "S"}:
                raise ValueError(
                    "Unsafe feature_names dtype in long-history panel: "
                    f"{feature_names.dtype}"
                )
            return LongHistoryPanel(
                dates=pd.DatetimeIndex(pd.to_datetime(dates.astype(str))),
                values=data["values"].astype(np.float32),
                masks=data["masks"].astype(np.float32),
                feature_names=feature_names.astype(str).tolist(),
                target_returns=data["target_returns"].astype(np.float32),
            )
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError(
                "The long-history panel was written with object-string arrays by "
                "v1.5.2. Rebuild it with v1.5.3 so dates and feature names are "
                "stored as fixed-width Unicode."
            ) from exc
        raise


def _market_frame(raw_dir: Path, ticker: str) -> pd.DataFrame:
    path = raw_dir / "market" / f"{ticker}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing market file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.drop_duplicates("Date").set_index("Date").sort_index()
    return frame


def _fred_frame(raw_dir: Path, series_id: str, alias: str) -> pd.Series:
    path = raw_dir / "fred" / f"{series_id}_{alias}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FRED file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"]).drop_duplicates("Date")
    return frame.set_index("Date")[alias].sort_index().astype(float)


def _shift_and_coalesce_index(
    values: pd.Series | pd.DataFrame,
    days: int,
) -> pd.Series | pd.DataFrame:
    """Shift observations and coalesce collisions using the latest source row.

    A business-day offset can map multiple source dates to one availability date.
    This occurs for calendar-daily FRED data and also for historical Kenneth French
    observations from the era when U.S. exchanges traded on Saturdays. Stable
    sorting followed by ``keep="last"`` retains the chronologically latest source
    observation without moving any observation backwards.
    """

    shifted = values.copy()
    shifted.index = pd.DatetimeIndex(pd.to_datetime(shifted.index)).tz_localize(None)
    shifted = shifted.sort_index(kind="stable")
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    shifted.index = shifted.index + pd.offsets.BDay(max(0, int(days)))
    shifted = shifted.sort_index(kind="stable")
    return shifted[~shifted.index.duplicated(keep="last")]


def _business_shift(
    series: pd.Series,
    days: int,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Apply a conservative release lag to a scalar source."""

    numeric = pd.to_numeric(series, errors="coerce")
    shifted = _shift_and_coalesce_index(numeric, days)
    target = pd.DatetimeIndex(pd.to_datetime(target_index)).tz_localize(None)
    return shifted.reindex(target).ffill(limit=15)


def _business_shift_frame(
    frame: pd.DataFrame,
    days: int,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Apply a conservative release lag to a multi-column daily source.

    Daily return factors are not forward-filled because a missing trading-day
    return is not a level observation. The caller retains explicit availability
    masks for any missing target-calendar dates.
    """

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    shifted = _shift_and_coalesce_index(numeric, days)
    target = pd.DatetimeIndex(pd.to_datetime(target_index)).tz_localize(None)
    return shifted.reindex(target)


def _safe_log_change(series: pd.Series, periods: int) -> pd.Series:
    positive = series.where(series > 0.0)
    return np.log(positive).diff(periods)


def _append_feature(
    values: list[np.ndarray],
    masks: list[np.ndarray],
    names: list[str],
    series: pd.Series,
    name: str,
) -> None:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    available = numeric.notna().to_numpy(dtype=np.float32)
    values.append(numeric.fillna(0.0).to_numpy(dtype=np.float32))
    masks.append(available)
    names.append(name)


def parse_french_csv_zip(payload: bytes) -> pd.DataFrame:
    """Parse a Kenneth French daily CSV ZIP into decimal returns.

    Files contain prose before and after the numeric table.  The parser locates the
    first YYYYMMDD row and stops when the date field ceases to be eight digits.
    """

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError("Kenneth French archive contains no CSV file")
        text = archive.read(members[0]).decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    start = next(
        (index for index, line in enumerate(lines) if re.match(r"^\d{8},", line)),
        None,
    )
    if start is None:
        raise ValueError("Could not locate daily factor observations")
    header_index = start - 1
    header = [part.strip() or "Date" for part in lines[header_index].split(",")]
    rows: list[str] = []
    for line in lines[start:]:
        if not re.match(r"^\d{8},", line):
            break
        rows.append(line)
    frame = pd.read_csv(io.StringIO("\n".join([",".join(header), *rows])))
    date_name = frame.columns[0]
    frame = frame.rename(columns={date_name: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"].astype(str), format="%Y%m%d")
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / 100.0
    return frame.drop_duplicates("Date").set_index("Date").sort_index()


def download_french_factors(
    specifications: dict[str, dict[str, str]],
    raw_dir: str | Path,
    timeout: int = 90,
) -> None:
    output = Path(raw_dir) / "french"
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "RiskGraphResearch/1.5.3"})
    errors: list[str] = []
    for alias, record in specifications.items():
        try:
            response = session.get(str(record["url"]), timeout=timeout)
            response.raise_for_status()
            frame = parse_french_csv_zip(response.content)
            target = output / f"{alias}.csv"
            frame.reset_index().to_csv(target, index=False)
            print(
                f"saved French {alias}: {len(frame):,} rows "
                f"({frame.index.min().date()} to {frame.index.max().date()}) -> {target}"
            )
        except Exception as exc:
            errors.append(f"{alias}: {exc}")
    if errors:
        raise RuntimeError("French factor downloads failed:\n" + "\n".join(errors))



def download_fred_series_v150(
    series: dict[str, str],
    raw_dir: str | Path,
    start: str,
    end: str,
    timeout: int = 90,
) -> None:
    """Download FRED series while allowing legitimate partial histories.

    The v1.5 long-history panel carries availability masks, so a late-inception
    source (for example the broad dollar index) must not truncate or abort the
    complete experiment. Every series is still required to have a current tail
    and a meaningful number of numeric observations.
    """

    output = Path(raw_dir) / "fred"
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "RiskGraphResearch/1.5.3"})
    errors: list[str] = []
    for series_id, alias in series.items():
        try:
            response = session.get(
                FRED_CSV,
                params={"id": series_id, "cosd": start, "coed": end},
                timeout=timeout,
            )
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            if frame.shape[1] < 2:
                raise ValueError("unexpected FRED CSV format")
            frame = frame.iloc[:, :2].copy()
            frame.columns = ["Date", alias]
            frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
            frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
            frame = (
                frame.dropna(subset=["Date"])
                .drop_duplicates("Date")
                .sort_values("Date")
            )
            valid = frame.dropna(subset=[alias])
            if len(valid) < 100:
                raise ValueError(f"only {len(valid)} numeric observations")
            last = pd.Timestamp(valid["Date"].max())
            if last < pd.Timestamp(end) - pd.Timedelta(days=60):
                raise ValueError(
                    f"coverage ends at {last.date()}, too early for requested end {end}"
                )
            target = output / f"{series_id}_{alias}.csv"
            frame.to_csv(target, index=False)
            print(
                f"saved FRED {series_id}: {len(valid):,} numeric rows "
                f"({valid['Date'].min().date()} to {valid['Date'].max().date()}) "
                f"-> {target}"
            )
        except Exception as exc:
            errors.append(f"{series_id}: {exc}")
    if errors:
        raise RuntimeError("FRED downloads failed:\n" + "\n".join(errors))


def _load_french(raw_dir: Path, alias: str) -> pd.DataFrame:
    path = raw_dir / "french" / f"{alias}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing French factor file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    return frame.drop_duplicates("Date", keep="last").set_index("Date").sort_index()


def build_long_history_panel(
    config: dict[str, Any],
    raw_dir: str | Path,
    output_path: str | Path,
    metadata_path: str | Path,
) -> None:
    settings = config["performance_v150"]
    raw = Path(raw_dir)
    target_ticker = str(settings["target_ticker"])
    target_market = _market_frame(raw, target_ticker)
    dates = target_market.index
    target_engineered = asset_features(target_market).reindex(dates)
    target_returns = target_engineered["ret_1d"].to_numpy(dtype=np.float32)

    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    names: list[str] = []

    for column in target_engineered.columns:
        _append_feature(values, masks, names, target_engineered[column], f"target:{column}")

    market_tickers = list(
        dict.fromkeys(
            [*settings["core_market_tickers"], *settings["optional_market_tickers"]]
        )
    )
    core_market = {str(value) for value in settings["core_market_tickers"]}
    for ticker in market_tickers:
        path = raw / "market" / f"{ticker}.csv"
        if not path.is_file():
            if str(ticker) in core_market:
                raise FileNotFoundError(f"Missing required core market file: {path}")
            print(f"optional market source unavailable; omitting {ticker}: {path}")
            continue
        market = _market_frame(raw, str(ticker)).reindex(dates)
        close = market["Close"].astype(float)
        ret = np.log(close.where(close > 0.0)).diff()
        _append_feature(values, masks, names, ret, f"market_return:{ticker}")
        rolling = ret.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0)
        _append_feature(values, masks, names, rolling, f"market_vol21:{ticker}")

    all_fred = {
        **config["data"]["fred_series"],
        **settings.get("optional_fred_series", {}),
    }
    lags = settings.get("fred_release_lag_business_days", {})
    for series_id, alias in all_fred.items():
        raw_series = _fred_frame(raw, str(series_id), str(alias))
        series = _business_shift(raw_series, int(lags.get(series_id, 1)), dates)
        _append_feature(values, masks, names, series, f"fred_level:{alias}")
        _append_feature(values, masks, names, series.diff(), f"fred_change1:{alias}")
        _append_feature(values, masks, names, series.diff(5), f"fred_change5:{alias}")
        if str(alias) in {"wti_oil", "broad_dollar"}:
            _append_feature(
                values,
                masks,
                names,
                _safe_log_change(series, 1),
                f"fred_logret1:{alias}",
            )

    french = settings["french_factors"]
    release_lag = int(french.get("release_lag_business_days", 1))
    factor_columns = {
        "ff3_daily": {"mkt_rf", "smb", "hml", "rf"},
        "ff5_daily": {"rmw", "cma"},
        "momentum_daily": {"mom", "mom_"},
    }
    for alias in ("ff3_daily", "ff5_daily", "momentum_daily"):
        frame = _load_french(raw, alias)
        frame = _business_shift_frame(frame, release_lag, dates)
        for column in frame.columns:
            clean_name = re.sub(r"[^A-Za-z0-9]+", "_", str(column)).strip("_").lower()
            if clean_name not in factor_columns[alias]:
                continue
            series = frame[column].astype(float)
            _append_feature(values, masks, names, series, f"french_return:{alias}:{clean_name}")
            _append_feature(
                values,
                masks,
                names,
                series.rolling(21, min_periods=10).std(ddof=0) * np.sqrt(252.0),
                f"french_vol21:{alias}:{clean_name}",
            )

    matrix = np.stack(values, axis=1).astype(np.float32)
    mask_matrix = np.stack(masks, axis=1).astype(np.float32)
    # Retain the target calendar; only require a valid target return and one year of
    # target history.  Satellite channels remain optional and are accompanied by masks.
    target_valid = np.isfinite(target_returns)
    warmup = np.arange(len(dates)) >= 252
    keep = target_valid & warmup
    dates = dates[keep]
    matrix = matrix[keep]
    mask_matrix = mask_matrix[keep]
    target_returns = target_returns[keep]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    date_strings = _fixed_width_unicode(
        pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist()
    )
    feature_name_strings = _fixed_width_unicode(names)
    np.savez_compressed(
        output,
        dates=date_strings,
        values=matrix,
        masks=mask_matrix,
        feature_names=feature_name_strings,
        target_returns=target_returns,
    )
    coverage = {
        name: float(mask_matrix[:, index].mean()) for index, name in enumerate(names)
    }
    metadata = {
        "dates": [str(dates.min().date()), str(dates.max().date())],
        "rows": int(len(dates)),
        "features": names,
        "feature_coverage": coverage,
        "target_calendar": target_ticker,
        "causality": (
            "FRED and French observations are shifted by configured business-day "
            "release lags; optional market channels retain explicit availability masks."
        ),
    }
    meta = Path(metadata_path)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved long-history panel: {matrix.shape} -> {output}")
    print(f"saved long-history metadata -> {meta}")


def build_v150_panels(config: dict[str, Any], raw_dir: str | Path) -> None:
    data = config["data"]
    settings = config["performance_v150"]
    build_panel(
        tickers=[str(value) for value in data["tickers"]],
        target_ticker=str(data["target_ticker"]),
        fred_series={str(k): str(v) for k, v in data["fred_series"].items()},
        raw_dir=raw_dir,
        output_path=data["processed_path"],
        metadata_path=data["metadata_path"],
        correlation_window=int(config["features"]["correlation_window"]),
        graph_top_k=int(config["features"]["graph_top_k"]),
        macro_release_lag_days=int(data.get("macro_release_lag_days", 1)),
    )
    build_long_history_panel(
        config,
        raw_dir,
        settings["long_panel_path"],
        settings["long_panel_metadata_path"],
    )


def split_long_origins(
    panel: LongHistoryPanel,
    fold: Fold,
    lookback: int,
    horizons: list[int],
    embargo_days: int,
    common_max_horizon: int | None = None,
) -> dict[str, np.ndarray]:
    maximum = max(max(horizons), int(common_max_horizon or 0))
    valid = np.arange(lookback - 1, len(panel.dates) - maximum, dtype=np.int64)
    origin_dates = panel.dates[valid]
    target_end = panel.dates[valid + maximum]
    train_end = pd.Timestamp(fold.train_end)
    validation_end = pd.Timestamp(fold.validation_end)
    test_end = pd.Timestamp(fold.test_end)
    train = valid[target_end <= train_end]
    validation_start = train_end + pd.offsets.BDay(int(embargo_days))
    validation = valid[(origin_dates > validation_start) & (target_end <= validation_end)]
    test_start = validation_end + pd.offsets.BDay(int(embargo_days))
    test = valid[(origin_dates > test_start) & (target_end <= test_end)]
    groups = {"train": train, "validation": validation, "test": test}
    for name, values_ in groups.items():
        if len(values_) == 0:
            raise ValueError(f"No {name} long-history origins for fold {fold.name}")
    return groups


def long_targets(
    panel: LongHistoryPanel,
    origins: np.ndarray,
    horizons: list[int],
) -> np.ndarray:
    output = np.empty((len(origins), len(horizons)), dtype=np.float32)
    for row, origin in enumerate(np.asarray(origins, dtype=np.int64)):
        for column, horizon in enumerate(horizons):
            output[row, column] = panel.target_returns[origin + 1 : origin + horizon + 1].sum()
    return output
