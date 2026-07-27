from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASSET_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "abs_ret_1d",
    "vol_5d",
    "vol_21d",
    "vol_63d",
    "downside_vol_21d",
    "drawdown_252d",
    "volume_z_21d",
    "intraday_range",
    "rsi_14",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def asset_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().sort_index()
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)
    ret = np.log(close).diff()
    output = pd.DataFrame(index=frame.index)
    output["ret_1d"] = ret
    output["ret_5d"] = np.log(close).diff(5)
    output["ret_21d"] = np.log(close).diff(21)
    output["abs_ret_1d"] = ret.abs()
    output["vol_5d"] = ret.rolling(5).std(ddof=0) * np.sqrt(252.0)
    output["vol_21d"] = ret.rolling(21).std(ddof=0) * np.sqrt(252.0)
    output["vol_63d"] = ret.rolling(63).std(ddof=0) * np.sqrt(252.0)
    output["downside_vol_21d"] = (
        ret.clip(upper=0.0).pow(2).rolling(21).mean().pow(0.5) * np.sqrt(252.0)
    )
    output["drawdown_252d"] = close / close.rolling(252, min_periods=63).max() - 1.0
    log_volume = np.log1p(volume)
    output["volume_z_21d"] = (
        log_volume - log_volume.rolling(21).mean()
    ) / log_volume.rolling(21).std(ddof=0).replace(0.0, np.nan)
    output["intraday_range"] = (frame["High"] - frame["Low"]) / close.replace(0.0, np.nan)
    output["rsi_14"] = _rsi(close) / 100.0
    return output.replace([np.inf, -np.inf], np.nan)


def _load_market(raw_dir: Path, ticker: str) -> pd.DataFrame:
    path = raw_dir / "market" / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing market file: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame.set_index("Date").sort_index()
    return frame


def _load_macro(
    raw_dir: Path,
    series: dict[str, str],
    index: pd.DatetimeIndex,
    release_lag_days: int = 1,
) -> pd.DataFrame:
    macro = pd.DataFrame(index=index)
    for series_id, alias in series.items():
        path = raw_dir / "fred" / f"{series_id}_{alias}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing FRED file: {path}")
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        macro[alias] = frame[alias].reindex(index).ffill(limit=10).shift(release_lag_days)
    if {"treasury_10y", "treasury_2y"}.issubset(macro.columns):
        macro["curve_slope_10y_2y"] = macro["treasury_10y"] - macro["treasury_2y"]
    for name in list(macro.columns):
        macro[f"{name}_change_1d"] = macro[name].diff()
    return macro.replace([np.inf, -np.inf], np.nan)


def build_dynamic_graph(returns: np.ndarray, top_k: int) -> np.ndarray:
    if returns.ndim != 2:
        raise ValueError("returns must have shape [time, assets]")
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    strengths = np.abs(corr)
    np.fill_diagonal(strengths, 0.0)
    n_assets = strengths.shape[0]
    adjacency = np.zeros_like(corr, dtype=np.float32)
    k = min(max(1, top_k), max(1, n_assets - 1))
    for row in range(n_assets):
        neighbours = np.argpartition(strengths[row], -k)[-k:]
        adjacency[row, neighbours] = corr[row, neighbours]
    transpose = adjacency.T.copy()
    use_transpose = np.abs(transpose) > np.abs(adjacency)
    adjacency[use_transpose] = transpose[use_transpose]
    np.fill_diagonal(adjacency, 1.0)
    return adjacency.astype(np.float32)


def build_panel(
    tickers: list[str],
    target_ticker: str,
    fred_series: dict[str, str],
    raw_dir: str | Path,
    output_path: str | Path,
    metadata_path: str | Path,
    correlation_window: int,
    graph_top_k: int,
    macro_release_lag_days: int = 1,
) -> None:
    raw = Path(raw_dir)
    market = {ticker: _load_market(raw, ticker) for ticker in tickers}
    common_index = market[target_ticker].index
    for ticker in tickers:
        common_index = common_index.intersection(market[ticker].index)
    common_index = common_index.sort_values()
    if len(common_index) < 1000:
        raise ValueError(f"Only {len(common_index)} common market dates; check downloads")

    feature_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        feature_frames.append(asset_features(market[ticker].reindex(common_index)))
    macro = _load_macro(raw, fred_series, common_index, macro_release_lag_days)

    valid = macro.notna().all(axis=1)
    for frame in feature_frames:
        valid &= frame.notna().all(axis=1)
    common_index = common_index[valid]
    feature_frames = [frame.reindex(common_index) for frame in feature_frames]
    macro = macro.reindex(common_index)
    if len(common_index) < 1000:
        raise ValueError(
            f"Only {len(common_index)} fully aligned dates remain after feature and macro "
            "filtering. Inspect each FRED file's first and last numeric observation; a "
            "short-history macro series can silently truncate the whole experiment."
        )
    print(
        "fully aligned panel coverage: "
        f"{common_index.min().date()} to {common_index.max().date()} "
        f"({len(common_index):,} rows)"
    )

    asset_cube = np.stack([frame[ASSET_FEATURES].to_numpy(np.float32) for frame in feature_frames], axis=1)
    macro_matrix = macro.to_numpy(np.float32)
    returns = np.stack([frame["ret_1d"].to_numpy(np.float32) for frame in feature_frames], axis=1)

    adjacency = np.zeros((len(common_index), len(tickers), len(tickers)), dtype=np.float32)
    for i in range(len(common_index)):
        start = max(0, i - correlation_window + 1)
        if i - start + 1 < max(20, min(correlation_window, 20)):
            adjacency[i] = np.eye(len(tickers), dtype=np.float32)
        else:
            adjacency[i] = build_dynamic_graph(returns[start : i + 1], graph_top_k)

    target_index = tickers.index(target_ticker)
    target_returns = returns[:, target_index]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dates=np.asarray(common_index.strftime("%Y-%m-%d"), dtype="U10"),
        asset_features=asset_cube,
        macro_features=macro_matrix,
        adjacency=adjacency,
        target_returns=target_returns.astype(np.float32),
        tickers=np.asarray(tickers, dtype="U32"),
        asset_feature_names=np.asarray(ASSET_FEATURES, dtype="U64"),
        macro_feature_names=np.asarray(list(macro.columns), dtype="U64"),
        target_index=np.asarray(target_index, dtype=np.int64),
    )

    metadata: dict[str, Any] = {
        "rows": len(common_index),
        "start": str(common_index.min().date()),
        "end": str(common_index.max().date()),
        "tickers": tickers,
        "target_ticker": target_ticker,
        "target_index": target_index,
        "asset_features": ASSET_FEATURES,
        "macro_features": list(macro.columns),
        "correlation_window": correlation_window,
        "graph_top_k": graph_top_k,
        "macro_release_lag_days": macro_release_lag_days,
        "causality": "market features and graphs at t use dates <= t; macro values are lagged",
    }
    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved panel: {asset_cube.shape} -> {output}")
    print(f"saved metadata -> {metadata_output}")
