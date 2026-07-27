from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskgraph.data.features import ASSET_FEATURES, build_dynamic_graph


def create_synthetic_panel(
    output_path: str | Path,
    metadata_path: str | Path,
    seed: int = 42,
    rows: int = 1200,
    assets: int = 8,
    asset_features: int = 12,
    macro_features: int = 6,
) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=rows)
    regime = np.zeros(rows)
    regime[300:430] = 1.0
    regime[760:860] = 1.0
    market = rng.standard_t(df=6, size=rows) * (0.006 + 0.012 * regime)
    loadings = rng.uniform(0.5, 1.2, size=assets)
    returns = market[:, None] * loadings[None, :] + rng.normal(0, 0.004, size=(rows, assets))
    returns[:, 0] += 0.25 * np.roll(returns[:, 1], 1)
    returns[0] = 0.0

    features = np.zeros((rows, assets, asset_features), dtype=np.float32)
    for node in range(assets):
        series = pd.Series(returns[:, node], index=dates)
        values = [
            series,
            series.rolling(5).sum(),
            series.rolling(21).sum(),
            series.abs(),
            series.rolling(5).std(),
            series.rolling(21).std(),
            series.rolling(63).std(),
            series.clip(upper=0).pow(2).rolling(21).mean().pow(0.5),
            series.rolling(63).sum(),
            pd.Series(rng.normal(size=rows), index=dates).rolling(21).mean(),
            series.abs().rolling(3).mean(),
            series.ewm(span=14, adjust=False).mean(),
        ]
        matrix = np.column_stack([value.to_numpy() for value in values])
        features[:, node, : min(asset_features, matrix.shape[1])] = matrix[:, :asset_features]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    macro = np.column_stack(
        [
            regime,
            pd.Series(market).rolling(21).std().fillna(0).to_numpy(),
            np.cumsum(rng.normal(0, 0.01, rows)),
            rng.normal(size=rows),
            pd.Series(market).rolling(5).mean().fillna(0).to_numpy(),
            pd.Series(regime).rolling(10).mean().fillna(0).to_numpy(),
        ]
    ).astype(np.float32)[:, :macro_features]

    adjacency = np.zeros((rows, assets, assets), dtype=np.float32)
    for i in range(rows):
        start = max(0, i - 59)
        if i - start < 20:
            adjacency[i] = np.eye(assets, dtype=np.float32)
        else:
            adjacency[i] = build_dynamic_graph(returns[start : i + 1], top_k=min(3, assets - 1))

    tickers = ["SPY", *[f"A{i}" for i in range(1, assets)]]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),
        asset_features=features,
        macro_features=macro,
        adjacency=adjacency,
        target_returns=returns[:, 0].astype(np.float32),
        tickers=np.asarray(tickers, dtype="U32"),
        asset_feature_names=np.asarray([*ASSET_FEATURES[:asset_features], *[f"asset_feature_{i}" for i in range(len(ASSET_FEATURES), asset_features)]], dtype="U64"),
        macro_feature_names=np.asarray(["vix", "synthetic_vol", "synthetic_trend", "synthetic_noise", "synthetic_momentum", "synthetic_regime_average"][:macro_features], dtype="U64"),
        target_index=np.asarray(0, dtype=np.int64),
    )
    metadata = {
        "synthetic": True,
        "rows": rows,
        "assets": assets,
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
