from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.performance_v150.data import LongHistoryPanel


@dataclass(frozen=True)
class RegimeFeatureSpec:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    stress_weights: np.ndarray
    thresholds: tuple[float, float]

    def export(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "stress_weights": self.stress_weights.tolist(),
            "thresholds": list(self.thresholds),
            "causality": "Every descriptor uses observations no later than the forecast origin.",
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> RegimeFeatureSpec:
        return cls(
            names=tuple(str(value) for value in record["names"]),
            mean=np.asarray(record["mean"], dtype=np.float64),
            std=np.asarray(record["std"], dtype=np.float64),
            stress_weights=np.asarray(record["stress_weights"], dtype=np.float64),
            thresholds=tuple(float(value) for value in record["thresholds"]),
        )


def _safe_std(values: np.ndarray) -> float:
    value = float(np.nanstd(values))
    return value if np.isfinite(value) and value > 1e-8 else 1.0


def _feature_value(panel: LongHistoryPanel, name: str, origin: int) -> float:
    try:
        index = panel.feature_names.index(name)
    except ValueError:
        return 0.0
    if panel.masks[origin, index] < 0.5:
        return 0.0
    return float(panel.values[origin, index])


def raw_regime_features(
    panel: LongHistoryPanel,
    origins: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    returns = np.asarray(panel.target_returns, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    names = (
        "log_vol_ratio_21_126",
        "downside_variance_share_63",
        "drawdown_252",
        "absolute_return_z_21",
        "vix_level",
        "credit_spread_level",
        "market_dispersion",
    )
    market_columns = [
        index
        for index, name in enumerate(panel.feature_names)
        if name.startswith("market_return:") and name != "market_return:SPY"
    ]
    output = np.zeros((len(origins), len(names)), dtype=np.float64)
    for row, origin_value in enumerate(origins):
        origin = int(origin_value)
        history_21 = returns[max(0, origin - 20) : origin + 1]
        history_63 = returns[max(0, origin - 62) : origin + 1]
        history_126 = returns[max(0, origin - 125) : origin + 1]
        history_252 = returns[max(0, origin - 251) : origin + 1]
        vol_21 = _safe_std(history_21)
        vol_126 = _safe_std(history_126)
        output[row, 0] = np.log(max(vol_21, 1e-8) / max(vol_126, 1e-8))
        total_variance = float(np.sum(np.square(history_63)))
        downside_variance = float(np.sum(np.square(np.minimum(history_63, 0.0))))
        output[row, 1] = downside_variance / max(total_variance, 1e-12)
        cumulative = np.exp(np.cumsum(history_252))
        output[row, 2] = float(cumulative[-1] / max(cumulative.max(), 1e-12) - 1.0)
        current = abs(float(returns[origin]))
        output[row, 3] = (current - float(np.mean(np.abs(history_21)))) / max(
            float(np.std(np.abs(history_21))), 1e-8
        )
        output[row, 4] = _feature_value(panel, "fred_level:vix", origin)
        output[row, 5] = _feature_value(
            panel, "fred_level:baa_credit_spread", origin
        )
        if market_columns:
            observed = [
                float(panel.values[origin, index])
                for index in market_columns
                if panel.masks[origin, index] >= 0.5
            ]
            output[row, 6] = float(np.std(observed)) if len(observed) >= 3 else 0.0
    return np.nan_to_num(output), names


def fit_regime_feature_spec(
    panel: LongHistoryPanel,
    train_origins: np.ndarray,
    thresholds: tuple[float, float] = (0.55, 0.85),
) -> RegimeFeatureSpec:
    raw, names = raw_regime_features(panel, train_origins)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std >= 1e-8, std, 1.0)
    standardized = (raw - mean) / std
    # Signs make larger scores correspond to a more stressed state.
    weights = np.asarray([1.0, 0.7, -1.0, 0.5, 0.8, 0.7, 0.5], dtype=np.float64)
    weights /= np.linalg.norm(weights)
    score = standardized @ weights
    fitted_thresholds = tuple(float(np.quantile(score, value)) for value in thresholds)
    return RegimeFeatureSpec(
        names=names,
        mean=mean,
        std=std,
        stress_weights=weights,
        thresholds=fitted_thresholds,
    )


def transform_regime_features(
    panel: LongHistoryPanel,
    origins: np.ndarray,
    spec: RegimeFeatureSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, names = raw_regime_features(panel, origins)
    if names != spec.names:
        raise ValueError("Regime feature names do not match fitted specification")
    standardized = (raw - spec.mean[None, :]) / spec.std[None, :]
    score = standardized @ spec.stress_weights
    regime = np.digitize(score, np.asarray(spec.thresholds), right=False).astype(np.int64)
    return standardized.astype(np.float32), score.astype(np.float32), regime


def regime_frame(
    dates: pd.DatetimeIndex,
    features: np.ndarray,
    score: np.ndarray,
    regime: np.ndarray,
    names: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=list(names))
    frame.insert(0, "date", pd.DatetimeIndex(dates).strftime("%Y-%m-%d"))
    frame["stress_score"] = score
    frame["regime"] = regime
    return frame


def save_regime_spec(path: str | Path, spec: RegimeFeatureSpec) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec.export(), indent=2, sort_keys=True), encoding="utf-8")
