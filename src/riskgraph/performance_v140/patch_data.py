from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel, split_origins, targets_for_origins
from riskgraph.performance_v140.baselines import ewma_student_t_forecast


@dataclass(frozen=True)
class PatchFeatureSpec:
    channel_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    lookback: int

    def export(self) -> dict[str, Any]:
        return {
            "channel_names": list(self.channel_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "lookback": int(self.lookback),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> PatchFeatureSpec:
        return cls(
            channel_names=tuple(str(value) for value in record["channel_names"]),
            mean=np.asarray(record["mean"], dtype=np.float32),
            std=np.asarray(record["std"], dtype=np.float32),
            lookback=int(record["lookback"]),
        )


@dataclass(frozen=True)
class PatchArrays:
    histories: np.ndarray
    baselines: np.ndarray
    targets: np.ndarray
    origins: np.ndarray
    dates: np.ndarray


class PatchForecastDataset(Dataset):
    def __init__(self, arrays: PatchArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return int(len(self.arrays.origins))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": torch.from_numpy(self.arrays.histories[index]),
            "baseline": torch.from_numpy(self.arrays.baselines[index]),
            "target": torch.from_numpy(self.arrays.targets[index]),
            "origin": torch.tensor(int(self.arrays.origins[index]), dtype=torch.long),
        }


def _return_index(panel: Panel) -> int:
    if "ret_1d" not in panel.asset_feature_names:
        raise ValueError("Panel is missing ret_1d")
    return panel.asset_feature_names.index("ret_1d")


def _feature_matrix(
    panel: Panel,
    asset_tickers: list[str],
    macro_features: list[str],
    target_asset_features: list[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    channels: list[np.ndarray] = []
    names: list[str] = []
    return_index = _return_index(panel)
    for ticker in asset_tickers:
        if ticker not in panel.tickers:
            continue
        index = panel.tickers.index(ticker)
        channels.append(panel.asset_features[:, index, return_index].astype(np.float32))
        names.append(f"asset_return:{ticker}")
    for feature in target_asset_features:
        if feature not in panel.asset_feature_names:
            continue
        index = panel.asset_feature_names.index(feature)
        channels.append(panel.asset_features[:, panel.target_index, index].astype(np.float32))
        names.append(f"target_feature:{feature}")
    for feature in macro_features:
        if feature not in panel.macro_feature_names:
            continue
        index = panel.macro_feature_names.index(feature)
        channels.append(panel.macro_features[:, index].astype(np.float32))
        names.append(f"macro:{feature}")
    if not channels:
        raise ValueError("No requested patch-model channels exist in the panel")
    matrix = np.stack(channels, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix.astype(np.float32), tuple(names)


def fit_patch_feature_spec(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
) -> PatchFeatureSpec:
    patch = config["performance_v140"]["patch_transformer"]
    matrix, names = _feature_matrix(
        panel,
        asset_tickers=[str(value) for value in patch["asset_tickers"]],
        macro_features=[str(value) for value in patch["macro_features"]],
        target_asset_features=[str(value) for value in patch["target_asset_features"]],
    )
    train_mask = panel.dates <= pd.Timestamp(fold.train_end)
    values = matrix[train_mask]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return PatchFeatureSpec(
        channel_names=names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        lookback=int(patch["lookback"]),
    )


def _matrix_from_spec(panel: Panel, spec: PatchFeatureSpec) -> np.ndarray:
    channels: list[np.ndarray] = []
    for name in spec.channel_names:
        family, feature = name.split(":", maxsplit=1)
        if family == "asset_return":
            ticker_index = panel.tickers.index(feature)
            channels.append(
                panel.asset_features[:, ticker_index, _return_index(panel)].astype(np.float32)
            )
        elif family == "target_feature":
            feature_index = panel.asset_feature_names.index(feature)
            channels.append(
                panel.asset_features[:, panel.target_index, feature_index].astype(np.float32)
            )
        elif family == "macro":
            feature_index = panel.macro_feature_names.index(feature)
            channels.append(panel.macro_features[:, feature_index].astype(np.float32))
        else:
            raise ValueError(f"Unknown channel family: {family}")
    matrix = np.stack(channels, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return ((matrix - spec.mean[None, :]) / spec.std[None, :]).astype(np.float32)


def patch_fold_origins(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    horizons = [int(value) for value in config["features"]["horizons"]]
    lookback = int(config["performance_v140"]["patch_transformer"]["lookback"])
    return split_origins(
        panel,
        fold,
        lookback=lookback,
        horizons=horizons,
        embargo_days=int(config["splits"].get("embargo_days", 0)),
    )


def build_patch_arrays(
    panel: Panel,
    fold: Fold,
    origins: np.ndarray,
    config: dict[str, Any],
    spec: PatchFeatureSpec,
) -> PatchArrays:
    origins = np.asarray(origins, dtype=np.int64)
    matrix = _matrix_from_spec(panel, spec)
    histories = np.stack(
        [matrix[int(origin) - spec.lookback + 1 : int(origin) + 1] for origin in origins],
        axis=0,
    ).astype(np.float32)
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    train_end_index = int(np.flatnonzero(panel.dates <= pd.Timestamp(fold.train_end))[-1])
    performance = config["performance_v140"]
    baselines, _ = ewma_student_t_forecast(
        panel.target_returns,
        origins,
        horizons,
        quantiles,
        train_end_index=train_end_index,
        mean_decay=float(performance["ewma_mean_decay"]),
        variance_decay=float(performance["ewma_variance_decay"]),
    )
    targets = targets_for_origins(panel, origins, horizons).astype(np.float32)
    return PatchArrays(
        histories=histories,
        baselines=baselines,
        targets=targets,
        origins=origins,
        dates=np.asarray([str(panel.dates[int(origin)].date()) for origin in origins]),
    )
