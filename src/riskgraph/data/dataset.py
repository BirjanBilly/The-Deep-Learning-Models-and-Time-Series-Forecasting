from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from riskgraph.config import Fold


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    asset_features: np.ndarray
    macro_features: np.ndarray
    adjacency: np.ndarray
    target_returns: np.ndarray
    tickers: list[str]
    asset_feature_names: list[str]
    macro_feature_names: list[str]
    target_index: int


@dataclass(frozen=True)
class Scalers:
    asset_mean: np.ndarray
    asset_std: np.ndarray
    macro_mean: np.ndarray
    macro_std: np.ndarray


def load_panel(path: str | Path) -> Panel:
    with np.load(path, allow_pickle=False) as data:
        return Panel(
            dates=pd.DatetimeIndex(pd.to_datetime(data["dates"].astype(str))),
            asset_features=data["asset_features"].astype(np.float32),
            macro_features=data["macro_features"].astype(np.float32),
            adjacency=data["adjacency"].astype(np.float32),
            target_returns=data["target_returns"].astype(np.float32),
            tickers=data["tickers"].astype(str).tolist(),
            asset_feature_names=data["asset_feature_names"].astype(str).tolist(),
            macro_feature_names=data["macro_feature_names"].astype(str).tolist(),
            target_index=int(data["target_index"]),
        )


def fit_scalers(panel: Panel, train_end: str) -> Scalers:
    mask = panel.dates <= pd.Timestamp(train_end)
    if mask.sum() < 100:
        raise ValueError("Too few training rows to fit scalers")
    asset = panel.asset_features[mask]
    macro = panel.macro_features[mask]
    asset_mean = asset.mean(axis=(0, 1), keepdims=False)
    asset_std = asset.std(axis=(0, 1), keepdims=False)
    macro_mean = macro.mean(axis=0)
    macro_std = macro.std(axis=0)
    asset_std = np.where(asset_std < 1e-6, 1.0, asset_std)
    macro_std = np.where(macro_std < 1e-6, 1.0, macro_std)
    return Scalers(
        asset_mean=asset_mean.astype(np.float32),
        asset_std=asset_std.astype(np.float32),
        macro_mean=macro_mean.astype(np.float32),
        macro_std=macro_std.astype(np.float32),
    )


def _add_business_days(date: pd.Timestamp, days: int) -> pd.Timestamp:
    return date + pd.offsets.BDay(days)


def split_origins(
    panel: Panel,
    fold: Fold,
    lookback: int,
    horizons: list[int],
    embargo_days: int,
) -> dict[str, np.ndarray]:
    max_horizon = max(horizons)
    valid = np.arange(lookback - 1, len(panel.dates) - max_horizon, dtype=np.int64)
    origin_dates = panel.dates[valid]
    target_end_dates = panel.dates[valid + max_horizon]
    train_end = pd.Timestamp(fold.train_end)
    validation_end = pd.Timestamp(fold.validation_end)
    test_end = pd.Timestamp(fold.test_end)
    train = valid[target_end_dates <= train_end]
    val_start = _add_business_days(train_end, embargo_days)
    validation = valid[(origin_dates > val_start) & (target_end_dates <= validation_end)]
    test_start = _add_business_days(validation_end, embargo_days)
    test = valid[(origin_dates > test_start) & (target_end_dates <= test_end)]
    for name, indices in {"train": train, "validation": validation, "test": test}.items():
        if len(indices) == 0:
            raise ValueError(f"No {name} origins for fold {fold.name}")
    return {"train": train, "validation": validation, "test": test}


class MarketWindowDataset(Dataset):
    def __init__(
        self,
        panel: Panel,
        origins: np.ndarray,
        scalers: Scalers,
        lookback: int,
        horizons: list[int],
        graph_mode: str = "dynamic",
        macro_mode: str = "enabled",
        static_graph: np.ndarray | None = None,
    ) -> None:
        self.panel = panel
        self.origins = np.asarray(origins, dtype=np.int64)
        self.scalers = scalers
        self.lookback = int(lookback)
        self.horizons = [int(value) for value in horizons]
        self.graph_mode = graph_mode
        self.macro_mode = macro_mode
        self.static_graph = static_graph
        if graph_mode == "static" and static_graph is None:
            raise ValueError("static_graph is required when graph_mode='static'")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        origin = int(self.origins[index])
        start = origin - self.lookback + 1
        asset = self.panel.asset_features[start : origin + 1]
        macro = self.panel.macro_features[start : origin + 1]
        asset = (asset - self.scalers.asset_mean[None, None, :]) / self.scalers.asset_std[None, None, :]
        macro = (macro - self.scalers.macro_mean[None, :]) / self.scalers.macro_std[None, :]
        if self.macro_mode == "disabled":
            macro = np.zeros_like(macro)
        if self.graph_mode == "dynamic":
            adjacency = self.panel.adjacency[origin]
        elif self.graph_mode == "static":
            adjacency = self.static_graph
        elif self.graph_mode == "identity":
            adjacency = np.eye(len(self.panel.tickers), dtype=np.float32)
        else:
            raise ValueError(f"Unknown graph_mode: {self.graph_mode}")
        targets = []
        for horizon in self.horizons:
            targets.append(self.panel.target_returns[origin + 1 : origin + horizon + 1].sum())
        target = np.asarray(targets, dtype=np.float32)
        direction = np.asarray(target[0] > 0.0, dtype=np.float32)
        return {
            "asset": torch.from_numpy(asset.astype(np.float32)),
            "macro": torch.from_numpy(macro.astype(np.float32)),
            "adjacency": torch.from_numpy(adjacency.astype(np.float32)),
            "target": torch.from_numpy(target),
            "direction": torch.from_numpy(direction),
            "origin_index": torch.tensor(origin, dtype=torch.long),
            "date": str(self.panel.dates[origin].date()),
        }


def mean_training_graph(panel: Panel, train_origins: np.ndarray) -> np.ndarray:
    graph = panel.adjacency[np.asarray(train_origins, dtype=np.int64)].mean(axis=0)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 1.0)
    return graph.astype(np.float32)


@dataclass(frozen=True)
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray

    def _numpy_shape(self, target: np.ndarray) -> tuple[int, ...]:
        if target.ndim == 2:
            return (1, -1)
        if target.ndim == 3:
            return (1, -1, 1)
        raise ValueError("Target arrays must have shape [samples, horizons] or [samples, horizons, quantiles]")

    def transform_numpy(self, target: np.ndarray) -> np.ndarray:
        values = np.asarray(target)
        shape = self._numpy_shape(values)
        return (values - self.mean.reshape(shape)) / self.std.reshape(shape)

    def inverse_numpy(self, target: np.ndarray) -> np.ndarray:
        values = np.asarray(target)
        shape = self._numpy_shape(values)
        return values * self.std.reshape(shape) + self.mean.reshape(shape)

    def _tensor_shape(self, target: torch.Tensor) -> tuple[int, ...]:
        if target.ndim == 2:
            return (1, -1)
        if target.ndim == 3:
            return (1, -1, 1)
        raise ValueError("Target tensors must have shape [samples, horizons] or [samples, horizons, quantiles]")

    def transform_tensor(self, target: torch.Tensor) -> torch.Tensor:
        shape = self._tensor_shape(target)
        mean = torch.as_tensor(self.mean, dtype=target.dtype, device=target.device).reshape(shape)
        std = torch.as_tensor(self.std, dtype=target.dtype, device=target.device).reshape(shape)
        return (target - mean) / std

    def inverse_tensor(self, target: torch.Tensor) -> torch.Tensor:
        shape = self._tensor_shape(target)
        mean = torch.as_tensor(self.mean, dtype=target.dtype, device=target.device).reshape(shape)
        std = torch.as_tensor(self.std, dtype=target.dtype, device=target.device).reshape(shape)
        return target * std + mean


def targets_for_origins(panel: Panel, origins: np.ndarray, horizons: list[int]) -> np.ndarray:
    values = np.empty((len(origins), len(horizons)), dtype=np.float32)
    for row, origin in enumerate(np.asarray(origins, dtype=np.int64)):
        for column, horizon in enumerate(horizons):
            values[row, column] = panel.target_returns[origin + 1 : origin + horizon + 1].sum()
    return values


def fit_target_scaler(panel: Panel, train_origins: np.ndarray, horizons: list[int]) -> TargetScaler:
    target = targets_for_origins(panel, train_origins, horizons)
    mean = target.mean(axis=0)
    std = target.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return TargetScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def stress_mask_for_origins(
    panel: Panel,
    origins: np.ndarray,
    train_end: str,
    quantile: float = 0.80,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    if "vix" in panel.macro_feature_names:
        series = panel.macro_features[:, panel.macro_feature_names.index("vix")]
    else:
        target_feature = "vol_21d"
        if target_feature not in panel.asset_feature_names:
            return np.zeros(len(origins), dtype=bool)
        feature_index = panel.asset_feature_names.index(target_feature)
        series = panel.asset_features[:, panel.target_index, feature_index]
    training_values = series[panel.dates <= pd.Timestamp(train_end)]
    threshold = float(np.nanquantile(training_values, quantile))
    return np.asarray(series[origins] >= threshold, dtype=bool)
