from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from riskgraph.data.dataset import Panel, Scalers, fit_scalers, mean_training_graph
from riskgraph.performance_v140.baselines import weighted_ewma_correlation


@dataclass(frozen=True)
class TailWindowSet:
    """Future multi-asset paths and conditioning state for scenario learning."""

    normalized_paths: np.ndarray  # [samples, horizon, assets]
    actual_paths: np.ndarray  # [samples, horizon, assets]
    scales: np.ndarray  # [samples, assets]
    regimes: np.ndarray  # [samples]
    origins: np.ndarray  # [samples]
    dates: np.ndarray  # ISO dates
    regime_edges: np.ndarray  # training-fitted cut points
    state_asset: np.ndarray | None = None  # [samples, lookback, assets, features]
    state_macro: np.ndarray | None = None  # [samples, lookback, macro features]
    state_adjacency: np.ndarray | None = None  # [samples, assets, assets]
    state_embedding: np.ndarray | None = None  # [samples, hidden]
    teacher_quantiles: np.ndarray | None = None  # [samples, horizons, quantiles]
    baseline_cholesky: np.ndarray | None = None  # [samples, assets, assets]

    @property
    def horizon(self) -> int:
        return int(self.actual_paths.shape[1])

    @property
    def n_assets(self) -> int:
        return int(self.actual_paths.shape[2])


class TailWindowDataset(Dataset):
    def __init__(self, windows: TailWindowSet) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return int(len(self.windows.origins))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "normalized_path": torch.from_numpy(self.windows.normalized_paths[index]),
            "actual_path": torch.from_numpy(self.windows.actual_paths[index]),
            "scale": torch.from_numpy(self.windows.scales[index]),
            "regime": torch.tensor(int(self.windows.regimes[index]), dtype=torch.long),
            "origin": torch.tensor(int(self.windows.origins[index]), dtype=torch.long),
            "date": str(self.windows.dates[index]),
        }
        if self.windows.state_asset is not None:
            assert self.windows.state_macro is not None
            assert self.windows.state_adjacency is not None
            item["state_asset"] = torch.from_numpy(self.windows.state_asset[index])
            item["state_macro"] = torch.from_numpy(self.windows.state_macro[index])
            item["state_adjacency"] = torch.from_numpy(self.windows.state_adjacency[index])
        if self.windows.baseline_cholesky is not None:
            item["baseline_cholesky"] = torch.from_numpy(
                self.windows.baseline_cholesky[index]
            )
        if self.windows.state_embedding is not None:
            assert self.windows.teacher_quantiles is not None
            item["state_embedding"] = torch.from_numpy(
                self.windows.state_embedding[index]
            )
            item["teacher_quantiles"] = torch.from_numpy(
                self.windows.teacher_quantiles[index]
            )
        return item


class RegimeBatchSampler(Sampler[list[int]]):
    """Yield fixed-size batches whose observations all share one regime.

    Tail-GAN's discriminator receives a sample from one distribution. Mixing
    calm and stressed windows inside one discriminator sample would blur the
    conditional distribution. This sampler keeps the sample coherent.
    """

    def __init__(
        self,
        regimes: np.ndarray,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        self.regimes = np.asarray(regimes, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._groups = {
            int(regime): np.flatnonzero(self.regimes == regime)
            for regime in np.unique(self.regimes)
        }
        if not self._groups:
            raise ValueError("No regimes are available")
        for regime, indices in self._groups.items():
            if len(indices) < self.batch_size:
                raise ValueError(
                    f"Regime {regime} has {len(indices)} samples, fewer than batch_size={self.batch_size}"
                )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self._groups.values():
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            limit = len(shuffled)
            if self.drop_last:
                limit = (limit // self.batch_size) * self.batch_size
            for start in range(0, limit, self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch.tolist())
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self._groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += int(np.ceil(len(indices) / self.batch_size))
        return total


def _return_feature_index(panel: Panel) -> int:
    if "ret_1d" not in panel.asset_feature_names:
        raise ValueError("Panel must include asset feature 'ret_1d'")
    return panel.asset_feature_names.index("ret_1d")


def _regime_series(panel: Panel) -> np.ndarray:
    if "vix" in panel.macro_feature_names:
        return panel.macro_features[:, panel.macro_feature_names.index("vix")].astype(float)
    if "vol_21d" in panel.asset_feature_names:
        index = panel.asset_feature_names.index("vol_21d")
        return panel.asset_features[:, panel.target_index, index].astype(float)
    returns = panel.asset_features[:, panel.target_index, _return_feature_index(panel)].astype(float)
    values = np.full(len(returns), np.nan, dtype=float)
    for i in range(20, len(returns)):
        values[i] = np.std(returns[i - 20 : i + 1], ddof=0)
    first = np.flatnonzero(np.isfinite(values))
    fill = values[first[0]] if len(first) else 1.0
    return np.nan_to_num(values, nan=fill)


def fit_regime_edges(
    panel: Panel,
    train_origins: np.ndarray,
    quantiles: tuple[float, ...] = (0.33, 0.67),
) -> np.ndarray:
    series = _regime_series(panel)
    values = series[np.asarray(train_origins, dtype=np.int64)]
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise ValueError("Too few finite observations to fit regimes")
    return np.quantile(values, quantiles).astype(np.float32)


def build_tail_windows(
    panel: Panel,
    origins: np.ndarray,
    horizon: int,
    scale_lookback: int = 21,
    regime_edges: np.ndarray | None = None,
    regime_fit_origins: np.ndarray | None = None,
    scale_floor: float = 1e-4,
    baseline_cov_lookback: int | None = None,
    baseline_cov_decay: float = 0.97,
    baseline_cov_shrinkage: float = 0.15,
) -> TailWindowSet:
    """Construct leakage-safe future paths from a processed RiskGraph panel.

    Volatility scales and regime labels use information available at the
    forecast origin. The future path always begins at origin + 1.
    """

    origins = np.asarray(origins, dtype=np.int64)
    horizon = int(horizon)
    scale_lookback = int(scale_lookback)
    if horizon < 1 or scale_lookback < 2:
        raise ValueError("horizon must be positive and scale_lookback must be at least 2")
    if len(origins) == 0:
        raise ValueError("origins cannot be empty")
    if origins.min() < scale_lookback - 1:
        raise ValueError("An origin does not have enough scale history")
    if origins.max() + horizon >= len(panel.dates):
        raise ValueError("An origin does not have a complete future path")

    return_index = _return_feature_index(panel)
    returns = panel.asset_features[:, :, return_index].astype(np.float32)
    actual = np.empty((len(origins), horizon, len(panel.tickers)), dtype=np.float32)
    scales = np.empty((len(origins), len(panel.tickers)), dtype=np.float32)
    for row, origin in enumerate(origins):
        actual[row] = returns[origin + 1 : origin + horizon + 1]
        history = returns[origin - scale_lookback + 1 : origin + 1]
        scale = np.std(history, axis=0, ddof=0)
        scales[row] = np.maximum(scale, scale_floor)
    normalized = actual / scales[:, None, :]

    baseline_cholesky = None
    if baseline_cov_lookback is not None:
        baseline_cholesky = np.stack(
            [
                weighted_ewma_correlation(
                    returns,
                    int(origin),
                    lookback=int(baseline_cov_lookback),
                    decay=float(baseline_cov_decay),
                    shrinkage=float(baseline_cov_shrinkage),
                )
                for origin in origins
            ],
            axis=0,
        ).astype(np.float32)

    if regime_edges is None:
        fit_origins = origins if regime_fit_origins is None else np.asarray(regime_fit_origins)
        regime_edges = fit_regime_edges(panel, fit_origins)
    regime_edges = np.asarray(regime_edges, dtype=np.float32)
    state = _regime_series(panel)[origins]
    regimes = np.digitize(state, regime_edges, right=False).astype(np.int64)
    dates = np.asarray([str(panel.dates[int(origin)].date()) for origin in origins])
    return TailWindowSet(
        normalized_paths=normalized.astype(np.float32),
        actual_paths=actual.astype(np.float32),
        scales=scales.astype(np.float32),
        regimes=regimes,
        origins=origins,
        dates=dates,
        regime_edges=regime_edges,
        baseline_cholesky=baseline_cholesky,
    )


def attach_conditioning_state(
    panel: Panel,
    windows: TailWindowSet,
    train_end: str,
    train_origins: np.ndarray,
    lookback: int,
    graph_mode: str = "dynamic",
    macro_mode: str = "enabled",
    scalers: Scalers | None = None,
) -> TailWindowSet:
    """Attach leakage-safe RiskGraph input histories to scenario windows."""

    lookback = int(lookback)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if windows.origins.min() < lookback - 1:
        raise ValueError("A conditioning origin does not have enough history")
    if graph_mode not in {"dynamic", "static", "identity"}:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    if macro_mode not in {"enabled", "disabled"}:
        raise ValueError(f"Unknown macro_mode: {macro_mode}")
    fitted = fit_scalers(panel, train_end) if scalers is None else scalers
    static_graph = (
        mean_training_graph(panel, np.asarray(train_origins, dtype=np.int64))
        if graph_mode == "static"
        else None
    )
    count = len(windows.origins)
    asset = np.empty(
        (count, lookback, len(panel.tickers), panel.asset_features.shape[-1]),
        dtype=np.float32,
    )
    macro = np.empty(
        (count, lookback, panel.macro_features.shape[-1]),
        dtype=np.float32,
    )
    adjacency = np.empty(
        (count, len(panel.tickers), len(panel.tickers)),
        dtype=np.float32,
    )
    identity = np.eye(len(panel.tickers), dtype=np.float32)
    for row, origin_value in enumerate(windows.origins):
        origin = int(origin_value)
        start = origin - lookback + 1
        asset_window = panel.asset_features[start : origin + 1]
        macro_window = panel.macro_features[start : origin + 1]
        asset[row] = (
            asset_window - fitted.asset_mean[None, None, :]
        ) / fitted.asset_std[None, None, :]
        macro_scaled = (
            macro_window - fitted.macro_mean[None, :]
        ) / fitted.macro_std[None, :]
        macro[row] = 0.0 if macro_mode == "disabled" else macro_scaled
        if graph_mode == "dynamic":
            adjacency[row] = panel.adjacency[origin]
        elif graph_mode == "static":
            assert static_graph is not None
            adjacency[row] = static_graph
        else:
            adjacency[row] = identity
    return replace(
        windows,
        state_asset=asset,
        state_macro=macro,
        state_adjacency=adjacency,
    )
