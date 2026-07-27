from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StrategyBank:
    static_weights: np.ndarray  # [static_strategies, assets]
    static_names: list[str]
    include_mean_reversion: bool = True
    include_trend_following: bool = True
    signal_window: int = 5
    signal_temperature: float = 0.75

    @property
    def n_assets(self) -> int:
        return int(self.static_weights.shape[1])

    @property
    def names(self) -> list[str]:
        result = list(self.static_names)
        if self.include_mean_reversion:
            result.extend([f"mean_reversion::{index}" for index in range(self.n_assets)])
        if self.include_trend_following:
            result.extend([f"trend_following::{index}" for index in range(self.n_assets)])
        return result

    def pnl(self, paths: torch.Tensor) -> torch.Tensor:
        """Compute differentiable terminal PnLs for non-anticipative strategies.

        paths contains log-return increments with shape [samples, horizon, assets].
        """

        if paths.ndim != 3 or paths.shape[-1] != self.n_assets:
            raise ValueError("paths must have shape [samples, horizon, assets]")
        weights = torch.as_tensor(self.static_weights, dtype=paths.dtype, device=paths.device)
        terminal_returns = paths.sum(dim=1)
        outputs = [terminal_returns @ weights.transpose(0, 1)]

        if self.include_mean_reversion or self.include_trend_following:
            cumulative = paths.cumsum(dim=1)
            series = cumulative.transpose(1, 2)  # [B, A, T]
            window = max(2, min(int(self.signal_window), paths.shape[1]))
            padded = F.pad(series, (window - 1, 0), mode="replicate")
            moving_average = F.avg_pool1d(padded, kernel_size=window, stride=1)
            deviation = series - moving_average
            rolling_scale = series.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            signal = torch.tanh(deviation / (float(self.signal_temperature) * rolling_scale))
            position_signal = torch.zeros_like(signal)
            position_signal[:, :, 1:] = signal[:, :, :-1]
            asset_returns = paths.transpose(1, 2)
            if self.include_mean_reversion:
                outputs.append((-position_signal * asset_returns).sum(dim=-1))
            if self.include_trend_following:
                outputs.append((position_signal * asset_returns).sum(dim=-1))
        return torch.cat(outputs, dim=1)


def _canonicalize_sign(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0:
        result *= -1.0
    return result


def _normalize_l1(weights: np.ndarray) -> np.ndarray:
    denominator = np.sum(np.abs(weights), axis=-1, keepdims=True)
    denominator = np.where(denominator < 1e-12, 1.0, denominator)
    return weights / denominator


def eigenportfolio_weights(flat_returns: np.ndarray, count: int) -> np.ndarray:
    returns = np.asarray(flat_returns, dtype=float)
    if returns.ndim != 2:
        raise ValueError("flat_returns must have shape [observations, assets]")
    std = returns.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    values, vectors = np.linalg.eigh(corr)
    order = np.argsort(values)[::-1]
    selected = vectors[:, order[: min(int(count), returns.shape[1])]].T
    selected = np.stack([_canonicalize_sign(vector) for vector in selected])
    weights = selected / std[None, :]
    return _normalize_l1(weights).astype(np.float32)


def build_strategy_bank(
    training_paths: np.ndarray,
    mode: str = "full",
    num_eigenportfolios: int = 8,
    num_random_portfolios: int = 8,
    seed: int = 42,
    signal_window: int = 5,
) -> StrategyBank:
    paths = np.asarray(training_paths, dtype=np.float32)
    if paths.ndim != 3:
        raise ValueError("training_paths must have shape [samples, horizon, assets]")
    n_assets = paths.shape[-1]
    mode = str(mode)
    if mode not in {"raw", "static", "full"}:
        raise ValueError("mode must be raw, static or full")

    weights = [np.eye(n_assets, dtype=np.float32)]
    names = [[f"buy_hold::{index}" for index in range(n_assets)]]
    if mode in {"static", "full"}:
        flattened = paths.reshape(-1, n_assets)
        eigen = eigenportfolio_weights(flattened, count=num_eigenportfolios)
        weights.append(eigen)
        names.append([f"eigenportfolio::{index + 1}" for index in range(len(eigen))])
        rng = np.random.default_rng(seed)
        random_weights = _normalize_l1(rng.normal(size=(num_random_portfolios, n_assets))).astype(np.float32)
        weights.append(random_weights)
        names.append([f"random_portfolio::{index + 1}" for index in range(len(random_weights))])

    return StrategyBank(
        static_weights=np.concatenate(weights, axis=0),
        static_names=sum(names, []),
        include_mean_reversion=mode == "full",
        include_trend_following=mode == "full",
        signal_window=int(signal_window),
    )
