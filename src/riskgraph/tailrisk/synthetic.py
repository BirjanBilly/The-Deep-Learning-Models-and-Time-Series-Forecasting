from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskgraph.tailrisk.data import TailWindowSet


@dataclass(frozen=True)
class SyntheticTailSplit:
    train: TailWindowSet
    validation: TailWindowSet
    test: TailWindowSet


def _simulate_path(
    rng: np.random.Generator,
    horizon: int,
    n_assets: int,
    regime: int,
    correlation: np.ndarray,
) -> np.ndarray:
    """Generate one path with mixed AR and GARCH-like dynamics."""

    volatility_multiplier = (0.70, 1.00, 1.75)[int(regime)]
    covariance = 0.5 * (correlation + correlation.T)
    cholesky = np.linalg.cholesky(covariance + np.eye(n_assets) * 1e-5)
    innovations = (rng.standard_normal((horizon, n_assets)) @ cholesky.T).astype(np.float32)
    heavy = rng.standard_t(df=5.0, size=(horizon, n_assets)).astype(np.float32)
    innovations[:, -2:] = 0.55 * innovations[:, -2:] + 0.45 * heavy[:, -2:]

    path = np.zeros((horizon, n_assets), dtype=np.float32)
    variance = np.full(n_assets, 0.8, dtype=np.float32)
    ar = np.zeros(n_assets, dtype=np.float32)
    if n_assets > 1:
        ar[1] = 0.50
    if n_assets > 2:
        ar[2] = -0.15

    previous = np.zeros(n_assets, dtype=np.float32)
    for time in range(horizon):
        variance = 0.03 + 0.10 * np.square(previous) + 0.87 * variance
        scale = np.sqrt(np.maximum(variance, 1e-5))
        value = ar * previous + scale * innovations[time]
        value *= volatility_multiplier
        path[time] = value
        previous = value
    return path


def make_synthetic_tail_split(
    train_samples: int = 576,
    validation_samples: int = 144,
    test_samples: int = 144,
    horizon: int = 20,
    n_assets: int = 5,
    seed: int = 42,
) -> SyntheticTailSplit:
    """Create a deterministic three-regime dataset for integration tests.

    The generated paths combine Gaussian, autoregressive and heavy-tailed
    GARCH-like components. They are not intended as a market benchmark; they
    only exercise the complete training and evaluation pipeline.
    """

    if n_assets < 3:
        raise ValueError("n_assets must be at least 3")
    total = int(train_samples + validation_samples + test_samples)
    rng = np.random.default_rng(seed)
    base = 0.28 * np.ones((n_assets, n_assets), dtype=np.float32)
    np.fill_diagonal(base, 1.0)
    # A small sector block makes eigenportfolio checks meaningful.
    if n_assets >= 4:
        base[:2, :2] = 0.65
        base[2:4, 2:4] = 0.55
        np.fill_diagonal(base, 1.0)
    # Positive-definite stabilization.
    values, vectors = np.linalg.eigh(base)
    base = (vectors * np.maximum(values, 0.05)) @ vectors.T
    diagonal = np.sqrt(np.diag(base))
    correlation = base / diagonal[:, None] / diagonal[None, :]
    correlation = 0.5 * (correlation + correlation.T)
    corr_values, corr_vectors = np.linalg.eigh(correlation)
    correlation = (corr_vectors * np.maximum(corr_values, 1e-5)) @ corr_vectors.T
    corr_diagonal = np.sqrt(np.diag(correlation))
    correlation = correlation / corr_diagonal[:, None] / corr_diagonal[None, :]

    regimes = np.tile(np.arange(3, dtype=np.int64), int(np.ceil(total / 3)))[:total]
    rng.shuffle(regimes)
    normalized = np.stack(
        [_simulate_path(rng, horizon, n_assets, int(regime), correlation) for regime in regimes]
    )
    scales = rng.lognormal(mean=-4.45, sigma=0.18, size=(total, n_assets)).astype(np.float32)
    actual = normalized * scales[:, None, :]
    origins = np.arange(total, dtype=np.int64)
    start = np.datetime64("2010-01-04")
    dates = np.asarray(
        [str(np.busday_offset(start, int(index), roll="forward"))[:10] for index in origins]
    )
    edges = np.asarray([0.33, 0.67], dtype=np.float32)

    def subset(start_index: int, stop_index: int) -> TailWindowSet:
        selector = slice(start_index, stop_index)
        return TailWindowSet(
            normalized_paths=normalized[selector].astype(np.float32),
            actual_paths=actual[selector].astype(np.float32),
            scales=scales[selector].astype(np.float32),
            regimes=regimes[selector].astype(np.int64),
            origins=origins[selector],
            dates=dates[selector],
            regime_edges=edges,
        )

    train_stop = train_samples
    validation_stop = train_samples + validation_samples
    return SyntheticTailSplit(
        train=subset(0, train_stop),
        validation=subset(train_stop, validation_stop),
        test=subset(validation_stop, total),
    )
