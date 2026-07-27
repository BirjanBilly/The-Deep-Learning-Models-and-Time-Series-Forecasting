from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import t


@dataclass(frozen=True)
class EWMATState:
    """Leakage-safe EWMA location/scale state and fitted Student-t tail shape."""

    mean: np.ndarray
    variance: np.ndarray
    degrees_of_freedom: float


def ewma_location_variance(
    returns: np.ndarray,
    mean_decay: float = 0.985,
    variance_decay: float = 0.94,
    variance_floor: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=float).reshape(-1)
    if len(values) < 2:
        raise ValueError("At least two returns are required")
    mean = np.empty_like(values)
    variance = np.empty_like(values)
    # The state at index t must use observations available no later than t.
    mean[0] = float(values[0])
    variance[0] = max(float(values[0] * values[0]), variance_floor)
    for index in range(1, len(values)):
        previous = values[index - 1]
        mean[index] = mean_decay * mean[index - 1] + (1.0 - mean_decay) * previous
        innovation = previous - mean[index - 1]
        variance[index] = (
            variance_decay * variance[index - 1]
            + (1.0 - variance_decay) * innovation * innovation
        )
    return mean, np.maximum(variance, variance_floor)


def _student_t_loglikelihood(standardized: np.ndarray, degrees_of_freedom: float) -> float:
    residual = np.asarray(standardized, dtype=float)
    residual = residual[np.isfinite(residual)]
    nu = float(degrees_of_freedom)
    if len(residual) == 0 or nu <= 2.0:
        return float("-inf")
    # Scale to unit variance: raw Student-t scale is sqrt((nu - 2) / nu).
    scale = np.sqrt((nu - 2.0) / nu)
    z = residual / scale
    constant = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )
    return float(np.sum(constant - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu)))


def fit_student_t_df(
    returns: np.ndarray,
    train_end_index: int,
    mean_decay: float = 0.985,
    variance_decay: float = 0.94,
    candidates: tuple[float, ...] = (3.5, 4.0, 5.0, 7.0, 10.0, 15.0, 30.0, 100.0),
) -> EWMATState:
    values = np.asarray(returns, dtype=float).reshape(-1)
    mean, variance = ewma_location_variance(
        values,
        mean_decay=mean_decay,
        variance_decay=variance_decay,
    )
    end = max(2, min(int(train_end_index), len(values) - 1))
    standardized = (values[1 : end + 1] - mean[:end]) / np.sqrt(variance[:end])
    scores = {
        float(nu): _student_t_loglikelihood(standardized, float(nu))
        for nu in candidates
    }
    best = max(scores, key=scores.get)
    return EWMATState(mean=mean, variance=variance, degrees_of_freedom=float(best))


def ewma_student_t_forecast(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: list[float] | np.ndarray,
    train_end_index: int,
    mean_decay: float = 0.985,
    variance_decay: float = 0.94,
    degrees_of_freedom: float | None = None,
) -> tuple[np.ndarray, EWMATState]:
    values = np.asarray(returns, dtype=float).reshape(-1)
    origins = np.asarray(origins, dtype=np.int64)
    q = np.asarray(quantiles, dtype=float)
    state = fit_student_t_df(
        values,
        train_end_index=train_end_index,
        mean_decay=mean_decay,
        variance_decay=variance_decay,
    )
    nu = state.degrees_of_freedom if degrees_of_freedom is None else float(degrees_of_freedom)
    scale_correction = np.sqrt((nu - 2.0) / nu) if nu < 1e5 else 1.0
    z = t.ppf(q, df=nu) * scale_correction
    prediction = np.empty((len(origins), len(horizons), len(q)), dtype=np.float32)
    for row, origin in enumerate(origins):
        location = float(state.mean[int(origin)])
        daily_variance = float(state.variance[int(origin)])
        for h_index, horizon in enumerate(horizons):
            sigma = np.sqrt(max(float(horizon) * daily_variance, 1e-12))
            prediction[row, h_index] = (
                float(horizon) * location + sigma * z
            ).astype(np.float32)
    return prediction, EWMATState(
        mean=state.mean,
        variance=state.variance,
        degrees_of_freedom=nu,
    )


def weighted_ewma_correlation(
    returns: np.ndarray,
    origin: int,
    lookback: int = 126,
    decay: float = 0.97,
    shrinkage: float = 0.15,
    jitter: float = 1e-5,
) -> np.ndarray:
    """Return a positive-definite, history-only correlation Cholesky factor."""

    matrix = np.asarray(returns, dtype=float)
    start = max(0, int(origin) - int(lookback) + 1)
    history = matrix[start : int(origin) + 1]
    if history.ndim != 2 or len(history) < 5:
        return np.eye(matrix.shape[1], dtype=np.float32)
    weights = decay ** np.arange(len(history) - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    mean = np.sum(history * weights[:, None], axis=0)
    centered = history - mean
    covariance = (centered * weights[:, None]).T @ centered
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance / np.maximum(diagonal[:, None] * diagonal[None, :], 1e-12)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = 0.5 * (correlation + correlation.T)
    correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(len(diagonal))
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, jitter)
    correlation = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(correlation), jitter))
    correlation = correlation / (diagonal[:, None] * diagonal[None, :])
    np.fill_diagonal(correlation, 1.0)
    return np.linalg.cholesky(correlation + jitter * np.eye(len(diagonal))).astype(np.float32)
