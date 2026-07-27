from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class BaselineForecast:
    predictions: np.ndarray
    fallback_count: int
    fit_count: int


def _historical_sums(returns: np.ndarray, horizon: int) -> np.ndarray:
    if len(returns) < horizon:
        return np.asarray([], dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(returns, dtype=float)])
    return cumulative[horizon:] - cumulative[:-horizon]


def historical_simulation(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: np.ndarray,
    window: int,
) -> BaselineForecast:
    result = np.empty((len(origins), len(horizons), len(quantiles)), dtype=np.float32)
    for row, origin in enumerate(origins):
        history = returns[max(0, int(origin) - window + 1) : int(origin) + 1]
        for h_index, horizon in enumerate(horizons):
            samples = _historical_sums(history, horizon)
            if len(samples) < 30:
                samples = _historical_sums(returns[: int(origin) + 1], horizon)
            result[row, h_index] = np.quantile(samples, quantiles).astype(np.float32)
    return BaselineForecast(result, fallback_count=0, fit_count=0)


def _ewma_variance_path(returns: np.ndarray, decay: float) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    initial = float(np.var(returns[: min(60, len(returns))]))
    initial = max(initial, 1e-8)
    variance = np.empty(len(returns), dtype=float)
    variance[0] = initial
    for index in range(1, len(returns)):
        variance[index] = decay * variance[index - 1] + (1.0 - decay) * returns[index - 1] ** 2
    return np.maximum(variance, 1e-10)


def ewma_gaussian(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: np.ndarray,
    decay: float,
) -> BaselineForecast:
    variance = _ewma_variance_path(returns, decay)
    z = norm.ppf(quantiles)
    result = np.empty((len(origins), len(horizons), len(quantiles)), dtype=np.float32)
    for row, origin in enumerate(origins):
        history = returns[max(0, int(origin) - 252 + 1) : int(origin) + 1]
        mean = float(np.mean(history)) if len(history) else 0.0
        for h_index, horizon in enumerate(horizons):
            sigma = np.sqrt(float(horizon) * variance[int(origin)])
            result[row, h_index] = (horizon * mean + sigma * z).astype(np.float32)
    return BaselineForecast(result, fallback_count=0, fit_count=0)


def _fit_variance_model(method: str, log_squared_returns: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "ets_volatility":
            return SimpleExpSmoothing(log_squared_returns, initialization_method="estimated").fit(
                optimized=True,
                remove_bias=False,
            )
        if method == "sarima_volatility":
            return SARIMAX(
                log_squared_returns,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 0, 5),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
    raise ValueError(f"Unknown statistical baseline: {method}")


def statistical_volatility_gaussian(
    method: str,
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: np.ndarray,
    window: int,
    refit_every: int,
    ewma_decay: float = 0.94,
) -> BaselineForecast:
    origins = np.asarray(origins, dtype=np.int64)
    z = norm.ppf(quantiles)
    result = np.empty((len(origins), len(horizons), len(quantiles)), dtype=np.float32)
    ewma_variance = _ewma_variance_path(returns, ewma_decay)
    max_horizon = max(horizons)
    variance_forecast: np.ndarray | None = None
    fallback_count = 0
    fit_count = 0

    for row, origin in enumerate(origins):
        should_refit = variance_forecast is None or row % max(1, refit_every) == 0
        if should_refit:
            history = returns[max(0, int(origin) - window + 1) : int(origin) + 1]
            squared = np.maximum(history**2, 1e-10)
            log_squared = np.log(squared)
            try:
                fitted = _fit_variance_model(method, log_squared)
                log_forecast = np.asarray(fitted.forecast(max_horizon), dtype=float)
                variance_forecast = np.maximum(np.exp(np.clip(log_forecast, -25.0, 2.0)), 1e-10)
                fit_count += 1
            except Exception:
                variance_forecast = np.full(max_horizon, ewma_variance[int(origin)], dtype=float)
                fallback_count += 1
        assert variance_forecast is not None
        history_mean = returns[max(0, int(origin) - 252 + 1) : int(origin) + 1]
        mean = float(np.mean(history_mean)) if len(history_mean) else 0.0
        for h_index, horizon in enumerate(horizons):
            cumulative_variance = float(np.sum(variance_forecast[:horizon]))
            sigma = np.sqrt(max(cumulative_variance, 1e-10))
            result[row, h_index] = (horizon * mean + sigma * z).astype(np.float32)
    return BaselineForecast(result, fallback_count=fallback_count, fit_count=fit_count)


def forecast_baseline(
    method: str,
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: list[float],
    historical_window: int = 750,
    ewma_lambda: float = 0.94,
    statistical_window: int = 1500,
    refit_every: int = 21,
) -> BaselineForecast:
    q = np.asarray(quantiles, dtype=float)
    if method == "historical_simulation":
        return historical_simulation(returns, origins, horizons, q, historical_window)
    if method == "ewma":
        return ewma_gaussian(returns, origins, horizons, q, ewma_lambda)
    if method in {"ets_volatility", "sarima_volatility"}:
        return statistical_volatility_gaussian(
            method,
            returns,
            origins,
            horizons,
            q,
            statistical_window,
            refit_every,
            ewma_decay=ewma_lambda,
        )
    raise ValueError(f"Unknown baseline method: {method}")
