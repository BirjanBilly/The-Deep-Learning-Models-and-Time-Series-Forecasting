from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

_EPS = 1e-12


@dataclass(frozen=True)
class BacktestResult:
    statistic: float
    p_value: float
    observations: int
    exceptions: int
    expected_rate: float
    observed_rate: float


def _safe_xlogy(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    return float(count) * math.log(max(min(float(probability), 1.0 - _EPS), _EPS))


def pinball_loss_numpy(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    error = target[..., None] - prediction
    loss = np.maximum(quantiles * error, (quantiles - 1.0) * error)
    return float(np.mean(loss))


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    target = np.asarray(target, dtype=float)
    width = upper - lower
    below = (target < lower).astype(float)
    above = (target > upper).astype(float)
    return width + (2.0 / alpha) * (lower - target) * below + (2.0 / alpha) * (target - upper) * above


def kupiec_unconditional_coverage(exceptions: np.ndarray, expected_rate: float) -> BacktestResult:
    flags = np.asarray(exceptions, dtype=bool).reshape(-1)
    n = int(flags.size)
    x = int(flags.sum())
    if n == 0:
        return BacktestResult(float("nan"), float("nan"), 0, 0, expected_rate, float("nan"))
    observed = x / n
    null_log_likelihood = _safe_xlogy(n - x, 1.0 - expected_rate) + _safe_xlogy(x, expected_rate)
    alt_log_likelihood = _safe_xlogy(n - x, 1.0 - observed) + _safe_xlogy(x, observed)
    statistic = max(0.0, -2.0 * (null_log_likelihood - alt_log_likelihood))
    return BacktestResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, df=1)),
        observations=n,
        exceptions=x,
        expected_rate=float(expected_rate),
        observed_rate=float(observed),
    )


def christoffersen_independence(exceptions: np.ndarray) -> BacktestResult:
    flags = np.asarray(exceptions, dtype=np.int8).reshape(-1)
    if flags.size < 2:
        return BacktestResult(float("nan"), float("nan"), int(flags.size), int(flags.sum()), float("nan"), float("nan"))
    previous = flags[:-1]
    current = flags[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    total_zero = n00 + n01
    total_one = n10 + n11
    pi01 = n01 / total_zero if total_zero else 0.0
    pi11 = n11 / total_one if total_one else 0.0
    total_transitions = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total_transitions if total_transitions else 0.0
    independent_ll = _safe_xlogy(n00 + n10, 1.0 - pi) + _safe_xlogy(n01 + n11, pi)
    markov_ll = (
        _safe_xlogy(n00, 1.0 - pi01)
        + _safe_xlogy(n01, pi01)
        + _safe_xlogy(n10, 1.0 - pi11)
        + _safe_xlogy(n11, pi11)
    )
    statistic = max(0.0, -2.0 * (independent_ll - markov_ll))
    return BacktestResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, df=1)),
        observations=int(flags.size),
        exceptions=int(flags.sum()),
        expected_rate=float(pi),
        observed_rate=float(flags.mean()),
    )


def _quantile_index(quantiles: np.ndarray, value: float, tolerance: float = 1e-8) -> int | None:
    candidates = np.flatnonzero(np.isclose(quantiles, value, atol=tolerance, rtol=0.0))
    return int(candidates[0]) if len(candidates) else None


def approximate_expected_shortfall(predictions: np.ndarray, quantiles: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    mask = quantiles <= alpha + 1e-12
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(quantiles - alpha)))
        return predictions[..., nearest]
    return np.mean(predictions[..., mask], axis=-1)


def evaluate_forecasts(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float] | np.ndarray,
    horizons: list[int],
    stress_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if pred.ndim != 3:
        raise ValueError("prediction must have shape [samples, horizons, quantiles]")
    if truth.shape != pred.shape[:2]:
        raise ValueError(f"target shape {truth.shape} does not match {pred.shape[:2]}")
    if pred.shape[1] != len(horizons) or pred.shape[2] != len(q):
        raise ValueError("horizon or quantile dimensions do not match configuration")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Predictions and targets must be finite")
    if np.any(np.diff(pred, axis=-1) < -1e-8):
        raise ValueError("Predicted quantiles are not monotone")

    overall: dict[str, Any] = {
        "samples": int(pred.shape[0]),
        "mean_pinball": pinball_loss_numpy(pred, truth, q),
    }
    rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    median_index = int(np.argmin(np.abs(q - 0.5)))

    for horizon_index, horizon in enumerate(horizons):
        y = truth[:, horizon_index]
        p = pred[:, horizon_index, :]
        median = p[:, median_index]
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "samples": int(len(y)),
            "pinball": pinball_loss_numpy(p[:, None, :], y[:, None], q),
            "mae_median": float(np.mean(np.abs(y - median))),
            "rmse_median": float(np.sqrt(np.mean((y - median) ** 2))),
            "direction_accuracy_median": float(np.mean((median > 0.0) == (y > 0.0))),
        }

        interval_scores: list[np.ndarray] = []
        for coverage in (0.90, 0.95):
            alpha = 1.0 - coverage
            lower_index = _quantile_index(q, alpha / 2.0)
            upper_index = _quantile_index(q, 1.0 - alpha / 2.0)
            if lower_index is None or upper_index is None:
                continue
            lower = p[:, lower_index]
            upper = p[:, upper_index]
            inside = (y >= lower) & (y <= upper)
            score = interval_score(lower, upper, y, alpha)
            interval_scores.append(score)
            label = int(round(coverage * 100))
            row[f"coverage_{label}"] = float(inside.mean())
            row[f"width_{label}"] = float(np.mean(upper - lower))
            row[f"interval_score_{label}"] = float(np.mean(score))
        if interval_scores:
            row["mean_interval_score"] = float(np.mean(np.stack(interval_scores, axis=0)))

        var_index = _quantile_index(q, 0.05)
        if var_index is not None:
            var = p[:, var_index]
            exceptions = y < var
            kupiec = kupiec_unconditional_coverage(exceptions, expected_rate=0.05)
            christoffersen = christoffersen_independence(exceptions)
            es = approximate_expected_shortfall(p, q, alpha=0.05)
            tail = y[exceptions]
            row.update(
                {
                    "var_05_exception_rate": float(exceptions.mean()),
                    "var_05_kupiec_p": kupiec.p_value,
                    "var_05_christoffersen_p": christoffersen.p_value,
                    "predicted_es_05_mean": float(np.mean(es)),
                    "realized_tail_mean": float(np.mean(tail)) if len(tail) else float("nan"),
                }
            )
            backtest_rows.extend(
                [
                    {
                        "horizon": int(horizon),
                        "test": "Kupiec unconditional coverage",
                        **kupiec.__dict__,
                    },
                    {
                        "horizon": int(horizon),
                        "test": "Christoffersen independence",
                        **christoffersen.__dict__,
                    },
                ]
            )

        if stress_mask is not None:
            mask = np.asarray(stress_mask, dtype=bool)
            if mask.shape != (len(y),):
                raise ValueError("stress_mask must have shape [samples]")
            for regime_name, regime_mask in (("stress", mask), ("normal", ~mask)):
                if not np.any(regime_mask):
                    continue
                row[f"{regime_name}_samples"] = int(regime_mask.sum())
                row[f"{regime_name}_pinball"] = pinball_loss_numpy(
                    p[regime_mask, None, :], y[regime_mask, None], q
                )
                row[f"{regime_name}_mae_median"] = float(
                    np.mean(np.abs(y[regime_mask] - median[regime_mask]))
                )
        rows.append(row)

    detail = pd.DataFrame(rows)
    backtests = pd.DataFrame(backtest_rows)
    overall["horizons"] = detail.to_dict(orient="records")
    return overall, detail, backtests


def moving_block_bootstrap_difference(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    block_length: int = 20,
    draws: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    a = np.asarray(losses_a, dtype=float).reshape(-1)
    b = np.asarray(losses_b, dtype=float).reshape(-1)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("Loss arrays must have equal length and at least two observations")
    difference = a - b
    n = len(difference)
    block_length = max(1, min(int(block_length), n))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_length + 1)
    samples = np.empty(draws, dtype=float)
    blocks_needed = math.ceil(n / block_length)
    for draw in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        resampled = np.concatenate([difference[start : start + block_length] for start in chosen])[:n]
        samples[draw] = np.mean(resampled)
    return {
        "mean_difference": float(np.mean(difference)),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_a_worse_than_b": float(np.mean(samples > 0.0)),
    }
