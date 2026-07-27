from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2, norm

from riskgraph.evaluation.metrics import evaluate_forecasts
from riskgraph.tailrisk.data import TailWindowSet
from riskgraph.tailrisk.models import (
    EWMABackboneFactorScaleGenerator,
    StableEWMABackboneFactorScaleGenerator,
    EWMABackboneTailGenerator,
    RegimeTailGenerator,
    RiskGraphConditionedScenarioGenerator,
    RiskGraphEWMABackboneScenarioGenerator,
    sample_noise,
)
from riskgraph.tailrisk.score import empirical_var_es, joint_var_es_score
from riskgraph.tailrisk.strategies import StrategyBank


@dataclass(frozen=True)
class ScenarioEvaluation:
    strategy_metrics: pd.DataFrame
    structural_metrics: dict[str, Any]
    rank_frequency: pd.DataFrame


def _numpy_pnl(bank: StrategyBank, paths: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(np.asarray(paths, dtype=np.float32)).to(device)
        return bank.pnl(tensor).cpu().numpy()


def _autocorrelation(paths: np.ndarray, max_lag: int = 10) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    horizon = values.shape[1]
    max_lag = min(int(max_lag), horizon - 1)
    result = np.zeros((values.shape[2], max_lag + 1), dtype=float)
    result[:, 0] = 1.0
    for asset in range(values.shape[2]):
        series = values[:, :, asset]
        for lag in range(1, max_lag + 1):
            left = series[:, :-lag].reshape(-1)
            right = series[:, lag:].reshape(-1)
            if left.std() < 1e-12 or right.std() < 1e-12:
                result[asset, lag] = 0.0
            else:
                result[asset, lag] = np.corrcoef(left, right)[0, 1]
    return np.nan_to_num(result)


def _kupiec_p_value(exceptions: np.ndarray, alpha: float) -> float:
    flags = np.asarray(exceptions, dtype=bool)
    n = len(flags)
    x = int(flags.sum())
    if n == 0:
        return float("nan")
    observed = x / n
    eps = 1e-12
    null = (n - x) * math.log(max(1 - alpha, eps)) + x * math.log(max(alpha, eps))
    alt = (n - x) * math.log(max(1 - observed, eps)) + x * math.log(max(observed, eps))
    statistic = max(0.0, -2.0 * (null - alt))
    return float(chi2.sf(statistic, 1))


def _score_test_p_value(
    real_pnl: np.ndarray,
    generated_var: np.ndarray,
    generated_es: np.ndarray,
    reference_var: np.ndarray,
    reference_es: np.ndarray,
    alpha: float,
    weight: float,
) -> np.ndarray:
    p_values = []
    for strategy in range(real_pnl.shape[1]):
        x = torch.tensor(real_pnl[:, strategy], dtype=torch.float64)
        fake_score = joint_var_es_score(
            torch.tensor(generated_var[strategy], dtype=torch.float64),
            torch.tensor(generated_es[strategy], dtype=torch.float64),
            x,
            alpha=alpha,
            weight=weight,
            reduction="none",
        ).numpy()
        reference_score = joint_var_es_score(
            torch.tensor(reference_var[strategy], dtype=torch.float64),
            torch.tensor(reference_es[strategy], dtype=torch.float64),
            x,
            alpha=alpha,
            weight=weight,
            reduction="none",
        ).numpy()
        difference = fake_score - reference_score
        standard_error = difference.std(ddof=1) / math.sqrt(max(1, len(difference)))
        statistic = difference.mean() / standard_error if standard_error > 1e-12 else 0.0
        p_values.append(float(2.0 * norm.sf(abs(statistic))))
    return np.asarray(p_values)


def evaluate_scenario_generator(
    real_paths: np.ndarray,
    generated_paths: np.ndarray,
    strategy_bank: StrategyBank,
    alphas: list[float] | tuple[float, ...],
    device: torch.device,
    weight: float = 10.0,
    quantile_grid: np.ndarray | None = None,
) -> ScenarioEvaluation:
    real_paths = np.asarray(real_paths, dtype=np.float32)
    generated_paths = np.asarray(generated_paths, dtype=np.float32)
    real_pnl = _numpy_pnl(strategy_bank, real_paths, device)
    generated_pnl = _numpy_pnl(strategy_bank, generated_paths, device)
    real_var, real_es = empirical_var_es(real_pnl, alphas)
    fake_var, fake_es = empirical_var_es(generated_pnl, alphas)

    rows: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(alphas):
        score_p = _score_test_p_value(
            real_pnl,
            fake_var[:, alpha_index],
            fake_es[:, alpha_index],
            real_var[:, alpha_index],
            real_es[:, alpha_index],
            alpha=float(alpha),
            weight=weight,
        )
        for strategy, name in enumerate(strategy_bank.names):
            denominator_var = max(abs(real_var[strategy, alpha_index]), 1e-8)
            denominator_es = max(abs(real_es[strategy, alpha_index]), 1e-8)
            exceptions = real_pnl[:, strategy] <= fake_var[strategy, alpha_index]
            rows.append(
                {
                    "strategy": name,
                    "alpha": float(alpha),
                    "real_var": float(real_var[strategy, alpha_index]),
                    "generated_var": float(fake_var[strategy, alpha_index]),
                    "var_relative_error": float(abs(fake_var[strategy, alpha_index] - real_var[strategy, alpha_index]) / denominator_var),
                    "real_es": float(real_es[strategy, alpha_index]),
                    "generated_es": float(fake_es[strategy, alpha_index]),
                    "es_relative_error": float(abs(fake_es[strategy, alpha_index] - real_es[strategy, alpha_index]) / denominator_es),
                    "coverage_rate": float(exceptions.mean()),
                    "coverage_kupiec_p": _kupiec_p_value(exceptions, float(alpha)),
                    "score_test_p": float(score_p[strategy]),
                }
            )
    strategy_metrics = pd.DataFrame(rows)

    real_corr = np.nan_to_num(
        np.corrcoef(real_paths.reshape(-1, real_paths.shape[-1]), rowvar=False)
    )
    fake_corr = np.nan_to_num(
        np.corrcoef(generated_paths.reshape(-1, generated_paths.shape[-1]), rowvar=False)
    )
    real_auto = _autocorrelation(real_paths)
    fake_auto = _autocorrelation(generated_paths)
    structural = {
        "correlation_l1_error": float(np.sum(np.abs(real_corr - fake_corr))),
        "autocorrelation_l1_error": float(np.sum(np.abs(real_auto - fake_auto))),
        "real_correlation": np.nan_to_num(real_corr).tolist(),
        "generated_correlation": np.nan_to_num(fake_corr).tolist(),
        "real_autocorrelation": real_auto.tolist(),
        "generated_autocorrelation": fake_auto.tolist(),
        "mean_tail_relative_error": float(
            strategy_metrics[["var_relative_error", "es_relative_error"]].to_numpy().mean()
        ),
        "coverage_rejection_rate_5pct": float((strategy_metrics["coverage_kupiec_p"] < 0.05).mean()),
        "score_rejection_rate_5pct": float((strategy_metrics["score_test_p"] < 0.05).mean()),
    }

    if quantile_grid is None:
        quantile_grid = np.unique(np.concatenate([np.geomspace(0.001, 0.1, 30), np.linspace(0.11, 0.99, 30)]))
    rank_rows = []
    for strategy, name in enumerate(strategy_bank.names):
        for quantile in quantile_grid:
            rank_rows.append(
                {
                    "strategy": name,
                    "quantile": float(quantile),
                    "real_pnl_quantile": float(np.quantile(real_pnl[:, strategy], quantile)),
                    "generated_pnl_quantile": float(np.quantile(generated_pnl[:, strategy], quantile)),
                }
            )
    return ScenarioEvaluation(
        strategy_metrics=strategy_metrics,
        structural_metrics=structural,
        rank_frequency=pd.DataFrame(rank_rows),
    )


def generate_paths_for_windows(
    generator: torch.nn.Module,
    windows: TailWindowSet,
    scenarios_per_origin: int,
    latent_dim: int,
    device: torch.device,
    noise_distribution: str,
    degrees_of_freedom: float,
    chunk_size: int = 256,
) -> np.ndarray:
    """Generate origin-specific actual-return scenarios using current volatility scales."""

    generator.eval()
    result = np.empty(
        (len(windows.origins), scenarios_per_origin, windows.horizon, windows.n_assets),
        dtype=np.float32,
    )
    with torch.no_grad():
        for origin_index in range(len(windows.origins)):
            parts = []
            regime_value = int(windows.regimes[origin_index])
            for start in range(0, scenarios_per_origin, chunk_size):
                size = min(chunk_size, scenarios_per_origin - start)
                regime = torch.full((size,), regime_value, dtype=torch.long, device=device)
                noise = sample_noise(
                    size,
                    latent_dim,
                    device,
                    distribution=noise_distribution,
                    degrees_of_freedom=degrees_of_freedom,
                )
                baseline = None
                if isinstance(
                    generator,
                    (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator, RiskGraphEWMABackboneScenarioGenerator),
                ):
                    if windows.baseline_cholesky is None:
                        raise ValueError("EWMA backbone requires baseline Cholesky factors")
                    baseline = (
                        torch.from_numpy(
                            windows.baseline_cholesky[origin_index : origin_index + 1]
                        )
                        .to(device)
                        .expand(size, -1, -1)
                    )
                if isinstance(
                    generator,
                    (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
                ):
                    if windows.state_embedding is not None:
                        state = (
                            torch.from_numpy(
                                windows.state_embedding[origin_index : origin_index + 1]
                            )
                            .to(device)
                            .expand(size, -1)
                        )
                    else:
                        if windows.state_asset is None:
                            raise ValueError("Conditioned generator requires state histories")
                        assert windows.state_macro is not None
                        assert windows.state_adjacency is not None
                        state, _ = generator.encode_state(
                            torch.from_numpy(
                                windows.state_asset[origin_index : origin_index + 1]
                            )
                            .to(device)
                            .expand(size, -1, -1, -1),
                            torch.from_numpy(
                                windows.state_macro[origin_index : origin_index + 1]
                            )
                            .to(device)
                            .expand(size, -1, -1),
                            torch.from_numpy(
                                windows.state_adjacency[origin_index : origin_index + 1]
                            )
                            .to(device)
                            .expand(size, -1, -1),
                        )
                    if isinstance(generator, RiskGraphEWMABackboneScenarioGenerator):
                        assert baseline is not None
                        normalized = generator.decode(noise, regime, baseline, state)
                    else:
                        normalized = generator.decode(noise, regime, state)
                elif isinstance(
                    generator,
                    (
                        EWMABackboneTailGenerator,
                        EWMABackboneFactorScaleGenerator,
                        StableEWMABackboneFactorScaleGenerator,
                    ),
                ):
                    assert baseline is not None
                    normalized = generator(noise, regime, baseline)
                else:
                    assert isinstance(generator, RegimeTailGenerator)
                    normalized = generator(noise, regime)
                scale = torch.as_tensor(windows.scales[origin_index], device=device).view(1, 1, -1)
                parts.append((normalized * scale).cpu().numpy())
            result[origin_index] = np.concatenate(parts, axis=0)
    return result


def scenario_forecasts(
    generated: np.ndarray,
    windows: TailWindowSet,
    target_index: int,
    horizons: list[int],
    quantiles: list[float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    prediction = np.empty((len(windows.origins), len(horizons), len(quantiles)), dtype=np.float32)
    target = np.empty((len(windows.origins), len(horizons)), dtype=np.float32)
    rows = []
    for row in range(len(windows.origins)):
        for h_index, horizon in enumerate(horizons):
            scenario_return = generated[row, :, :horizon, target_index].sum(axis=1)
            prediction[row, h_index] = np.quantile(scenario_return, quantiles)
            target[row, h_index] = windows.actual_paths[row, :horizon, target_index].sum()
            record = {
                "date": str(windows.dates[row]),
                "origin_index": int(windows.origins[row]),
                "regime": int(windows.regimes[row]),
                "horizon": int(horizon),
                "target": float(target[row, h_index]),
            }
            record.update({f"q_{q:g}": float(prediction[row, h_index, index]) for index, q in enumerate(quantiles)})
            rows.append(record)
    return prediction, target, pd.DataFrame(rows)


def evaluate_scenario_forecasts(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    return evaluate_forecasts(prediction, target, quantiles, horizons)


def generate_regime_samples(
    generator: torch.nn.Module,
    windows: TailWindowSet,
    scenarios_per_regime: int,
    latent_dim: int,
    device: torch.device,
    noise_distribution: str,
    degrees_of_freedom: float,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray]:
    """Create matched real/generated scenario samples within each regime."""

    rng = np.random.default_rng(seed)
    real_parts: list[np.ndarray] = []
    generated_parts: list[np.ndarray] = []
    generator.eval()
    with torch.no_grad():
        for regime_value in np.unique(windows.regimes):
            indices = np.flatnonzero(windows.regimes == regime_value)
            if len(indices) == 0:
                continue
            selected = rng.choice(indices, size=int(scenarios_per_regime), replace=True)
            regime = torch.full(
                (int(scenarios_per_regime),),
                int(regime_value),
                dtype=torch.long,
                device=device,
            )
            noise = sample_noise(
                int(scenarios_per_regime),
                int(latent_dim),
                device,
                distribution=noise_distribution,
                degrees_of_freedom=degrees_of_freedom,
            )
            baseline = None
            if isinstance(
                generator,
                (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator, RiskGraphEWMABackboneScenarioGenerator),
            ):
                if windows.baseline_cholesky is None:
                    raise ValueError("EWMA backbone requires baseline Cholesky factors")
                baseline = torch.from_numpy(windows.baseline_cholesky[selected]).to(device)
            if isinstance(
                generator,
                (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
            ):
                if windows.state_embedding is not None:
                    state = torch.from_numpy(windows.state_embedding[selected]).to(device)
                else:
                    if windows.state_asset is None:
                        raise ValueError("Conditioned generator requires state histories")
                    assert windows.state_macro is not None
                    assert windows.state_adjacency is not None
                    state, _ = generator.encode_state(
                        torch.from_numpy(windows.state_asset[selected]).to(device),
                        torch.from_numpy(windows.state_macro[selected]).to(device),
                        torch.from_numpy(windows.state_adjacency[selected]).to(device),
                    )
                if isinstance(generator, RiskGraphEWMABackboneScenarioGenerator):
                    assert baseline is not None
                    normalized = generator.decode(noise, regime, baseline, state)
                else:
                    normalized = generator.decode(noise, regime, state)
            elif isinstance(
                    generator,
                    (
                        EWMABackboneTailGenerator,
                        EWMABackboneFactorScaleGenerator,
                        StableEWMABackboneFactorScaleGenerator,
                    ),
                ):
                assert baseline is not None
                normalized = generator(noise, regime, baseline)
            else:
                assert isinstance(generator, RegimeTailGenerator)
                normalized = generator(noise, regime)
            scale = torch.from_numpy(windows.scales[selected]).to(device)
            generated_parts.append((normalized * scale[:, None, :]).cpu().numpy())
            real_parts.append(windows.actual_paths[selected])
    if not real_parts:
        raise ValueError("No regimes available for matched scenario generation")
    return np.concatenate(real_parts), np.concatenate(generated_parts)


def riskgraph_teacher_quantiles_for_windows(
    generator: torch.nn.Module,
    windows: TailWindowSet,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray | None:
    """Return direct RiskGraph teacher quantiles embedded in a conditioned checkpoint."""

    if not isinstance(
        generator,
        (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
    ):
        return None
    if windows.teacher_quantiles is not None:
        return np.asarray(windows.teacher_quantiles, dtype=np.float32)
    if windows.state_asset is None:
        raise ValueError("Conditioned generator requires state histories")
    assert windows.state_macro is not None
    assert windows.state_adjacency is not None
    values: list[np.ndarray] = []
    generator.eval()
    with torch.no_grad():
        for start in range(0, len(windows.origins), int(batch_size)):
            end = min(start + int(batch_size), len(windows.origins))
            _, quantiles = generator.encode_state(
                torch.from_numpy(windows.state_asset[start:end]).to(device),
                torch.from_numpy(windows.state_macro[start:end]).to(device),
                torch.from_numpy(windows.state_adjacency[start:end]).to(device),
            )
            values.append(quantiles.cpu().numpy())
    return np.concatenate(values, axis=0)


def teacher_consistency_metrics(
    scenario_quantiles: np.ndarray,
    teacher_quantiles: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    scenario = np.asarray(scenario_quantiles, dtype=float)
    teacher = np.asarray(teacher_quantiles, dtype=float)
    if scenario.shape != teacher.shape:
        raise ValueError(
            f"Scenario and teacher quantiles must have equal shape, got {scenario.shape} and {teacher.shape}"
        )
    spread = np.maximum(np.abs(teacher[..., -1] - teacher[..., 0]), epsilon)
    absolute = np.abs(scenario - teacher)
    return {
        "teacher_quantile_mae": float(absolute.mean()),
        "teacher_quantile_normalized_mae": float(
            (absolute / spread[..., None]).mean()
        ),
        "teacher_median_mae": float(
            np.abs(
                scenario[..., scenario.shape[-1] // 2]
                - teacher[..., teacher.shape[-1] // 2]
            ).mean()
        ),
    }
