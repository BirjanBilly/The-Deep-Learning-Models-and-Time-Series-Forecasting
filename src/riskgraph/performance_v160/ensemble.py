from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.io import write_json
from riskgraph.performance_v140.calibration import (
    monotone_rearrange,
    pinball_loss_numpy,
)
from riskgraph.performance_v150.ensemble import (
    _align_baseline_to_model,
    _arrays_to_frame,
    _candidate_weights,
    _date_loss,
    _frame_to_arrays,
    _validate_alignment,
    _weighted_prediction,
    circular_block_bootstrap_lower_bound,
)
from riskgraph.tailrisk.evaluation import evaluate_scenario_forecasts


@dataclass(frozen=True)
class RegimeEnsembleGate:
    seed_weights: np.ndarray
    regime_residual_weights: np.ndarray
    spread_scale: float
    quantile_offsets: np.ndarray
    bootstrap_lower_bound: float
    episode_improvements: tuple[float, ...]
    regime_confirmation_improvements: tuple[float, ...]
    risk_calibration_delta: float
    accepted: bool
    safety_reason: str

    def export(self) -> dict[str, Any]:
        record = asdict(self)
        record["seed_weights"] = self.seed_weights.tolist()
        record["regime_residual_weights"] = self.regime_residual_weights.tolist()
        record["quantile_offsets"] = self.quantile_offsets.tolist()
        return record

    @property
    def fallback_to_baseline(self) -> bool:
        return not self.accepted


def _regime_array(frame: pd.DataFrame, dates: list[str]) -> np.ndarray:
    source = frame.copy()
    source["date"] = source["date"].astype(str)
    lookup = source.drop_duplicates("date").set_index("date")["regime"]
    missing = [value for value in dates if value not in lookup.index]
    if missing:
        raise ValueError(f"Regime features miss {len(missing)} forecast dates")
    return lookup.loc[dates].to_numpy(dtype=np.int64)


def _identity_gate(shape: tuple[int, int], reason: str, seed_count: int, regimes: int) -> RegimeEnsembleGate:
    return RegimeEnsembleGate(
        seed_weights=np.full(seed_count, 1.0 / seed_count, dtype=np.float32),
        regime_residual_weights=np.zeros(regimes, dtype=np.float32),
        spread_scale=1.0,
        quantile_offsets=np.zeros(shape, dtype=np.float32),
        bootstrap_lower_bound=0.0,
        episode_improvements=tuple(),
        regime_confirmation_improvements=tuple(),
        risk_calibration_delta=0.0,
        accepted=False,
        safety_reason=reason,
    )


def apply_regime_gate(
    raw_ensemble: np.ndarray,
    baseline: np.ndarray,
    regimes: np.ndarray,
    gate: RegimeEnsembleGate,
) -> np.ndarray:
    if not gate.accepted:
        return np.asarray(baseline, dtype=np.float32).copy()
    residual = gate.regime_residual_weights[np.asarray(regimes, dtype=np.int64)]
    candidate = baseline + residual[:, None, None] * (raw_ensemble - baseline)
    median_index = candidate.shape[-1] // 2
    median = candidate[..., median_index : median_index + 1]
    candidate = median + float(gate.spread_scale) * (candidate - median)
    candidate = candidate + gate.quantile_offsets[None, ...]
    return monotone_rearrange(candidate)


def _risk_calibration_error(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
) -> float:
    _, detail, _ = evaluate_scenario_forecasts(prediction, target, quantiles, horizons)
    values: list[float] = []
    for column, nominal in (("var_05_exception_rate", 0.05), ("coverage_90", 0.90), ("coverage_95", 0.95)):
        if column in detail:
            values.extend(abs(detail[column].dropna().to_numpy(float) - nominal).tolist())
    return float(np.mean(values)) if values else 0.0


def _episode_improvements(
    baseline_daily: np.ndarray,
    candidate_daily: np.ndarray,
    count: int,
) -> tuple[float, ...]:
    count = max(2, min(int(count), len(baseline_daily) // 15))
    boundaries = np.linspace(0, len(baseline_daily), count + 1).round().astype(int)
    output: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        base = baseline_daily[left:right]
        candidate = candidate_daily[left:right]
        if len(base) < 10:
            continue
        output.append(float((base.mean() - candidate.mean()) / max(abs(base.mean()), 1e-12)))
    return tuple(output)


def fit_regime_ensemble_gate(
    seed_predictions: list[np.ndarray],
    baseline: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    settings: dict[str, Any],
    seed: int = 160,
) -> RegimeEnsembleGate:
    models = np.stack([monotone_rearrange(value) for value in seed_predictions], axis=0)
    baseline = monotone_rearrange(baseline)
    truth = np.asarray(target, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    n = len(truth)
    regime_count = int(max(int(np.max(regimes)) + 1, int(settings.get("regime_count", 3))))
    identity = _identity_gate(baseline.shape[1:], "not_evaluated", len(models), regime_count)
    if n < 100:
        return RegimeEnsembleGate(**{**identity.__dict__, "safety_reason": "insufficient_validation_dates"})
    selection_end = max(40, int(round(n * float(settings["selection_fraction"]))))
    tuning_end = max(selection_end + 20, int(round(n * (float(settings["selection_fraction"]) + float(settings["tuning_fraction"])))))
    tuning_end = min(tuning_end, n - 25)
    selection = slice(0, selection_end)
    tuning = slice(selection_end, tuning_end)
    confirmation = slice(tuning_end, n)

    model_losses = np.asarray(
        [pinball_loss_numpy(model[selection], truth[selection], q) for model in models]
    )
    baseline_selection = pinball_loss_numpy(baseline[selection], truth[selection], q)
    best_seed_weights = np.full(len(models), 1.0 / len(models))
    best_global = baseline_selection
    for weights in _candidate_weights(len(models), model_losses):
        ensemble = _weighted_prediction(models[:, selection], weights)
        for residual in settings["residual_weights"]:
            candidate = baseline[selection] + float(residual) * (ensemble - baseline[selection])
            score = pinball_loss_numpy(candidate, truth[selection], q)
            if score < best_global - 1e-12:
                best_global = float(score)
                best_seed_weights = weights.copy()
    ensemble_all = _weighted_prediction(models, best_seed_weights)
    regime_weights = np.zeros(regime_count, dtype=np.float32)
    minimum_regime_samples = int(settings.get("minimum_regime_samples", 20))
    for regime_value in range(regime_count):
        mask = (regimes[selection] == regime_value)
        if int(mask.sum()) < minimum_regime_samples:
            continue
        best_score = pinball_loss_numpy(
            baseline[selection][mask], truth[selection][mask], q
        )
        for residual in settings["residual_weights"]:
            candidate = baseline[selection][mask] + float(residual) * (
                ensemble_all[selection][mask] - baseline[selection][mask]
            )
            score = pinball_loss_numpy(candidate, truth[selection][mask], q)
            if score < best_score - 1e-12:
                best_score = float(score)
                regime_weights[regime_value] = float(residual)
    selection_candidate = baseline[selection] + regime_weights[regimes[selection]][:, None, None] * (
        ensemble_all[selection] - baseline[selection]
    )
    selection_score = pinball_loss_numpy(selection_candidate, truth[selection], q)
    selection_improvement = (baseline_selection - selection_score) / max(abs(baseline_selection), 1e-12)
    if selection_improvement < float(settings["minimum_selection_improvement"]):
        return RegimeEnsembleGate(
            **{
                **identity.__dict__,
                "seed_weights": best_seed_weights.astype(np.float32),
                "safety_reason": "regime_selection_improvement_below_threshold",
            }
        )

    tuning_uncalibrated = baseline[tuning] + regime_weights[regimes[tuning]][:, None, None] * (
        ensemble_all[tuning] - baseline[tuning]
    )
    baseline_tuning = pinball_loss_numpy(baseline[tuning], truth[tuning], q)
    best_tuning = baseline_tuning
    best_scale = 1.0
    best_offsets = np.zeros(baseline.shape[1:], dtype=np.float32)
    median_index = int(np.argmin(np.abs(q - 0.5)))
    for scale in settings["spread_scales"]:
        median = tuning_uncalibrated[..., median_index : median_index + 1]
        spread = median + float(scale) * (tuning_uncalibrated - median)
        errors = truth[tuning, :, None] - spread
        offsets = np.empty(errors.shape[1:], dtype=np.float32)
        for h in range(errors.shape[1]):
            for qi, tau in enumerate(q):
                offsets[h, qi] = float(np.quantile(errors[:, h, qi], tau))
        for shrinkage in settings["offset_shrinkages"]:
            candidate_offsets = float(shrinkage) * offsets
            candidate = monotone_rearrange(spread + candidate_offsets[None, ...])
            score = pinball_loss_numpy(candidate, truth[tuning], q)
            if score < best_tuning - 1e-12:
                best_tuning = float(score)
                best_scale = float(scale)
                best_offsets = candidate_offsets.astype(np.float32)
    provisional = RegimeEnsembleGate(
        seed_weights=best_seed_weights.astype(np.float32),
        regime_residual_weights=regime_weights,
        spread_scale=best_scale,
        quantile_offsets=best_offsets,
        bootstrap_lower_bound=0.0,
        episode_improvements=tuple(),
        regime_confirmation_improvements=tuple(),
        risk_calibration_delta=0.0,
        accepted=True,
        safety_reason="pending_confirmation",
    )
    candidate_confirmation = apply_regime_gate(
        ensemble_all[confirmation], baseline[confirmation], regimes[confirmation], provisional
    )
    baseline_confirmation = pinball_loss_numpy(baseline[confirmation], truth[confirmation], q)
    selected_confirmation = pinball_loss_numpy(candidate_confirmation, truth[confirmation], q)
    confirmation_improvement = (baseline_confirmation - selected_confirmation) / max(abs(baseline_confirmation), 1e-12)
    baseline_daily = _date_loss(baseline[confirmation], truth[confirmation], q)
    candidate_daily = _date_loss(candidate_confirmation, truth[confirmation], q)
    lower = circular_block_bootstrap_lower_bound(
        baseline_daily - candidate_daily,
        int(settings["block_length"]),
        int(settings["bootstrap_repetitions"]),
        float(settings["one_sided_confidence"]),
        seed,
    )
    episodes = _episode_improvements(
        baseline_daily,
        candidate_daily,
        int(settings.get("confirmation_episodes", 4)),
    )
    positive_fraction = float(np.mean(np.asarray(episodes) >= 0.0)) if episodes else 0.0
    regime_improvements: list[float] = []
    for regime_value in range(regime_count):
        mask = regimes[confirmation] == regime_value
        if int(mask.sum()) < minimum_regime_samples:
            regime_improvements.append(0.0)
            continue
        base_loss = pinball_loss_numpy(baseline[confirmation][mask], truth[confirmation][mask], q)
        model_loss = pinball_loss_numpy(candidate_confirmation[mask], truth[confirmation][mask], q)
        regime_improvements.append((base_loss - model_loss) / max(abs(base_loss), 1e-12))
    baseline_risk = _risk_calibration_error(
        baseline[confirmation], truth[confirmation], quantiles, horizons
    )
    candidate_risk = _risk_calibration_error(
        candidate_confirmation, truth[confirmation], quantiles, horizons
    )
    risk_delta = candidate_risk - baseline_risk
    accepted = (
        confirmation_improvement >= float(settings["minimum_confirmation_improvement"])
        and lower > 0.0
        and positive_fraction >= float(settings["minimum_positive_episode_fraction"])
        and (min(episodes) if episodes else -np.inf) >= float(settings["maximum_episode_degradation"])
        and min(regime_improvements) >= float(settings["maximum_regime_degradation"])
        and risk_delta <= float(settings["maximum_risk_calibration_degradation"])
    )
    if not accepted:
        if lower <= 0.0:
            reason = "regime_bootstrap_lower_bound_nonpositive"
        elif risk_delta > float(settings["maximum_risk_calibration_degradation"]):
            reason = "regime_risk_calibration_rejected"
        else:
            reason = "regime_episode_or_confirmation_rejected"
        return RegimeEnsembleGate(
            seed_weights=best_seed_weights.astype(np.float32),
            regime_residual_weights=np.zeros(regime_count, dtype=np.float32),
            spread_scale=1.0,
            quantile_offsets=np.zeros(baseline.shape[1:], dtype=np.float32),
            bootstrap_lower_bound=float(lower),
            episode_improvements=episodes,
            regime_confirmation_improvements=tuple(float(value) for value in regime_improvements),
            risk_calibration_delta=float(risk_delta),
            accepted=False,
            safety_reason=reason,
        )
    return RegimeEnsembleGate(
        seed_weights=best_seed_weights.astype(np.float32),
        regime_residual_weights=regime_weights,
        spread_scale=best_scale,
        quantile_offsets=best_offsets,
        bootstrap_lower_bound=float(lower),
        episode_improvements=episodes,
        regime_confirmation_improvements=tuple(float(value) for value in regime_improvements),
        risk_calibration_delta=float(risk_delta),
        accepted=True,
        safety_reason="regime_multi_episode_bootstrap_confirmed",
    )


def evaluate_regime_seed_ensemble(
    seed_directories: list[str | Path],
    output_dir: str | Path,
    horizons: list[int],
    settings: dict[str, Any],
    family: str,
    fold: str,
    variant: str,
    statistical_directory: str | Path,
) -> Path:
    directories = [Path(value) for value in seed_directories]
    statistical = Path(statistical_directory)
    validation_raw = [pd.read_csv(path / "validation_raw_predictions.csv") for path in directories]
    test_raw = [pd.read_csv(path / "test_raw_predictions.csv") for path in directories]
    validation_source = pd.read_csv(statistical / "validation_baseline_predictions.csv")
    test_source = pd.read_csv(statistical / "test_baseline_predictions.csv")
    validation_baseline = [_align_baseline_to_model(validation_source, model) for model in validation_raw]
    test_baseline = [_align_baseline_to_model(test_source, model) for model in test_raw]
    _validate_alignment([*validation_raw, *validation_baseline])
    _validate_alignment([*test_raw, *test_baseline])
    validation_models = [_frame_to_arrays(frame)[0] for frame in validation_raw]
    validation_base, validation_target, quantiles = _frame_to_arrays(validation_baseline[0])
    validation_dates = sorted(validation_raw[0]["date"].astype(str).unique().tolist())
    test_dates = sorted(test_raw[0]["date"].astype(str).unique().tolist())
    regime_validation_frame = pd.read_csv(statistical / "validation_regime_features.csv")
    regime_test_frame = pd.read_csv(statistical / "test_regime_features.csv")
    validation_regimes = _regime_array(regime_validation_frame, validation_dates)
    test_regimes = _regime_array(regime_test_frame, test_dates)
    gate = fit_regime_ensemble_gate(
        validation_models,
        validation_base,
        validation_target,
        validation_regimes,
        quantiles,
        horizons,
        settings,
    )
    test_models = np.stack([_frame_to_arrays(frame)[0] for frame in test_raw], axis=0)
    test_base, test_target, _ = _frame_to_arrays(test_baseline[0])
    raw_ensemble = _weighted_prediction(test_models, gate.seed_weights)
    final = apply_regime_gate(raw_ensemble, test_base, test_regimes, gate)
    metrics, detail, var = evaluate_scenario_forecasts(final, test_target, quantiles, horizons)
    raw_metrics, raw_detail, raw_var = evaluate_scenario_forecasts(raw_ensemble, test_target, quantiles, horizons)
    baseline_metrics, baseline_detail, baseline_var = evaluate_scenario_forecasts(test_base, test_target, quantiles, horizons)
    improvement = 100.0 * (baseline_metrics["mean_pinball"] - metrics["mean_pinball"]) / max(abs(baseline_metrics["mean_pinball"]), 1e-12)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _arrays_to_frame(test_raw[0], final).to_csv(output / "predictions.csv", index=False)
    _arrays_to_frame(test_raw[0], raw_ensemble).to_csv(output / "raw_ensemble_predictions.csv", index=False)
    _arrays_to_frame(test_baseline[0], test_base).to_csv(output / "baseline_predictions.csv", index=False)
    pd.concat(
        [
            detail.assign(model="regime_gated"),
            raw_detail.assign(model="raw_seed_ensemble"),
            baseline_detail.assign(model="statistical_expert_ensemble"),
        ],
        ignore_index=True,
    ).to_csv(output / "metrics_comparison_by_horizon.csv", index=False)
    pd.concat(
        [
            var.assign(model="regime_gated"),
            raw_var.assign(model="raw_seed_ensemble"),
            baseline_var.assign(model="statistical_expert_ensemble"),
        ],
        ignore_index=True,
    ).to_csv(output / "var_backtests.csv", index=False)
    validation_ensemble = _weighted_prediction(np.stack(validation_models), gate.seed_weights)
    validation_final = apply_regime_gate(validation_ensemble, validation_base, validation_regimes, gate)
    _arrays_to_frame(validation_raw[0], validation_final).to_csv(output / "validation_predictions.csv", index=False)
    _arrays_to_frame(validation_baseline[0], validation_base).to_csv(output / "validation_baseline_predictions.csv", index=False)
    daily = pd.DataFrame(
        {
            "date": test_dates,
            "regime": test_regimes,
            "regime_gated_loss": _date_loss(final, test_target, np.asarray(quantiles)),
            "raw_ensemble_loss": _date_loss(raw_ensemble, test_target, np.asarray(quantiles)),
            "statistical_baseline_loss": _date_loss(test_base, test_target, np.asarray(quantiles)),
        }
    )
    daily["gated_improvement"] = daily["statistical_baseline_loss"] - daily["regime_gated_loss"]
    daily.to_csv(output / "daily_loss_comparison.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "raw_model_metrics.json", raw_metrics)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    write_json(output / "ensemble_gate.json", gate.export())
    write_json(
        output / "evaluation_summary.json",
        {
            "family": family,
            "fold": fold,
            "variant": variant,
            "mean_pinball": float(metrics["mean_pinball"]),
            "raw_ensemble_pinball": float(raw_metrics["mean_pinball"]),
            "statistical_baseline_pinball": float(baseline_metrics["mean_pinball"]),
            "improvement_vs_statistical_pct": float(improvement),
            "fallback_to_baseline": bool(gate.fallback_to_baseline),
            "seed_weights": gate.seed_weights.tolist(),
            "regime_residual_weights": gate.regime_residual_weights.tolist(),
            "bootstrap_lower_bound": float(gate.bootstrap_lower_bound),
            "risk_calibration_delta": float(gate.risk_calibration_delta),
            "safety_reason": gate.safety_reason,
            "test_origins": int(len(test_target)),
        },
    )
    return output
