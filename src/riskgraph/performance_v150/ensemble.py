from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.io import write_json
from riskgraph.performance_v140.calibration import (
    ForecastGate,
    apply_forecast_gate,
    monotone_rearrange,
    pinball_loss_numpy,
)
from riskgraph.tailrisk.evaluation import evaluate_scenario_forecasts


@dataclass(frozen=True)
class EnsembleGate:
    seed_weights: np.ndarray
    forecast_gate: ForecastGate
    bootstrap_lower_bound: float
    bootstrap_confidence: float
    bootstrap_repetitions: int
    block_length: int
    subblock_improvements: tuple[float, float]
    accepted: bool
    safety_reason: str

    def export(self) -> dict[str, Any]:
        record = asdict(self)
        record["seed_weights"] = self.seed_weights.tolist()
        record["forecast_gate"] = self.forecast_gate.export()
        return record


def _quantile_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("q_")],
        key=lambda name: float(name[2:]),
    )


def _frame_to_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[float]]:
    frame = frame.sort_values(["date", "horizon"]).reset_index(drop=True)
    columns = _quantile_columns(frame)
    prediction = frame[columns].to_numpy(dtype=np.float32)
    target = frame["target"].to_numpy(dtype=np.float32)
    horizons = sorted(frame["horizon"].unique().astype(int).tolist())
    dates = frame["date"].astype(str).drop_duplicates().tolist()
    expected = len(dates) * len(horizons)
    if len(frame) != expected:
        raise ValueError("Prediction frame is not a complete date-by-horizon grid")
    prediction = prediction.reshape(len(dates), len(horizons), len(columns))
    target = target.reshape(len(dates), len(horizons))
    return prediction, target, [float(column[2:]) for column in columns]


def _arrays_to_frame(template: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    output = template.copy()
    columns = _quantile_columns(output)
    output[columns] = prediction.reshape(len(output), len(columns))
    return output


def _validate_alignment(frames: list[pd.DataFrame]) -> None:
    if not frames:
        raise ValueError("No prediction frames supplied")
    key = (
        frames[0][["date", "horizon", "target"]]
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    for frame in frames[1:]:
        other = (
            frame[["date", "horizon", "target"]]
            .sort_values(["date", "horizon"])
            .reset_index(drop=True)
        )
        if not key.equals(other):
            raise ValueError("Seed predictions are not aligned on identical origins")



def _align_baseline_to_model(
    baseline: pd.DataFrame, model: pd.DataFrame
) -> pd.DataFrame:
    keys = model[["date", "horizon"]].copy()
    keys["date"] = keys["date"].astype(str)
    source = baseline.copy()
    source["date"] = source["date"].astype(str)
    aligned = keys.merge(source, on=["date", "horizon"], how="left", validate="one_to_one")
    required = ["target", *_quantile_columns(source)]
    if aligned[required].isna().any().any():
        raise ValueError("Canonical baseline does not cover every model forecast origin")
    model_sorted = model.sort_values(["date", "horizon"]).reset_index(drop=True)
    aligned = aligned.sort_values(["date", "horizon"]).reset_index(drop=True)
    if not np.allclose(
        aligned["target"].to_numpy(float),
        model_sorted["target"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Canonical baseline target mismatch")
    return aligned

def _candidate_weights(model_count: int, model_losses: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    candidates.append(np.full(model_count, 1.0 / model_count, dtype=np.float64))
    for index in range(model_count):
        vector = np.zeros(model_count, dtype=np.float64)
        vector[index] = 1.0
        candidates.append(vector)
    if model_count >= 2:
        for left in range(model_count):
            for right in range(left + 1, model_count):
                vector = np.zeros(model_count, dtype=np.float64)
                vector[left] = 0.5
                vector[right] = 0.5
                candidates.append(vector)
    centered = model_losses - np.min(model_losses)
    for temperature in (0.25, 0.5, 1.0, 2.0):
        scale = max(float(np.std(model_losses)), 1e-8) * temperature
        logits = -centered / scale
        weights = np.exp(logits - np.max(logits))
        candidates.append(weights / weights.sum())
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.allclose(candidate, existing, atol=1e-10) for existing in unique):
            unique.append(candidate)
    return unique


def _weighted_prediction(models: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, models, axes=(0, 0)).astype(np.float32)


def _date_loss(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    error = target[..., None] - prediction
    loss = np.maximum(quantiles[None, None, :] * error, (quantiles[None, None, :] - 1.0) * error)
    return loss.mean(axis=(1, 2))


def circular_block_bootstrap_lower_bound(
    improvement: np.ndarray,
    block_length: int,
    repetitions: int,
    confidence: float,
    seed: int,
) -> float:
    values = np.asarray(improvement, dtype=np.float64)
    if len(values) < max(20, block_length * 2):
        return float("-inf")
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_length), len(values)))
    blocks = int(np.ceil(len(values) / block))
    means = np.empty(int(repetitions), dtype=np.float64)
    offsets = np.arange(block)
    for iteration in range(int(repetitions)):
        starts = rng.integers(0, len(values), size=blocks)
        indices = np.concatenate([(start + offsets) % len(values) for start in starts])[
            : len(values)
        ]
        means[iteration] = values[indices].mean()
    alpha = 1.0 - float(confidence)
    return float(np.quantile(means, alpha))


def _identity_forecast_gate(shape: tuple[int, int], reason: str) -> ForecastGate:
    return ForecastGate(
        residual_weight=0.0,
        spread_scale=1.0,
        quantile_offsets=np.zeros(shape, dtype=np.float32),
        online_learning_rate=0.0,
        online_clip=2.0,
        baseline_validation_pinball=0.0,
        selected_validation_pinball=0.0,
        relative_improvement=0.0,
        fallback_to_baseline=True,
        selection_samples=0,
        calibration_samples=0,
        confirmation_samples=0,
        baseline_selection_pinball=0.0,
        selected_selection_pinball=0.0,
        selection_relative_improvement=0.0,
        baseline_calibration_pinball=0.0,
        selected_calibration_pinball=0.0,
        calibration_relative_improvement=0.0,
        baseline_confirmation_pinball=0.0,
        selected_confirmation_pinball=0.0,
        confirmation_relative_improvement=0.0,
        safety_reason=reason,
    )


def fit_ensemble_gate(
    seed_predictions: list[np.ndarray],
    baseline: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    settings: dict[str, Any],
    seed: int = 150,
) -> EnsembleGate:
    models = np.stack([monotone_rearrange(value) for value in seed_predictions], axis=0)
    baseline = monotone_rearrange(baseline)
    truth = np.asarray(target, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    n = len(truth)
    if n < 80:
        raise ValueError("Ensemble gate requires at least 80 chronological validation dates")
    selection_end = max(30, int(round(n * float(settings["selection_fraction"]))))
    tuning_end = max(selection_end + 20, int(round(n * (float(settings["selection_fraction"]) + float(settings["tuning_fraction"])))))
    tuning_end = min(tuning_end, n - 20)
    selection = slice(0, selection_end)
    tuning = slice(selection_end, tuning_end)
    confirmation = slice(tuning_end, n)

    model_losses = np.asarray(
        [pinball_loss_numpy(model[selection], truth[selection], q) for model in models],
        dtype=np.float64,
    )
    best_weights = np.full(len(models), 1.0 / len(models))
    best_residual = 0.0
    baseline_selection = pinball_loss_numpy(baseline[selection], truth[selection], q)
    best_selection = baseline_selection
    for weights in _candidate_weights(len(models), model_losses):
        ensemble = _weighted_prediction(models[:, selection], weights)
        for residual in settings["residual_weights"]:
            candidate = baseline[selection] + float(residual) * (ensemble - baseline[selection])
            score = pinball_loss_numpy(candidate, truth[selection], q)
            if score < best_selection - 1e-12:
                best_selection = float(score)
                best_weights = weights.copy()
                best_residual = float(residual)
    selection_improvement = (baseline_selection - best_selection) / max(abs(baseline_selection), 1e-12)
    identity = _identity_forecast_gate(baseline.shape[1:], "ensemble_selection_rejected")
    if selection_improvement < float(settings["minimum_selection_improvement"]):
        return EnsembleGate(
            seed_weights=best_weights.astype(np.float32),
            forecast_gate=identity,
            bootstrap_lower_bound=0.0,
            bootstrap_confidence=float(settings["one_sided_confidence"]),
            bootstrap_repetitions=int(settings["bootstrap_repetitions"]),
            block_length=int(settings["block_length"]),
            subblock_improvements=(0.0, 0.0),
            accepted=False,
            safety_reason="ensemble_selection_improvement_below_threshold",
        )

    ensemble_all = _weighted_prediction(models, best_weights)
    uncalibrated_tuning = baseline[tuning] + best_residual * (
        ensemble_all[tuning] - baseline[tuning]
    )
    baseline_tuning = pinball_loss_numpy(baseline[tuning], truth[tuning], q)
    best_tuning = baseline_tuning
    best_scale = 1.0
    best_offsets = np.zeros(baseline.shape[1:], dtype=np.float32)
    median_index = int(np.argmin(np.abs(q - 0.5)))
    for scale in settings["spread_scales"]:
        median = uncalibrated_tuning[..., median_index : median_index + 1]
        spread = median + float(scale) * (uncalibrated_tuning - median)
        errors = truth[tuning, :, None] - spread
        raw_offsets = np.empty((errors.shape[1], errors.shape[2]), dtype=np.float32)
        for horizon_index in range(errors.shape[1]):
            for quantile_index, tau in enumerate(q):
                raw_offsets[horizon_index, quantile_index] = float(
                    np.quantile(errors[:, horizon_index, quantile_index], tau)
                )
        for shrinkage in settings["offset_shrinkages"]:
            offsets = float(shrinkage) * raw_offsets
            candidate = monotone_rearrange(spread + offsets[None, ...])
            score = pinball_loss_numpy(candidate, truth[tuning], q)
            if score < best_tuning - 1e-12:
                best_tuning = float(score)
                best_scale = float(scale)
                best_offsets = offsets.astype(np.float32)
    if best_tuning > baseline_tuning + 1e-12:
        return EnsembleGate(
            seed_weights=best_weights.astype(np.float32),
            forecast_gate=identity,
            bootstrap_lower_bound=0.0,
            bootstrap_confidence=float(settings["one_sided_confidence"]),
            bootstrap_repetitions=int(settings["bootstrap_repetitions"]),
            block_length=int(settings["block_length"]),
            subblock_improvements=(0.0, 0.0),
            accepted=False,
            safety_reason="ensemble_calibration_worse_than_baseline",
        )

    gate = ForecastGate(
        residual_weight=best_residual,
        spread_scale=best_scale,
        quantile_offsets=best_offsets,
        online_learning_rate=0.0,
        online_clip=2.0,
        baseline_validation_pinball=0.0,
        selected_validation_pinball=0.0,
        relative_improvement=0.0,
        fallback_to_baseline=False,
        selection_samples=selection_end,
        calibration_samples=tuning_end - selection_end,
        confirmation_samples=n - tuning_end,
        baseline_selection_pinball=baseline_selection,
        selected_selection_pinball=best_selection,
        selection_relative_improvement=selection_improvement,
        baseline_calibration_pinball=baseline_tuning,
        selected_calibration_pinball=best_tuning,
        calibration_relative_improvement=(baseline_tuning - best_tuning) / max(abs(baseline_tuning), 1e-12),
        baseline_confirmation_pinball=0.0,
        selected_confirmation_pinball=0.0,
        confirmation_relative_improvement=0.0,
        safety_reason="pending_bootstrap_confirmation",
    )
    confirmation_candidate = apply_forecast_gate(
        ensemble_all[confirmation],
        baseline[confirmation],
        q,
        gate,
        observed_targets=None,
        horizons=horizons,
    )
    baseline_confirmation = pinball_loss_numpy(
        baseline[confirmation], truth[confirmation], q
    )
    selected_confirmation = pinball_loss_numpy(
        confirmation_candidate, truth[confirmation], q
    )
    confirmation_improvement = (
        baseline_confirmation - selected_confirmation
    ) / max(abs(baseline_confirmation), 1e-12)
    baseline_daily = _date_loss(baseline[confirmation], truth[confirmation], q)
    candidate_daily = _date_loss(confirmation_candidate, truth[confirmation], q)
    daily_improvement = baseline_daily - candidate_daily
    lower = circular_block_bootstrap_lower_bound(
        daily_improvement,
        int(settings["block_length"]),
        int(settings["bootstrap_repetitions"]),
        float(settings["one_sided_confidence"]),
        seed,
    )
    midpoint = len(daily_improvement) // 2
    subblocks: list[float] = []
    for part in (daily_improvement[:midpoint], daily_improvement[midpoint:]):
        base_part = baseline_daily[:midpoint] if len(subblocks) == 0 else baseline_daily[midpoint:]
        subblocks.append(float(part.mean() / max(abs(base_part.mean()), 1e-12)))
    accepted = (
        confirmation_improvement >= float(settings["minimum_confirmation_improvement"])
        and lower > 0.0
        and min(subblocks) >= float(settings["minimum_subblock_improvement"])
    )
    if not accepted:
        reason = (
            "ensemble_bootstrap_lower_bound_nonpositive"
            if lower <= 0.0
            else "ensemble_confirmation_or_subblock_rejected"
        )
        return EnsembleGate(
            seed_weights=best_weights.astype(np.float32),
            forecast_gate=identity,
            bootstrap_lower_bound=lower,
            bootstrap_confidence=float(settings["one_sided_confidence"]),
            bootstrap_repetitions=int(settings["bootstrap_repetitions"]),
            block_length=int(settings["block_length"]),
            subblock_improvements=(float(subblocks[0]), float(subblocks[1])),
            accepted=False,
            safety_reason=reason,
        )
    confirmed_gate = ForecastGate(
        **{
            **gate.__dict__,
            "baseline_validation_pinball": float(baseline_confirmation),
            "selected_validation_pinball": float(selected_confirmation),
            "relative_improvement": float(confirmation_improvement),
            "baseline_confirmation_pinball": float(baseline_confirmation),
            "selected_confirmation_pinball": float(selected_confirmation),
            "confirmation_relative_improvement": float(confirmation_improvement),
            "safety_reason": "bootstrap_confirmed_ensemble",
        }
    )
    return EnsembleGate(
        seed_weights=best_weights.astype(np.float32),
        forecast_gate=confirmed_gate,
        bootstrap_lower_bound=lower,
        bootstrap_confidence=float(settings["one_sided_confidence"]),
        bootstrap_repetitions=int(settings["bootstrap_repetitions"]),
        block_length=int(settings["block_length"]),
        subblock_improvements=(float(subblocks[0]), float(subblocks[1])),
        accepted=True,
        safety_reason="bootstrap_confirmed_ensemble",
    )


def evaluate_seed_ensemble(
    seed_directories: list[str | Path],
    output_dir: str | Path,
    horizons: list[int],
    settings: dict[str, Any],
    family: str,
    fold: str,
    variant: str,
    canonical_baseline_directories: list[str | Path] | None = None,
) -> Path:
    directories = [Path(value) for value in seed_directories]
    validation_raw = [pd.read_csv(path / "validation_raw_predictions.csv") for path in directories]
    baseline_directories = (
        [Path(value) for value in canonical_baseline_directories]
        if canonical_baseline_directories is not None
        else directories
    )
    if len(baseline_directories) != len(directories):
        raise ValueError("Canonical baseline directory count must match seed count")
    validation_baseline_source = [
        pd.read_csv(path / "validation_baseline_predictions.csv")
        for path in baseline_directories
    ]
    test_raw = [pd.read_csv(path / "test_raw_predictions.csv") for path in directories]
    test_baseline_source = [
        pd.read_csv(path / "test_baseline_predictions.csv")
        for path in baseline_directories
    ]
    validation_baseline = [
        _align_baseline_to_model(base, model)
        for base, model in zip(validation_baseline_source, validation_raw, strict=True)
    ]
    test_baseline = [
        _align_baseline_to_model(base, model)
        for base, model in zip(test_baseline_source, test_raw, strict=True)
    ]
    _validate_alignment([*validation_raw, *validation_baseline])
    _validate_alignment([*test_raw, *test_baseline])
    validation_models = [_frame_to_arrays(frame)[0] for frame in validation_raw]
    validation_base, validation_target, quantiles = _frame_to_arrays(validation_baseline[0])
    gate = fit_ensemble_gate(
        validation_models,
        validation_base,
        validation_target,
        quantiles,
        horizons,
        settings,
    )
    test_models = np.stack([_frame_to_arrays(frame)[0] for frame in test_raw], axis=0)
    test_base, test_target, _ = _frame_to_arrays(test_baseline[0])
    raw_ensemble = _weighted_prediction(test_models, gate.seed_weights)
    final = apply_forecast_gate(
        raw_ensemble,
        test_base,
        quantiles,
        gate.forecast_gate,
        observed_targets=None,
        horizons=horizons,
    )
    metrics, detail, var = evaluate_scenario_forecasts(final, test_target, quantiles, horizons)
    raw_metrics, raw_detail, raw_var = evaluate_scenario_forecasts(
        raw_ensemble, test_target, quantiles, horizons
    )
    baseline_metrics, baseline_detail, baseline_var = evaluate_scenario_forecasts(
        test_base, test_target, quantiles, horizons
    )
    improvement = 100.0 * (
        baseline_metrics["mean_pinball"] - metrics["mean_pinball"]
    ) / max(abs(baseline_metrics["mean_pinball"]), 1e-12)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _arrays_to_frame(test_raw[0], final).to_csv(output / "predictions.csv", index=False)
    _arrays_to_frame(test_raw[0], raw_ensemble).to_csv(
        output / "raw_ensemble_predictions.csv", index=False
    )
    _arrays_to_frame(test_baseline[0], test_base).to_csv(
        output / "baseline_predictions.csv", index=False
    )
    pd.concat(
        [
            detail.assign(model="ensemble_gated"),
            raw_detail.assign(model="raw_seed_ensemble"),
            baseline_detail.assign(model="ewma_student_t"),
        ],
        ignore_index=True,
    ).to_csv(output / "metrics_comparison_by_horizon.csv", index=False)
    pd.concat(
        [
            var.assign(model="ensemble_gated"),
            raw_var.assign(model="raw_seed_ensemble"),
            baseline_var.assign(model="ewma_student_t"),
        ],
        ignore_index=True,
    ).to_csv(output / "var_backtests.csv", index=False)
    validation_ensemble = _weighted_prediction(
        np.stack(validation_models, axis=0), gate.seed_weights
    )
    validation_final = apply_forecast_gate(
        validation_ensemble,
        validation_base,
        quantiles,
        gate.forecast_gate,
        observed_targets=None,
        horizons=horizons,
    )
    _arrays_to_frame(validation_raw[0], validation_final).to_csv(
        output / "validation_predictions.csv", index=False
    )
    _arrays_to_frame(validation_baseline[0], validation_base).to_csv(
        output / "validation_baseline_predictions.csv", index=False
    )
    daily = pd.DataFrame(
        {
            "date": sorted(test_raw[0]["date"].astype(str).unique().tolist()),
            "ensemble_gated_loss": _date_loss(final, test_target, np.asarray(quantiles)),
            "raw_ensemble_loss": _date_loss(
                raw_ensemble, test_target, np.asarray(quantiles)
            ),
            "ewma_student_t_loss": _date_loss(
                test_base, test_target, np.asarray(quantiles)
            ),
        }
    )
    daily["gated_improvement"] = (
        daily["ewma_student_t_loss"] - daily["ensemble_gated_loss"]
    )
    daily["raw_improvement"] = (
        daily["ewma_student_t_loss"] - daily["raw_ensemble_loss"]
    )
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
            "ewma_student_t_pinball": float(baseline_metrics["mean_pinball"]),
            "improvement_vs_ewma_pct": float(improvement),
            "fallback_to_baseline": bool(gate.forecast_gate.fallback_to_baseline),
            "seed_weights": gate.seed_weights.tolist(),
            "bootstrap_lower_bound": float(gate.bootstrap_lower_bound),
            "safety_reason": gate.safety_reason,
            "test_origins": int(len(test_target)),
        },
    )
    return output
