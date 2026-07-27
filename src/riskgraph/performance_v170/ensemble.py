from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.io import write_json
from riskgraph.performance_v140.calibration import monotone_rearrange, pinball_loss_numpy
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
class ProbabilisticEnsembleGate:
    seed_weights: np.ndarray
    regime_residual_weights: np.ndarray
    bootstrap_lower_bound: float
    tuning_improvement: float
    confirmation_improvement: float
    episode_improvements: tuple[float, ...]
    regime_confirmation_improvements: tuple[float, ...]
    risk_calibration_delta: float
    effective_seed_count: float
    accepted: bool
    safety_reason: str

    def export(self) -> dict[str, Any]:
        record = asdict(self)
        record["seed_weights"] = self.seed_weights.tolist()
        record["regime_residual_weights"] = self.regime_residual_weights.tolist()
        return record

    @property
    def fallback_to_baseline(self) -> bool:
        return not self.accepted


def _identity_gate(
    seed_count: int,
    regime_count: int,
    reason: str,
) -> ProbabilisticEnsembleGate:
    weights = np.full(seed_count, 1.0 / seed_count, dtype=np.float32)
    return ProbabilisticEnsembleGate(
        seed_weights=weights,
        regime_residual_weights=np.zeros(regime_count, dtype=np.float32),
        bootstrap_lower_bound=0.0,
        tuning_improvement=0.0,
        confirmation_improvement=0.0,
        episode_improvements=tuple(),
        regime_confirmation_improvements=tuple(),
        risk_calibration_delta=0.0,
        effective_seed_count=float(seed_count),
        accepted=False,
        safety_reason=reason,
    )


def _probability_arrays(frame: pd.DataFrame, dates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    source = frame.copy()
    source["date"] = source["date"].astype(str)
    source = source.drop_duplicates("date").set_index("date")
    missing = [value for value in dates if value not in source.index]
    if missing:
        raise ValueError(f"Probabilistic regime features miss {len(missing)} forecast dates")
    probabilities = source.loc[dates, ["p_calm", "p_transition", "p_stress"]].to_numpy(float)
    confidence = source.loc[dates, "regime_confidence"].to_numpy(float)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities.astype(np.float32), confidence.astype(np.float32)


def effective_seed_count(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    return float(1.0 / max(np.square(values).sum(), 1e-12))


def _regularize_seed_weights(
    weights: np.ndarray,
    shrinkage: float,
    maximum_weight: float,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    values = np.clip(values, 0.0, None)
    if values.sum() <= 0.0:
        values = np.ones_like(values)
    values /= values.sum()
    equal = np.full_like(values, 1.0 / len(values))
    values = (1.0 - shrinkage) * values + shrinkage * equal
    if maximum_weight < 1.0:
        # Alternating projection onto the simplex and box constraint.
        for _ in range(20):
            excess = np.maximum(values - maximum_weight, 0.0)
            if excess.sum() <= 1e-12:
                break
            values = np.minimum(values, maximum_weight)
            room = np.maximum(maximum_weight - values, 0.0)
            room_sum = room.sum()
            if room_sum <= 1e-12:
                break
            values += excess.sum() * room / room_sum
            values /= values.sum()
    return values / values.sum()


def _soft_strength(
    probabilities: np.ndarray,
    confidence: np.ndarray,
    regime_weights: np.ndarray,
    confidence_floor: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    weights = np.asarray(regime_weights, dtype=np.float64)
    base = probabilities @ weights
    if confidence_floor >= 1.0:
        return np.zeros_like(base)
    reliability = np.clip(
        (confidence - confidence_floor) / max(1.0 - confidence_floor, 1e-12),
        0.0,
        1.0,
    )
    return base * reliability


def apply_probabilistic_gate(
    raw_ensemble: np.ndarray,
    baseline: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    gate: ProbabilisticEnsembleGate,
    confidence_floor: float,
) -> np.ndarray:
    if not gate.accepted:
        return np.asarray(baseline, dtype=np.float32).copy()
    strength = _soft_strength(
        probabilities,
        confidence,
        gate.regime_residual_weights,
        confidence_floor,
    )
    candidate = np.asarray(baseline, dtype=np.float64) + strength[:, None, None] * (
        np.asarray(raw_ensemble, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    )
    return monotone_rearrange(candidate).astype(np.float32)


def _episode_improvements(
    baseline_daily: np.ndarray,
    candidate_daily: np.ndarray,
    count: int,
) -> tuple[float, ...]:
    count = max(2, min(int(count), max(2, len(baseline_daily) // 15)))
    boundaries = np.linspace(0, len(baseline_daily), count + 1).round().astype(int)
    output: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        if right - left < 10:
            continue
        base = baseline_daily[left:right]
        candidate = candidate_daily[left:right]
        output.append(
            float((base.mean() - candidate.mean()) / max(abs(base.mean()), 1e-12))
        )
    return tuple(output)


def _risk_calibration_error(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
) -> float:
    _, detail, _ = evaluate_scenario_forecasts(prediction, target, quantiles, horizons)
    values: list[float] = []
    for column, nominal in (
        ("var_05_exception_rate", 0.05),
        ("coverage_90", 0.90),
        ("coverage_95", 0.95),
    ):
        if column in detail:
            values.extend(abs(detail[column].dropna().to_numpy(float) - nominal).tolist())
    return float(np.mean(values)) if values else 0.0


def _family_caps(settings: dict[str, Any], family: str, regime_count: int) -> np.ndarray:
    mapping = settings.get("family_residual_caps", {})
    values = mapping.get(family, mapping.get("default", [0.2, 0.35, 0.5]))
    caps = np.asarray(values, dtype=np.float64)
    if caps.shape != (regime_count,):
        raise ValueError(
            f"family_residual_caps[{family!r}] must contain {regime_count} values"
        )
    return np.clip(caps, 0.0, 1.0)


def _residual_grid(settings: dict[str, Any], caps: np.ndarray) -> list[tuple[float, ...]]:
    values = [float(value) for value in settings.get("residual_weights", [0.0, 0.1, 0.2, 0.35, 0.5])]
    per_regime = [sorted({value for value in values if value <= cap + 1e-12} | {0.0}) for cap in caps]
    return list(product(*per_regime))


def fit_probabilistic_ensemble_gate(
    seed_predictions: list[np.ndarray],
    baseline: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
    settings: dict[str, Any],
    family: str,
    seed: int = 170,
) -> ProbabilisticEnsembleGate:
    models = np.stack([monotone_rearrange(value) for value in seed_predictions], axis=0)
    baseline = monotone_rearrange(baseline)
    truth = np.asarray(target, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = len(truth)
    regime_count = probabilities.shape[1]
    identity = _identity_gate(len(models), regime_count, "not_evaluated")
    if n < 120:
        return ProbabilisticEnsembleGate(
            **{**identity.__dict__, "safety_reason": "insufficient_validation_dates"}
        )

    selection_end = max(45, int(round(n * float(settings.get("selection_fraction", 0.45)))))
    tuning_end = max(
        selection_end + 25,
        int(
            round(
                n
                * (
                    float(settings.get("selection_fraction", 0.45))
                    + float(settings.get("tuning_fraction", 0.25))
                )
            )
        ),
    )
    tuning_end = min(tuning_end, n - 30)
    selection = slice(0, selection_end)
    tuning = slice(selection_end, tuning_end)
    confirmation = slice(tuning_end, n)

    model_losses = np.asarray(
        [pinball_loss_numpy(model[selection], truth[selection], q) for model in models],
        dtype=np.float64,
    )
    seed_candidates = _candidate_weights(len(models), model_losses)
    seed_shrinkage = float(settings.get("seed_weight_shrinkage", 0.15))
    max_seed_weight = float(settings.get("maximum_seed_weight", 0.80))
    concentration_penalty = float(settings.get("seed_concentration_penalty", 1e-5))
    confidence_floor = float(settings.get("confidence_floor", 0.45))
    caps = _family_caps(settings, family, regime_count)
    residual_candidates = _residual_grid(settings, caps)

    baseline_selection = pinball_loss_numpy(baseline[selection], truth[selection], q)
    best_score = baseline_selection
    best_weights = np.full(len(models), 1.0 / len(models), dtype=np.float64)
    best_regime_weights = np.zeros(regime_count, dtype=np.float64)
    for raw_weights in seed_candidates:
        weights = _regularize_seed_weights(raw_weights, seed_shrinkage, max_seed_weight)
        ensemble = _weighted_prediction(models[:, selection], weights)
        penalty = concentration_penalty * np.square(weights).sum()
        for candidate_weights in residual_candidates:
            regime_weights = np.asarray(candidate_weights, dtype=np.float64)
            strength = _soft_strength(
                probabilities[selection],
                confidence[selection],
                regime_weights,
                confidence_floor,
            )
            candidate = baseline[selection] + strength[:, None, None] * (
                ensemble - baseline[selection]
            )
            score = pinball_loss_numpy(candidate, truth[selection], q) + penalty
            if score < best_score - 1e-12:
                best_score = float(score)
                best_weights = weights.copy()
                best_regime_weights = regime_weights.copy()

    selected_ensemble = _weighted_prediction(models, best_weights)
    selected_strength = _soft_strength(
        probabilities[selection],
        confidence[selection],
        best_regime_weights,
        confidence_floor,
    )
    selected_candidate = baseline[selection] + selected_strength[:, None, None] * (
        selected_ensemble[selection] - baseline[selection]
    )
    selected_loss = pinball_loss_numpy(selected_candidate, truth[selection], q)
    selection_improvement = (
        baseline_selection - selected_loss
    ) / max(abs(baseline_selection), 1e-12)
    if selection_improvement < float(settings.get("minimum_selection_improvement", 0.0015)):
        return ProbabilisticEnsembleGate(
            **{
                **identity.__dict__,
                "seed_weights": best_weights.astype(np.float32),
                "effective_seed_count": effective_seed_count(best_weights),
                "safety_reason": "probabilistic_selection_improvement_below_threshold",
            }
        )

    def candidate_for(part: slice) -> np.ndarray:
        strength = _soft_strength(
            probabilities[part],
            confidence[part],
            best_regime_weights,
            confidence_floor,
        )
        return baseline[part] + strength[:, None, None] * (
            selected_ensemble[part] - baseline[part]
        )

    tuning_candidate = candidate_for(tuning)
    baseline_tuning = pinball_loss_numpy(baseline[tuning], truth[tuning], q)
    model_tuning = pinball_loss_numpy(tuning_candidate, truth[tuning], q)
    tuning_improvement = (
        baseline_tuning - model_tuning
    ) / max(abs(baseline_tuning), 1e-12)
    if tuning_improvement < float(settings.get("minimum_tuning_improvement", 0.0)):
        return ProbabilisticEnsembleGate(
            **{
                **identity.__dict__,
                "seed_weights": best_weights.astype(np.float32),
                "tuning_improvement": float(tuning_improvement),
                "effective_seed_count": effective_seed_count(best_weights),
                "safety_reason": "probabilistic_tuning_rejected",
            }
        )

    confirmation_candidate = candidate_for(confirmation)
    baseline_confirmation = pinball_loss_numpy(baseline[confirmation], truth[confirmation], q)
    model_confirmation = pinball_loss_numpy(
        confirmation_candidate, truth[confirmation], q
    )
    confirmation_improvement = (
        baseline_confirmation - model_confirmation
    ) / max(abs(baseline_confirmation), 1e-12)
    baseline_daily = _date_loss(baseline[confirmation], truth[confirmation], q)
    candidate_daily = _date_loss(confirmation_candidate, truth[confirmation], q)
    lower = circular_block_bootstrap_lower_bound(
        baseline_daily - candidate_daily,
        int(settings.get("block_length", 10)),
        int(settings.get("bootstrap_repetitions", 2000)),
        float(settings.get("one_sided_confidence", 0.80)),
        seed,
    )
    episodes = _episode_improvements(
        baseline_daily,
        candidate_daily,
        int(settings.get("confirmation_episodes", 4)),
    )
    positive_fraction = float(np.mean(np.asarray(episodes) >= 0.0)) if episodes else 0.0
    dominant = probabilities[confirmation].argmax(axis=1)
    regime_improvements: list[float] = []
    minimum_regime_samples = int(settings.get("minimum_regime_samples", 18))
    for regime_value in range(regime_count):
        mask = dominant == regime_value
        if int(mask.sum()) < minimum_regime_samples:
            regime_improvements.append(0.0)
            continue
        base_loss = pinball_loss_numpy(
            baseline[confirmation][mask], truth[confirmation][mask], q
        )
        model_loss = pinball_loss_numpy(
            confirmation_candidate[mask], truth[confirmation][mask], q
        )
        regime_improvements.append(
            (base_loss - model_loss) / max(abs(base_loss), 1e-12)
        )
    baseline_risk = _risk_calibration_error(
        baseline[confirmation], truth[confirmation], quantiles, horizons
    )
    model_risk = _risk_calibration_error(
        confirmation_candidate, truth[confirmation], quantiles, horizons
    )
    risk_delta = model_risk - baseline_risk
    effective = effective_seed_count(best_weights)
    accepted = (
        confirmation_improvement
        >= float(settings.get("minimum_confirmation_improvement", 0.0015))
        and lower > 0.0
        and tuning_improvement >= float(settings.get("minimum_tuning_improvement", 0.0))
        and positive_fraction
        >= float(settings.get("minimum_positive_episode_fraction", 0.75))
        and (min(episodes) if episodes else -np.inf)
        >= float(settings.get("maximum_episode_degradation", -0.0025))
        and min(regime_improvements)
        >= float(settings.get("maximum_regime_degradation", -0.005))
        and risk_delta
        <= float(settings.get("maximum_risk_calibration_degradation", 0.01))
        and effective >= float(settings.get("minimum_effective_seed_count", 1.35))
    )
    if not accepted:
        if effective < float(settings.get("minimum_effective_seed_count", 1.35)):
            reason = "probabilistic_seed_concentration_rejected"
        elif lower <= 0.0:
            reason = "probabilistic_bootstrap_lower_bound_nonpositive"
        elif risk_delta > float(settings.get("maximum_risk_calibration_degradation", 0.01)):
            reason = "probabilistic_risk_calibration_rejected"
        else:
            reason = "probabilistic_episode_regime_or_confirmation_rejected"
        return ProbabilisticEnsembleGate(
            seed_weights=best_weights.astype(np.float32),
            regime_residual_weights=np.zeros(regime_count, dtype=np.float32),
            bootstrap_lower_bound=float(lower),
            tuning_improvement=float(tuning_improvement),
            confirmation_improvement=float(confirmation_improvement),
            episode_improvements=episodes,
            regime_confirmation_improvements=tuple(float(v) for v in regime_improvements),
            risk_calibration_delta=float(risk_delta),
            effective_seed_count=float(effective),
            accepted=False,
            safety_reason=reason,
        )
    return ProbabilisticEnsembleGate(
        seed_weights=best_weights.astype(np.float32),
        regime_residual_weights=best_regime_weights.astype(np.float32),
        bootstrap_lower_bound=float(lower),
        tuning_improvement=float(tuning_improvement),
        confirmation_improvement=float(confirmation_improvement),
        episode_improvements=episodes,
        regime_confirmation_improvements=tuple(float(v) for v in regime_improvements),
        risk_calibration_delta=float(risk_delta),
        effective_seed_count=float(effective),
        accepted=True,
        safety_reason="probabilistic_multi_episode_bootstrap_confirmed",
    )


def evaluate_probabilistic_seed_ensemble(
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
    validation_raw = [
        pd.read_csv(path / "validation_raw_predictions.csv") for path in directories
    ]
    test_raw = [pd.read_csv(path / "test_raw_predictions.csv") for path in directories]
    validation_source = pd.read_csv(statistical / "validation_baseline_predictions.csv")
    test_source = pd.read_csv(statistical / "test_baseline_predictions.csv")
    validation_anchor_source = pd.read_csv(
        statistical / "validation_anchor_predictions.csv"
    )
    test_anchor_source = pd.read_csv(statistical / "test_anchor_predictions.csv")
    validation_baseline = [
        _align_baseline_to_model(validation_source, model) for model in validation_raw
    ]
    test_baseline = [
        _align_baseline_to_model(test_source, model) for model in test_raw
    ]
    validation_anchor = _align_baseline_to_model(
        validation_anchor_source, validation_raw[0]
    )
    test_anchor = _align_baseline_to_model(test_anchor_source, test_raw[0])
    _validate_alignment([*validation_raw, *validation_baseline, validation_anchor])
    _validate_alignment([*test_raw, *test_baseline, test_anchor])

    validation_models = [_frame_to_arrays(frame)[0] for frame in validation_raw]
    validation_base, validation_target, quantiles = _frame_to_arrays(
        validation_baseline[0]
    )
    validation_anchor_array, _, _ = _frame_to_arrays(validation_anchor)
    test_dates = sorted(test_raw[0]["date"].astype(str).unique().tolist())
    validation_dates = sorted(validation_raw[0]["date"].astype(str).unique().tolist())
    regime_validation_frame = pd.read_csv(
        statistical / "validation_regime_features.csv"
    )
    regime_test_frame = pd.read_csv(statistical / "test_regime_features.csv")
    validation_probabilities, validation_confidence = _probability_arrays(
        regime_validation_frame, validation_dates
    )
    test_probabilities, test_confidence = _probability_arrays(
        regime_test_frame, test_dates
    )
    gate = fit_probabilistic_ensemble_gate(
        validation_models,
        validation_base,
        validation_target,
        validation_probabilities,
        validation_confidence,
        quantiles,
        horizons,
        settings,
        family,
    )

    test_models = np.stack([_frame_to_arrays(frame)[0] for frame in test_raw], axis=0)
    test_base, test_target, _ = _frame_to_arrays(test_baseline[0])
    test_anchor_array, _, _ = _frame_to_arrays(test_anchor)
    raw_ensemble = _weighted_prediction(test_models, gate.seed_weights)
    final = apply_probabilistic_gate(
        raw_ensemble,
        test_base,
        test_probabilities,
        test_confidence,
        gate,
        float(settings.get("confidence_floor", 0.45)),
    )
    metrics, detail, var = evaluate_scenario_forecasts(
        final, test_target, quantiles, horizons
    )
    raw_metrics, raw_detail, raw_var = evaluate_scenario_forecasts(
        raw_ensemble, test_target, quantiles, horizons
    )
    baseline_metrics, baseline_detail, baseline_var = evaluate_scenario_forecasts(
        test_base, test_target, quantiles, horizons
    )
    anchor_metrics, anchor_detail, anchor_var = evaluate_scenario_forecasts(
        test_anchor_array, test_target, quantiles, horizons
    )
    improvement_champion = 100.0 * (
        baseline_metrics["mean_pinball"] - metrics["mean_pinball"]
    ) / max(abs(baseline_metrics["mean_pinball"]), 1e-12)
    improvement_anchor = 100.0 * (
        anchor_metrics["mean_pinball"] - metrics["mean_pinball"]
    ) / max(abs(anchor_metrics["mean_pinball"]), 1e-12)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_baseline_frame = (
        test_baseline[0]
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    test_anchor_frame = (
        test_anchor
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    test_prediction_frame = (
        test_baseline_frame.copy()
        if gate.fallback_to_baseline
        else _arrays_to_frame(test_raw[0], final)
    )

    test_prediction_frame.to_csv(
        output / "predictions.csv",
        index=False,
    )
    _arrays_to_frame(test_raw[0], raw_ensemble).to_csv(
        output / "raw_ensemble_predictions.csv",
        index=False,
    )
    test_baseline_frame.to_csv(
        output / "baseline_predictions.csv",
        index=False,
    )
    test_anchor_frame.to_csv(
        output / "anchor_predictions.csv",
        index=False,
    )
    pd.concat(
        [
            detail.assign(model="probabilistic_gated"),
            raw_detail.assign(model="raw_seed_ensemble"),
            baseline_detail.assign(model="statistical_champion"),
            anchor_detail.assign(model="frozen_v154_anchor"),
        ],
        ignore_index=True,
    ).to_csv(output / "metrics_comparison_by_horizon.csv", index=False)
    pd.concat(
        [
            var.assign(model="probabilistic_gated"),
            raw_var.assign(model="raw_seed_ensemble"),
            baseline_var.assign(model="statistical_champion"),
            anchor_var.assign(model="frozen_v154_anchor"),
        ],
        ignore_index=True,
    ).to_csv(output / "var_backtests.csv", index=False)

    validation_ensemble = _weighted_prediction(
        np.stack(validation_models), gate.seed_weights
    )
    validation_final = apply_probabilistic_gate(
        validation_ensemble,
        validation_base,
        validation_probabilities,
        validation_confidence,
        gate,
        float(settings.get("confidence_floor", 0.45)),
    )
    validation_baseline_frame = (
        validation_baseline[0]
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    validation_anchor_frame = (
        validation_anchor
        .sort_values(["date", "horizon"])
        .reset_index(drop=True)
    )
    validation_prediction_frame = (
        validation_baseline_frame.copy()
        if gate.fallback_to_baseline
        else _arrays_to_frame(
            validation_raw[0],
            validation_final,
        )
    )

    validation_prediction_frame.to_csv(
        output / "validation_predictions.csv",
        index=False,
    )
    validation_baseline_frame.to_csv(
        output / "validation_baseline_predictions.csv",
        index=False,
    )
    validation_anchor_frame.to_csv(
        output / "validation_anchor_predictions.csv",
        index=False,
    )

    dominant = test_probabilities.argmax(axis=1)
    daily = pd.DataFrame(
        {
            "date": test_dates,
            "dominant_regime": dominant,
            "regime_confidence": test_confidence,
            "probabilistic_gated_loss": _date_loss(final, test_target, np.asarray(quantiles)),
            "raw_ensemble_loss": _date_loss(raw_ensemble, test_target, np.asarray(quantiles)),
            "statistical_champion_loss": _date_loss(test_base, test_target, np.asarray(quantiles)),
            "frozen_anchor_loss": _date_loss(test_anchor_array, test_target, np.asarray(quantiles)),
        }
    )
    daily["improvement_vs_champion"] = (
        daily["statistical_champion_loss"] - daily["probabilistic_gated_loss"]
    )
    daily["improvement_vs_anchor"] = (
        daily["frozen_anchor_loss"] - daily["probabilistic_gated_loss"]
    )
    daily.to_csv(output / "daily_loss_comparison.csv", index=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "raw_model_metrics.json", raw_metrics)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    write_json(output / "anchor_metrics.json", anchor_metrics)
    write_json(output / "ensemble_gate.json", gate.export())
    write_json(
        output / "evaluation_summary.json",
        {
            "family": family,
            "fold": fold,
            "variant": variant,
            "mean_pinball": float(metrics["mean_pinball"]),
            "raw_ensemble_pinball": float(raw_metrics["mean_pinball"]),
            "statistical_champion_pinball": float(baseline_metrics["mean_pinball"]),
            "frozen_v154_anchor_pinball": float(anchor_metrics["mean_pinball"]),
            "improvement_vs_statistical_champion_pct": float(improvement_champion),
            "improvement_vs_frozen_v154_anchor_pct": float(improvement_anchor),
            "fallback_to_baseline": bool(gate.fallback_to_baseline),
            "seed_weights": gate.seed_weights.tolist(),
            "effective_seed_count": float(gate.effective_seed_count),
            "regime_residual_weights": gate.regime_residual_weights.tolist(),
            "bootstrap_lower_bound": float(gate.bootstrap_lower_bound),
            "risk_calibration_delta": float(gate.risk_calibration_delta),
            "safety_reason": gate.safety_reason,
            "test_origins": int(len(test_target)),
        },
    )
    return output
