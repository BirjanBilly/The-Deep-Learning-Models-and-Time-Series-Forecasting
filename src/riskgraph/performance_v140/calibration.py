from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from riskgraph.evaluation.metrics import pinball_loss_numpy


@dataclass(frozen=True)
class ForecastGate:
    """Validation-only residual shrinkage and causal quantile calibration.

    v1.4.1 uses three chronological validation blocks:

    1. residual-weight selection,
    2. calibration tuning,
    3. untouched safety confirmation.

    The final gate is the identity EWMA baseline unless the learned candidate clears
    both the residual-selection and confirmation thresholds.  This does not promise
    an out-of-sample test win, which would be impossible without looking at test
    outcomes, but it prevents a candidate already known to be worse on validation
    from being labelled "baseline safe".
    """

    residual_weight: float
    spread_scale: float
    quantile_offsets: np.ndarray
    online_learning_rate: float
    online_clip: float
    baseline_validation_pinball: float
    selected_validation_pinball: float
    relative_improvement: float
    fallback_to_baseline: bool
    selection_samples: int
    calibration_samples: int
    confirmation_samples: int = 0
    baseline_selection_pinball: float = float("nan")
    selected_selection_pinball: float = float("nan")
    selection_relative_improvement: float = float("nan")
    baseline_calibration_pinball: float = float("nan")
    selected_calibration_pinball: float = float("nan")
    calibration_relative_improvement: float = float("nan")
    baseline_confirmation_pinball: float = float("nan")
    selected_confirmation_pinball: float = float("nan")
    confirmation_relative_improvement: float = float("nan")
    safety_reason: str = "legacy_record"

    def export(self) -> dict[str, Any]:
        record = asdict(self)
        record["quantile_offsets"] = self.quantile_offsets.tolist()
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ForecastGate:
        baseline_validation = float(record["baseline_validation_pinball"])
        selected_validation = float(record["selected_validation_pinball"])
        relative = float(record["relative_improvement"])
        return cls(
            residual_weight=float(record["residual_weight"]),
            spread_scale=float(record["spread_scale"]),
            quantile_offsets=np.asarray(record["quantile_offsets"], dtype=np.float32),
            online_learning_rate=float(record.get("online_learning_rate", 0.0)),
            online_clip=float(record.get("online_clip", 2.0)),
            baseline_validation_pinball=baseline_validation,
            selected_validation_pinball=selected_validation,
            relative_improvement=relative,
            fallback_to_baseline=bool(record["fallback_to_baseline"]),
            selection_samples=int(record["selection_samples"]),
            calibration_samples=int(record["calibration_samples"]),
            confirmation_samples=int(record.get("confirmation_samples", 0)),
            baseline_selection_pinball=float(
                record.get("baseline_selection_pinball", baseline_validation)
            ),
            selected_selection_pinball=float(
                record.get("selected_selection_pinball", selected_validation)
            ),
            selection_relative_improvement=float(
                record.get("selection_relative_improvement", relative)
            ),
            baseline_calibration_pinball=float(
                record.get("baseline_calibration_pinball", baseline_validation)
            ),
            selected_calibration_pinball=float(
                record.get("selected_calibration_pinball", selected_validation)
            ),
            calibration_relative_improvement=float(
                record.get("calibration_relative_improvement", relative)
            ),
            baseline_confirmation_pinball=float(
                record.get("baseline_confirmation_pinball", baseline_validation)
            ),
            selected_confirmation_pinball=float(
                record.get("selected_confirmation_pinball", selected_validation)
            ),
            confirmation_relative_improvement=float(
                record.get("confirmation_relative_improvement", relative)
            ),
            safety_reason=str(record.get("safety_reason", "legacy_record")),
        )


def monotone_rearrange(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    return np.sort(values, axis=-1).astype(np.float32)


def _relative_improvement(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(abs(baseline), 1e-12))


def _blend(
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    residual_weight: float,
) -> np.ndarray:
    return monotone_rearrange(
        baseline_prediction
        + float(residual_weight) * (model_prediction - baseline_prediction)
    )


def _apply_spread(prediction: np.ndarray, quantiles: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    median = values[..., median_index : median_index + 1]
    return monotone_rearrange(median + float(scale) * (values - median))


def _quantile_bias_offsets(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    offsets = np.empty(pred.shape[1:], dtype=np.float32)
    for h_index in range(pred.shape[1]):
        for q_index, quantile in enumerate(quantiles):
            residual = truth[:, h_index] - pred[:, h_index, q_index]
            offsets[h_index, q_index] = float(np.quantile(residual, quantile))
    return offsets


def _coverage_penalty(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    penalties: list[float] = []
    for coverage in (0.90, 0.95):
        alpha = 1.0 - coverage
        lower_candidates = np.flatnonzero(np.isclose(quantiles, alpha / 2.0, atol=1e-8))
        upper_candidates = np.flatnonzero(
            np.isclose(quantiles, 1.0 - alpha / 2.0, atol=1e-8)
        )
        if not len(lower_candidates) or not len(upper_candidates):
            continue
        lower = pred[..., int(lower_candidates[0])]
        upper = pred[..., int(upper_candidates[0])]
        observed = np.mean((truth >= lower) & (truth <= upper))
        penalties.append(abs(float(observed) - coverage))
    return float(np.mean(penalties)) if penalties else 0.0


def causal_online_quantile_calibration(
    prediction: np.ndarray,
    observed_targets: np.ndarray,
    quantiles: list[float] | np.ndarray,
    horizons: list[int] | np.ndarray,
    learning_rate: float,
    clip_multiple: float = 2.0,
) -> np.ndarray:
    """Prequential quantile calibration using only resolved prior outcomes.

    For horizon h, the update from origin j is first available at origin j+h.
    Passing the retrospective target array is safe because this delay is enforced.
    """

    static = monotone_rearrange(prediction)
    truth = np.asarray(observed_targets, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    horizon_values = np.asarray(horizons, dtype=int)
    if truth.shape != static.shape[:2]:
        raise ValueError("Observed targets have an incompatible shape")
    if len(horizon_values) != static.shape[1]:
        raise ValueError("Horizon count does not match forecast shape")
    rate = float(learning_rate)
    if rate <= 0.0:
        return static
    output = np.empty_like(static, dtype=np.float32)
    offsets = np.zeros(static.shape[1:], dtype=np.float64)
    width = np.maximum(static[..., -1] - static[..., 0], 1e-5)
    reference_width = np.maximum(np.median(width, axis=0), 1e-5)
    for origin_index in range(len(static)):
        for h_index, horizon in enumerate(horizon_values):
            resolved_index = origin_index - max(1, int(horizon))
            if resolved_index >= 0:
                prior = output[resolved_index, h_index]
                realized = truth[resolved_index, h_index]
                indicator = (realized <= prior).astype(float)
                step_scale = max(float(width[resolved_index, h_index]), 1e-5)
                offsets[h_index] += rate * step_scale * (q - indicator)
                limit = float(clip_multiple) * float(reference_width[h_index])
                offsets[h_index] = np.clip(offsets[h_index], -limit, limit)
            output[origin_index, h_index] = np.sort(
                static[origin_index, h_index] + offsets[h_index]
            )
    return output


def _apply_candidate(
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    quantiles: np.ndarray,
    horizons: np.ndarray,
    residual_weight: float,
    spread_scale: float,
    quantile_offsets: np.ndarray,
    online_learning_rate: float,
    online_clip: float,
    observed_targets: np.ndarray | None,
) -> np.ndarray:
    blended = _blend(model_prediction, baseline_prediction, residual_weight)
    spread = _apply_spread(blended, quantiles, spread_scale)
    static = monotone_rearrange(spread + quantile_offsets[None, ...])
    if observed_targets is None or online_learning_rate <= 0.0:
        return static
    return causal_online_quantile_calibration(
        static,
        observed_targets,
        quantiles,
        horizons,
        learning_rate=online_learning_rate,
        clip_multiple=online_clip,
    )


def _identity_gate(
    shape: tuple[int, ...],
    online_clip: float,
    baseline_selection: float,
    selected_selection: float,
    selection_relative: float,
    baseline_calibration: float,
    selected_calibration: float,
    calibration_relative: float,
    baseline_confirmation: float,
    selection_samples: int,
    calibration_samples: int,
    confirmation_samples: int,
    reason: str,
    selected_confirmation: float | None = None,
    confirmation_relative: float | None = None,
) -> ForecastGate:
    return ForecastGate(
        residual_weight=0.0,
        spread_scale=1.0,
        quantile_offsets=np.zeros(shape, dtype=np.float32),
        online_learning_rate=0.0,
        online_clip=float(online_clip),
        baseline_validation_pinball=float(baseline_confirmation),
        selected_validation_pinball=float(baseline_confirmation),
        relative_improvement=0.0,
        fallback_to_baseline=True,
        selection_samples=int(selection_samples),
        calibration_samples=int(calibration_samples),
        confirmation_samples=int(confirmation_samples),
        baseline_selection_pinball=float(baseline_selection),
        selected_selection_pinball=float(selected_selection),
        selection_relative_improvement=float(selection_relative),
        baseline_calibration_pinball=float(baseline_calibration),
        selected_calibration_pinball=float(selected_calibration),
        calibration_relative_improvement=float(calibration_relative),
        baseline_confirmation_pinball=float(baseline_confirmation),
        selected_confirmation_pinball=float(
            baseline_confirmation
            if selected_confirmation is None
            else selected_confirmation
        ),
        confirmation_relative_improvement=float(
            0.0 if confirmation_relative is None else confirmation_relative
        ),
        safety_reason=str(reason),
    )


def fit_forecast_gate(
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float] | np.ndarray,
    horizons: list[int] | np.ndarray | None = None,
    residual_weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
    spread_scales: tuple[float, ...] = (0.9, 1.0, 1.1, 1.25, 1.5, 1.8),
    calibration_fraction: float = 0.40,
    confirmation_fraction_of_calibration: float = 0.50,
    offset_shrinkages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    online_learning_rates: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.10),
    online_clip: float = 2.0,
    coverage_penalty_weight: float = 0.02,
    min_relative_improvement: float = 0.0025,
    min_confirmation_improvement: float | None = None,
) -> ForecastGate:
    """Fit a chronologically confirmed baseline-safe gate.

    The baseline is the only admissible fallback.  Calibration parameters may trade
    among candidates only when their tuning pinball is no worse than the tuning
    baseline.  The selected candidate is then accepted only when it improves an
    untouched final validation block by the requested threshold.
    """

    model = monotone_rearrange(model_prediction)
    baseline = monotone_rearrange(baseline_prediction)
    truth = np.asarray(target, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    horizon_values = (
        np.arange(1, model.shape[1] + 1, dtype=int)
        if horizons is None
        else np.asarray(horizons, dtype=int)
    )
    if model.shape != baseline.shape or truth.shape != model.shape[:2]:
        raise ValueError("Forecast and target shapes are incompatible")
    n = len(truth)
    if n < 30:
        raise ValueError("At least 30 validation observations are required")
    if not 0.20 <= calibration_fraction <= 0.70:
        raise ValueError("calibration_fraction must lie in [0.20, 0.70]")
    if not 0.25 <= confirmation_fraction_of_calibration <= 0.75:
        raise ValueError(
            "confirmation_fraction_of_calibration must lie in [0.25, 0.75]"
        )

    selection_end = max(
        10,
        min(n - 20, int(round(n * (1.0 - calibration_fraction)))),
    )
    remaining = n - selection_end
    confirmation_size = max(
        10,
        min(
            remaining - 10,
            int(round(remaining * confirmation_fraction_of_calibration)),
        ),
    )
    tuning_size = remaining - confirmation_size
    tuning_end = selection_end + tuning_size

    selection = slice(0, selection_end)
    tuning = slice(selection_end, tuning_end)
    confirmation = slice(tuning_end, n)

    baseline_selection = pinball_loss_numpy(baseline[selection], truth[selection], q)
    selected_weight = 0.0
    selected_selection = baseline_selection
    for weight in residual_weights:
        candidate = _blend(model[selection], baseline[selection], weight)
        score = pinball_loss_numpy(candidate, truth[selection], q)
        if score < selected_selection - 1e-12:
            selected_selection = float(score)
            selected_weight = float(weight)
    selection_relative = _relative_improvement(
        baseline_selection,
        selected_selection,
    )

    baseline_tuning = pinball_loss_numpy(baseline[tuning], truth[tuning], q)
    baseline_confirmation = pinball_loss_numpy(
        baseline[confirmation],
        truth[confirmation],
        q,
    )
    zero_offsets = np.zeros(model.shape[1:], dtype=np.float32)

    if selection_relative < float(min_relative_improvement):
        return _identity_gate(
            model.shape[1:],
            online_clip,
            baseline_selection,
            selected_selection,
            selection_relative,
            baseline_tuning,
            baseline_tuning,
            0.0,
            baseline_confirmation,
            selection_end,
            tuning_size,
            confirmation_size,
            "selection_improvement_below_threshold",
        )

    uncalibrated_tuning = _blend(
        model[tuning],
        baseline[tuning],
        selected_weight,
    )
    baseline_composite = baseline_tuning + float(
        coverage_penalty_weight
    ) * _coverage_penalty(baseline[tuning], truth[tuning], q)
    best_composite = float(baseline_composite)
    selected_scale = 1.0
    selected_offsets = zero_offsets.copy()
    selected_online_rate = 0.0
    selected_tuning = float(baseline_tuning)
    calibration_candidate_found = False

    for spread_scale in spread_scales:
        spread = _apply_spread(uncalibrated_tuning, q, spread_scale)
        raw_offsets = _quantile_bias_offsets(spread, truth[tuning], q)
        for shrinkage in offset_shrinkages:
            offsets = float(shrinkage) * raw_offsets
            static_candidate = monotone_rearrange(spread + offsets[None, ...])
            for online_rate in online_learning_rates:
                candidate = causal_online_quantile_calibration(
                    static_candidate,
                    truth[tuning],
                    q,
                    horizon_values,
                    learning_rate=float(online_rate),
                    clip_multiple=float(online_clip),
                )
                pinball = float(pinball_loss_numpy(candidate, truth[tuning], q))
                # Coverage may break a tie, but it may not buy a pinball deterioration.
                if pinball > baseline_tuning + 1e-12:
                    continue
                composite = pinball + float(coverage_penalty_weight) * _coverage_penalty(
                    candidate,
                    truth[tuning],
                    q,
                )
                if composite < best_composite - 1e-12:
                    best_composite = float(composite)
                    selected_scale = float(spread_scale)
                    selected_offsets = offsets.astype(np.float32)
                    selected_online_rate = float(online_rate)
                    selected_tuning = pinball
                    calibration_candidate_found = True

    calibration_relative = _relative_improvement(
        baseline_tuning,
        selected_tuning,
    )

    if not calibration_candidate_found:
        return _identity_gate(
            model.shape[1:],
            online_clip,
            baseline_selection,
            selected_selection,
            selection_relative,
            baseline_tuning,
            baseline_tuning,
            0.0,
            baseline_confirmation,
            selection_end,
            tuning_size,
            confirmation_size,
            "no_calibration_candidate_beats_baseline",
        )

    confirmation_candidate = _apply_candidate(
        model[confirmation],
        baseline[confirmation],
        q,
        horizon_values,
        selected_weight,
        selected_scale,
        selected_offsets,
        selected_online_rate,
        online_clip,
        truth[confirmation],
    )
    selected_confirmation = float(
        pinball_loss_numpy(confirmation_candidate, truth[confirmation], q)
    )
    confirmation_relative = _relative_improvement(
        baseline_confirmation,
        selected_confirmation,
    )
    confirmation_threshold = (
        float(min_relative_improvement)
        if min_confirmation_improvement is None
        else float(min_confirmation_improvement)
    )

    if selected_tuning > baseline_tuning + 1e-12:
        return _identity_gate(
            model.shape[1:],
            online_clip,
            baseline_selection,
            selected_selection,
            selection_relative,
            baseline_tuning,
            selected_tuning,
            calibration_relative,
            baseline_confirmation,
            selection_end,
            tuning_size,
            confirmation_size,
            "calibration_pinball_worse_than_baseline",
        )
    if confirmation_relative < confirmation_threshold:
        return _identity_gate(
            model.shape[1:],
            online_clip,
            baseline_selection,
            selected_selection,
            selection_relative,
            baseline_tuning,
            selected_tuning,
            calibration_relative,
            baseline_confirmation,
            selection_end,
            tuning_size,
            confirmation_size,
            "confirmation_improvement_below_threshold",
            selected_confirmation=selected_confirmation,
            confirmation_relative=confirmation_relative,
        )

    return ForecastGate(
        residual_weight=float(selected_weight),
        spread_scale=float(selected_scale),
        quantile_offsets=selected_offsets.astype(np.float32),
        online_learning_rate=float(selected_online_rate),
        online_clip=float(online_clip),
        baseline_validation_pinball=float(baseline_confirmation),
        selected_validation_pinball=float(selected_confirmation),
        relative_improvement=float(confirmation_relative),
        fallback_to_baseline=False,
        selection_samples=int(selection_end),
        calibration_samples=int(tuning_size),
        confirmation_samples=int(confirmation_size),
        baseline_selection_pinball=float(baseline_selection),
        selected_selection_pinball=float(selected_selection),
        selection_relative_improvement=float(selection_relative),
        baseline_calibration_pinball=float(baseline_tuning),
        selected_calibration_pinball=float(selected_tuning),
        calibration_relative_improvement=float(calibration_relative),
        baseline_confirmation_pinball=float(baseline_confirmation),
        selected_confirmation_pinball=float(selected_confirmation),
        confirmation_relative_improvement=float(confirmation_relative),
        safety_reason="confirmed_candidate",
    )


def apply_forecast_gate(
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    quantiles: list[float] | np.ndarray,
    gate: ForecastGate,
    observed_targets: np.ndarray | None = None,
    horizons: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    if gate.fallback_to_baseline:
        # A true fallback must preserve the exact baseline array, including dtype and
        # rounding, so saved final predictions can be audited byte-for-byte against
        # the separately saved EWMA Student-t prediction surface.
        return np.asarray(baseline_prediction).copy()
    baseline = monotone_rearrange(baseline_prediction)
    q = np.asarray(quantiles, dtype=float)
    horizon_values = (
        np.arange(1, baseline.shape[1] + 1, dtype=int)
        if horizons is None
        else np.asarray(horizons, dtype=int)
    )
    return _apply_candidate(
        model_prediction,
        baseline,
        q,
        horizon_values,
        gate.residual_weight,
        gate.spread_scale,
        gate.quantile_offsets,
        gate.online_learning_rate,
        gate.online_clip,
        observed_targets,
    )
