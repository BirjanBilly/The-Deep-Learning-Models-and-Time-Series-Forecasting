from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

from riskgraph.config import Fold
from riskgraph.io import predictions_frame, write_json
from riskgraph.performance_v140.calibration import pinball_loss_numpy
from riskgraph.performance_v150.data import LongHistoryPanel, long_targets, split_long_origins
from riskgraph.performance_v160.regime import (
    fit_regime_feature_spec,
    regime_frame,
    save_regime_spec,
    transform_regime_features,
)


@dataclass(frozen=True)
class StatisticalExpertSpec:
    name: str
    mean_decay: float
    variance_decay: float
    degrees_of_freedom: float
    asymmetric: bool = False
    downside_decay: float = 0.90
    upside_decay: float = 0.97
    leverage: float = 0.0


@dataclass(frozen=True)
class StatisticalEnsembleState:
    expert_specs: tuple[StatisticalExpertSpec, ...]
    weights: np.ndarray  # [horizon, expert]
    episode_losses: np.ndarray  # [horizon, episode]
    selection_origins: np.ndarray
    train_end_index: int
    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]

    def export(self) -> dict[str, Any]:
        return {
            "expert_specs": [asdict(value) for value in self.expert_specs],
            "weights": self.weights.tolist(),
            "episode_losses": self.episode_losses.tolist(),
            "selection_origins": self.selection_origins.tolist(),
            "train_end_index": int(self.train_end_index),
            "horizons": list(self.horizons),
            "quantiles": list(self.quantiles),
            "causality": (
                "Expert states use observations available no later than each origin. "
                "Mixture weights are fitted only on origins ending by train_end."
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StatisticalEnsembleState:
        return cls(
            expert_specs=tuple(
                StatisticalExpertSpec(**dict(value))
                for value in record["expert_specs"]
            ),
            weights=np.asarray(record["weights"], dtype=np.float64),
            episode_losses=np.asarray(record["episode_losses"], dtype=np.float64),
            selection_origins=np.asarray(record["selection_origins"], dtype=np.int64),
            train_end_index=int(record["train_end_index"]),
            horizons=tuple(int(value) for value in record["horizons"]),
            quantiles=tuple(float(value) for value in record["quantiles"]),
        )


def default_expert_specs() -> tuple[StatisticalExpertSpec, ...]:
    """A deliberately small, interpretable heavy-tail expert pool."""

    return (
        StatisticalExpertSpec("ewma_fast_t4", 0.970, 0.900, 4.0),
        StatisticalExpertSpec("ewma_standard_t5", 0.985, 0.940, 5.0),
        StatisticalExpertSpec("ewma_medium_t7", 0.992, 0.970, 7.0),
        StatisticalExpertSpec("ewma_slow_t10", 0.997, 0.985, 10.0),
        StatisticalExpertSpec(
            "ewma_leverage_t5",
            0.985,
            0.955,
            5.0,
            leverage=0.20,
        ),
        StatisticalExpertSpec(
            "split_scale_t5",
            0.985,
            0.955,
            5.0,
            asymmetric=True,
            downside_decay=0.90,
            upside_decay=0.97,
        ),
    )


def _causal_location_scales(
    returns: np.ndarray,
    spec: StatisticalExpertSpec,
    floor: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        raise ValueError("At least two returns are required")
    mean = np.empty_like(values)
    lower = np.empty_like(values)
    upper = np.empty_like(values)
    mean[0] = values[0]
    initial = max(values[0] * values[0], floor)
    lower[0] = initial
    upper[0] = initial
    for index in range(1, len(values)):
        previous = values[index - 1]
        innovation = previous - mean[index - 1]
        mean[index] = (
            spec.mean_decay * mean[index - 1]
            + (1.0 - spec.mean_decay) * previous
        )
        squared = innovation * innovation
        leverage_multiplier = 1.0 + spec.leverage * float(innovation < 0.0)
        if spec.asymmetric:
            lower_innovation = squared if innovation < 0.0 else 0.35 * squared
            upper_innovation = squared if innovation >= 0.0 else 0.35 * squared
            lower[index] = (
                spec.downside_decay * lower[index - 1]
                + (1.0 - spec.downside_decay) * lower_innovation
            )
            upper[index] = (
                spec.upside_decay * upper[index - 1]
                + (1.0 - spec.upside_decay) * upper_innovation
            )
        else:
            variance = (
                spec.variance_decay * lower[index - 1]
                + (1.0 - spec.variance_decay) * squared * leverage_multiplier
            )
            lower[index] = variance
            upper[index] = variance
    return mean, np.maximum(lower, floor), np.maximum(upper, floor)


def expert_forecast(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: Iterable[int],
    quantiles: Iterable[float],
    spec: StatisticalExpertSpec,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    horizons_tuple = tuple(int(value) for value in horizons)
    q = np.asarray(tuple(float(value) for value in quantiles), dtype=np.float64)
    mean, lower_variance, upper_variance = _causal_location_scales(returns, spec)
    nu = float(spec.degrees_of_freedom)
    unit_scale = np.sqrt((nu - 2.0) / nu)
    z = t.ppf(q, df=nu) * unit_scale
    output = np.empty(
        (len(origins), len(horizons_tuple), len(q)), dtype=np.float32
    )
    for row, origin in enumerate(origins):
        origin_index = int(origin)
        for horizon_index, horizon in enumerate(horizons_tuple):
            negative = np.sqrt(float(horizon) * lower_variance[origin_index])
            positive = np.sqrt(float(horizon) * upper_variance[origin_index])
            scale = np.where(z < 0.0, negative, positive)
            output[row, horizon_index] = (
                float(horizon) * mean[origin_index] + scale * z
            ).astype(np.float32)
    return np.sort(output, axis=-1)


def expert_forecast_bank(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: Iterable[int],
    quantiles: Iterable[float],
    specs: Iterable[StatisticalExpertSpec],
) -> np.ndarray:
    return np.stack(
        [expert_forecast(returns, origins, horizons, quantiles, spec) for spec in specs],
        axis=0,
    )


def _episode_slices(length: int, episodes: int, minimum: int = 30) -> list[slice]:
    episodes = max(2, min(int(episodes), max(2, length // minimum)))
    boundaries = np.linspace(0, length, episodes + 1).round().astype(int)
    result = [
        slice(int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(episodes)
        if boundaries[index + 1] - boundaries[index] >= minimum
    ]
    if len(result) < 2:
        midpoint = length // 2
        return [slice(0, midpoint), slice(midpoint, length)]
    return result


def _candidate_weights(expert_losses: np.ndarray) -> list[np.ndarray]:
    count = len(expert_losses)
    candidates: list[np.ndarray] = [np.full(count, 1.0 / count)]
    for index in range(count):
        single = np.zeros(count)
        single[index] = 1.0
        candidates.append(single)
    centered = expert_losses - np.nanmin(expert_losses)
    scale = max(float(np.nanstd(expert_losses)), 1e-10)
    for temperature in (0.25, 0.5, 1.0, 2.0, 4.0):
        logits = -centered / (scale * temperature)
        weights = np.exp(logits - np.max(logits))
        candidates.append(weights / weights.sum())
    inverse = 1.0 / np.maximum(expert_losses, 1e-12)
    candidates.append(inverse / inverse.sum())
    return candidates


def _robust_weight_objective(
    weights: np.ndarray,
    experts: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
    episode_slices: list[slice],
    worst_episode_weight: float,
    l2_weight: float,
) -> float:
    prediction = np.tensordot(weights, experts, axes=(0, 0))
    losses = np.asarray(
        [
            pinball_loss_numpy(prediction[part], target[part], quantiles)
            for part in episode_slices
        ],
        dtype=np.float64,
    )
    equal = np.full_like(weights, 1.0 / len(weights))
    return float(
        losses.mean()
        + float(worst_episode_weight) * losses.max()
        + float(l2_weight) * np.square(weights - equal).sum()
    )


def fit_statistical_weights(
    expert_predictions: np.ndarray,
    target: np.ndarray,
    quantiles: Iterable[float],
    episodes: int = 6,
    worst_episode_weight: float = 0.35,
    l2_weight: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one convex expert mixture per forecast horizon."""

    experts = np.asarray(expert_predictions, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    q = np.asarray(tuple(float(value) for value in quantiles), dtype=np.float64)
    if experts.ndim != 4:
        raise ValueError("expert_predictions must have [expert, date, horizon, quantile]")
    slices = _episode_slices(experts.shape[1], episodes)
    weights = np.empty((experts.shape[2], experts.shape[0]), dtype=np.float64)
    episode_losses = np.empty((experts.shape[2], len(slices)), dtype=np.float64)
    for horizon_index in range(experts.shape[2]):
        horizon_experts = experts[:, :, horizon_index : horizon_index + 1, :]
        horizon_target = truth[:, horizon_index : horizon_index + 1]
        single_losses = np.asarray(
            [
                pinball_loss_numpy(value, horizon_target, q)
                for value in horizon_experts
            ]
        )
        candidates = _candidate_weights(single_losses)
        best = min(
            candidates,
            key=lambda value: _robust_weight_objective(
                value,
                horizon_experts,
                horizon_target,
                q,
                slices,
                worst_episode_weight,
                l2_weight,
            ),
        )
        result = minimize(
            _robust_weight_objective,
            x0=best,
            args=(
                horizon_experts,
                horizon_target,
                q,
                slices,
                worst_episode_weight,
                l2_weight,
            ),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(best),
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"maxiter": 250, "ftol": 1e-12},
        )
        chosen = result.x if result.success else best
        chosen = np.clip(chosen, 0.0, 1.0)
        chosen /= chosen.sum()
        weights[horizon_index] = chosen
        prediction = np.tensordot(chosen, horizon_experts, axes=(0, 0))
        episode_losses[horizon_index] = [
            pinball_loss_numpy(prediction[part], horizon_target[part], q)
            for part in slices
        ]
    return weights.astype(np.float32), episode_losses.astype(np.float64)


def apply_statistical_weights(
    expert_predictions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    experts = np.asarray(expert_predictions, dtype=np.float64)
    selected = np.asarray(weights, dtype=np.float64)
    if selected.shape != (experts.shape[2], experts.shape[0]):
        raise ValueError("weights must have shape [horizon, expert]")
    output = np.empty(experts.shape[1:], dtype=np.float32)
    for horizon_index in range(experts.shape[2]):
        output[:, horizon_index] = np.tensordot(
            selected[horizon_index],
            experts[:, :, horizon_index],
            axes=(0, 0),
        )
    return np.sort(output, axis=-1)


def build_statistical_ensemble(
    panel: LongHistoryPanel,
    fold: Fold,
    horizons: list[int],
    quantiles: list[float],
    lookback: int,
    embargo_days: int,
    common_max_horizon: int | None,
    settings: dict[str, Any],
) -> tuple[StatisticalEnsembleState, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    groups = split_long_origins(
        panel,
        fold,
        lookback,
        horizons,
        embargo_days,
        common_max_horizon=common_max_horizon,
    )
    selection_count = min(
        int(settings.get("selection_origins", 1500)),
        len(groups["train"]),
    )
    if selection_count < 180:
        raise ValueError("Statistical expert selection requires at least 180 train origins")
    selection_origins = groups["train"][-selection_count:]
    specs = default_expert_specs()
    selection_bank = expert_forecast_bank(
        panel.target_returns,
        selection_origins,
        horizons,
        quantiles,
        specs,
    )
    selection_targets = long_targets(panel, selection_origins, horizons)
    weights, episode_losses = fit_statistical_weights(
        selection_bank,
        selection_targets,
        quantiles,
        episodes=int(settings.get("episodes", 6)),
        worst_episode_weight=float(settings.get("worst_episode_weight", 0.35)),
        l2_weight=float(settings.get("l2_weight", 1e-7)),
    )
    train_end_index = int(
        np.flatnonzero(panel.dates <= pd.Timestamp(fold.train_end))[-1]
    )
    state = StatisticalEnsembleState(
        expert_specs=specs,
        weights=weights,
        episode_losses=episode_losses,
        selection_origins=selection_origins,
        train_end_index=train_end_index,
        horizons=tuple(horizons),
        quantiles=tuple(quantiles),
    )
    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, origins in groups.items():
        bank = expert_forecast_bank(
            panel.target_returns,
            origins,
            horizons,
            quantiles,
            specs,
        )
        outputs[split_name] = (
            origins,
            long_targets(panel, origins, horizons),
            apply_statistical_weights(bank, weights),
        )
    return state, outputs


def save_statistical_fold(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    performance = config["performance_v160"]
    settings = performance["statistical_experts"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    state, outputs = build_statistical_ensemble(
        panel,
        fold,
        horizons,
        quantiles,
        int(performance["structured_model"]["lookback"]),
        int(config["splits"].get("embargo_days", 0)),
        int(config["tailrisk"]["scenario_horizon"])
        if bool(performance.get("common_origin_required", True))
        else None,
        settings,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "statistical_ensemble.json", state.export())
    pd.DataFrame(
        state.weights,
        index=[f"horizon_{value}" for value in horizons],
        columns=[value.name for value in state.expert_specs],
    ).to_csv(output / "statistical_weights.csv")
    regime_spec = fit_regime_feature_spec(panel, outputs["train"][0])
    save_regime_spec(output / "regime_feature_spec.json", regime_spec)
    for split_name, (origins, targets, predictions) in outputs.items():
        predictions_frame(
            panel.dates[origins],
            targets,
            predictions,
            horizons,
            quantiles,
        ).to_csv(output / f"{split_name}_baseline_predictions.csv", index=False)
        features, score, regime = transform_regime_features(panel, origins, regime_spec)
        regime_frame(
            panel.dates[origins], features, score, regime, regime_spec.names
        ).to_csv(output / f"{split_name}_regime_features.csv", index=False)
    return output


def load_statistical_state(path: str | Path) -> StatisticalEnsembleState:
    import json

    return StatisticalEnsembleState.from_record(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
