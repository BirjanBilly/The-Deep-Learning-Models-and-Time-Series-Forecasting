from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from riskgraph.config import Fold
from riskgraph.io import predictions_frame, write_json
from riskgraph.performance_v140.baselines import ewma_student_t_forecast
from riskgraph.performance_v140.calibration import monotone_rearrange, pinball_loss_numpy
from riskgraph.performance_v150.data import LongHistoryPanel, long_targets, split_long_origins
from riskgraph.performance_v150.ensemble import circular_block_bootstrap_lower_bound
from riskgraph.performance_v160.statistical import (
    StatisticalExpertSpec,
    apply_statistical_weights,
    default_expert_specs,
    expert_forecast_bank,
    fit_statistical_weights,
)
from riskgraph.performance_v170.regime import (
    ProbabilisticRegimeSpec,
    fit_probabilistic_regime_spec,
    probabilistic_regime_frame,
    save_probabilistic_regime_spec,
    transform_probabilistic_regime_features,
)


@dataclass(frozen=True)
class StatisticalChampionState:
    """Frozen EWMA anchor plus a conservatively admitted adaptive expert pool."""

    expert_specs: tuple[StatisticalExpertSpec, ...]
    weights: np.ndarray  # adaptive pool weights [horizon, expert]
    anchor_degrees_of_freedom: float
    champion_blend: np.ndarray  # [horizon], 0 means exact frozen anchor
    spread_scales: np.ndarray  # [horizon]
    quantile_offsets: np.ndarray  # [horizon, quantile]
    accepted: np.ndarray  # [horizon]
    confirmation_improvements: np.ndarray
    bootstrap_lower_bounds: np.ndarray
    episode_improvements: tuple[tuple[float, ...], ...]
    selection_origins: np.ndarray
    train_end_index: int
    horizons: tuple[int, ...]
    quantiles: tuple[float, ...]

    def export(self) -> dict[str, Any]:
        return {
            "expert_specs": [spec.__dict__ for spec in self.expert_specs],
            "weights": self.weights.tolist(),
            "anchor_degrees_of_freedom": float(self.anchor_degrees_of_freedom),
            "champion_blend": self.champion_blend.tolist(),
            "spread_scales": self.spread_scales.tolist(),
            "quantile_offsets": self.quantile_offsets.tolist(),
            "accepted": self.accepted.tolist(),
            "confirmation_improvements": self.confirmation_improvements.tolist(),
            "bootstrap_lower_bounds": self.bootstrap_lower_bounds.tolist(),
            "episode_improvements": [list(values) for values in self.episode_improvements],
            "selection_origins": self.selection_origins.tolist(),
            "train_end_index": int(self.train_end_index),
            "horizons": list(self.horizons),
            "quantiles": list(self.quantiles),
            "anchor": {
                "name": "frozen_v154_ewma_student_t",
                "mean_decay": 0.985,
                "variance_decay": 0.94,
                "degrees_of_freedom_fitted_on_train_only": True,
            },
            "causality": (
                "Adaptive expert weights, champion blending, calibration and confirmation "
                "use disjoint chronological subsets ending no later than train_end."
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "StatisticalChampionState":
        return cls(
            expert_specs=tuple(StatisticalExpertSpec(**dict(value)) for value in record["expert_specs"]),
            weights=np.asarray(record["weights"], dtype=np.float64),
            anchor_degrees_of_freedom=float(record["anchor_degrees_of_freedom"]),
            champion_blend=np.asarray(record["champion_blend"], dtype=np.float64),
            spread_scales=np.asarray(record["spread_scales"], dtype=np.float64),
            quantile_offsets=np.asarray(record["quantile_offsets"], dtype=np.float64),
            accepted=np.asarray(record["accepted"], dtype=bool),
            confirmation_improvements=np.asarray(record["confirmation_improvements"], dtype=np.float64),
            bootstrap_lower_bounds=np.asarray(record["bootstrap_lower_bounds"], dtype=np.float64),
            episode_improvements=tuple(tuple(float(x) for x in row) for row in record["episode_improvements"]),
            selection_origins=np.asarray(record["selection_origins"], dtype=np.int64),
            train_end_index=int(record["train_end_index"]),
            horizons=tuple(int(value) for value in record["horizons"]),
            quantiles=tuple(float(value) for value in record["quantiles"]),
        )


# Backward-compatible name used by the v1.7 structured checkpoint code.
StatisticalEnsembleState = StatisticalChampionState


def anchor_forecast(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: Iterable[int],
    quantiles: Iterable[float],
    train_end_index: int,
    degrees_of_freedom: float | None = None,
) -> tuple[np.ndarray, float]:
    prediction, state = ewma_student_t_forecast(
        returns,
        np.asarray(origins, dtype=np.int64),
        [int(value) for value in horizons],
        [float(value) for value in quantiles],
        train_end_index=int(train_end_index),
        mean_decay=0.985,
        variance_decay=0.94,
        degrees_of_freedom=degrees_of_freedom,
    )
    return monotone_rearrange(prediction), float(state.degrees_of_freedom)


def _episode_slices(length: int, count: int, minimum: int = 20) -> list[slice]:
    count = max(2, min(int(count), max(2, length // minimum)))
    boundaries = np.linspace(0, length, count + 1).round().astype(int)
    return [
        slice(int(left), int(right))
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        if right - left >= minimum
    ]


def _date_pinball(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    error = target[..., None] - prediction
    losses = np.maximum(quantiles * error, (quantiles - 1.0) * error)
    return losses.mean(axis=(1, 2))


def _fit_calibration(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: np.ndarray,
    spread_grid: Iterable[float],
    shrinkage_grid: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    horizons = prediction.shape[1]
    spread = np.ones(horizons, dtype=np.float64)
    offsets = np.zeros((horizons, prediction.shape[2]), dtype=np.float64)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    for h in range(horizons):
        baseline_loss = pinball_loss_numpy(
            prediction[:, h : h + 1], target[:, h : h + 1], quantiles
        )
        best = baseline_loss
        for scale in spread_grid:
            median = prediction[:, h : h + 1, median_index : median_index + 1]
            candidate = median + float(scale) * (prediction[:, h : h + 1] - median)
            errors = target[:, h : h + 1, None] - candidate
            raw_offsets = np.asarray(
                [np.quantile(errors[:, 0, qi], tau) for qi, tau in enumerate(quantiles)],
                dtype=np.float64,
            )
            for shrinkage in shrinkage_grid:
                proposed = monotone_rearrange(
                    candidate + float(shrinkage) * raw_offsets[None, None, :]
                )
                score = pinball_loss_numpy(proposed, target[:, h : h + 1], quantiles)
                if score < best - 1e-12:
                    best = float(score)
                    spread[h] = float(scale)
                    offsets[h] = float(shrinkage) * raw_offsets
    return spread, offsets


def apply_champion_state(
    anchor: np.ndarray,
    adaptive_pool: np.ndarray,
    state: StatisticalChampionState,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float64)
    pool = np.asarray(adaptive_pool, dtype=np.float64)
    output = np.empty_like(anchor)
    median_index = int(np.argmin(np.abs(np.asarray(state.quantiles) - 0.5)))
    for h in range(anchor.shape[1]):
        candidate = anchor[:, h] + float(state.champion_blend[h]) * (
            pool[:, h] - anchor[:, h]
        )
        median = candidate[:, median_index : median_index + 1]
        candidate = median + float(state.spread_scales[h]) * (candidate - median)
        candidate = candidate + state.quantile_offsets[h][None, :]
        output[:, h] = candidate
    return monotone_rearrange(output).astype(np.float32)


def champion_forecast(
    returns: np.ndarray,
    origins: np.ndarray,
    horizons: Iterable[int],
    quantiles: Iterable[float],
    state: StatisticalChampionState,
) -> np.ndarray:
    bank = expert_forecast_bank(
        returns, origins, horizons, quantiles, state.expert_specs
    )
    pool = apply_statistical_weights(bank, state.weights)
    anchor, _ = anchor_forecast(
        returns,
        origins,
        horizons,
        quantiles,
        state.train_end_index,
        state.anchor_degrees_of_freedom,
    )
    return apply_champion_state(anchor, pool, state)


def build_statistical_champion(
    panel: LongHistoryPanel,
    fold: Fold,
    horizons: list[int],
    quantiles: list[float],
    lookback: int,
    embargo_days: int,
    common_max_horizon: int | None,
    settings: dict[str, Any],
) -> tuple[
    StatisticalChampionState,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    groups = split_long_origins(
        panel,
        fold,
        lookback,
        horizons,
        embargo_days,
        common_max_horizon=common_max_horizon,
    )
    count = min(int(settings.get("selection_origins", 1800)), len(groups["train"]))
    if count < 600:
        raise ValueError("Statistical champion requires at least 600 training origins")
    selection_origins = groups["train"][-count:]
    train_end_index = int(np.flatnonzero(panel.dates <= pd.Timestamp(fold.train_end))[-1])
    fit_end = max(240, int(round(count * float(settings.get("pool_fit_fraction", 0.55)))))
    select_end = max(fit_end + 100, int(round(count * float(settings.get("blend_selection_end_fraction", 0.75)))))
    calibration_end = max(select_end + 60, int(round(count * float(settings.get("calibration_end_fraction", 0.85)))))
    calibration_end = min(calibration_end, count - 120)
    pool_fit_origins = selection_origins[:fit_end]
    blend_origins = selection_origins[fit_end:select_end]
    calibration_origins = selection_origins[select_end:calibration_end]
    confirmation_origins = selection_origins[calibration_end:]

    specs = default_expert_specs()
    fit_bank = expert_forecast_bank(
        panel.target_returns, pool_fit_origins, horizons, quantiles, specs
    )
    fit_targets = long_targets(panel, pool_fit_origins, horizons)
    weights, _ = fit_statistical_weights(
        fit_bank,
        fit_targets,
        quantiles,
        episodes=int(settings.get("pool_episodes", 6)),
        worst_episode_weight=float(settings.get("pool_worst_episode_weight", 0.35)),
        l2_weight=float(settings.get("pool_l2_weight", 1e-7)),
    )
    _, anchor_df = anchor_forecast(
        panel.target_returns,
        pool_fit_origins,
        horizons,
        quantiles,
        train_end_index,
    )

    q = np.asarray(quantiles, dtype=np.float64)
    blend_bank = expert_forecast_bank(panel.target_returns, blend_origins, horizons, quantiles, specs)
    blend_pool = apply_statistical_weights(blend_bank, weights)
    blend_anchor, _ = anchor_forecast(
        panel.target_returns, blend_origins, horizons, quantiles, train_end_index, anchor_df
    )
    blend_target = long_targets(panel, blend_origins, horizons)
    blend_values = np.zeros(len(horizons), dtype=np.float64)
    blend_grid = [float(value) for value in settings.get("blend_weights", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])]
    for h in range(len(horizons)):
        base_loss = pinball_loss_numpy(blend_anchor[:, h : h + 1], blend_target[:, h : h + 1], q)
        best = base_loss
        for alpha in blend_grid:
            candidate = blend_anchor[:, h : h + 1] + alpha * (
                blend_pool[:, h : h + 1] - blend_anchor[:, h : h + 1]
            )
            score = pinball_loss_numpy(candidate, blend_target[:, h : h + 1], q)
            if score < best - 1e-12:
                best = float(score)
                blend_values[h] = alpha

    calibration_bank = expert_forecast_bank(
        panel.target_returns, calibration_origins, horizons, quantiles, specs
    )
    calibration_pool = apply_statistical_weights(calibration_bank, weights)
    calibration_anchor, _ = anchor_forecast(
        panel.target_returns, calibration_origins, horizons, quantiles, train_end_index, anchor_df
    )
    calibration_pre = calibration_anchor + blend_values[None, :, None] * (
        calibration_pool - calibration_anchor
    )
    spread_scales, offsets = _fit_calibration(
        calibration_pre,
        long_targets(panel, calibration_origins, horizons),
        q,
        settings.get("spread_scales", [0.95, 1.0, 1.05]),
        settings.get("offset_shrinkages", [0.0, 0.25, 0.5]),
    )

    provisional = StatisticalChampionState(
        expert_specs=specs,
        weights=weights,
        anchor_degrees_of_freedom=anchor_df,
        champion_blend=blend_values.copy(),
        spread_scales=spread_scales.copy(),
        quantile_offsets=offsets.copy(),
        accepted=np.ones(len(horizons), dtype=bool),
        confirmation_improvements=np.zeros(len(horizons), dtype=np.float64),
        bootstrap_lower_bounds=np.zeros(len(horizons), dtype=np.float64),
        episode_improvements=tuple(tuple() for _ in horizons),
        selection_origins=selection_origins,
        train_end_index=train_end_index,
        horizons=tuple(horizons),
        quantiles=tuple(quantiles),
    )
    confirmation_bank = expert_forecast_bank(
        panel.target_returns, confirmation_origins, horizons, quantiles, specs
    )
    confirmation_pool = apply_statistical_weights(confirmation_bank, weights)
    confirmation_anchor, _ = anchor_forecast(
        panel.target_returns, confirmation_origins, horizons, quantiles, train_end_index, anchor_df
    )
    confirmation_target = long_targets(panel, confirmation_origins, horizons)
    confirmation_candidate = apply_champion_state(confirmation_anchor, confirmation_pool, provisional)

    accepted = np.zeros(len(horizons), dtype=bool)
    confirmation_improvements = np.zeros(len(horizons), dtype=np.float64)
    lower_bounds = np.zeros(len(horizons), dtype=np.float64)
    episode_records: list[tuple[float, ...]] = []
    for h in range(len(horizons)):
        base_loss = pinball_loss_numpy(
            confirmation_anchor[:, h : h + 1], confirmation_target[:, h : h + 1], q
        )
        candidate_loss = pinball_loss_numpy(
            confirmation_candidate[:, h : h + 1], confirmation_target[:, h : h + 1], q
        )
        improvement = (base_loss - candidate_loss) / max(abs(base_loss), 1e-12)
        base_daily = _date_pinball(
            confirmation_anchor[:, h : h + 1], confirmation_target[:, h : h + 1], q
        )
        candidate_daily = _date_pinball(
            confirmation_candidate[:, h : h + 1], confirmation_target[:, h : h + 1], q
        )
        lower = circular_block_bootstrap_lower_bound(
            base_daily - candidate_daily,
            int(settings.get("block_length", 10)),
            int(settings.get("bootstrap_repetitions", 1000)),
            float(settings.get("one_sided_confidence", 0.80)),
            170 + h,
        )
        episodes = tuple(
            float((base_daily[part].mean() - candidate_daily[part].mean()) / max(abs(base_daily[part].mean()), 1e-12))
            for part in _episode_slices(len(base_daily), int(settings.get("confirmation_episodes", 4)))
        )
        positive_fraction = float(np.mean(np.asarray(episodes) >= 0.0)) if episodes else 0.0
        is_accepted = (
            blend_values[h] > 0.0
            and improvement >= float(settings.get("minimum_confirmation_improvement", 0.001))
            and lower > 0.0
            and positive_fraction >= float(settings.get("minimum_positive_episode_fraction", 0.75))
            and (min(episodes) if episodes else -np.inf) >= float(settings.get("maximum_episode_degradation", -0.0025))
        )
        accepted[h] = is_accepted
        confirmation_improvements[h] = improvement
        lower_bounds[h] = lower
        episode_records.append(episodes)
        if not is_accepted:
            blend_values[h] = 0.0
            spread_scales[h] = 1.0
            offsets[h] = 0.0

    state = StatisticalChampionState(
        expert_specs=specs,
        weights=weights,
        anchor_degrees_of_freedom=anchor_df,
        champion_blend=blend_values.astype(np.float32),
        spread_scales=spread_scales.astype(np.float32),
        quantile_offsets=offsets.astype(np.float32),
        accepted=accepted,
        confirmation_improvements=confirmation_improvements,
        bootstrap_lower_bounds=lower_bounds,
        episode_improvements=tuple(episode_records),
        selection_origins=selection_origins,
        train_end_index=train_end_index,
        horizons=tuple(horizons),
        quantiles=tuple(quantiles),
    )

    outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, origins in groups.items():
        bank = expert_forecast_bank(panel.target_returns, origins, horizons, quantiles, specs)
        pool = apply_statistical_weights(bank, weights)
        anchor, _ = anchor_forecast(
            panel.target_returns, origins, horizons, quantiles, train_end_index, anchor_df
        )
        champion = apply_champion_state(anchor, pool, state)
        outputs[split_name] = (
            origins,
            long_targets(panel, origins, horizons),
            champion,
            anchor,
            pool,
        )
    return state, outputs


def save_statistical_fold(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    performance = config["performance_v170"]
    settings = performance["statistical_champion"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    state, outputs = build_statistical_champion(
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
    write_json(output / "statistical_champion.json", state.export())
    # Compatibility filename used by structured checkpoints and tooling.
    write_json(output / "statistical_ensemble.json", state.export())
    pd.DataFrame(
        state.weights,
        index=[f"horizon_{value}" for value in horizons],
        columns=[value.name for value in state.expert_specs],
    ).to_csv(output / "statistical_weights.csv")
    pd.DataFrame(
        {
            "horizon": horizons,
            "adaptive_pool_blend": state.champion_blend,
            "spread_scale": state.spread_scales,
            "accepted": state.accepted,
            "confirmation_improvement": state.confirmation_improvements,
            "bootstrap_lower_bound": state.bootstrap_lower_bounds,
        }
    ).to_csv(output / "statistical_champion_weights.csv", index=False)

    regime_spec: ProbabilisticRegimeSpec = fit_probabilistic_regime_spec(
        panel, outputs["train"][0], performance["probabilistic_regime"]
    )
    save_probabilistic_regime_spec(output / "regime_feature_spec.json", regime_spec)
    for split_name, (origins, targets, champion, anchor, pool) in outputs.items():
        dates = panel.dates[origins]
        predictions_frame(dates, targets, champion, horizons, quantiles).to_csv(
            output / f"{split_name}_baseline_predictions.csv", index=False
        )
        predictions_frame(dates, targets, anchor, horizons, quantiles).to_csv(
            output / f"{split_name}_anchor_predictions.csv", index=False
        )
        predictions_frame(dates, targets, pool, horizons, quantiles).to_csv(
            output / f"{split_name}_adaptive_pool_predictions.csv", index=False
        )
        features, score, probabilities, confidence, entropy, regime = (
            transform_probabilistic_regime_features(panel, origins, regime_spec)
        )
        probabilistic_regime_frame(
            dates,
            features,
            score,
            probabilities,
            confidence,
            entropy,
            regime,
            regime_spec.feature_spec.names,
        ).to_csv(output / f"{split_name}_regime_features.csv", index=False)
    return output


def load_statistical_state(path: str | Path) -> StatisticalChampionState:
    import json

    return StatisticalChampionState.from_record(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
