from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.performance_v150.data import LongHistoryPanel
from riskgraph.performance_v160.regime import (
    RegimeFeatureSpec,
    fit_regime_feature_spec,
    raw_regime_features,
)


@dataclass(frozen=True)
class ProbabilisticRegimeSpec:
    """Causal three-state regime model with uncertainty and persistence.

    The underlying stress score is fitted on training origins only. Probabilities
    are smooth functions of that score, then filtered forward in time so no
    future observation can affect the current regime allocation.
    """

    feature_spec: RegimeFeatureSpec
    temperature: float
    persistence: float
    confidence_floor: float

    def export(self) -> dict[str, Any]:
        return {
            "feature_spec": self.feature_spec.export(),
            "temperature": float(self.temperature),
            "persistence": float(self.persistence),
            "confidence_floor": float(self.confidence_floor),
            "states": ["calm", "transition", "stress"],
            "causality": (
                "Feature standardisation and score thresholds are fitted only on "
                "training origins. Probability filtering is strictly forward-only."
            ),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ProbabilisticRegimeSpec":
        return cls(
            feature_spec=RegimeFeatureSpec.from_record(record["feature_spec"]),
            temperature=float(record["temperature"]),
            persistence=float(record["persistence"]),
            confidence_floor=float(record["confidence_floor"]),
        )


def fit_probabilistic_regime_spec(
    panel: LongHistoryPanel,
    train_origins: np.ndarray,
    settings: dict[str, Any],
) -> ProbabilisticRegimeSpec:
    base = fit_regime_feature_spec(
        panel,
        train_origins,
        thresholds=tuple(float(value) for value in settings.get("threshold_quantiles", (0.55, 0.85))),
    )
    gap = max(float(base.thresholds[1] - base.thresholds[0]), 1e-3)
    temperature = float(settings.get("temperature", 0.35 * gap))
    return ProbabilisticRegimeSpec(
        feature_spec=base,
        temperature=max(temperature, 1e-3),
        persistence=float(np.clip(settings.get("persistence", 0.85), 0.0, 0.995)),
        confidence_floor=float(np.clip(settings.get("confidence_floor", 0.45), 0.0, 0.95)),
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def _raw_probabilities(score: np.ndarray, spec: ProbabilisticRegimeSpec) -> np.ndarray:
    low, high = spec.feature_spec.thresholds
    temperature = max(float(spec.temperature), 1e-6)
    calm = _sigmoid((low - score) / temperature)
    stress = _sigmoid((score - high) / temperature)
    transition = np.maximum(1.0 - calm - stress, 1e-6)
    probabilities = np.stack([calm, transition, stress], axis=1)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _causal_smooth(probabilities: np.ndarray, persistence: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if len(values) == 0:
        return values
    output = np.empty_like(values)
    output[0] = values[0]
    for index in range(1, len(values)):
        output[index] = persistence * output[index - 1] + (1.0 - persistence) * values[index]
        output[index] /= output[index].sum()
    return output


def transform_probabilistic_regime_features(
    panel: LongHistoryPanel,
    origins: np.ndarray,
    spec: ProbabilisticRegimeSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, names = raw_regime_features(panel, origins)
    if names != spec.feature_spec.names:
        raise ValueError("Regime feature names do not match fitted specification")
    standardized = (
        raw - spec.feature_spec.mean[None, :]
    ) / spec.feature_spec.std[None, :]
    score = standardized @ spec.feature_spec.stress_weights
    probabilities = _causal_smooth(_raw_probabilities(score, spec), spec.persistence)
    confidence = probabilities.max(axis=1)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    regime = probabilities.argmax(axis=1).astype(np.int64)
    return (
        standardized.astype(np.float32),
        score.astype(np.float32),
        probabilities.astype(np.float32),
        confidence.astype(np.float32),
        entropy.astype(np.float32),
        regime,
    )


def probabilistic_regime_frame(
    dates: pd.DatetimeIndex,
    features: np.ndarray,
    score: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    regime: np.ndarray,
    names: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=list(names))
    frame.insert(0, "date", pd.DatetimeIndex(dates).strftime("%Y-%m-%d"))
    frame["stress_score"] = score
    frame["p_calm"] = probabilities[:, 0]
    frame["p_transition"] = probabilities[:, 1]
    frame["p_stress"] = probabilities[:, 2]
    frame["regime_confidence"] = confidence
    frame["regime_entropy"] = entropy
    frame["regime"] = regime
    return frame


def save_probabilistic_regime_spec(path: str | Path, spec: ProbabilisticRegimeSpec) -> None:
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec.export(), indent=2, sort_keys=True), encoding="utf-8")
