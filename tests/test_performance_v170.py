from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from riskgraph.config import Fold
from riskgraph.performance_v150.data import LongHistoryPanel
from riskgraph.performance_v170.ensemble import (
    apply_probabilistic_gate,
    effective_seed_count,
    fit_probabilistic_ensemble_gate,
)
from riskgraph.performance_v170.regime import (
    fit_probabilistic_regime_spec,
    transform_probabilistic_regime_features,
)
from riskgraph.performance_v170.settings import get_probabilistic_gate_settings
from riskgraph.performance_v170.statistical import (
    StatisticalChampionState,
    anchor_forecast,
    apply_champion_state,
    build_statistical_champion,
    default_expert_specs,
)
from riskgraph.performance_v170.structured import (
    SSLRegimePatchTransformer,
    StructuredV170Config,
)
from riskgraph.tailrisk.data import TailWindowSet
from riskgraph.tailrisk.evaluation import generate_paths_for_windows
from riskgraph.tailrisk.models import StableEWMABackboneFactorScaleGenerator


def _long_panel(rows: int = 1050, features: int = 8) -> LongHistoryPanel:
    rng = np.random.default_rng(170)
    returns = rng.standard_t(df=6, size=rows).astype(np.float32) * 0.008
    values = rng.normal(size=(rows, features)).astype(np.float32)
    values[:, 0] = returns
    masks = np.ones_like(values, dtype=np.float32)
    names = ("target:ret_1d", *[f"feature:{index}" for index in range(1, features)])
    return LongHistoryPanel(
        dates=pd.bdate_range("2010-01-04", periods=rows),
        values=values,
        masks=masks,
        feature_names=names,
        target_returns=returns,
    )


def _champion_settings() -> dict[str, object]:
    return {
        "selection_origins": 650,
        "pool_fit_fraction": 0.55,
        "blend_selection_end_fraction": 0.75,
        "calibration_end_fraction": 0.85,
        "pool_episodes": 4,
        "pool_worst_episode_weight": 0.2,
        "pool_l2_weight": 1e-7,
        "blend_weights": [0.0, 0.5, 1.0],
        "spread_scales": [1.0],
        "offset_shrinkages": [0.0],
        "block_length": 5,
        "bootstrap_repetitions": 50,
        "one_sided_confidence": 0.75,
        "confirmation_episodes": 3,
        "minimum_confirmation_improvement": 1.0,
        "minimum_positive_episode_fraction": 1.0,
        "maximum_episode_degradation": 0.0,
    }


def test_statistical_champion_rejected_horizons_equal_anchor() -> None:
    panel = _long_panel()
    fold = Fold(
        name="synthetic",
        train_end=str(panel.dates[800].date()),
        validation_end=str(panel.dates[920].date()),
        test_end=str(panel.dates[-1].date()),
    )
    state, outputs = build_statistical_champion(
        panel,
        fold,
        [1, 5],
        [0.05, 0.25, 0.5, 0.75, 0.95],
        lookback=60,
        embargo_days=5,
        common_max_horizon=5,
        settings=_champion_settings(),
    )
    assert not state.accepted.any()
    for _, _, champion, anchor, _ in outputs.values():
        np.testing.assert_allclose(champion, anchor, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(state.weights.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(state.weights >= 0.0)


def test_anchor_forecast_is_monotone() -> None:
    panel = _long_panel(500)
    origins = np.arange(252, 450)
    forecast, degrees = anchor_forecast(
        panel.target_returns,
        origins,
        [1, 5],
        [0.05, 0.25, 0.5, 0.75, 0.95],
        train_end_index=400,
    )
    assert degrees > 2.0
    assert forecast.shape == (len(origins), 2, 5)
    assert np.all(np.diff(forecast, axis=-1) >= -1e-8)


def test_apply_champion_state_can_blend_but_preserves_monotonicity() -> None:
    anchor = np.asarray([[[-2.0, -1.0, 0.0, 1.0, 2.0]]], dtype=np.float32)
    pool = np.asarray([[[-1.5, -0.5, 0.2, 0.8, 1.6]]], dtype=np.float32)
    state = StatisticalChampionState(
        expert_specs=default_expert_specs(),
        weights=np.full((1, len(default_expert_specs())), 1 / len(default_expert_specs())),
        anchor_degrees_of_freedom=5.0,
        champion_blend=np.asarray([0.5]),
        spread_scales=np.asarray([1.0]),
        quantile_offsets=np.zeros((1, 5)),
        accepted=np.asarray([True]),
        confirmation_improvements=np.asarray([0.01]),
        bootstrap_lower_bounds=np.asarray([0.001]),
        episode_improvements=((0.01, 0.02),),
        selection_origins=np.arange(10),
        train_end_index=100,
        horizons=(1,),
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    )
    result = apply_champion_state(anchor, pool, state)
    assert np.all(np.diff(result, axis=-1) >= -1e-8)
    assert not np.allclose(result, anchor)


def test_probabilistic_regime_is_normalized_and_causal() -> None:
    panel = _long_panel(500)
    origins = np.arange(252, 450)
    spec = fit_probabilistic_regime_spec(
        panel,
        origins[:120],
        {"persistence": 0.8, "confidence_floor": 0.4},
    )
    features, score, probabilities, confidence, entropy, regime = (
        transform_probabilistic_regime_features(panel, origins, spec)
    )
    assert features.shape == (len(origins), 7)
    assert np.isfinite(features).all()
    assert np.isfinite(score).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
    assert np.all(entropy >= 0.0)
    assert set(np.unique(regime)).issubset({0, 1, 2})


def test_ssl_structured_adapter_and_future_head_support_backward() -> None:
    config = StructuredV170Config(
        value_channels=4,
        lookback=20,
        horizons=2,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        ssl_target_dim=26,
        regime_target_dim=5,
        future_target_dim=6,
        patch_lengths=(5, 10),
        patch_strides=(5, 5),
        channel_dim=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    model = SSLRegimePatchTransformer(config)
    history = torch.randn(4, 20, 8)
    history[..., 4:] = 1.0
    baseline = torch.tensor(
        [[[-0.04, -0.01, 0.0, 0.01, 0.04], [-0.08, -0.02, 0.0, 0.02, 0.08]]]
    ).expand(4, -1, -1)
    prediction, parameters, regime = model(history, baseline)
    torch.testing.assert_close(prediction, baseline, atol=1e-6, rtol=0.0)
    reconstruction, regime_ssl, future, representation = model.ssl_outputs(history, 0.25)
    loss = (
        prediction.mean()
        + parameters.square().mean()
        + regime.square().mean()
        + reconstruction.square().mean()
        + regime_ssl.square().mean()
        + future.square().mean()
        + representation.square().mean()
    )
    loss.backward()
    assert future.shape == (4, 6)
    assert any(parameter.grad is not None for parameter in model.parameters())


def _gate_settings() -> dict[str, object]:
    return {
        "selection_fraction": 0.45,
        "tuning_fraction": 0.25,
        "minimum_selection_improvement": 0.0,
        "minimum_tuning_improvement": -1.0,
        "minimum_confirmation_improvement": -1.0,
        "residual_weights": [0.0, 0.5, 1.0],
        "block_length": 5,
        "bootstrap_repetitions": 100,
        "one_sided_confidence": 0.75,
        "minimum_regime_samples": 10,
        "confirmation_episodes": 3,
        "minimum_positive_episode_fraction": 0.0,
        "maximum_episode_degradation": -1.0,
        "maximum_regime_degradation": -1.0,
        "maximum_risk_calibration_degradation": 1.0,
        "seed_weight_shrinkage": 0.2,
        "maximum_seed_weight": 0.8,
        "seed_concentration_penalty": 0.0,
        "minimum_effective_seed_count": 1.0,
        "confidence_floor": 0.4,
        "family_residual_caps": {"structured": [1.0, 1.0, 1.0]},
    }


def test_probabilistic_gate_fallback_is_exact() -> None:
    rng = np.random.default_rng(171)
    n = 160
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    target = rng.normal(0.0, 0.01, size=(n, 2)).astype(np.float32)
    baseline = np.empty((n, 2, len(quantiles)), dtype=np.float32)
    for index, tau in enumerate(quantiles):
        baseline[..., index] = target + (tau - 0.5) * 0.05
    worse = baseline + 0.02
    probabilities = np.full((n, 3), 1 / 3, dtype=np.float32)
    confidence = np.full(n, 1 / 3, dtype=np.float32)
    settings = _gate_settings()
    settings["minimum_selection_improvement"] = 0.05
    gate = fit_probabilistic_ensemble_gate(
        [worse, worse, worse],
        baseline,
        target,
        probabilities,
        confidence,
        quantiles,
        [1, 5],
        settings,
        family="structured",
    )
    final = apply_probabilistic_gate(
        worse,
        baseline,
        probabilities,
        confidence,
        gate,
        confidence_floor=0.4,
    )
    assert gate.fallback_to_baseline
    np.testing.assert_allclose(final, baseline, atol=0.0, rtol=0.0)


def test_seed_weight_regularization_prevents_single_seed_effective_count() -> None:
    assert effective_seed_count(np.asarray([0.8, 0.1, 0.1])) > 1.5
    assert effective_seed_count(np.asarray([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_probabilistic_gate_settings_uses_canonical_key() -> None:
    settings = get_probabilistic_gate_settings(
        {"performance_v170": {"probabilistic_gate": {"block_length": 5}}}
    )
    assert settings["block_length"] == 5


def test_probabilistic_gate_settings_reports_missing_mapping() -> None:
    with pytest.raises(KeyError, match="probabilistic_gate"):
        get_probabilistic_gate_settings({"performance_v170": {}})


def test_stable_factor_scale_generator_starts_at_backbone() -> None:
    torch.manual_seed(170)
    generator = StableEWMABackboneFactorScaleGenerator(
        latent_dim=12,
        horizon=5,
        n_assets=4,
        n_regimes=3,
        hidden_size=16,
        layers=1,
        factor_rank=2,
        degrees_of_freedom=5.0,
        initial_gate=0.015,
        factor_scale_limit=0.16,
        idio_scale_limit=0.10,
        drift_limit=0.02,
        skew_limit=0.06,
    )
    noise = torch.randn(6, 12)
    regime = torch.arange(6) % 3
    cholesky = torch.eye(4).expand(6, -1, -1)
    output = generator(noise, regime, cholesky)
    normalized_weight = F.normalize(generator.noise_projection.weight, dim=1)
    innovation = F.linear(
        noise * generator.noise_unit_scale,
        normalized_weight,
        generator.noise_projection.bias,
    ).view(6, 5, 4)
    backbone = torch.einsum("bhj,bij->bhi", innovation, cholesky)
    torch.testing.assert_close(output, backbone, atol=1e-6, rtol=1e-6)


def test_stable_generator_supported_by_scenario_evaluation() -> None:
    generator = StableEWMABackboneFactorScaleGenerator(
        latent_dim=8,
        horizon=5,
        n_assets=3,
        n_regimes=2,
        hidden_size=12,
        layers=1,
        factor_rank=2,
    )
    windows = TailWindowSet(
        normalized_paths=np.zeros((4, 5, 3), dtype=np.float32),
        actual_paths=np.zeros((4, 5, 3), dtype=np.float32),
        scales=np.ones((4, 3), dtype=np.float32),
        regimes=np.asarray([0, 1, 0, 1], dtype=np.int64),
        origins=np.arange(4, dtype=np.int64),
        dates=np.asarray(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
        regime_edges=np.asarray([0.5], dtype=np.float32),
        baseline_cholesky=np.repeat(np.eye(3, dtype=np.float32)[None, ...], 4, axis=0),
    )
    generated = generate_paths_for_windows(
        generator,
        windows,
        scenarios_per_origin=7,
        latent_dim=8,
        device=torch.device("cpu"),
        noise_distribution="student_t",
        degrees_of_freedom=5.0,
        chunk_size=4,
    )
    assert generated.shape == (4, 7, 5, 3)
    assert np.isfinite(generated).all()
