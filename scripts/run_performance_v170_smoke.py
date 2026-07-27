#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from riskgraph.config import Fold
from riskgraph.performance_v150.data import LongHistoryPanel
from riskgraph.performance_v170.ensemble import fit_probabilistic_ensemble_gate
from riskgraph.performance_v170.regime import (
    fit_probabilistic_regime_spec,
    transform_probabilistic_regime_features,
)
from riskgraph.performance_v170.statistical import build_statistical_champion
from riskgraph.performance_v170.structured import (
    SSLRegimePatchTransformer,
    StructuredV170Config,
)
from riskgraph.tailrisk.evaluation import generate_paths_for_windows
from riskgraph.tailrisk.models import StableEWMABackboneFactorScaleGenerator
from riskgraph.tailrisk.strategies import build_strategy_bank
from riskgraph.tailrisk.synthetic import make_synthetic_tail_split
from riskgraph.tailrisk.trainer import (
    TailTrainingConfig,
    load_generator_from_checkpoint,
    train_tail_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic v1.7 integration smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/performance_v170_smoke")
    args = parser.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(170)

    dates = pd.bdate_range("2010-01-04", periods=1100)
    target_returns = rng.standard_t(df=6, size=len(dates)).astype(np.float32) * 0.008
    values = rng.normal(size=(len(dates), 8)).astype(np.float32)
    values[:, 0] = target_returns
    panel = LongHistoryPanel(
        dates=dates,
        values=values,
        masks=np.ones_like(values, dtype=np.float32),
        feature_names=("target:ret_1d", *[f"feature:{i}" for i in range(1, 8)]),
        target_returns=target_returns,
    )
    fold = Fold(
        name="synthetic",
        train_end=str(dates[820].date()),
        validation_end=str(dates[950].date()),
        test_end=str(dates[-1].date()),
    )
    settings = {
        "selection_origins": 650,
        "pool_fit_fraction": 0.55,
        "blend_selection_end_fraction": 0.75,
        "calibration_end_fraction": 0.85,
        "pool_episodes": 4,
        "pool_worst_episode_weight": 0.2,
        "pool_l2_weight": 1e-7,
        "blend_weights": [0.0, 0.25, 0.5, 1.0],
        "spread_scales": [1.0],
        "offset_shrinkages": [0.0],
        "block_length": 5,
        "bootstrap_repetitions": 100,
        "one_sided_confidence": 0.75,
        "confirmation_episodes": 3,
        "minimum_confirmation_improvement": -1.0,
        "minimum_positive_episode_fraction": 0.0,
        "maximum_episode_degradation": -1.0,
    }
    champion, outputs = build_statistical_champion(
        panel,
        fold,
        horizons=[1, 5],
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        lookback=60,
        embargo_days=5,
        common_max_horizon=5,
        settings=settings,
    )
    test_origins, targets, baseline, anchor, _ = outputs["test"]
    assert baseline.shape == anchor.shape
    assert np.all(np.diff(baseline, axis=-1) >= -1e-8)
    regime_spec = fit_probabilistic_regime_spec(
        panel,
        outputs["train"][0],
        {"persistence": 0.8, "confidence_floor": 0.4},
    )
    _, _, probabilities, confidence, _, _ = transform_probabilistic_regime_features(
        panel, test_origins, regime_spec
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    structured = SSLRegimePatchTransformer(
        StructuredV170Config(
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
    ).to(device)
    history = torch.randn(4, 20, 8, device=device)
    history[..., 4:] = 1.0
    baseline_tensor = torch.from_numpy(baseline[:4]).to(device)
    prediction, parameters, _ = structured(history, baseline_tensor)
    np.testing.assert_allclose(prediction.detach().cpu().numpy(), baseline[:4], atol=1e-6)
    reconstruction, regime_output, future_output, representation = structured.ssl_outputs(
        history, 0.20
    )
    (
        prediction.mean()
        + parameters.square().mean()
        + reconstruction.square().mean()
        + regime_output.square().mean()
        + future_output.square().mean()
        + representation.square().mean()
    ).backward()

    n = 180
    synthetic_target = rng.normal(0.0, 0.01, size=(n, 2)).astype(np.float32)
    q = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])
    synthetic_base = np.empty((n, 2, len(q)), dtype=np.float32)
    for index, tau in enumerate(q):
        synthetic_base[..., index] = synthetic_target + (tau - 0.5) * 0.05 + 0.002
    stronger = synthetic_base + 0.5 * (
        synthetic_target[..., None] - synthetic_base[..., 2:3]
    )
    seed_predictions = [
        stronger + rng.normal(0.0, 0.00005, stronger.shape) for _ in range(3)
    ]
    soft = np.zeros((n, 3), dtype=np.float32)
    soft[np.arange(n), np.arange(n) % 3] = 0.8
    soft += 0.2 / 3.0
    soft /= soft.sum(axis=1, keepdims=True)
    gate = fit_probabilistic_ensemble_gate(
        seed_predictions,
        synthetic_base,
        synthetic_target,
        soft,
        np.full(n, 0.8, dtype=np.float32),
        q.tolist(),
        [1, 5],
        {
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
            "minimum_effective_seed_count": 1.0,
            "confidence_floor": 0.4,
            "family_residual_caps": {"structured": [1.0, 1.0, 1.0]},
        },
        family="structured",
    )
    assert np.isclose(gate.seed_weights.sum(), 1.0)

    output = Path(args.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    split = make_synthetic_tail_split(
        train_samples=144,
        validation_samples=60,
        test_samples=36,
        horizon=6,
        n_assets=4,
        seed=170,
    )
    identity = np.eye(4, dtype=np.float32)
    train = replace(
        split.train,
        baseline_cholesky=np.repeat(identity[None, ...], len(split.train.origins), axis=0),
    )
    validation = replace(
        split.validation,
        baseline_cholesky=np.repeat(identity[None, ...], len(split.validation.origins), axis=0),
    )
    test = replace(
        split.test,
        baseline_cholesky=np.repeat(identity[None, ...], len(split.test.origins), axis=0),
    )
    bank_strategy = build_strategy_bank(
        train.actual_paths,
        mode="full",
        num_eigenportfolios=2,
        num_random_portfolios=2,
        seed=170,
        signal_window=3,
    )
    tail_run = output / "stress_shrunk_gom" / "seed_170"
    train_tail_model(
        train,
        validation,
        bank_strategy,
        alphas=[0.05],
        output_dir=tail_run,
        seed=170,
        config=TailTrainingConfig(
            objective="gom",
            epochs=1,
            minimum_epochs=1,
            batch_size=20,
            latent_dim=24,
            validation_scenarios=40,
            patience=2,
            generator_backbone="ewma_factor_scale_stable",
            baseline_hidden_size=24,
            baseline_gru_layers=1,
            factor_rank=2,
            baseline_initial_gate=0.015,
            factor_scale_limit=0.16,
            idio_scale_limit=0.10,
            drift_limit=0.02,
            skew_limit=0.06,
            correlation_loss_weight=0.05,
            autocorrelation_loss_weight=0.05,
            marginal_loss_weight=0.05,
            energy_loss_weight=0.01,
            generator_ema_decay=0.90,
        ),
        device_name=args.device,
    )
    loaded, checkpoint, loaded_device = load_generator_from_checkpoint(
        tail_run / "best_checkpoint.pt", device_name=args.device
    )
    assert isinstance(loaded, StableEWMABackboneFactorScaleGenerator)
    scenarios = generate_paths_for_windows(
        loaded,
        test,
        scenarios_per_origin=10,
        latent_dim=int(checkpoint["training_config"]["latent_dim"]),
        device=loaded_device,
        noise_distribution=str(checkpoint["training_config"]["noise_distribution"]),
        degrees_of_freedom=float(checkpoint["training_config"]["degrees_of_freedom"]),
        chunk_size=5,
    )
    assert scenarios.shape == (36, 10, 6, 4)
    assert np.isfinite(scenarios).all()
    (output / "SMOKE_PASSED.txt").write_text(
        "PERFORMANCE V1.7 SMOKE PASSED\n", encoding="utf-8"
    )
    print("PERFORMANCE V1.7 SMOKE PASSED")
    print(f"outputs -> {output}")


if __name__ == "__main__":
    main()
