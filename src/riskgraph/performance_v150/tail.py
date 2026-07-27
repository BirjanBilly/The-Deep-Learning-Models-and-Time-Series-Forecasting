from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel
from riskgraph.io import predictions_frame, write_json
from riskgraph.performance_v140.baselines import ewma_student_t_forecast
from riskgraph.tailrisk.evaluation import (
    evaluate_scenario_generator,
    generate_paths_for_windows,
    generate_regime_samples,
    scenario_forecasts,
)
from riskgraph.tailrisk.experiment import (
    prepare_fold_windows,
    strategy_bank_from_checkpoint,
    strategy_bank_from_config,
)
from riskgraph.tailrisk.trainer import load_generator_from_checkpoint


def refined_tail_config_v150(config: dict[str, Any]) -> dict[str, Any]:
    refined = copy.deepcopy(config)
    settings = refined["performance_v150"]["tail_factor_scale"]
    refined["tailrisk"].update(
        {
            "output_dir": refined["performance_v150"]["output_dir"],
            "generator_backbone": "ewma_factor_scale",
            "strategy_mode": str(settings["strategy_mode"]),
            "epochs": int(settings["epochs"]),
            "patience": int(settings["patience"]),
            "batch_size": int(settings["batch_size"]),
            "latent_dim": int(settings["latent_dim"]),
            "baseline_cov_lookback": int(settings["baseline_cov_lookback"]),
            "baseline_cov_decay": float(settings["baseline_cov_decay"]),
            "baseline_cov_shrinkage": float(settings["baseline_cov_shrinkage"]),
            "baseline_hidden_size": int(settings["hidden_size"]),
            "baseline_gru_layers": int(settings["gru_layers"]),
            "baseline_initial_gate": float(settings["initial_gate"]),
            "factor_rank": int(settings["factor_rank"]),
            "factor_scale_limit": float(settings["factor_scale_limit"]),
            "idio_scale_limit": float(settings["idio_scale_limit"]),
            "drift_limit": float(settings["drift_limit"]),
            "skew_limit": float(settings["skew_limit"]),
            "correlation_loss_weight": float(settings["correlation_loss_weight"]),
            "autocorrelation_loss_weight": float(settings["autocorrelation_loss_weight"]),
            "marginal_loss_weight": float(settings["marginal_loss_weight"]),
            "energy_loss_weight": float(settings["energy_loss_weight"]),
            "validation_correlation_weight": float(
                settings["validation_correlation_weight"]
            ),
            "validation_autocorrelation_weight": float(
                settings["validation_autocorrelation_weight"]
            ),
            "validation_scenarios": int(settings["validation_scenarios"]),
            "conditioning_mode": "none",
            "quantile_consistency_weight": 0.0,
        }
    )
    return refined


def _target_returns(panel: Panel) -> np.ndarray:
    index = panel.asset_feature_names.index("ret_1d")
    return panel.asset_features[:, panel.target_index, index].astype(np.float64)


def _train_end_index(panel: Panel, fold: Fold) -> int:
    return int(
        np.searchsorted(panel.dates.values, np.datetime64(fold.train_end), side="right")
        - 1
    )


def export_tail_seed_predictions_v150(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device_name: str = "auto",
) -> Path:
    started = time.time()
    refined = refined_tail_config_v150(config)
    settings = refined["performance_v150"]["tail_factor_scale"]
    generator, checkpoint, device = load_generator_from_checkpoint(
        checkpoint_path,
        device_name=device_name,
    )
    _, train_windows, validation_windows, test_windows = prepare_fold_windows(
        panel,
        fold,
        refined,
        conditioning_spec=checkpoint.get("conditioner_spec"),
    )
    latent_dim = int(checkpoint["training_config"]["latent_dim"])
    noise_distribution = str(checkpoint["training_config"]["noise_distribution"])
    degrees_of_freedom = float(checkpoint["training_config"]["degrees_of_freedom"])
    horizons = [int(value) for value in refined["tailrisk"]["forecast_horizons"]]
    quantiles = [float(value) for value in refined["features"]["quantiles"]]
    target_returns = _target_returns(panel)
    train_end = _train_end_index(panel, fold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for split_name, windows, scenarios in (
        (
            "validation",
            validation_windows,
            int(settings["validation_scenarios_per_origin"]),
        ),
        ("test", test_windows, int(settings["evaluation_scenarios_per_origin"])),
    ):
        paths = generate_paths_for_windows(
            generator,
            windows,
            scenarios_per_origin=scenarios,
            latent_dim=latent_dim,
            device=device,
            noise_distribution=noise_distribution,
            degrees_of_freedom=degrees_of_freedom,
            chunk_size=int(settings["generation_chunk_size"]),
        )
        model_prediction, target, _ = scenario_forecasts(
            paths,
            windows,
            target_index=panel.target_index,
            horizons=horizons,
            quantiles=quantiles,
        )
        baseline_prediction, state = ewma_student_t_forecast(
            target_returns,
            windows.origins,
            horizons,
            quantiles,
            train_end_index=train_end,
            mean_decay=float(refined["performance_v140"]["ewma_mean_decay"]),
            variance_decay=float(refined["performance_v140"]["ewma_variance_decay"]),
        )
        predictions_frame(
            panel.dates[windows.origins],
            target,
            model_prediction,
            horizons,
            quantiles,
        ).to_csv(output / f"{split_name}_raw_predictions.csv", index=False)
        predictions_frame(
            panel.dates[windows.origins],
            target,
            baseline_prediction,
            horizons,
            quantiles,
        ).to_csv(output / f"{split_name}_baseline_predictions.csv", index=False)
        if split_name == "validation":
            write_json(
                output / "baseline_state.json",
                {"student_t_degrees_of_freedom": float(state.degrees_of_freedom)},
            )

    training_bank = strategy_bank_from_checkpoint(checkpoint)
    unseen_bank = strategy_bank_from_config(
        train_windows.actual_paths,
        refined["tailrisk"],
        mode=str(settings["strategy_mode"]),
        seed=seed + 100_000,
    )
    real_paths, generated_paths = generate_regime_samples(
        generator,
        test_windows,
        scenarios_per_regime=int(settings["evaluation_scenarios_per_regime"]),
        latent_dim=latent_dim,
        device=device,
        noise_distribution=noise_distribution,
        degrees_of_freedom=degrees_of_freedom,
        seed=seed,
    )
    training_eval = evaluate_scenario_generator(
        real_paths,
        generated_paths,
        training_bank,
        [float(value) for value in refined["tailrisk"]["alphas"]],
        device,
        weight=float(checkpoint["training_config"]["score_weight"]),
    )
    unseen_eval = evaluate_scenario_generator(
        real_paths,
        generated_paths,
        unseen_bank,
        [float(value) for value in refined["tailrisk"]["alphas"]],
        device,
        weight=float(checkpoint["training_config"]["score_weight"]),
    )
    write_json(output / "scenario_metrics.json", unseen_eval.structural_metrics)
    write_json(
        output / "scenario_metrics_training_bank.json",
        training_eval.structural_metrics,
    )
    unseen_eval.strategy_metrics.to_csv(
        output / "strategy_tail_metrics_unseen_bank.csv", index=False
    )
    training_eval.strategy_metrics.to_csv(
        output / "strategy_tail_metrics_training_bank.csv", index=False
    )
    unseen_eval.rank_frequency.to_csv(output / "rank_frequency.csv", index=False)
    write_json(
        output / "seed_evaluation_metadata.json",
        {
            "fold": fold.name,
            "seed": int(seed),
            "model_family": "tail_factor_scale_v150",
            "generator_type": checkpoint.get("generator_type"),
            "elapsed_seconds": time.time() - started,
        },
    )
    return output
