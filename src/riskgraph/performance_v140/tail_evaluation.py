from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel, stress_mask_for_origins
from riskgraph.io import predictions_frame, write_json
from riskgraph.performance_v140.baselines import ewma_student_t_forecast
from riskgraph.performance_v140.calibration import apply_forecast_gate, fit_forecast_gate
from riskgraph.tailrisk.evaluation import (
    evaluate_scenario_forecasts,
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
from riskgraph.tailrisk.reporting import (
    plot_forecast_fan,
    plot_matrix_pair,
    plot_rank_frequency,
    plot_training_history,
)
from riskgraph.tailrisk.trainer import load_generator_from_checkpoint


def refined_tail_config(config: dict[str, Any]) -> dict[str, Any]:
    refined = copy.deepcopy(config)
    settings = refined["performance_v140"]["tailgan"]
    refined["tailrisk"].update(
        {
            "generator_backbone": "ewma_gru",
            "baseline_cov_lookback": int(settings["baseline_cov_lookback"]),
            "baseline_cov_decay": float(settings["baseline_cov_decay"]),
            "baseline_cov_shrinkage": float(settings["baseline_cov_shrinkage"]),
        }
    )
    return refined


def _target_returns(panel: Panel) -> np.ndarray:
    return_index = panel.asset_feature_names.index("ret_1d")
    return panel.asset_features[:, panel.target_index, return_index].astype(np.float64)


def _train_end_index(panel: Panel, fold: Fold) -> int:
    return int(np.searchsorted(panel.dates.values, np.datetime64(fold.train_end), side="right") - 1)


def _forecast_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
):
    return evaluate_scenario_forecasts(prediction, target, quantiles, horizons)


def evaluate_tail_fold_v140(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device_name: str = "auto",
) -> Path:
    started = time.time()
    refined = refined_tail_config(config)
    settings = refined["performance_v140"]["tailgan"]
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
    degrees_of_freedom = float(
        checkpoint["training_config"]["degrees_of_freedom"]
    )
    validation_scenarios = int(settings["validation_scenarios_per_origin"])
    test_scenarios = int(settings["evaluation_scenarios_per_origin"])
    chunk_size = int(settings["generation_chunk_size"])
    horizons = [int(value) for value in refined["tailrisk"]["forecast_horizons"]]
    quantiles = [float(value) for value in refined["features"]["quantiles"]]

    validation_paths = generate_paths_for_windows(
        generator,
        validation_windows,
        scenarios_per_origin=validation_scenarios,
        latent_dim=latent_dim,
        device=device,
        noise_distribution=noise_distribution,
        degrees_of_freedom=degrees_of_freedom,
        chunk_size=chunk_size,
    )
    validation_model, validation_target, _ = scenario_forecasts(
        validation_paths,
        validation_windows,
        target_index=panel.target_index,
        horizons=horizons,
        quantiles=quantiles,
    )
    target_returns = _target_returns(panel)
    train_end = _train_end_index(panel, fold)
    validation_baseline, baseline_state = ewma_student_t_forecast(
        target_returns,
        validation_windows.origins,
        horizons,
        quantiles,
        train_end_index=train_end,
        mean_decay=float(refined["performance_v140"]["ewma_mean_decay"]),
        variance_decay=float(refined["performance_v140"]["ewma_variance_decay"]),
    )
    gate = fit_forecast_gate(
        validation_model,
        validation_baseline,
        validation_target,
        quantiles,
        horizons=horizons,
        min_relative_improvement=float(
            refined["performance_v140"]["minimum_validation_improvement"]
        ),
        min_confirmation_improvement=float(
            refined["performance_v140"]["minimum_confirmation_improvement"]
        ),
        calibration_fraction=float(
            refined["performance_v140"]["gate_calibration_fraction"]
        ),
        confirmation_fraction_of_calibration=float(
            refined["performance_v140"][
                "gate_confirmation_fraction_of_calibration"
            ]
        ),
        coverage_penalty_weight=float(
            refined["performance_v140"]["coverage_penalty_weight"]
        ),
    )

    test_paths = generate_paths_for_windows(
        generator,
        test_windows,
        scenarios_per_origin=test_scenarios,
        latent_dim=latent_dim,
        device=device,
        noise_distribution=noise_distribution,
        degrees_of_freedom=degrees_of_freedom,
        chunk_size=chunk_size,
    )
    test_model, test_target, _ = scenario_forecasts(
        test_paths,
        test_windows,
        target_index=panel.target_index,
        horizons=horizons,
        quantiles=quantiles,
    )
    test_baseline, _ = ewma_student_t_forecast(
        target_returns,
        test_windows.origins,
        horizons,
        quantiles,
        train_end_index=train_end,
        mean_decay=float(refined["performance_v140"]["ewma_mean_decay"]),
        variance_decay=float(refined["performance_v140"]["ewma_variance_decay"]),
        degrees_of_freedom=baseline_state.degrees_of_freedom,
    )
    calibrated = apply_forecast_gate(
        test_model,
        test_baseline,
        quantiles,
        gate,
        observed_targets=test_target,
        horizons=horizons,
    )
    metrics, detail, var_backtests = _forecast_metrics(
        calibrated,
        test_target,
        quantiles,
        horizons,
    )
    raw_metrics, raw_detail, raw_var = _forecast_metrics(
        test_model,
        test_target,
        quantiles,
        horizons,
    )
    baseline_metrics, baseline_detail, baseline_var = _forecast_metrics(
        test_baseline,
        test_target,
        quantiles,
        horizons,
    )

    training_bank = strategy_bank_from_checkpoint(checkpoint)
    unseen_bank = strategy_bank_from_config(
        train_windows.actual_paths,
        refined["tailrisk"],
        mode=str(refined["performance_v140"]["tailgan"]["strategy_mode"]),
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

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.update(
        {
            "fold": fold.name,
            "seed": int(seed),
            "variant": output.parent.name,
            "model_family": "tailgan_ewma_gru_residual_v140",
            "raw_model_pinball": float(raw_metrics["mean_pinball"]),
            "ewma_student_t_pinball": float(baseline_metrics["mean_pinball"]),
            "test_improvement_vs_ewma_pct": float(
                100.0
                * (baseline_metrics["mean_pinball"] - metrics["mean_pinball"])
                / max(abs(baseline_metrics["mean_pinball"]), 1e-12)
            ),
            "performance_gate": gate.export(),
            "student_t_degrees_of_freedom": baseline_state.degrees_of_freedom,
            "elapsed_seconds": time.time() - started,
        }
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "raw_model_metrics.json", raw_metrics)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    write_json(output / "performance_gate.json", gate.export())
    write_json(output / "scenario_metrics.json", unseen_eval.structural_metrics)
    write_json(
        output / "scenario_metrics_training_bank.json",
        training_eval.structural_metrics,
    )
    pd.concat(
        [
            detail.assign(model="tail_gated_calibrated"),
            raw_detail.assign(model="tail_raw"),
            baseline_detail.assign(model="ewma_student_t"),
        ],
        ignore_index=True,
    ).to_csv(output / "metrics_comparison_by_horizon.csv", index=False)
    pd.concat(
        [
            var_backtests.assign(model="tail_gated_calibrated"),
            raw_var.assign(model="tail_raw"),
            baseline_var.assign(model="ewma_student_t"),
        ],
        ignore_index=True,
    ).to_csv(output / "var_backtests.csv", index=False)
    stress = stress_mask_for_origins(panel, test_windows.origins, fold.train_end)
    final_frame = predictions_frame(
        panel.dates[test_windows.origins],
        test_target,
        calibrated,
        horizons,
        quantiles,
        stress_mask=stress,
    )
    final_frame.to_csv(output / "predictions.csv", index=False)
    predictions_frame(
        panel.dates[test_windows.origins],
        test_target,
        test_model,
        horizons,
        quantiles,
        stress_mask=stress,
    ).to_csv(output / "raw_predictions.csv", index=False)
    predictions_frame(
        panel.dates[test_windows.origins],
        test_target,
        test_baseline,
        horizons,
        quantiles,
        stress_mask=stress,
    ).to_csv(output / "baseline_predictions.csv", index=False)
    training_eval.strategy_metrics.to_csv(
        output / "strategy_tail_metrics_training_bank.csv",
        index=False,
    )
    unseen_eval.strategy_metrics.to_csv(
        output / "strategy_tail_metrics_unseen_bank.csv",
        index=False,
    )
    unseen_eval.rank_frequency.to_csv(output / "rank_frequency.csv", index=False)
    if (output / "training_history.csv").is_file():
        plot_training_history(
            pd.read_csv(output / "training_history.csv"),
            output / "training_history.png",
        )
    plot_rank_frequency(unseen_eval.rank_frequency, output / "rank_frequency.png")
    plot_matrix_pair(
        np.asarray(unseen_eval.structural_metrics["real_correlation"]),
        np.asarray(unseen_eval.structural_metrics["generated_correlation"]),
        output / "correlation_comparison.png",
        "EWMA-backbone cross-asset correlation",
    )
    plot_matrix_pair(
        np.asarray(unseen_eval.structural_metrics["real_autocorrelation"]),
        np.asarray(unseen_eval.structural_metrics["generated_autocorrelation"]),
        output / "autocorrelation_comparison.png",
        "EWMA-backbone temporal autocorrelation",
    )
    plot_forecast_fan(final_frame, output / "forecast_fan.png", horizon=horizons[0])
    write_json(
        output / "evaluation_summary.json",
        {
            "fold": fold.name,
            "seed": int(seed),
            "variant": output.parent.name,
            "mean_pinball": float(metrics["mean_pinball"]),
            "raw_model_pinball": float(raw_metrics["mean_pinball"]),
            "ewma_student_t_pinball": float(baseline_metrics["mean_pinball"]),
            "improvement_vs_ewma_pct": metrics["test_improvement_vs_ewma_pct"],
            "gate": gate.export(),
            "unseen_strategy_tail_relative_error": unseen_eval.structural_metrics[
                "mean_tail_relative_error"
            ],
            "correlation_l1_error": unseen_eval.structural_metrics[
                "correlation_l1_error"
            ],
            "autocorrelation_l1_error": unseen_eval.structural_metrics[
                "autocorrelation_l1_error"
            ],
            "test_origins": len(test_windows.origins),
        },
    )
    return output
