from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel, stress_mask_for_origins
from riskgraph.evaluation.metrics import evaluate_forecasts, pinball_loss_numpy
from riskgraph.io import predictions_frame, write_json
from riskgraph.performance_v140.calibration import apply_forecast_gate, fit_forecast_gate
from riskgraph.performance_v140.patch_data import (
    PatchForecastDataset,
    build_patch_arrays,
    patch_fold_origins,
)
from riskgraph.performance_v140.patch_training import load_patch_checkpoint, resolve_device


def _predict(
    model,
    arrays,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        PatchForecastDataset(arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["history"].to(device),
                batch["baseline"].to(device),
            )
            parts.append(prediction.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _loss_detail(
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    calibrated_prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    horizons: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    q = np.asarray(quantiles, dtype=float)
    for h_index, horizon in enumerate(horizons):
        for name, prediction in (
            ("ewma_student_t", baseline_prediction),
            ("patch_raw", model_prediction),
            ("patch_gated_calibrated", calibrated_prediction),
        ):
            pinball = pinball_loss_numpy(
                prediction[:, h_index : h_index + 1, :],
                target[:, h_index : h_index + 1],
                q,
            )
            rows.append(
                {
                    "model": name,
                    "horizon": int(horizon),
                    "pinball": float(pinball),
                    "quantile_crps_approx": float(2.0 * pinball),
                }
            )
    return pd.DataFrame(rows)


def _plot_forecast_fan(
    dates: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
    output: Path,
    horizon_index: int = 0,
) -> None:
    q = np.asarray(quantiles, dtype=float)
    median = int(np.argmin(np.abs(q - 0.5)))
    lower = int(np.argmin(np.abs(q - 0.05)))
    upper = int(np.argmin(np.abs(q - 0.95)))
    figure, axis = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(dates)
    axis.fill_between(
        x,
        prediction[:, horizon_index, lower],
        prediction[:, horizon_index, upper],
        alpha=0.25,
        label="90% interval",
    )
    axis.plot(x, prediction[:, horizon_index, median], linewidth=1.2, label="median")
    axis.plot(x, target[:, horizon_index], linewidth=0.9, label="realized")
    axis.axhline(0.0, linewidth=0.7)
    axis.set_title("Baseline-anchored patch Transformer forecast")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def evaluate_patch_fold(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    model, spec, checkpoint = load_patch_checkpoint(checkpoint_path, device)
    split = patch_fold_origins(panel, fold, config)
    validation = build_patch_arrays(panel, fold, split["validation"], config, spec)
    test = build_patch_arrays(panel, fold, split["test"], config, spec)
    patch = config["performance_v140"]["patch_transformer"]
    batch_size = int(patch.get("evaluation_batch_size", patch["batch_size"]))
    started = time.time()
    validation_model = _predict(model, validation, batch_size, device)
    test_model = _predict(model, test, batch_size, device)
    quantiles = [float(value) for value in checkpoint["quantiles"]]
    horizons = [int(value) for value in checkpoint["horizons"]]
    gate = fit_forecast_gate(
        validation_model,
        validation.baselines,
        validation.targets,
        quantiles,
        horizons=horizons,
        min_relative_improvement=float(
            config["performance_v140"]["minimum_validation_improvement"]
        ),
        min_confirmation_improvement=float(
            config["performance_v140"]["minimum_confirmation_improvement"]
        ),
        calibration_fraction=float(
            config["performance_v140"]["gate_calibration_fraction"]
        ),
        confirmation_fraction_of_calibration=float(
            config["performance_v140"][
                "gate_confirmation_fraction_of_calibration"
            ]
        ),
        coverage_penalty_weight=float(
            config["performance_v140"]["coverage_penalty_weight"]
        ),
    )
    calibrated = apply_forecast_gate(
        test_model,
        test.baselines,
        quantiles,
        gate,
        observed_targets=test.targets,
        horizons=horizons,
    )
    stress = stress_mask_for_origins(panel, test.origins, fold.train_end)
    metrics, detail, var_backtests = evaluate_forecasts(
        calibrated,
        test.targets,
        quantiles,
        horizons,
        stress_mask=stress,
    )
    raw_metrics, raw_detail, raw_var = evaluate_forecasts(
        test_model,
        test.targets,
        quantiles,
        horizons,
        stress_mask=stress,
    )
    baseline_metrics, baseline_detail, baseline_var = evaluate_forecasts(
        test.baselines,
        test.targets,
        quantiles,
        horizons,
        stress_mask=stress,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.update(
        {
            "fold": fold.name,
            "seed": int(checkpoint["seed"]),
            "variant": output.parent.name,
            "model_family": "llmtime_patch_residual_v140",
            "quantile_crps_approx": float(2.0 * metrics["mean_pinball"]),
            "raw_model_pinball": float(raw_metrics["mean_pinball"]),
            "ewma_student_t_pinball": float(baseline_metrics["mean_pinball"]),
            "test_improvement_vs_ewma_pct": float(
                100.0
                * (baseline_metrics["mean_pinball"] - metrics["mean_pinball"])
                / max(abs(baseline_metrics["mean_pinball"]), 1e-12)
            ),
            "elapsed_seconds": time.time() - started,
            "performance_gate": gate.export(),
        }
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "raw_model_metrics.json", raw_metrics)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    write_json(output / "performance_gate.json", gate.export())
    detail.assign(model="patch_gated_calibrated").to_csv(
        output / "metrics_by_horizon.csv", index=False
    )
    pd.concat(
        [
            raw_detail.assign(model="patch_raw"),
            baseline_detail.assign(model="ewma_student_t"),
            detail.assign(model="patch_gated_calibrated"),
        ],
        ignore_index=True,
    ).to_csv(output / "metrics_comparison_by_horizon.csv", index=False)
    pd.concat(
        [
            raw_var.assign(model="patch_raw"),
            baseline_var.assign(model="ewma_student_t"),
            var_backtests.assign(model="patch_gated_calibrated"),
        ],
        ignore_index=True,
    ).to_csv(output / "var_backtests.csv", index=False)
    _loss_detail(
        test_model,
        test.baselines,
        calibrated,
        test.targets,
        quantiles,
        horizons,
    ).to_csv(output / "loss_comparison.csv", index=False)
    predictions_frame(
        panel.dates[test.origins],
        test.targets,
        calibrated,
        horizons,
        quantiles,
        stress_mask=stress,
    ).to_csv(output / "predictions.csv", index=False)
    predictions_frame(
        panel.dates[test.origins],
        test.targets,
        test_model,
        horizons,
        quantiles,
        stress_mask=stress,
    ).to_csv(output / "raw_predictions.csv", index=False)
    predictions_frame(
        panel.dates[test.origins],
        test.targets,
        test.baselines,
        horizons,
        quantiles,
        stress_mask=stress,
    ).to_csv(output / "baseline_predictions.csv", index=False)
    _plot_forecast_fan(
        test.dates,
        calibrated,
        test.targets,
        quantiles,
        output / "forecast_fan.png",
    )
    write_json(
        output / "evaluation_summary.json",
        {
            "fold": fold.name,
            "seed": int(checkpoint["seed"]),
            "variant": output.parent.name,
            "mean_pinball": float(metrics["mean_pinball"]),
            "quantile_crps_approx": float(2.0 * metrics["mean_pinball"]),
            "ewma_student_t_pinball": float(baseline_metrics["mean_pinball"]),
            "improvement_vs_ewma_pct": metrics["test_improvement_vs_ewma_pct"],
            "gate": gate.export(),
            "test_origins": len(test.origins),
        },
    )
    return output
