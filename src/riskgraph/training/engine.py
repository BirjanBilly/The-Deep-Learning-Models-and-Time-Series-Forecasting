from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from riskgraph.config import Fold
from riskgraph.data.dataset import (
    MarketWindowDataset,
    Panel,
    TargetScaler,
    fit_scalers,
    fit_target_scaler,
    mean_training_graph,
    split_origins,
    stress_mask_for_origins,
)
from riskgraph.evaluation.metrics import evaluate_forecasts
from riskgraph.io import file_sha256, predictions_frame, runtime_metadata, write_json
from riskgraph.models.hybrid import TemporalGraphQuantileNet
from riskgraph.models.losses import combined_loss
from riskgraph.models.temporal import ModelOutput, TemporalFusionQuantileNet
from riskgraph.repro import seed_everything


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    model_name: str,
    panel: Panel,
    config: dict[str, Any],
) -> nn.Module:
    model_config = config["model"]
    features = config["features"]
    common = dict(
        asset_features=panel.asset_features.shape[-1],
        macro_features=panel.macro_features.shape[-1],
        hidden_size=int(model_config["hidden_size"]),
        lstm_layers=int(model_config["lstm_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        horizons=len(features["horizons"]),
        quantile_levels=[float(value) for value in features["quantiles"]],
        target_index=panel.target_index,
    )
    if model_name == "temporal":
        return TemporalFusionQuantileNet(**common)
    if model_name == "hybrid":
        return TemporalGraphQuantileNet(
            **common,
            graph_heads=int(model_config["graph_heads"]),
            graph_signal_mode=str(model_config.get("graph_signal_mode", "direction")),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _loader(
    dataset: MarketWindowDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


def _autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_scaler: TargetScaler,
    quantiles: torch.Tensor,
    direction_weight: float,
    gradient_clip: float,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "pinball": 0.0, "direction": 0.0, "samples": 0.0}
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, precision):
            output: ModelOutput = model(batch["asset"], batch["macro"], batch["adjacency"])
            target_scaled = target_scaler.transform_tensor(batch["target"])
            loss, components = combined_loss(
                output.quantiles,
                target_scaled,
                quantiles,
                output.direction_logit,
                batch["direction"],
                direction_weight,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        count = int(batch["target"].shape[0])
        totals["loss"] += float(loss.detach()) * count
        totals["pinball"] += components["pinball"] * count
        totals["direction"] += components["direction"] * count
        totals["samples"] += count
    samples = max(totals.pop("samples"), 1.0)
    return {name: value / samples for name, value in totals.items()}


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: TargetScaler,
    quantiles: torch.Tensor,
    direction_weight: float,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "pinball": 0.0, "direction": 0.0, "samples": 0.0}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            with _autocast_context(device, precision):
                output: ModelOutput = model(batch["asset"], batch["macro"], batch["adjacency"])
                target_scaled = target_scaler.transform_tensor(batch["target"])
                loss, components = combined_loss(
                    output.quantiles,
                    target_scaled,
                    quantiles,
                    output.direction_logit,
                    batch["direction"],
                    direction_weight,
                )
            count = int(batch["target"].shape[0])
            totals["loss"] += float(loss.detach()) * count
            totals["pinball"] += components["pinball"] * count
            totals["direction"] += components["direction"] * count
            totals["samples"] += count
    samples = max(totals.pop("samples"), 1.0)
    return {name: value / samples for name, value in totals.items()}


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: TargetScaler,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}

    def add_summary(name: str, value: torch.Tensor | None, dimensions: tuple[int, ...]) -> None:
        if value is None:
            return
        reduced = value.float().mean(dim=dimensions).detach().cpu().numpy()
        batch_count = int(value.shape[0])
        if name not in sums:
            sums[name] = reduced * batch_count
            counts[name] = batch_count
        else:
            sums[name] += reduced * batch_count
            counts[name] += batch_count

    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            with _autocast_context(device, precision):
                output: ModelOutput = model(batch["asset"], batch["macro"], batch["adjacency"])
            prediction = target_scaler.inverse_tensor(output.quantiles.float())
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target"].float().cpu().numpy())
            origins.append(batch["origin_index"].cpu().numpy())
            add_summary("asset_variable_weights", output.asset_variable_weights, (0, 1, 2))
            add_summary("macro_variable_weights", output.macro_variable_weights, (0, 1))
            add_summary("temporal_attention", output.temporal_attention, (0, 1, 2))
            add_summary("graph_attention", output.graph_attention, (0, 1))
            add_summary("fusion_gate", output.fusion_gate, (0,))

    interpretation = {name: value / max(counts[name], 1) for name, value in sums.items()}
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(origins, axis=0),
        interpretation,
    )


def train_fold(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    config_path: str | Path,
    model_name: str,
    seed: int,
    output_dir: str | Path,
    graph_mode: str = "dynamic",
    macro_mode: str = "enabled",
) -> Path:
    seed_everything(seed)
    device = select_device()
    features = config["features"]
    training = config["training"]
    horizons = [int(value) for value in features["horizons"]]
    quantile_levels = [float(value) for value in features["quantiles"]]
    origin_groups = split_origins(
        panel,
        fold,
        lookback=int(features["lookback"]),
        horizons=horizons,
        embargo_days=int(config["splits"].get("embargo_days", 0)),
    )
    scalers = fit_scalers(panel, fold.train_end)
    target_scaler = fit_target_scaler(panel, origin_groups["train"], horizons)
    static_graph = mean_training_graph(panel, origin_groups["train"])

    datasets = {
        split: MarketWindowDataset(
            panel,
            origins,
            scalers,
            lookback=int(features["lookback"]),
            horizons=horizons,
            graph_mode=graph_mode,
            macro_mode=macro_mode,
            static_graph=static_graph if graph_mode == "static" else None,
        )
        for split, origins in origin_groups.items()
    }
    loaders = {
        split: _loader(
            dataset,
            batch_size=int(training["batch_size"]),
            num_workers=int(training.get("num_workers", 0)),
            shuffle=split == "train",
            seed=seed,
            device=device,
        )
        for split, dataset in datasets.items()
    }

    model = build_model(model_name, panel, config).to(device)
    if bool(training.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    quantile_tensor = torch.tensor(quantile_levels, dtype=torch.float32, device=device)
    direction_weight = float(config["model"].get("direction_loss_weight", 0.0)) if model_name == "hybrid" else 0.0
    precision = str(training.get("precision", "fp32")).lower()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation = float("inf")
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_metrics = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            target_scaler,
            quantile_tensor,
            direction_weight,
            float(training["gradient_clip"]),
            device,
            precision,
        )
        validation_metrics = _evaluate_loss(
            model,
            loaders["validation"],
            target_scaler,
            quantile_tensor,
            direction_weight,
            device,
            precision,
        )
        record: dict[str, float | int] = {"epoch": epoch}
        record.update({f"train_{key}": value for key, value in train_metrics.items()})
        record.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(record)
        print(
            f"{fold.name} {model_name} seed={seed} epoch={epoch:03d} "
            f"train_pinball={train_metrics['pinball']:.6f} "
            f"validation_pinball={validation_metrics['pinball']:.6f}"
        )
        if validation_metrics["pinball"] < best_validation - 1e-6:
            best_validation = validation_metrics["pinball"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    prediction, target, test_origins, interpretation = _collect_predictions(
        model,
        loaders["test"],
        target_scaler,
        device,
        precision,
    )
    stress_mask = stress_mask_for_origins(panel, test_origins, fold.train_end)
    metrics, detail, backtests = evaluate_forecasts(
        prediction,
        target,
        quantile_levels,
        horizons,
        stress_mask=stress_mask,
    )
    metrics.update(
        {
            "fold": fold.name,
            "model": model_name,
            "seed": seed,
            "graph_mode": graph_mode,
            "macro_mode": macro_mode,
            "graph_signal_mode": str(config["model"].get("graph_signal_mode", "direction")),
            "best_epoch": best_epoch,
            "best_validation_pinball_scaled": best_validation,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
    )

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_to_save = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(
        {
            "model_state_dict": state_to_save,
            "model": model_name,
            "fold": asdict(fold),
            "seed": seed,
            "target_mean": target_scaler.mean,
            "target_std": target_scaler.std,
            "quantiles": quantile_levels,
            "horizons": horizons,
        },
        run_dir / "checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    prediction_dates = panel.dates[test_origins]
    predictions_frame(
        prediction_dates,
        target,
        prediction,
        horizons,
        quantile_levels,
        stress_mask=stress_mask,
    ).to_csv(run_dir / "predictions.csv", index=False)
    detail.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    backtests.to_csv(run_dir / "var_backtests.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(
        run_dir / "interpretability.npz",
        **interpretation,
        asset_feature_names=np.asarray(panel.asset_feature_names),
        macro_feature_names=np.asarray(panel.macro_feature_names),
        tickers=np.asarray(panel.tickers),
    )
    metadata = runtime_metadata(seed, device)
    metadata.update(
        {
            "fold": asdict(fold),
            "model": model_name,
            "graph_mode": graph_mode,
            "macro_mode": macro_mode,
            "graph_signal_mode": str(config["model"].get("graph_signal_mode", "direction")),
            "train_samples": len(datasets["train"]),
            "validation_samples": len(datasets["validation"]),
            "test_samples": len(datasets["test"]),
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": file_sha256(config_path),
            "target_scaler": {
                "mean": target_scaler.mean.tolist(),
                "std": target_scaler.std.tolist(),
            },
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)
    return run_dir
