from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel
from riskgraph.io import runtime_metadata, write_json
from riskgraph.performance_v140.patch_data import (
    PatchFeatureSpec,
    PatchForecastDataset,
    build_patch_arrays,
    fit_patch_feature_spec,
    patch_fold_origins,
)
from riskgraph.performance_v140.patch_model import (
    ResidualPatchConfig,
    ResidualPatchQuantileTransformer,
    pinball_loss,
)
from riskgraph.repro import seed_everything


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _epoch(
    model: ResidualPatchQuantileTransformer,
    loader: DataLoader,
    quantiles: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    correction_penalty: float,
) -> float:
    model.train(optimizer is not None)
    values: list[float] = []
    for batch in loader:
        history = batch["history"].to(device)
        baseline = batch["baseline"].to(device)
        target = batch["target"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(history, baseline)
        loss = pinball_loss(prediction, target, quantiles)
        loss = loss + float(correction_penalty) * torch.mean(
            torch.abs(prediction - baseline)
            / (baseline[..., -1:] - baseline[..., :1]).abs().clamp_min(1e-4)
        )
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


def train_patch_fold(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
    device_name: str = "auto",
    max_epochs: int | None = None,
) -> Path:
    seed_everything(seed)
    device = resolve_device(device_name)
    performance = config["performance_v140"]
    patch = performance["patch_transformer"]
    split = patch_fold_origins(panel, fold, config)
    stride = max(1, int(patch.get("train_stride", 1)))
    train_origins = split["train"][::stride]
    validation_origins = split["validation"]
    spec = fit_patch_feature_spec(panel, fold, config)
    train_arrays = build_patch_arrays(panel, fold, train_origins, config, spec)
    validation_arrays = build_patch_arrays(panel, fold, validation_origins, config, spec)
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantile_values = [float(value) for value in config["features"]["quantiles"]]
    model_config = ResidualPatchConfig(
        channels=len(spec.channel_names),
        lookback=spec.lookback,
        horizons=len(horizons),
        quantiles=len(quantile_values),
        patch_length=int(patch["patch_length"]),
        patch_stride=int(patch["patch_stride"]),
        d_model=int(patch["d_model"]),
        n_heads=int(patch["n_heads"]),
        n_layers=int(patch["n_layers"]),
        d_ff=int(patch["d_ff"]),
        dropout=float(patch["dropout"]),
        correction_limit=float(patch["correction_limit"]),
    )
    model = ResidualPatchQuantileTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(patch["learning_rate"]),
        weight_decay=float(patch["weight_decay"]),
    )
    batch_size = int(patch["batch_size"])
    train_loader = DataLoader(
        PatchForecastDataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(patch.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        PatchForecastDataset(validation_arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(patch.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    quantiles = torch.tensor(quantile_values, dtype=torch.float32, device=device).view(1, 1, -1)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    wait = 0
    history: list[dict[str, float | int]] = []
    epochs = int(max_epochs if max_epochs is not None else patch["epochs"])
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(
            model,
            train_loader,
            quantiles,
            device,
            optimizer,
            float(patch["gradient_clip"]),
            float(patch["correction_penalty"]),
        )
        with torch.no_grad():
            validation_loss = _epoch(
                model,
                validation_loader,
                quantiles,
                device,
                None,
                float(patch["gradient_clip"]),
                float(patch["correction_penalty"]),
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "mean_gate": float(torch.sigmoid(model.gate_logits).mean().detach().cpu()),
            }
        )
        print(
            f"{fold.name} patch-v140 seed={seed} epoch={epoch:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )
        if validation_loss < best - 1e-7:
            best = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(patch["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No finite patch-transformer checkpoint was produced")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = {
        "package_version": "1.4.1",
        "model_family": "llmtime_patch_residual",
        "fold": asdict(fold),
        "seed": int(seed),
        "model_config": model.export_config(),
        "model_state_dict": best_state,
        "feature_spec": spec.export(),
        "horizons": horizons,
        "quantiles": quantile_values,
        "best_epoch": best_epoch,
        "best_validation_loss": best,
    }
    torch.save(checkpoint, output / "best_checkpoint.pt")
    write_json(
        output / "run_metadata.json",
        {
            **runtime_metadata(seed, device),
            "fold": fold.name,
            "variant": output.parent.name,
            "model_family": "llmtime_patch_residual",
            "best_epoch": best_epoch,
            "best_validation_loss": best,
            "train_origins": len(train_origins),
            "validation_origins": len(validation_origins),
            "feature_channels": list(spec.channel_names),
            "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "elapsed_seconds": time.time() - started,
            "baseline_anchor": "EWMA Student-t quantile skip connection",
            "causality": "feature normalization and Student-t degrees of freedom use training dates only",
        },
    )
    write_json(
        output / "data_split.json",
        {
            "fold": asdict(fold),
            "train_origins": train_origins.tolist(),
            "validation_origins": validation_origins.tolist(),
            "test_origins": split["test"].tolist(),
        },
    )
    return output


def load_patch_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ResidualPatchQuantileTransformer, PatchFeatureSpec, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResidualPatchQuantileTransformer(
        ResidualPatchConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, PatchFeatureSpec.from_record(checkpoint["feature_spec"]), checkpoint
