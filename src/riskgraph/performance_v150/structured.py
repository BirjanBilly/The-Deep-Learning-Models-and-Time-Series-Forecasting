from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from riskgraph.config import Fold
from riskgraph.io import predictions_frame, runtime_metadata, write_json
from riskgraph.performance_v140.baselines import ewma_student_t_forecast
from riskgraph.performance_v150.data import (
    LongHistoryPanel,
    long_targets,
    split_long_origins,
)
from riskgraph.repro import seed_everything


@dataclass(frozen=True)
class LongFeatureSpec:
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    lookback: int

    def export(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_names": list(self.feature_names),
            "lookback": int(self.lookback),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> LongFeatureSpec:
        return cls(
            mean=np.asarray(record["mean"], dtype=np.float32),
            std=np.asarray(record["std"], dtype=np.float32),
            feature_names=tuple(str(value) for value in record["feature_names"]),
            lookback=int(record["lookback"]),
        )


@dataclass(frozen=True)
class StructuredArrays:
    histories: np.ndarray
    baselines: np.ndarray
    targets: np.ndarray
    origins: np.ndarray


class StructuredDataset(Dataset):
    def __init__(self, arrays: StructuredArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return int(len(self.arrays.origins))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": torch.from_numpy(self.arrays.histories[index]),
            "baseline": torch.from_numpy(self.arrays.baselines[index]),
            "target": torch.from_numpy(self.arrays.targets[index]),
            "origin": torch.tensor(int(self.arrays.origins[index]), dtype=torch.long),
        }


def fit_feature_spec(
    panel: LongHistoryPanel,
    fold: Fold,
    lookback: int,
) -> LongFeatureSpec:
    train = panel.dates <= pd.Timestamp(fold.train_end)
    values = panel.values[train].astype(np.float64)
    masks = panel.masks[train].astype(bool)
    mean = np.zeros(values.shape[1], dtype=np.float64)
    std = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if len(observed) < 20:
            continue
        mean[column] = float(np.mean(observed))
        candidate = float(np.std(observed))
        std[column] = candidate if candidate >= 1e-6 else 1.0
    return LongFeatureSpec(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=tuple(panel.feature_names),
        lookback=int(lookback),
    )


def build_structured_arrays(
    panel: LongHistoryPanel,
    fold: Fold,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: list[float],
    spec: LongFeatureSpec,
    ewma_mean_decay: float,
    ewma_variance_decay: float,
) -> tuple[StructuredArrays, float]:
    origins = np.asarray(origins, dtype=np.int64)
    standardized = (panel.values - spec.mean[None, :]) / spec.std[None, :]
    standardized = np.where(panel.masks > 0.5, standardized, 0.0)
    combined = np.concatenate([standardized, panel.masks], axis=1).astype(np.float32)
    histories = np.stack(
        [combined[o - spec.lookback + 1 : o + 1] for o in origins],
        axis=0,
    ).astype(np.float32)
    train_end_index = int(
        np.flatnonzero(panel.dates <= pd.Timestamp(fold.train_end))[-1]
    )
    baseline, state = ewma_student_t_forecast(
        panel.target_returns,
        origins,
        horizons,
        quantiles,
        train_end_index=train_end_index,
        mean_decay=float(ewma_mean_decay),
        variance_decay=float(ewma_variance_decay),
    )
    targets = long_targets(panel, origins, horizons)
    return (
        StructuredArrays(
            histories=histories,
            baselines=baseline.astype(np.float32),
            targets=targets.astype(np.float32),
            origins=origins,
        ),
        float(state.degrees_of_freedom),
    )


@dataclass(frozen=True)
class StructuredTransformerConfig:
    channels: int
    lookback: int
    horizons: int
    quantiles: tuple[float, ...]
    patch_lengths: tuple[int, ...] = (5, 21, 63)
    patch_strides: tuple[int, ...] = (5, 10, 21)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    dropout: float = 0.15
    location_limit: float = 0.35
    log_scale_limit: float = 0.40


class StructuredPatchDistributionTransformer(nn.Module):
    """Low-dimensional distribution adapter around an EWMA Student-t forecast.

    The model predicts only location, lower-tail scale and upper-tail scale per
    horizon. Positive scale transforms preserve quantile ordering by construction.
    """

    def __init__(self, config: StructuredTransformerConfig) -> None:
        super().__init__()
        self.config = config
        if len(config.patch_lengths) != len(config.patch_strides):
            raise ValueError("patch_lengths and patch_strides must have equal length")
        self.patch_counts = tuple(
            1 + (config.lookback - length) // stride
            for length, stride in zip(
                config.patch_lengths, config.patch_strides, strict=True
            )
        )
        if any(count <= 0 for count in self.patch_counts):
            raise ValueError("Every patch length must fit within lookback")
        # One token represents a multivariate temporal patch. This keeps the token
        # count equal to the number of time patches (about 84 in the formal setup),
        # rather than channels × patches, which would make attention quadratic in
        # thousands of tokens.
        self.patch_projections = nn.ModuleList(
            [
                nn.Linear(length * config.channels, config.d_model)
                for length in config.patch_lengths
            ]
        )
        self.feature_gate = nn.Parameter(torch.zeros(config.channels))
        self.scale_embedding = nn.Embedding(len(config.patch_lengths), config.d_model)
        self.position_embeddings = nn.ModuleList(
            [nn.Embedding(count, config.d_model) for count in self.patch_counts]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.parameter_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.horizons * 3),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for layer in self.patch_projections:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        final = self.parameter_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def export_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _tokens(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (
            self.config.lookback,
            self.config.channels,
        ):
            raise ValueError("history has an incompatible shape")
        # The first half contains standardized numeric values and the second half
        # contains binary availability masks. Normalize only the numeric channels so
        # the masks remain exact 0/1 indicators.
        if self.config.channels % 2 != 0:
            raise ValueError("structured history must contain values plus matching masks")
        value_channels = self.config.channels // 2
        numeric = history[..., :value_channels]
        masks = history[..., value_channels:]
        mean = numeric.mean(dim=1, keepdim=True)
        std = numeric.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = torch.cat([(numeric - mean) / std, masks], dim=-1)
        normalized = normalized * (1.0 + 0.1 * torch.tanh(self.feature_gate))[None, None, :]
        output: list[torch.Tensor] = []
        for scale_index, (length, stride, projection, position, count) in enumerate(
            zip(
                self.config.patch_lengths,
                self.config.patch_strides,
                self.patch_projections,
                self.position_embeddings,
                self.patch_counts,
                strict=True,
            )
        ):
            # unfold over time -> [batch, patches, channels, patch_length]
            patches = normalized.unfold(1, length, stride).contiguous()
            patches = patches.reshape(history.shape[0], count, -1)
            embedded = projection(patches)
            pos = torch.arange(count, device=history.device)
            embedded = (
                embedded
                + self.scale_embedding.weight[scale_index][None, None, :]
                + position(pos)[None, :, :]
            )
            output.append(embedded)
        return torch.cat(output, dim=1)

    def forward(
        self,
        history: torch.Tensor,
        baseline_quantiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokens(history)
        cls = self.cls_token.expand(history.shape[0], -1, -1)
        hidden = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        raw = self.parameter_head(self.norm(hidden)).view(
            history.shape[0], self.config.horizons, 3
        )
        quantiles = torch.as_tensor(
            self.config.quantiles,
            dtype=baseline_quantiles.dtype,
            device=baseline_quantiles.device,
        )
        median_index = int(torch.argmin(torch.abs(quantiles - 0.5)).item())
        median = baseline_quantiles[..., median_index : median_index + 1]
        total_spread = (
            baseline_quantiles[..., -1:] - baseline_quantiles[..., :1]
        ).abs().clamp_min(1e-5)
        location = (
            torch.tanh(raw[..., 0:1])
            * total_spread
            * float(self.config.location_limit)
        )
        lower_scale = torch.exp(
            torch.tanh(raw[..., 1:2]) * float(self.config.log_scale_limit)
        )
        upper_scale = torch.exp(
            torch.tanh(raw[..., 2:3]) * float(self.config.log_scale_limit)
        )
        centered = baseline_quantiles - median
        scale = torch.where(
            quantiles[None, None, :] < 0.5,
            lower_scale,
            upper_scale,
        )
        prediction = median + location + centered * scale
        return torch.sort(prediction, dim=-1).values, raw


def weighted_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    tail_weight: float,
) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    weights = torch.ones_like(quantiles)
    weights = torch.where(
        (quantiles <= 0.10) | (quantiles >= 0.90),
        torch.full_like(weights, float(tail_weight)),
        weights,
    )
    return torch.sum(loss * weights) / (loss.shape[0] * loss.shape[1] * weights.sum())


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _epoch(
    model: StructuredPatchDistributionTransformer,
    loader: DataLoader,
    quantiles: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    parameter_penalty: float,
    tail_weight: float,
) -> float:
    model.train(optimizer is not None)
    losses: list[float] = []
    for batch in loader:
        history = batch["history"].to(device)
        baseline = batch["baseline"].to(device)
        target = batch["target"].to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        prediction, parameters = model(history, baseline)
        loss = weighted_pinball_loss(prediction, target, quantiles, tail_weight)
        loss = loss + float(parameter_penalty) * parameters.pow(2).mean()
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_structured_fold(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
    device_name: str = "auto",
    max_epochs: int | None = None,
) -> Path:
    seed_everything(seed)
    settings = config["performance_v150"]["structured_transformer"]
    performance = config["performance_v140"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    q_values = [float(value) for value in config["features"]["quantiles"]]
    lookback = int(settings["lookback"])
    groups = split_long_origins(
        panel,
        fold,
        lookback,
        horizons,
        int(config["splits"].get("embargo_days", 0)),
        common_max_horizon=(
            int(config["tailrisk"]["scenario_horizon"])
            if bool(config["performance_v150"].get("common_origin_required", True))
            else None
        ),
    )
    spec = fit_feature_spec(panel, fold, lookback)
    stride = max(1, int(settings.get("train_stride", 1)))
    train_arrays, degrees = build_structured_arrays(
        panel,
        fold,
        groups["train"][::stride],
        horizons,
        q_values,
        spec,
        float(performance["ewma_mean_decay"]),
        float(performance["ewma_variance_decay"]),
    )
    validation_arrays, _ = build_structured_arrays(
        panel,
        fold,
        groups["validation"],
        horizons,
        q_values,
        spec,
        float(performance["ewma_mean_decay"]),
        float(performance["ewma_variance_decay"]),
    )
    model_config = StructuredTransformerConfig(
        channels=train_arrays.histories.shape[-1],
        lookback=lookback,
        horizons=len(horizons),
        quantiles=tuple(q_values),
        patch_lengths=tuple(int(value) for value in settings["patch_lengths"]),
        patch_strides=tuple(int(value) for value in settings["patch_strides"]),
        d_model=int(settings["d_model"]),
        n_heads=int(settings["n_heads"]),
        n_layers=int(settings["n_layers"]),
        d_ff=int(settings["d_ff"]),
        dropout=float(settings["dropout"]),
        location_limit=float(settings["location_limit"]),
        log_scale_limit=float(settings["log_scale_limit"]),
    )
    device = _device(device_name)
    model = StructuredPatchDistributionTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    train_loader = DataLoader(
        StructuredDataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        StructuredDataset(validation_arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    quantiles = torch.tensor(q_values, dtype=torch.float32, device=device).view(1, 1, -1)
    epochs = int(max_epochs if max_epochs is not None else settings["epochs"])
    patience = int(settings["patience"])
    best = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(
            model,
            train_loader,
            quantiles,
            device,
            optimizer,
            float(settings["gradient_clip"]),
            float(settings["parameter_penalty"]),
            float(settings["tail_weight"]),
        )
        with torch.no_grad():
            validation_loss = _epoch(
                model,
                validation_loader,
                quantiles,
                device,
                None,
                float(settings["gradient_clip"]),
                float(settings["parameter_penalty"]),
                float(settings["tail_weight"]),
            )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        print(
            f"{fold.name} structured-v150 seed={seed} epoch={epoch:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )
        if validation_loss < best - 1e-10:
            best = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Structured model produced no checkpoint")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": best_state,
        "model_config": model.export_config(),
        "feature_spec": spec.export(),
        "fold": asdict(fold),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best),
        "student_t_degrees_of_freedom": float(degrees),
        "version": "1.5.0",
    }
    torch.save(checkpoint, output / "best_checkpoint.pt")
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    write_json(
        output / "run_metadata.json",
        {
            **runtime_metadata(seed, device),
            "seed": int(seed),
            "fold": asdict(fold),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best),
            "training_origins": int(len(train_arrays.origins)),
            "validation_origins": int(len(validation_arrays.origins)),
            "feature_count_with_masks": int(train_arrays.histories.shape[-1]),
            "model_family": "structured_patch_distribution_transformer_v150",
        },
    )
    write_json(
        output / "data_split.json",
        {
            "fold": asdict(fold),
            "train_origins": groups["train"].tolist(),
            "validation_origins": groups["validation"].tolist(),
            "test_origins": groups["test"].tolist(),
        },
    )
    write_json(output / "feature_spec.json", spec.export())
    return output


def load_structured_checkpoint(
    path: str | Path,
    device_name: str = "auto",
) -> tuple[StructuredPatchDistributionTransformer, LongFeatureSpec, dict[str, Any], torch.device]:
    device = _device(device_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    record = dict(checkpoint["model_config"])
    record["quantiles"] = tuple(float(value) for value in record["quantiles"])
    record["patch_lengths"] = tuple(int(value) for value in record["patch_lengths"])
    record["patch_strides"] = tuple(int(value) for value in record["patch_strides"])
    model = StructuredPatchDistributionTransformer(
        StructuredTransformerConfig(**record)
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, LongFeatureSpec.from_record(checkpoint["feature_spec"]), checkpoint, device


def predict_structured(
    model: StructuredPatchDistributionTransformer,
    arrays: StructuredArrays,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(StructuredDataset(arrays), batch_size=batch_size, shuffle=False)
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction, _ = model(
                batch["history"].to(device),
                batch["baseline"].to(device),
            )
            output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def export_seed_predictions(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> Path:
    started = time.time()
    model, spec, checkpoint, device = load_structured_checkpoint(
        checkpoint_path, device_name
    )
    settings = config["performance_v150"]["structured_transformer"]
    performance = config["performance_v140"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    groups = split_long_origins(
        panel,
        fold,
        spec.lookback,
        horizons,
        int(config["splits"].get("embargo_days", 0)),
        common_max_horizon=(
            int(config["tailrisk"]["scenario_horizon"])
            if bool(config["performance_v150"].get("common_origin_required", True))
            else None
        ),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("validation", "test"):
        arrays, _ = build_structured_arrays(
            panel,
            fold,
            groups[split_name],
            horizons,
            quantiles,
            spec,
            float(performance["ewma_mean_decay"]),
            float(performance["ewma_variance_decay"]),
        )
        raw = predict_structured(
            model,
            arrays,
            device,
            int(settings["evaluation_batch_size"]),
        )
        predictions_frame(
            panel.dates[arrays.origins],
            arrays.targets,
            raw,
            horizons,
            quantiles,
        ).to_csv(output / f"{split_name}_raw_predictions.csv", index=False)
        predictions_frame(
            panel.dates[arrays.origins],
            arrays.targets,
            arrays.baselines,
            horizons,
            quantiles,
        ).to_csv(output / f"{split_name}_baseline_predictions.csv", index=False)
    write_json(
        output / "seed_evaluation_metadata.json",
        {
            "fold": fold.name,
            "seed": int(checkpoint["seed"]),
            "elapsed_seconds": time.time() - started,
            "model_family": "structured_patch_distribution_transformer_v150",
        },
    )
    return output
