from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from riskgraph.config import Fold
from riskgraph.io import predictions_frame, runtime_metadata, write_json
from riskgraph.performance_v150.data import LongHistoryPanel, long_targets, split_long_origins
from riskgraph.performance_v150.structured import LongFeatureSpec, fit_feature_spec
from riskgraph.performance_v160.statistical import (
    StatisticalEnsembleState,
    apply_statistical_weights,
    expert_forecast_bank,
)
from riskgraph.repro import seed_everything


@dataclass(frozen=True)
class StructuredArraysV160:
    histories: np.ndarray
    baselines: np.ndarray
    targets: np.ndarray
    ssl_targets: np.ndarray
    regime_targets: np.ndarray
    origins: np.ndarray


class StructuredDatasetV160(Dataset):
    def __init__(self, arrays: StructuredArraysV160) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return int(len(self.arrays.origins))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": torch.from_numpy(self.arrays.histories[index]),
            "baseline": torch.from_numpy(self.arrays.baselines[index]),
            "target": torch.from_numpy(self.arrays.targets[index]),
            "ssl_target": torch.from_numpy(self.arrays.ssl_targets[index]),
            "regime_target": torch.from_numpy(self.arrays.regime_targets[index]),
            "origin": torch.tensor(int(self.arrays.origins[index]), dtype=torch.long),
        }


def _ssl_targets(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Targets derived only from the history visible at the forecast origin."""

    values = np.asarray(history, dtype=np.float32)
    target_return = values[:, -21:, 0]
    recent_5 = target_return[:, -5:]
    recent_21 = target_return
    cumulative = np.exp(np.cumsum(values[:, -63:, 0], axis=1))
    drawdown = cumulative[:, -1] / np.maximum(cumulative.max(axis=1), 1e-6) - 1.0
    downside = np.square(np.minimum(recent_21, 0.0)).sum(axis=1)
    total = np.square(recent_21).sum(axis=1)
    descriptors = np.stack(
        [
            recent_5.mean(axis=1),
            recent_5.std(axis=1),
            recent_21.std(axis=1),
            downside / np.maximum(total, 1e-6),
            drawdown,
        ],
        axis=1,
    ).astype(np.float32)
    reconstruction = np.concatenate([target_return, descriptors], axis=1).astype(
        np.float32
    )
    return reconstruction, descriptors


def build_structured_arrays_v160(
    panel: LongHistoryPanel,
    origins: np.ndarray,
    horizons: list[int],
    quantiles: list[float],
    spec: LongFeatureSpec,
    statistical_state: StatisticalEnsembleState,
) -> StructuredArraysV160:
    origins = np.asarray(origins, dtype=np.int64)
    standardized = (panel.values - spec.mean[None, :]) / spec.std[None, :]
    standardized = np.where(panel.masks > 0.5, standardized, 0.0)
    combined = np.concatenate([standardized, panel.masks], axis=1).astype(np.float32)
    histories = np.stack(
        [combined[o - spec.lookback + 1 : o + 1] for o in origins], axis=0
    ).astype(np.float32)
    expert_bank = expert_forecast_bank(
        panel.target_returns,
        origins,
        horizons,
        quantiles,
        statistical_state.expert_specs,
    )
    baselines = apply_statistical_weights(expert_bank, statistical_state.weights)
    ssl_target, regime_target = _ssl_targets(histories)
    return StructuredArraysV160(
        histories=histories,
        baselines=baselines.astype(np.float32),
        targets=long_targets(panel, origins, horizons).astype(np.float32),
        ssl_targets=ssl_target,
        regime_targets=regime_target,
        origins=origins,
    )


@dataclass(frozen=True)
class StructuredV160Config:
    value_channels: int
    lookback: int
    horizons: int
    quantiles: tuple[float, ...]
    ssl_target_dim: int = 26
    regime_target_dim: int = 5
    patch_lengths: tuple[int, ...] = (5, 21, 63)
    patch_strides: tuple[int, ...] = (5, 10, 21)
    channel_dim: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    dropout: float = 0.12
    location_limit: float = 0.25
    log_scale_limit: float = 0.30
    tail_power_limit: float = 0.20


class SSLRegimePatchTransformer(nn.Module):
    """Channel-independent multi-scale encoder with a structured distribution head.

    Temporal projection weights are shared across channels. A learned attention pool
    combines channels within each patch, avoiding a large dense projection over all
    feature-by-time coordinates. The forecast head changes only location, lower and
    upper scale, and monotone tail curvature around the statistical expert ensemble.
    """

    def __init__(self, config: StructuredV160Config) -> None:
        super().__init__()
        self.config = config
        if len(config.patch_lengths) != len(config.patch_strides):
            raise ValueError("patch lengths and strides must have equal length")
        self.patch_counts = tuple(
            1 + (config.lookback - length) // stride
            for length, stride in zip(
                config.patch_lengths, config.patch_strides, strict=True
            )
        )
        if any(value <= 0 for value in self.patch_counts):
            raise ValueError("Every patch length must fit within lookback")
        self.value_projections = nn.ModuleList(
            [nn.Linear(length, config.channel_dim) for length in config.patch_lengths]
        )
        self.mask_projections = nn.ModuleList(
            [nn.Linear(length, config.channel_dim) for length in config.patch_lengths]
        )
        self.feature_embedding = nn.Embedding(config.value_channels, config.channel_dim)
        self.channel_norm = nn.LayerNorm(config.channel_dim)
        self.channel_score = nn.Linear(config.channel_dim, 1)
        self.channel_to_model = nn.Linear(config.channel_dim, config.d_model)
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
            nn.Linear(config.d_model, config.horizons * 4),
        )
        self.ssl_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.ssl_target_dim),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.regime_target_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for layer in [*self.value_projections, *self.mask_projections]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        final = self.parameter_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def export_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def _corrupt(self, history: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0.0:
            return history
        channels = self.config.value_channels
        numeric = history[..., :channels]
        masks = history[..., channels:]
        corruption = (
            torch.rand_like(numeric) < float(probability)
        ) & (masks > 0.5)
        return torch.cat(
            [numeric.masked_fill(corruption, 0.0), masks.masked_fill(corruption, 0.0)],
            dim=-1,
        )

    def encode(
        self,
        history: torch.Tensor,
        corruption_probability: float = 0.0,
    ) -> torch.Tensor:
        expected = (self.config.lookback, 2 * self.config.value_channels)
        if history.ndim != 3 or tuple(history.shape[1:]) != expected:
            raise ValueError(f"history must have shape [batch, {expected[0]}, {expected[1]}]")
        history = self._corrupt(history, corruption_probability)
        channels = self.config.value_channels
        numeric = history[..., :channels]
        masks = history[..., channels:]
        count = masks.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (numeric * masks).sum(dim=1, keepdim=True) / count
        variance = ((numeric - mean).square() * masks).sum(dim=1, keepdim=True) / count
        normalized = (numeric - mean) / variance.sqrt().clamp_min(1e-5)
        normalized = normalized * masks
        feature_ids = torch.arange(channels, device=history.device)
        feature_embedding = self.feature_embedding(feature_ids)[None, None, :, :]
        tokens: list[torch.Tensor] = []
        for scale_index, (
            length,
            stride,
            value_projection,
            mask_projection,
            position,
            count_patches,
        ) in enumerate(
            zip(
                self.config.patch_lengths,
                self.config.patch_strides,
                self.value_projections,
                self.mask_projections,
                self.position_embeddings,
                self.patch_counts,
                strict=True,
            )
        ):
            value_patch = normalized.unfold(1, length, stride).contiguous()
            mask_patch = masks.unfold(1, length, stride).contiguous()
            channel_repr = (
                value_projection(value_patch)
                + mask_projection(mask_patch)
                + feature_embedding
            )
            channel_repr = F.gelu(self.channel_norm(channel_repr))
            availability = mask_patch.mean(dim=-1).clamp_min(1e-4)
            score = self.channel_score(channel_repr).squeeze(-1) + availability.log()
            weights = torch.softmax(score, dim=2)
            pooled = torch.sum(channel_repr * weights[..., None], dim=2)
            embedded = self.channel_to_model(pooled)
            positions = torch.arange(count_patches, device=history.device)
            embedded = (
                embedded
                + self.scale_embedding.weight[scale_index][None, None, :]
                + position(positions)[None, :, :]
            )
            tokens.append(embedded)
        temporal = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(history.shape[0], -1, -1)
        return self.norm(self.encoder(torch.cat([cls, temporal], dim=1))[:, 0])

    def ssl_outputs(
        self,
        history: torch.Tensor,
        corruption_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.encode(history, corruption_probability)
        return (
            self.ssl_head(representation),
            self.regime_head(representation),
            representation,
        )

    def forward(
        self,
        history: torch.Tensor,
        baseline_quantiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representation = self.encode(history)
        raw = self.parameter_head(representation).view(
            history.shape[0], self.config.horizons, 4
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
        ).abs().clamp_min(1e-6)
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
        power = torch.exp(
            torch.tanh(raw[..., 3:4]) * float(self.config.tail_power_limit)
        )
        centered = baseline_quantiles - median
        scale = torch.where(
            quantiles[None, None, :] < 0.5, lower_scale, upper_scale
        )
        normalized_distance = centered.abs() / total_spread
        curved = centered.sign() * total_spread * normalized_distance.clamp_min(1e-8).pow(power)
        prediction = median + location + scale * curved
        return torch.sort(prediction, dim=-1).values, raw, self.regime_head(representation)


def weighted_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    tail_weight: float,
) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    weights = torch.where(
        (quantiles <= 0.10) | (quantiles >= 0.90),
        torch.full_like(quantiles, float(tail_weight)),
        torch.ones_like(quantiles),
    )
    return torch.sum(loss * weights) / (loss.shape[0] * loss.shape[1] * weights.sum())


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def _model_config(arrays: StructuredArraysV160, settings: dict[str, Any], horizons: list[int], quantiles: list[float]) -> StructuredV160Config:
    return StructuredV160Config(
        value_channels=arrays.histories.shape[-1] // 2,
        lookback=arrays.histories.shape[1],
        horizons=len(horizons),
        quantiles=tuple(quantiles),
        ssl_target_dim=arrays.ssl_targets.shape[1],
        regime_target_dim=arrays.regime_targets.shape[1],
        patch_lengths=tuple(int(value) for value in settings["patch_lengths"]),
        patch_strides=tuple(int(value) for value in settings["patch_strides"]),
        channel_dim=int(settings["channel_dim"]),
        d_model=int(settings["d_model"]),
        n_heads=int(settings["n_heads"]),
        n_layers=int(settings["n_layers"]),
        d_ff=int(settings["d_ff"]),
        dropout=float(settings["dropout"]),
        location_limit=float(settings["location_limit"]),
        log_scale_limit=float(settings["log_scale_limit"]),
        tail_power_limit=float(settings["tail_power_limit"]),
    )


def _groups(panel: LongHistoryPanel, fold: Fold, config: dict[str, Any], lookback: int) -> dict[str, np.ndarray]:
    return split_long_origins(
        panel,
        fold,
        lookback,
        [int(value) for value in config["features"]["horizons"]],
        int(config["splits"].get("embargo_days", 0)),
        common_max_horizon=(
            int(config["tailrisk"]["scenario_horizon"])
            if bool(config["performance_v160"].get("common_origin_required", True))
            else None
        ),
    )


def pretrain_structured_fold_v160(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    statistical_state: StatisticalEnsembleState,
    seed: int,
    output_dir: str | Path,
    device_name: str = "auto",
    max_epochs: int | None = None,
) -> Path:
    seed_everything(seed)
    settings = config["performance_v160"]["structured_model"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    lookback = int(settings["lookback"])
    groups = _groups(panel, fold, config, lookback)
    spec = fit_feature_spec(panel, fold, lookback)
    stride = max(1, int(settings.get("pretrain_stride", 2)))
    arrays = build_structured_arrays_v160(
        panel,
        groups["train"][::stride],
        horizons,
        quantiles,
        spec,
        statistical_state,
    )
    device = _device(device_name)
    model = SSLRegimePatchTransformer(
        _model_config(arrays, settings, horizons, quantiles)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["pretrain_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loader = DataLoader(
        StructuredDatasetV160(arrays),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    epochs = int(max_epochs or settings["pretrain_epochs"])
    history: list[dict[str, float | int]] = []
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            history_tensor = batch["history"].to(device)
            target = batch["ssl_target"].to(device)
            regime_target = batch["regime_target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction_a, regime_a, representation_a = model.ssl_outputs(
                history_tensor, float(settings["mask_probability"])
            )
            reconstruction_b, regime_b, representation_b = model.ssl_outputs(
                history_tensor, float(settings["mask_probability"])
            )
            reconstruction_loss = 0.5 * (
                F.smooth_l1_loss(reconstruction_a, target)
                + F.smooth_l1_loss(reconstruction_b, target)
            )
            regime_loss = 0.5 * (
                F.mse_loss(regime_a, regime_target)
                + F.mse_loss(regime_b, regime_target)
            )
            contrastive = 1.0 - F.cosine_similarity(
                F.normalize(representation_a, dim=-1),
                F.normalize(representation_b, dim=-1),
                dim=-1,
            ).mean()
            loss = (
                reconstruction_loss
                + float(settings["pretrain_regime_weight"]) * regime_loss
                + float(settings["pretrain_contrastive_weight"]) * contrastive
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["gradient_clip"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "pretrain_loss": mean_loss})
        print(
            f"{fold.name} structured-v160-pretrain seed={seed} "
            f"epoch={epoch:03d} loss={mean_loss:.6f}"
        )
        if mean_loss < best:
            best = mean_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Self-supervised pretraining produced no checkpoint")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_config": model.export_config(),
            "feature_spec": spec.export(),
            "fold": asdict(fold),
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_pretrain_loss": float(best),
            "version": "1.6.0",
        },
        output / "pretrain_checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(output / "pretraining_history.csv", index=False)
    return output


def _set_encoder_trainable(model: SSLRegimePatchTransformer, trainable: bool) -> None:
    modules: list[nn.Module] = [
        model.value_projections,
        model.mask_projections,
        model.feature_embedding,
        model.channel_norm,
        model.channel_score,
        model.channel_to_model,
        model.scale_embedding,
        model.position_embeddings,
        model.encoder,
        model.norm,
    ]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
    model.cls_token.requires_grad_(trainable)


def train_structured_fold_v160(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    statistical_state: StatisticalEnsembleState,
    seed: int,
    pretrain_checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    max_epochs: int | None = None,
) -> Path:
    seed_everything(seed)
    started = time.time()
    settings = config["performance_v160"]["structured_model"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    lookback = int(settings["lookback"])
    groups = _groups(panel, fold, config, lookback)
    checkpoint = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
    spec = LongFeatureSpec.from_record(checkpoint["feature_spec"])
    stride = max(1, int(settings.get("train_stride", 1)))
    train_arrays = build_structured_arrays_v160(
        panel,
        groups["train"][::stride],
        horizons,
        quantiles,
        spec,
        statistical_state,
    )
    validation_arrays = build_structured_arrays_v160(
        panel,
        groups["validation"],
        horizons,
        quantiles,
        spec,
        statistical_state,
    )
    device = _device(device_name)
    model_config = dict(checkpoint["model_config"])
    for key in ("quantiles", "patch_lengths", "patch_strides"):
        model_config[key] = tuple(model_config[key])
    model = SSLRegimePatchTransformer(StructuredV160Config(**model_config)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith(("parameter_head", "regime_head")):
            head_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(settings["learning_rate"])
                * float(settings["encoder_learning_rate_multiplier"]),
            },
            {"params": head_parameters, "lr": float(settings["learning_rate"])},
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    train_loader = DataLoader(
        StructuredDatasetV160(train_arrays),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        StructuredDatasetV160(validation_arrays),
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=device).view(1, 1, -1)
    best = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    history_rows: list[dict[str, float | int]] = []
    epochs = int(max_epochs or settings["epochs"])
    freeze_epochs = int(settings["freeze_encoder_epochs"])
    for epoch in range(1, epochs + 1):
        _set_encoder_trainable(model, epoch > freeze_epochs)
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, parameters, regime_prediction = model(
                batch["history"].to(device), batch["baseline"].to(device)
            )
            loss = weighted_pinball_loss(
                prediction,
                batch["target"].to(device),
                q_tensor,
                float(settings["tail_weight"]),
            )
            loss = (
                loss
                + float(settings["parameter_penalty"]) * parameters.square().mean()
                + float(settings["regime_auxiliary_weight"])
                * F.mse_loss(regime_prediction, batch["regime_target"].to(device))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                float(settings["gradient_clip"]),
            )
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                prediction, parameters, regime_prediction = model(
                    batch["history"].to(device), batch["baseline"].to(device)
                )
                loss = weighted_pinball_loss(
                    prediction,
                    batch["target"].to(device),
                    q_tensor,
                    float(settings["tail_weight"]),
                )
                loss = (
                    loss
                    + float(settings["parameter_penalty"])
                    * parameters.square().mean()
                    + float(settings["regime_auxiliary_weight"])
                    * F.mse_loss(
                        regime_prediction, batch["regime_target"].to(device)
                    )
                )
                validation_losses.append(float(loss.cpu()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "encoder_frozen": epoch <= freeze_epochs,
            }
        )
        print(
            f"{fold.name} structured-v160 seed={seed} epoch={epoch:03d} "
            f"train={train_loss:.6f} validation={validation_loss:.6f}"
        )
        if validation_loss < best - 1e-10:
            best = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Structured v1.6 model produced no checkpoint")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "model_config": model.export_config(),
            "feature_spec": spec.export(),
            "statistical_state": statistical_state.export(),
            "fold": asdict(fold),
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best),
            "version": "1.6.0",
        },
        output / "best_checkpoint.pt",
    )
    pd.DataFrame(history_rows).to_csv(output / "training_history.csv", index=False)
    write_json(
        output / "run_metadata.json",
        {
            **runtime_metadata(seed, device),
            "fold": asdict(fold),
            "model_family": "ssl_regime_patch_transformer_v160",
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best),
            "elapsed_seconds": time.time() - started,
            "pretraining_checkpoint": str(pretrain_checkpoint),
        },
    )
    write_json(
        output / "data_split.json",
        {name: value.tolist() for name, value in groups.items()},
    )
    write_json(output / "feature_spec.json", spec.export())
    return output


def load_structured_checkpoint_v160(
    path: str | Path,
    device_name: str = "auto",
) -> tuple[
    SSLRegimePatchTransformer,
    LongFeatureSpec,
    StatisticalEnsembleState,
    dict[str, Any],
    torch.device,
]:
    device = _device(device_name)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    record = dict(checkpoint["model_config"])
    for key in ("quantiles", "patch_lengths", "patch_strides"):
        record[key] = tuple(record[key])
    model = SSLRegimePatchTransformer(StructuredV160Config(**record)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return (
        model,
        LongFeatureSpec.from_record(checkpoint["feature_spec"]),
        StatisticalEnsembleState.from_record(checkpoint["statistical_state"]),
        checkpoint,
        device,
    )


def predict_structured_v160(
    model: SSLRegimePatchTransformer,
    arrays: StructuredArraysV160,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        StructuredDatasetV160(arrays), batch_size=batch_size, shuffle=False
    )
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            prediction, _, _ = model(
                batch["history"].to(device), batch["baseline"].to(device)
            )
            output.append(prediction.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def export_structured_seed_v160(
    panel: LongHistoryPanel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
) -> Path:
    started = time.time()
    model, spec, statistical_state, checkpoint, device = load_structured_checkpoint_v160(
        checkpoint_path, device_name
    )
    settings = config["performance_v160"]["structured_model"]
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    groups = _groups(panel, fold, config, spec.lookback)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("validation", "test"):
        arrays = build_structured_arrays_v160(
            panel,
            groups[split_name],
            horizons,
            quantiles,
            spec,
            statistical_state,
        )
        raw = predict_structured_v160(
            model, arrays, device, int(settings["evaluation_batch_size"])
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
            "model_family": "ssl_regime_patch_transformer_v160",
        },
    )
    return output
