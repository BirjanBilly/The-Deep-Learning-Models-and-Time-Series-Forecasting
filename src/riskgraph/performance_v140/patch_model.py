from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualPatchConfig:
    channels: int
    lookback: int
    horizons: int
    quantiles: int
    patch_length: int = 12
    patch_stride: int = 6
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    dropout: float = 0.10
    correction_limit: float = 1.5


class ResidualPatchQuantileTransformer(nn.Module):
    """Patch-token Transformer that learns bounded corrections to EWMA-t quantiles.

    The baseline is an explicit skip connection. The residual gate is initialized
    near zero so training begins close to the transparent EWMA benchmark rather
    than relearning volatility from scratch.
    """

    def __init__(self, config: ResidualPatchConfig) -> None:
        super().__init__()
        self.config = config
        if config.patch_length > config.lookback:
            raise ValueError("patch_length cannot exceed lookback")
        self.patch_count = 1 + (config.lookback - config.patch_length) // config.patch_stride
        self.patch_projection = nn.Linear(config.patch_length, config.d_model)
        self.channel_embedding = nn.Embedding(config.channels, config.d_model)
        self.position_embedding = nn.Embedding(self.patch_count, config.d_model)
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
        output_size = config.horizons * config.quantiles
        self.correction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, output_size),
        )
        self.gate_logits = nn.Parameter(torch.full((config.horizons, config.quantiles), -1.75))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        final = self.correction_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def export_config(self) -> dict[str, int | float]:
        return asdict(self.config)

    def _tokens(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, lookback, channels]")
        if history.shape[1:] != (self.config.lookback, self.config.channels):
            raise ValueError(
                f"Expected history (*, {self.config.lookback}, {self.config.channels}), "
                f"got {tuple(history.shape)}"
            )
        # Reversible instance normalization over the historical axis. The EWMA
        # skip path retains level/scale information for the target distribution.
        mean = history.mean(dim=1, keepdim=True)
        std = history.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        values = (history - mean) / std
        patches = values.transpose(1, 2).unfold(
            dimension=-1,
            size=self.config.patch_length,
            step=self.config.patch_stride,
        )
        embedded = self.patch_projection(patches)
        channel_index = torch.arange(self.config.channels, device=history.device)
        position_index = torch.arange(self.patch_count, device=history.device)
        embedded = (
            embedded
            + self.channel_embedding(channel_index)[None, :, None, :]
            + self.position_embedding(position_index)[None, None, :, :]
        )
        return embedded.flatten(1, 2)

    def forward(
        self,
        history: torch.Tensor,
        baseline_quantiles: torch.Tensor,
    ) -> torch.Tensor:
        if baseline_quantiles.shape[1:] != (
            self.config.horizons,
            self.config.quantiles,
        ):
            raise ValueError("baseline_quantiles has the wrong shape")
        tokens = self._tokens(history)
        cls = self.cls_token.expand(history.shape[0], -1, -1)
        hidden = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        correction = self.correction_head(self.norm(hidden)).view(
            history.shape[0], self.config.horizons, self.config.quantiles
        )
        spread = (
            baseline_quantiles[..., -1:] - baseline_quantiles[..., :1]
        ).abs().clamp_min(1e-4)
        correction = torch.tanh(correction) * spread * float(self.config.correction_limit)
        gate = torch.sigmoid(self.gate_logits)[None, ...]
        prediction = baseline_quantiles + gate * correction
        # Rearrangement gives a valid quantile function while preserving gradients.
        return torch.sort(prediction, dim=-1).values


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()
