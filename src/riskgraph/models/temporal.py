from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from riskgraph.models.layers import (
    GatedResidualNetwork,
    OrderedQuantileHead,
    VariableSelectionNetwork,
)


@dataclass
class ModelOutput:
    quantiles: torch.Tensor
    direction_logit: torch.Tensor | None
    asset_variable_weights: torch.Tensor
    macro_variable_weights: torch.Tensor
    temporal_attention: torch.Tensor
    graph_attention: torch.Tensor | None
    fusion_gate: torch.Tensor | None


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        asset_features: int,
        macro_features: int,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.asset_vsn = VariableSelectionNetwork(asset_features, hidden_size, dropout)
        self.macro_vsn = VariableSelectionNetwork(macro_features, hidden_size, dropout)
        self.input_fusion = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_post = GatedResidualNetwork(hidden_size * 2, hidden_size, hidden_size, dropout)

    def forward(
        self,
        asset: torch.Tensor,
        macro: torch.Tensor,
        target_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, lookback, nodes, _ = asset.shape
        asset_selected, asset_weights = self.asset_vsn(asset)
        macro_selected, macro_weights = self.macro_vsn(macro)
        macro_expanded = macro_selected.unsqueeze(2).expand(-1, -1, nodes, -1)
        fused = self.input_fusion(torch.cat([asset_selected, macro_expanded], dim=-1))
        recurrent_input = fused.permute(0, 2, 1, 3).reshape(batch * nodes, lookback, self.hidden_size)
        recurrent, _ = self.lstm(recurrent_input)
        recurrent = recurrent.view(batch, nodes, lookback, self.hidden_size)
        node_last = recurrent[:, :, -1, :]
        target_sequence = recurrent[:, target_index, :, :]
        query = target_sequence[:, -1:, :]
        attended, attention = self.temporal_attention(
            query=query,
            key=target_sequence,
            value=target_sequence,
            need_weights=True,
            average_attn_weights=False,
        )
        temporal_context = self.temporal_post(
            torch.cat([query.squeeze(1), attended.squeeze(1)], dim=-1)
        )
        return temporal_context, node_last, asset_weights, macro_weights, attention


class TemporalFusionQuantileNet(nn.Module):
    def __init__(
        self,
        asset_features: int,
        macro_features: int,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        dropout: float,
        horizons: int,
        quantile_levels: list[float],
        target_index: int,
    ) -> None:
        super().__init__()
        self.target_index = target_index
        self.encoder = TemporalEncoder(
            asset_features,
            macro_features,
            hidden_size,
            lstm_layers,
            attention_heads,
            dropout,
        )
        self.quantile_head = OrderedQuantileHead(hidden_size, horizons, quantile_levels)

    def encode_state(
        self, asset: torch.Tensor, macro: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del adjacency
        temporal, _, asset_weights, macro_weights, attention = self.encoder(
            asset, macro, self.target_index
        )
        return temporal, asset_weights, macro_weights, attention

    def forward(self, asset: torch.Tensor, macro: torch.Tensor, adjacency: torch.Tensor) -> ModelOutput:
        temporal, asset_weights, macro_weights, attention = self.encode_state(
            asset, macro, adjacency
        )
        quantiles = self.quantile_head(temporal)
        return ModelOutput(
            quantiles=quantiles,
            direction_logit=None,
            asset_variable_weights=asset_weights,
            macro_variable_weights=macro_weights,
            temporal_attention=attention,
            graph_attention=None,
            fusion_gate=None,
        )
