from __future__ import annotations

import torch
from torch import nn

from riskgraph.models.layers import DenseGraphAttention, GatedResidualNetwork, OrderedQuantileHead
from riskgraph.models.temporal import ModelOutput, TemporalEncoder


class TemporalGraphQuantileNet(nn.Module):
    def __init__(
        self,
        asset_features: int,
        macro_features: int,
        hidden_size: int,
        lstm_layers: int,
        attention_heads: int,
        graph_heads: int,
        dropout: float,
        horizons: int,
        quantile_levels: list[float],
        target_index: int,
        graph_signal_mode: str = "direction",
    ) -> None:
        super().__init__()
        if graph_signal_mode not in {"direction", "embedding"}:
            raise ValueError("graph_signal_mode must be 'direction' or 'embedding'")
        self.target_index = target_index
        self.graph_signal_mode = graph_signal_mode
        self.encoder = TemporalEncoder(
            asset_features,
            macro_features,
            hidden_size,
            lstm_layers,
            attention_heads,
            dropout,
        )
        self.graph = DenseGraphAttention(hidden_size, graph_heads, dropout)
        self.graph_post = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_post = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.quantile_head = OrderedQuantileHead(hidden_size, horizons, quantile_levels)
        self.direction_head = nn.Linear(hidden_size, 1)
        self.direction_to_hidden = nn.Linear(1, hidden_size)

    def encode_state(
        self, asset: torch.Tensor, macro: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        temporal, node_last, asset_weights, macro_weights, temporal_attention = self.encoder(
            asset, macro, self.target_index
        )
        graph_nodes, graph_attention = self.graph(node_last, adjacency)
        graph_target = self.graph_post(graph_nodes[:, self.target_index, :])
        direction_logit = self.direction_head(graph_target).squeeze(-1)
        if self.graph_signal_mode == "direction":
            graph_fusion = self.direction_to_hidden(torch.tanh(direction_logit).unsqueeze(-1))
        else:
            graph_fusion = graph_target
        gate = torch.sigmoid(self.gate(torch.cat([temporal, graph_fusion], dim=-1)))
        fused = self.fusion_post(gate * graph_fusion + (1.0 - gate) * temporal)
        return (
            fused,
            direction_logit,
            asset_weights,
            macro_weights,
            temporal_attention,
            graph_attention,
            gate,
        )

    def forward(self, asset: torch.Tensor, macro: torch.Tensor, adjacency: torch.Tensor) -> ModelOutput:
        (
            fused,
            direction_logit,
            asset_weights,
            macro_weights,
            temporal_attention,
            graph_attention,
            gate,
        ) = self.encode_state(asset, macro, adjacency)
        quantiles = self.quantile_head(fused)
        return ModelOutput(
            quantiles=quantiles,
            direction_logit=direction_logit,
            asset_variable_weights=asset_weights,
            macro_variable_weights=macro_weights,
            temporal_attention=temporal_attention,
            graph_attention=graph_attention,
            fusion_gate=gate,
        )
