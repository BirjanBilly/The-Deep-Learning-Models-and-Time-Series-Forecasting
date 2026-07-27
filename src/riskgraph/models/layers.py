from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int | None = None, dropout: float = 0.1):
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.embedders = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(n_features)])
        self.weight_net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features),
        )
        self.post = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, received {x.shape[-1]}")
        weights = torch.softmax(self.weight_net(x), dim=-1)
        embedded = torch.stack(
            [embedder(x[..., index : index + 1]) for index, embedder in enumerate(self.embedders)],
            dim=-2,
        )
        selected = (embedded * weights.unsqueeze(-1)).sum(dim=-2)
        return self.post(selected), weights


class DenseGraphAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.sign_scale = nn.Parameter(torch.tensor(0.25))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        return x.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._reshape(self.q_proj(node_states))
        k = self._reshape(self.k_proj(node_states))
        v = self._reshape(self.v_proj(node_states))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        edge_strength = adjacency.abs().clamp_min(1e-6)
        edge_bias = torch.log(edge_strength).unsqueeze(1)
        edge_bias = edge_bias + self.sign_scale * adjacency.sign().unsqueeze(1)
        scores = scores + edge_bias
        mask = adjacency.abs().unsqueeze(1) > 0
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(node_states.shape)
        output = self.norm(node_states + self.out_proj(context))
        return output, attention


class OrderedQuantileHead(nn.Module):
    """Monotone quantile head initialised as a Gaussian location-scale forecast.

    Adjacent quantile spacings are positive by construction.  At initialisation the
    spacings match standard-normal quantiles; the network then learns horizon-specific
    location, scale and positive spacing multipliers.
    """

    def __init__(self, hidden_size: int, horizons: int, quantile_levels: list[float]):
        super().__init__()
        if len(quantile_levels) < 2:
            raise ValueError("At least two quantiles are required")
        levels = torch.tensor(quantile_levels, dtype=torch.float32)
        if not torch.all((levels > 0.0) & (levels < 1.0)):
            raise ValueError("Quantiles must lie strictly between zero and one")
        if not torch.all(levels[1:] > levels[:-1]):
            raise ValueError("Quantiles must be strictly increasing")
        normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        base = normal.icdf(levels)
        self.horizons = int(horizons)
        self.quantiles = int(len(quantile_levels))
        self.location = nn.Linear(hidden_size, horizons)
        self.scale = nn.Linear(hidden_size, horizons)
        self.spacing = nn.Linear(hidden_size, horizons * (self.quantiles - 1))
        self.register_buffer("base_first", base[:1])
        self.register_buffer("base_deltas", base[1:] - base[:-1])
        nn.init.zeros_(self.spacing.weight)
        nn.init.zeros_(self.spacing.bias)
        nn.init.zeros_(self.location.bias)
        nn.init.zeros_(self.scale.weight)
        # softplus(0.5413) is approximately one target standard deviation.
        nn.init.constant_(self.scale.bias, 0.541324854612918)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        location = self.location(x)
        scale = F.softplus(self.scale(x)) + 1e-4
        spacing_raw = self.spacing(x).view(x.shape[0], self.horizons, self.quantiles - 1)
        spacing_multiplier = F.softplus(spacing_raw) / math.log(2.0)
        deltas = scale.unsqueeze(-1) * self.base_deltas.view(1, 1, -1) * spacing_multiplier
        first = location.unsqueeze(-1) + scale.unsqueeze(-1) * self.base_first.view(1, 1, 1)
        return torch.cat([first, first + torch.cumsum(deltas, dim=-1)], dim=-1)
