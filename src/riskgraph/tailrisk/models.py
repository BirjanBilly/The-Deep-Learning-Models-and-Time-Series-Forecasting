from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from riskgraph.tailrisk.conditioning import RiskGraphConditioner
from riskgraph.tailrisk.sorting import neural_sort


class RegimeTailGenerator(nn.Module):
    """Regime-conditioned generator for normalized multi-asset return paths."""

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        hidden_sizes: tuple[int, ...] = (128, 256, 512),
        output_clip: float = 8.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.output_clip = float(output_clip)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        layers: list[nn.Module] = []
        previous = self.latent_dim + int(regime_embedding_dim)
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.LeakyReLU(0.2),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, self.horizon * self.n_assets))
        self.network = nn.Sequential(*layers)
        self.asset_log_scale = nn.Parameter(torch.zeros(self.n_assets))

    def forward(self, noise: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        embedding = self.regime_embedding(regime)
        values = self.network(torch.cat([noise, embedding], dim=-1))
        values = values.view(noise.shape[0], self.horizon, self.n_assets)
        scale = torch.exp(self.asset_log_scale).clamp(0.25, 4.0)
        values = values * scale.view(1, 1, -1)
        return torch.clamp(values, -self.output_clip, self.output_clip)


class RiskGraphConditionedTailGenerator(nn.Module):
    """Decode multi-asset paths from latent noise and a frozen RiskGraph state."""

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        state_dim: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        state_projection_dim: int = 64,
        hidden_sizes: tuple[int, ...] = (256, 512, 512),
        output_clip: float = 8.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.state_dim = int(state_dim)
        self.output_clip = float(output_clip)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, int(state_projection_dim)),
            nn.LayerNorm(int(state_projection_dim)),
            nn.SiLU(),
        )
        layers: list[nn.Module] = []
        previous = self.latent_dim + int(regime_embedding_dim) + int(state_projection_dim)
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.LayerNorm(int(hidden)),
                    nn.LeakyReLU(0.2),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, self.horizon * self.n_assets))
        self.residual_network = nn.Sequential(*layers)
        self.state_location = nn.Linear(int(state_projection_dim), self.horizon * self.n_assets)
        self.residual_gate = nn.Sequential(
            nn.Linear(int(state_projection_dim), self.n_assets),
            nn.Sigmoid(),
        )
        self.asset_log_scale = nn.Parameter(torch.zeros(self.n_assets))

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if state_embedding.ndim != 2 or state_embedding.shape[0] != noise.shape[0]:
            raise ValueError(
                "state_embedding must have shape [batch, state_dim] and match noise batch"
            )
        projected = self.state_projection(state_embedding)
        regime_embedding = self.regime_embedding(regime)
        residual = self.residual_network(
            torch.cat([noise, regime_embedding, projected], dim=-1)
        ).view(noise.shape[0], self.horizon, self.n_assets)
        location = self.state_location(projected).view(
            noise.shape[0], self.horizon, self.n_assets
        )
        gate = self.residual_gate(projected).view(noise.shape[0], 1, self.n_assets)
        asset_scale = torch.exp(self.asset_log_scale).clamp(0.25, 4.0).view(1, 1, -1)
        values = location + gate * residual * asset_scale
        return torch.clamp(values, -self.output_clip, self.output_clip)


class RiskGraphConditionedScenarioGenerator(nn.Module):
    """Self-contained frozen RiskGraph conditioner plus trainable path decoder."""

    def __init__(
        self,
        conditioner: RiskGraphConditioner,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        state_projection_dim: int = 64,
        hidden_sizes: tuple[int, ...] = (256, 512, 512),
        output_clip: float = 8.0,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.decoder = RiskGraphConditionedTailGenerator(
            latent_dim=latent_dim,
            horizon=horizon,
            n_assets=n_assets,
            state_dim=conditioner.state_dim,
            n_regimes=n_regimes,
            regime_embedding_dim=regime_embedding_dim,
            state_projection_dim=state_projection_dim,
            hidden_sizes=hidden_sizes,
            output_clip=output_clip,
        )
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.conditioner.freeze()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.conditioner.freeze()
        self.conditioner.eval()
        return self

    def encode_state(
        self,
        asset: torch.Tensor,
        macro: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.conditioner(asset, macro, adjacency)

    def decode(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(noise, regime, state_embedding)

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        state_asset: torch.Tensor,
        state_macro: torch.Tensor,
        state_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, teacher_quantiles = self.encode_state(
            state_asset, state_macro, state_adjacency
        )
        return self.decode(noise, regime, state), teacher_quantiles


class EWMABackboneTailDecoder(nn.Module):
    """Temporal residual decoder around a correlated EWMA Student-t backbone."""

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        hidden_size: int = 192,
        layers: int = 2,
        state_dim: int = 0,
        state_projection_dim: int = 64,
        residual_clip: float = 4.0,
        output_clip: float = 8.0,
        initial_gate: float = 0.08,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.state_dim = int(state_dim)
        self.residual_clip = float(residual_clip)
        self.output_clip = float(output_clip)
        nu = float(degrees_of_freedom)
        self.noise_unit_scale = float(((nu - 2.0) / nu) ** 0.5) if nu > 2.0 else 1.0
        self.noise_projection = nn.Linear(self.latent_dim, self.horizon * self.n_assets)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        context_dim = int(regime_embedding_dim)
        self.state_projection: nn.Module | None = None
        if self.state_dim > 0:
            self.state_projection = nn.Sequential(
                nn.Linear(self.state_dim, int(state_projection_dim)),
                nn.LayerNorm(int(state_projection_dim)),
                nn.SiLU(),
            )
            context_dim += int(state_projection_dim)
        self.input_projection = nn.Linear(self.n_assets + context_dim, int(hidden_size))
        self.gru = nn.GRU(
            input_size=int(hidden_size),
            hidden_size=int(hidden_size),
            num_layers=int(layers),
            dropout=0.10 if int(layers) > 1 else 0.0,
            batch_first=True,
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), self.n_assets),
        )
        initial = max(min(float(initial_gate), 0.99), 0.01)
        logit = torch.log(torch.tensor(initial / (1.0 - initial)))
        self.gate_logits = nn.Parameter(logit.repeat(self.n_assets))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.noise_projection.weight)
        nn.init.zeros_(self.noise_projection.bias)
        final = self.residual_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        baseline_cholesky: torch.Tensor,
        state_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if baseline_cholesky.shape != (
            noise.shape[0],
            self.n_assets,
            self.n_assets,
        ):
            raise ValueError("baseline_cholesky has the wrong shape")
        normalized_weight = F.normalize(self.noise_projection.weight, dim=1)
        innovation = F.linear(
            noise * self.noise_unit_scale,
            normalized_weight,
            self.noise_projection.bias,
        ).view(noise.shape[0], self.horizon, self.n_assets)
        backbone = torch.einsum("bhj,bij->bhi", innovation, baseline_cholesky)
        context = [self.regime_embedding(regime)]
        if self.state_projection is not None:
            if state_embedding is None:
                raise ValueError("state_embedding is required for conditioned backbone")
            context.append(self.state_projection(state_embedding))
        expanded = torch.cat(context, dim=-1)[:, None, :].expand(-1, self.horizon, -1)
        hidden, _ = self.gru(self.input_projection(torch.cat([backbone, expanded], dim=-1)))
        residual = torch.tanh(self.residual_head(hidden)) * self.residual_clip
        gate = torch.sigmoid(self.gate_logits).view(1, 1, -1)
        return torch.clamp(backbone + gate * residual, -self.output_clip, self.output_clip)


class EWMABackboneTailGenerator(nn.Module):
    """Regime-conditioned baseline-anchored Tail-GAN generator."""

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        hidden_size: int = 192,
        layers: int = 2,
        initial_gate: float = 0.08,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        super().__init__()
        self.decoder = EWMABackboneTailDecoder(
            latent_dim=latent_dim,
            horizon=horizon,
            n_assets=n_assets,
            n_regimes=n_regimes,
            hidden_size=hidden_size,
            layers=layers,
            initial_gate=initial_gate,
            degrees_of_freedom=degrees_of_freedom,
        )
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        baseline_cholesky: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(noise, regime, baseline_cholesky)


class RiskGraphEWMABackboneScenarioGenerator(nn.Module):
    """Frozen RiskGraph FiLM-style context plus an EWMA residual path decoder."""

    def __init__(
        self,
        conditioner: RiskGraphConditioner,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        state_projection_dim: int = 64,
        hidden_size: int = 192,
        layers: int = 2,
        initial_gate: float = 0.08,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.decoder = EWMABackboneTailDecoder(
            latent_dim=latent_dim,
            horizon=horizon,
            n_assets=n_assets,
            n_regimes=n_regimes,
            hidden_size=hidden_size,
            layers=layers,
            state_dim=conditioner.state_dim,
            state_projection_dim=state_projection_dim,
            initial_gate=initial_gate,
            degrees_of_freedom=degrees_of_freedom,
        )
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.conditioner.freeze()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.conditioner.freeze()
        self.conditioner.eval()
        return self

    def encode_state(
        self,
        asset: torch.Tensor,
        macro: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.conditioner(asset, macro, adjacency)

    def decode(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        baseline_cholesky: torch.Tensor,
        state_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(noise, regime, baseline_cholesky, state_embedding)


class EWMABackboneFactorScaleGenerator(nn.Module):
    """Constrained factor/idio scale adapter around an EWMA Student-t backbone.

    The generator cannot emit an unrestricted additive path. It may only rescale
    low-rank covariance factors, rescale idiosyncratic innovations, add a bounded
    drift and apply a bounded downside asymmetry. All heads are zero-initialized,
    so the initial generator exactly equals the statistical backbone.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        hidden_size: int = 160,
        layers: int = 2,
        factor_rank: int = 4,
        initial_gate: float = 0.04,
        factor_scale_limit: float = 0.30,
        idio_scale_limit: float = 0.20,
        drift_limit: float = 0.08,
        skew_limit: float = 0.15,
        output_clip: float = 8.0,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.factor_rank = min(max(1, int(factor_rank)), self.n_assets)
        self.factor_scale_limit = float(factor_scale_limit)
        self.idio_scale_limit = float(idio_scale_limit)
        self.drift_limit = float(drift_limit)
        self.skew_limit = float(skew_limit)
        self.output_clip = float(output_clip)
        nu = float(degrees_of_freedom)
        self.noise_unit_scale = float(((nu - 2.0) / nu) ** 0.5) if nu > 2.0 else 1.0
        self.noise_projection = nn.Linear(self.latent_dim, self.horizon * self.n_assets)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        self.input_projection = nn.Linear(
            self.n_assets + int(regime_embedding_dim), int(hidden_size)
        )
        self.gru = nn.GRU(
            input_size=int(hidden_size),
            hidden_size=int(hidden_size),
            num_layers=int(layers),
            dropout=0.10 if int(layers) > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.factor_head = nn.Linear(int(hidden_size), self.factor_rank)
        self.idio_head = nn.Linear(int(hidden_size), self.n_assets)
        self.drift_head = nn.Linear(int(hidden_size), self.n_assets)
        self.skew_head = nn.Linear(int(hidden_size), self.n_assets)
        initial = max(min(float(initial_gate), 0.99), 0.001)
        logit = torch.log(torch.tensor(initial / (1.0 - initial)))
        self.factor_gate_logit = nn.Parameter(logit.clone())
        self.idio_gate_logit = nn.Parameter(logit.clone())
        self.drift_gate_logit = nn.Parameter(logit.clone())
        self.skew_gate_logit = nn.Parameter(logit.clone())
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.noise_projection.weight)
        nn.init.zeros_(self.noise_projection.bias)
        for head in (self.factor_head, self.idio_head, self.drift_head, self.skew_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        baseline_cholesky: torch.Tensor,
    ) -> torch.Tensor:
        if baseline_cholesky.shape != (
            noise.shape[0],
            self.n_assets,
            self.n_assets,
        ):
            raise ValueError("baseline_cholesky has the wrong shape")
        normalized_weight = F.normalize(self.noise_projection.weight, dim=1)
        innovation = F.linear(
            noise * self.noise_unit_scale,
            normalized_weight,
            self.noise_projection.bias,
        ).view(noise.shape[0], self.horizon, self.n_assets)
        backbone = torch.einsum("bhj,bij->bhi", innovation, baseline_cholesky)

        covariance = baseline_cholesky @ baseline_cholesky.transpose(-1, -2)
        # The covariance input is not trainable. Detaching eigenvectors avoids
        # unstable gradients near repeated eigenvalues without changing the model.
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[..., -self.factor_rank :].detach()
        factor_scores = torch.einsum("bhi,bik->bhk", backbone, basis)
        factor_component = torch.einsum("bhk,bik->bhi", factor_scores, basis)
        idiosyncratic = backbone - factor_component

        regime_context = self.regime_embedding(regime)[:, None, :].expand(
            -1, self.horizon, -1
        )
        hidden, _ = self.gru(
            self.input_projection(torch.cat([backbone, regime_context], dim=-1))
        )
        hidden = self.norm(hidden)
        factor_adjustment = (
            torch.sigmoid(self.factor_gate_logit)
            * self.factor_scale_limit
            * torch.tanh(self.factor_head(hidden))
        )
        idio_adjustment = (
            torch.sigmoid(self.idio_gate_logit)
            * self.idio_scale_limit
            * torch.tanh(self.idio_head(hidden))
        )
        drift = (
            torch.sigmoid(self.drift_gate_logit)
            * self.drift_limit
            * torch.tanh(self.drift_head(hidden))
        )
        skew = (
            torch.sigmoid(self.skew_gate_logit)
            * self.skew_limit
            * torch.tanh(self.skew_head(hidden))
        )
        scaled_factor = torch.einsum(
            "bhk,bik->bhi", factor_scores * (1.0 + factor_adjustment), basis
        )
        scaled_idio = idiosyncratic * (1.0 + idio_adjustment)
        downside = skew * torch.relu(-backbone)
        output = scaled_factor + scaled_idio + drift - downside
        return torch.clamp(output, -self.output_clip, self.output_clip)


class StableEWMABackboneFactorScaleGenerator(nn.Module):
    """Origin-level factor-scale adapter with bounded, time-consistent controls.

    Unlike the v1.5 decoder, the scale and skew controls are constant across the
    scenario horizon. This sharply reduces the generator's degrees of freedom and
    prevents it from creating unrelated adjustments at every time step. All heads
    are zero-initialized, so the initial output is exactly the EWMA Student-t
    backbone.
    """

    def __init__(
        self,
        latent_dim: int,
        horizon: int,
        n_assets: int,
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        hidden_size: int = 160,
        layers: int = 2,
        factor_rank: int = 4,
        initial_gate: float = 0.03,
        factor_scale_limit: float = 0.25,
        idio_scale_limit: float = 0.15,
        drift_limit: float = 0.04,
        skew_limit: float = 0.12,
        output_clip: float = 8.0,
        degrees_of_freedom: float = 5.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.horizon = int(horizon)
        self.n_assets = int(n_assets)
        self.factor_rank = min(max(1, int(factor_rank)), self.n_assets)
        self.factor_scale_limit = float(factor_scale_limit)
        self.idio_scale_limit = float(idio_scale_limit)
        self.drift_limit = float(drift_limit)
        self.skew_limit = float(skew_limit)
        self.output_clip = float(output_clip)
        nu = float(degrees_of_freedom)
        self.noise_unit_scale = float(((nu - 2.0) / nu) ** 0.5) if nu > 2.0 else 1.0
        self.noise_projection = nn.Linear(self.latent_dim, self.horizon * self.n_assets)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        self.input_projection = nn.Linear(
            self.n_assets + int(regime_embedding_dim), int(hidden_size)
        )
        self.gru = nn.GRU(
            input_size=int(hidden_size),
            hidden_size=int(hidden_size),
            num_layers=int(layers),
            dropout=0.10 if int(layers) > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(int(hidden_size))
        self.temporal_score = nn.Linear(int(hidden_size), 1)
        self.factor_head = nn.Linear(int(hidden_size), self.factor_rank)
        self.idio_head = nn.Linear(int(hidden_size), self.n_assets)
        self.drift_head = nn.Linear(int(hidden_size), self.n_assets)
        self.skew_head = nn.Linear(int(hidden_size), self.n_assets)
        initial = max(min(float(initial_gate), 0.99), 0.001)
        logit = torch.log(torch.tensor(initial / (1.0 - initial)))
        self.factor_gate_logit = nn.Parameter(logit.clone())
        self.idio_gate_logit = nn.Parameter(logit.clone())
        self.drift_gate_logit = nn.Parameter(logit.clone())
        self.skew_gate_logit = nn.Parameter(logit.clone())
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.noise_projection.weight)
        nn.init.zeros_(self.noise_projection.bias)
        nn.init.zeros_(self.temporal_score.bias)
        for head in (self.factor_head, self.idio_head, self.drift_head, self.skew_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        noise: torch.Tensor,
        regime: torch.Tensor,
        baseline_cholesky: torch.Tensor,
    ) -> torch.Tensor:
        if baseline_cholesky.shape != (noise.shape[0], self.n_assets, self.n_assets):
            raise ValueError("baseline_cholesky has the wrong shape")
        normalized_weight = F.normalize(self.noise_projection.weight, dim=1)
        innovation = F.linear(
            noise * self.noise_unit_scale,
            normalized_weight,
            self.noise_projection.bias,
        ).view(noise.shape[0], self.horizon, self.n_assets)
        backbone = torch.einsum("bhj,bij->bhi", innovation, baseline_cholesky)
        covariance = baseline_cholesky @ baseline_cholesky.transpose(-1, -2)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[..., -self.factor_rank :].detach()
        factor_scores = torch.einsum("bhi,bik->bhk", backbone, basis)
        factor_component = torch.einsum("bhk,bik->bhi", factor_scores, basis)
        idiosyncratic = backbone - factor_component
        regime_context = self.regime_embedding(regime)[:, None, :].expand(
            -1, self.horizon, -1
        )
        hidden, _ = self.gru(
            self.input_projection(torch.cat([backbone, regime_context], dim=-1))
        )
        hidden = self.norm(hidden)
        attention = torch.softmax(self.temporal_score(hidden).squeeze(-1), dim=1)
        context = torch.sum(hidden * attention[..., None], dim=1)
        factor_adjustment = (
            torch.sigmoid(self.factor_gate_logit)
            * self.factor_scale_limit
            * torch.tanh(self.factor_head(context))
        )
        idio_adjustment = (
            torch.sigmoid(self.idio_gate_logit)
            * self.idio_scale_limit
            * torch.tanh(self.idio_head(context))
        )
        drift = (
            torch.sigmoid(self.drift_gate_logit)
            * self.drift_limit
            * torch.tanh(self.drift_head(context))
        )
        skew = (
            torch.sigmoid(self.skew_gate_logit)
            * self.skew_limit
            * torch.tanh(self.skew_head(context))
        )
        scaled_factor = torch.einsum(
            "bhk,bik->bhi", factor_scores * (1.0 + factor_adjustment[:, None, :]), basis
        )
        scaled_idio = idiosyncratic * (1.0 + idio_adjustment[:, None, :])
        downside = skew[:, None, :] * torch.relu(-backbone)
        output = scaled_factor + scaled_idio + drift[:, None, :] - downside
        return torch.clamp(output, -self.output_clip, self.output_clip)


class TailRiskDiscriminator(nn.Module):
    """Map ranked strategy PnL samples to joint VaR/ES estimates."""

    def __init__(
        self,
        sample_size: int,
        n_strategies: int,
        alphas: list[float] | tuple[float, ...],
        n_regimes: int = 3,
        regime_embedding_dim: int = 8,
        temperature: float = 0.1,
        weight: float = 10.0,
        spectral_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.sample_size = int(sample_size)
        self.n_strategies = int(n_strategies)
        self.alphas = tuple(float(value) for value in alphas)
        self.temperature = float(temperature)
        self.weight = float(weight)
        self.regime_embedding = nn.Embedding(int(n_regimes), int(regime_embedding_dim))
        linear_1 = nn.Linear(self.sample_size + int(regime_embedding_dim), 256)
        linear_2 = nn.Linear(256, 128)
        linear_3 = nn.Linear(128, 2 * len(self.alphas))
        if spectral_normalization:
            linear_1 = nn.utils.parametrizations.spectral_norm(linear_1)
            linear_2 = nn.utils.parametrizations.spectral_norm(linear_2)
            linear_3 = nn.utils.parametrizations.spectral_norm(linear_3)
        self.network = nn.Sequential(
            linear_1,
            nn.LeakyReLU(0.2),
            linear_2,
            nn.LeakyReLU(0.2),
            linear_3,
        )

    def _parameterize(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw.view(self.n_strategies, len(self.alphas), 2)
        var = raw[..., 0]
        gap = F.softplus(raw[..., 1])
        expected_shortfall = var - gap
        lower_bound = self.weight * var
        constrained = torch.maximum(expected_shortfall, lower_bound)
        expected_shortfall = torch.where(var < 0.0, constrained, torch.minimum(expected_shortfall, var))
        return var, expected_shortfall

    def forward(self, pnl: torch.Tensor, regime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pnl.shape != (self.sample_size, self.n_strategies):
            raise ValueError(
                f"Expected pnl shape {(self.sample_size, self.n_strategies)}, received {tuple(pnl.shape)}"
            )
        if regime.ndim != 1 or len(regime) != self.sample_size:
            raise ValueError("regime must have shape [sample_size]")
        if not bool(torch.all(regime == regime[0])):
            raise ValueError("All samples passed to the discriminator must share one regime")
        ranked = neural_sort(pnl.transpose(0, 1), temperature=self.temperature, descending=False)
        embedding = self.regime_embedding(regime[0]).expand(self.n_strategies, -1)
        raw = self.network(torch.cat([ranked, embedding], dim=-1))
        return self._parameterize(raw)


class ConditionalPathCritic(nn.Module):
    """WGAN-GP baseline critic for normalized paths."""

    def __init__(self, horizon: int, n_assets: int, n_regimes: int = 3, embedding_dim: int = 8) -> None:
        super().__init__()
        self.regime_embedding = nn.Embedding(n_regimes, embedding_dim)
        dimension = int(horizon) * int(n_assets) + int(embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(dimension, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, path: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        flat = path.flatten(start_dim=1)
        embedding = self.regime_embedding(regime)
        return self.network(torch.cat([flat, embedding], dim=-1)).squeeze(-1)


def sample_noise(
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    distribution: str = "student_t",
    degrees_of_freedom: float = 5.0,
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(batch_size, latent_dim, device=device)
    if distribution == "student_t":
        df = torch.tensor(float(degrees_of_freedom), device=device)
        normal = torch.randn(batch_size, latent_dim, device=device)
        chi_square = torch.distributions.Chi2(df).sample((batch_size, latent_dim)).to(device)
        return normal / torch.sqrt(chi_square / df)
    raise ValueError(f"Unknown noise distribution: {distribution}")
