from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _flatten_paths(paths: torch.Tensor) -> torch.Tensor:
    if paths.ndim != 3:
        raise ValueError("paths must have shape [samples, horizon, assets]")
    return paths.reshape(-1, paths.shape[-1])


def differentiable_correlation(paths: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    values = _flatten_paths(paths)
    centered = values - values.mean(dim=0, keepdim=True)
    denominator = max(values.shape[0] - 1, 1)
    covariance = centered.transpose(0, 1) @ centered / denominator
    scale = torch.sqrt(torch.diag(covariance).clamp_min(epsilon))
    correlation = covariance / (scale[:, None] * scale[None, :]).clamp_min(epsilon)
    return correlation.clamp(-1.0, 1.0)


def correlation_matching_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(differentiable_correlation(fake), differentiable_correlation(real))


def differentiable_autocorrelation(
    paths: torch.Tensor,
    max_lag: int = 5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if paths.ndim != 3:
        raise ValueError("paths must have shape [samples, horizon, assets]")
    maximum = min(int(max_lag), int(paths.shape[1]) - 1)
    if maximum < 1:
        return paths.new_zeros((paths.shape[-1], 0))
    correlations = []
    for lag in range(1, maximum + 1):
        left = paths[:, :-lag, :].reshape(-1, paths.shape[-1])
        right = paths[:, lag:, :].reshape(-1, paths.shape[-1])
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        numerator = (left * right).mean(dim=0)
        denominator = torch.sqrt(
            left.square().mean(dim=0).clamp_min(epsilon)
            * right.square().mean(dim=0).clamp_min(epsilon)
        )
        correlations.append((numerator / denominator.clamp_min(epsilon)).clamp(-1.0, 1.0))
    return torch.stack(correlations, dim=-1)


def autocorrelation_matching_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    max_lag: int = 5,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        differentiable_autocorrelation(fake, max_lag=max_lag),
        differentiable_autocorrelation(real, max_lag=max_lag),
    )


def marginal_quantile_matching_loss(
    real: torch.Tensor,
    fake: torch.Tensor,
    quantile_levels: Sequence[float] = (0.01, 0.05, 0.5, 0.95, 0.99),
    epsilon: float = 1e-5,
) -> torch.Tensor:
    levels = torch.as_tensor(
        list(quantile_levels),
        dtype=real.dtype,
        device=real.device,
    )
    real_quantiles = torch.quantile(_flatten_paths(real), levels, dim=0)
    fake_quantiles = torch.quantile(_flatten_paths(fake), levels, dim=0)
    real_scale = (
        real_quantiles[-1] - real_quantiles[0]
    ).abs().clamp_min(epsilon)
    return F.smooth_l1_loss(
        fake_quantiles / real_scale.unsqueeze(0),
        real_quantiles / real_scale.unsqueeze(0),
    )


def generated_conditional_quantiles(
    decoder,
    state_embedding: torch.Tensor,
    regime: torch.Tensor,
    scale: torch.Tensor,
    target_index: int,
    horizons: Sequence[int],
    quantile_levels: Sequence[float],
    scenarios: int,
    latent_dim: int,
    noise_sampler,
    noise_distribution: str,
    degrees_of_freedom: float,
) -> torch.Tensor:
    """Generate several paths per origin and estimate differentiable return quantiles."""

    batch = state_embedding.shape[0]
    scenarios = int(scenarios)
    if scenarios < 2:
        raise ValueError("At least two consistency scenarios are required")
    repeated_state = state_embedding.repeat_interleave(scenarios, dim=0)
    repeated_regime = regime.repeat_interleave(scenarios, dim=0)
    noise = noise_sampler(
        batch * scenarios,
        latent_dim,
        state_embedding.device,
        noise_distribution,
        degrees_of_freedom,
    )
    normalized = decoder(noise, repeated_regime, repeated_state)
    repeated_scale = scale.repeat_interleave(scenarios, dim=0)
    actual = normalized * repeated_scale[:, None, :]
    actual = actual.view(batch, scenarios, actual.shape[1], actual.shape[2])
    cumulative = torch.stack(
        [actual[:, :, : int(horizon), target_index].sum(dim=2) for horizon in horizons],
        dim=-1,
    )
    levels = torch.as_tensor(
        list(quantile_levels),
        dtype=cumulative.dtype,
        device=cumulative.device,
    )
    values = torch.quantile(cumulative, levels, dim=1)
    return values.permute(1, 2, 0).contiguous()


def quantile_consistency_loss(
    generated_quantiles: torch.Tensor,
    teacher_quantiles: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    if generated_quantiles.shape != teacher_quantiles.shape:
        raise ValueError(
            "Generated and teacher quantiles must share shape; "
            f"received {generated_quantiles.shape} and {teacher_quantiles.shape}"
        )
    spread = (teacher_quantiles[..., -1] - teacher_quantiles[..., 0]).abs().clamp_min(epsilon)
    normalized_generated = generated_quantiles / spread.unsqueeze(-1)
    normalized_teacher = teacher_quantiles / spread.unsqueeze(-1)
    return F.smooth_l1_loss(normalized_generated, normalized_teacher)
