from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from riskgraph.io import write_json
from riskgraph.repro import seed_everything
from riskgraph.tailrisk.conditioning import RiskGraphConditioner
from riskgraph.tailrisk.data import RegimeBatchSampler, TailWindowDataset, TailWindowSet
from riskgraph.tailrisk.evaluation import evaluate_scenario_generator
from riskgraph.tailrisk.models import (
    ConditionalPathCritic,
    EWMABackboneFactorScaleGenerator,
    StableEWMABackboneFactorScaleGenerator,
    EWMABackboneTailGenerator,
    RegimeTailGenerator,
    RiskGraphConditionedScenarioGenerator,
    RiskGraphEWMABackboneScenarioGenerator,
    TailRiskDiscriminator,
    sample_noise,
)
from riskgraph.tailrisk.regularization import (
    autocorrelation_matching_loss,
    correlation_matching_loss,
    generated_conditional_quantiles,
    marginal_quantile_matching_loss,
    quantile_consistency_loss,
)
from riskgraph.tailrisk.score import multi_alpha_score
from riskgraph.tailrisk.sorting import soft_var_es
from riskgraph.tailrisk.strategies import StrategyBank


@dataclass(frozen=True)
class TailTrainingConfig:
    objective: str = "tailgan"  # tailgan, gom, wgan_gp
    epochs: int = 100
    batch_size: int = 128
    latent_dim: int = 128
    learning_rate_generator: float = 2e-4
    learning_rate_discriminator: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.999)
    discriminator_steps: int = 1
    generator_steps: int = 1
    temperature: float = 0.1
    dual_lambda: float = 1.0
    score_weight: float = 10.0
    gradient_clip: float = 5.0
    noise_distribution: str = "student_t"
    degrees_of_freedom: float = 5.0
    validation_scenarios: int = 1024
    patience: int = 15
    wgan_gradient_penalty: float = 10.0
    conditioning_mode: str = "none"  # none, riskgraph
    state_projection_dim: int = 64
    conditioned_hidden_sizes: tuple[int, ...] = (256, 512, 512)
    correlation_loss_weight: float = 0.0
    autocorrelation_loss_weight: float = 0.0
    marginal_loss_weight: float = 0.0
    quantile_consistency_weight: float = 0.0
    max_autocorrelation_lag: int = 5
    marginal_quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)
    consistency_scenarios: int = 16
    consistency_origins_per_batch: int = 16
    generator_backbone: str = "mlp"  # mlp, ewma_gru, ewma_factor_scale
    baseline_hidden_size: int = 192
    baseline_gru_layers: int = 2
    baseline_initial_gate: float = 0.08
    energy_loss_weight: float = 0.0
    validation_correlation_weight: float = 0.0
    validation_autocorrelation_weight: float = 0.0
    factor_rank: int = 4
    factor_scale_limit: float = 0.30
    idio_scale_limit: float = 0.20
    drift_limit: float = 0.08
    skew_limit: float = 0.15
    discriminator_spectral_normalization: bool = False
    discriminator_output_penalty: float = 0.0
    normalize_tail_score: bool = False
    tail_score_ema_decay: float = 0.98
    tail_score_clip: float = 20.0
    generator_ema_decay: float = 0.0
    minimum_epochs: int = 0
    validation_coverage_weight: float = 0.0
    validation_score_rejection_weight: float = 0.0
    validation_smoothing: float = 0.0


class _GeneratorEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        with torch.no_grad():
            for name, value in model.state_dict().items():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )

    @contextmanager
    def average_parameters(self, model: nn.Module):
        if self.decay <= 0.0:
            yield
            return
        state = model.state_dict()
        backup = {
            name: value.detach().clone()
            for name, value in state.items()
            if name in self.shadow
        }
        averaged = {
            name: self.shadow.get(name, value)
            for name, value in state.items()
        }
        model.load_state_dict(averaged, strict=True)
        try:
            yield
        finally:
            restored = {name: backup.get(name, value) for name, value in state.items()}
            model.load_state_dict(restored, strict=True)


def _device_from_string(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _is_conditioned(generator: nn.Module) -> bool:
    return isinstance(
        generator,
        (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
    )


def _require_conditioning_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    names = ("state_asset", "state_macro", "state_adjacency")
    missing = [name for name in names if name not in batch]
    if missing:
        raise ValueError(f"Conditioned training requires state tensors: {missing}")
    return tuple(batch[name].to(device) for name in names)  # type: ignore[return-value]


def _decode_batch(
    generator: nn.Module,
    noise: torch.Tensor,
    regime: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    baseline = None
    if isinstance(generator, (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator, RiskGraphEWMABackboneScenarioGenerator)):
        if "baseline_cholesky" not in batch:
            raise ValueError("EWMA backbone training requires baseline_cholesky")
        baseline = batch["baseline_cholesky"].to(device)
    if isinstance(generator, (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator)):
        if "state_embedding" in batch and "teacher_quantiles" in batch:
            state_embedding = batch["state_embedding"].to(device)
            teacher_quantiles = batch["teacher_quantiles"].to(device)
        else:
            state_asset, state_macro, state_adjacency = _require_conditioning_batch(
                batch, device
            )
            state_embedding, teacher_quantiles = generator.encode_state(
                state_asset, state_macro, state_adjacency
            )
        if isinstance(generator, RiskGraphEWMABackboneScenarioGenerator):
            assert baseline is not None
            decoded = generator.decode(noise, regime, baseline, state_embedding)
        else:
            decoded = generator.decode(noise, regime, state_embedding)
        return decoded, state_embedding, teacher_quantiles
    if isinstance(generator, (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator)):
        assert baseline is not None
        return generator(noise, regime, baseline), None, None
    assert isinstance(generator, RegimeTailGenerator)
    return generator(noise, regime), None, None


def _generate_validation_paths(
    generator: nn.Module,
    windows: TailWindowSet,
    scenarios_per_regime: int,
    latent_dim: int,
    device: torch.device,
    config: TailTrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    generated = []
    real = []
    generator.eval()
    rng = np.random.default_rng(12345)
    with torch.no_grad():
        for regime_value in np.unique(windows.regimes):
            indices = np.flatnonzero(windows.regimes == regime_value)
            if len(indices) == 0:
                continue
            selected = rng.choice(indices, size=scenarios_per_regime, replace=True)
            regime = torch.full(
                (scenarios_per_regime,),
                int(regime_value),
                dtype=torch.long,
                device=device,
            )
            noise = sample_noise(
                scenarios_per_regime,
                latent_dim,
                device,
                config.noise_distribution,
                config.degrees_of_freedom,
            )
            baseline = None
            if isinstance(
                generator,
                (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator, RiskGraphEWMABackboneScenarioGenerator),
            ):
                if windows.baseline_cholesky is None:
                    raise ValueError("Validation windows lack baseline Cholesky factors")
                baseline = torch.from_numpy(windows.baseline_cholesky[selected]).to(device)
            if isinstance(
                generator,
                (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
            ):
                if windows.state_embedding is not None:
                    state = torch.from_numpy(windows.state_embedding[selected]).to(device)
                else:
                    if windows.state_asset is None:
                        raise ValueError(
                            "Validation windows are missing RiskGraph conditioning state"
                        )
                    assert windows.state_macro is not None
                    assert windows.state_adjacency is not None
                    state, _ = generator.encode_state(
                        torch.from_numpy(windows.state_asset[selected]).to(device),
                        torch.from_numpy(windows.state_macro[selected]).to(device),
                        torch.from_numpy(windows.state_adjacency[selected]).to(device),
                    )
                if isinstance(generator, RiskGraphEWMABackboneScenarioGenerator):
                    assert baseline is not None
                    normalized = generator.decode(noise, regime, baseline, state)
                else:
                    normalized = generator.decode(noise, regime, state)
            elif isinstance(generator, (EWMABackboneTailGenerator, EWMABackboneFactorScaleGenerator, StableEWMABackboneFactorScaleGenerator)):
                assert baseline is not None
                normalized = generator(noise, regime, baseline)
            else:
                assert isinstance(generator, RegimeTailGenerator)
                normalized = generator(noise, regime)
            scales = torch.from_numpy(windows.scales[selected]).to(device)
            fake = normalized * scales[:, None, :]
            generated.append(fake.cpu().numpy())
            real.append(windows.actual_paths[selected])
    return np.concatenate(real, axis=0), np.concatenate(generated, axis=0)


def _gradient_penalty(
    critic: ConditionalPathCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    regime: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.rand(real.shape[0], 1, 1, device=real.device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)
    score = critic(interpolated, regime)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradient.flatten(start_dim=1).norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def _energy_distribution_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_flat = real.flatten(start_dim=1)
    fake_flat = fake.flatten(start_dim=1)
    cross = torch.cdist(fake_flat, real_flat).mean()
    fake_self = torch.cdist(fake_flat, fake_flat).mean()
    real_self = torch.cdist(real_flat, real_flat).mean()
    return 2.0 * cross - fake_self - real_self


def _auxiliary_generator_losses(
    generator: nn.Module,
    normalized_real: torch.Tensor,
    fake_normalized: torch.Tensor,
    scale: torch.Tensor,
    regime: torch.Tensor,
    state_embedding: torch.Tensor | None,
    teacher_quantiles: torch.Tensor | None,
    config: TailTrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = fake_normalized.new_zeros(())
    correlation = (
        correlation_matching_loss(normalized_real, fake_normalized)
        if config.correlation_loss_weight > 0.0
        else zero
    )
    autocorrelation = (
        autocorrelation_matching_loss(
            normalized_real,
            fake_normalized,
            max_lag=config.max_autocorrelation_lag,
        )
        if config.autocorrelation_loss_weight > 0.0
        else zero
    )
    marginal = (
        marginal_quantile_matching_loss(
            normalized_real,
            fake_normalized,
            quantile_levels=config.marginal_quantiles,
        )
        if config.marginal_loss_weight > 0.0
        else zero
    )
    energy = (
        _energy_distribution_loss(normalized_real, fake_normalized)
        if config.energy_loss_weight > 0.0
        else zero
    )
    quantile = zero
    if (
        config.quantile_consistency_weight > 0.0
        and isinstance(generator, RiskGraphConditionedScenarioGenerator)
        and state_embedding is not None
        and teacher_quantiles is not None
    ):
        origins = min(
            int(config.consistency_origins_per_batch),
            state_embedding.shape[0],
        )
        indices = torch.randperm(
            state_embedding.shape[0],
            device=state_embedding.device,
        )[:origins]
        spec = generator.conditioner.spec
        generated_quantiles = generated_conditional_quantiles(
            generator.decoder,
            state_embedding[indices],
            regime[indices],
            scale[indices],
            target_index=int(spec.constructor["target_index"]),
            horizons=spec.horizons,
            quantile_levels=spec.quantile_levels,
            scenarios=config.consistency_scenarios,
            latent_dim=config.latent_dim,
            noise_sampler=sample_noise,
            noise_distribution=config.noise_distribution,
            degrees_of_freedom=config.degrees_of_freedom,
        )
        quantile = quantile_consistency_loss(
            generated_quantiles,
            teacher_quantiles[indices],
        )
    total = (
        config.correlation_loss_weight * correlation
        + config.autocorrelation_loss_weight * autocorrelation
        + config.marginal_loss_weight * marginal
        + config.quantile_consistency_weight * quantile
        + config.energy_loss_weight * energy
    )
    return total, {
        "correlation": correlation,
        "autocorrelation": autocorrelation,
        "marginal": marginal,
        "quantile_consistency": quantile,
        "energy": energy,
    }


def _build_generator(
    train_windows: TailWindowSet,
    validation_windows: TailWindowSet,
    config: TailTrainingConfig,
    conditioner: RiskGraphConditioner | None,
    device: torch.device,
) -> nn.Module:
    regimes = int(max(train_windows.regimes.max(), validation_windows.regimes.max()) + 1)
    if config.generator_backbone not in {
        "mlp", "ewma_gru", "ewma_factor_scale", "ewma_factor_scale_stable"
    }:
        raise ValueError(
            "generator_backbone must be mlp, ewma_gru, ewma_factor_scale or "
            "ewma_factor_scale_stable"
        )
    if config.generator_backbone == "ewma_factor_scale_stable":
        if train_windows.baseline_cholesky is None or validation_windows.baseline_cholesky is None:
            raise ValueError("stable factor-scale backbone requires baseline Cholesky factors")
        if config.conditioning_mode != "none":
            raise ValueError("stable factor-scale backbone supports conditioning_mode='none'")
        return StableEWMABackboneFactorScaleGenerator(
            latent_dim=config.latent_dim,
            horizon=train_windows.horizon,
            n_assets=train_windows.n_assets,
            n_regimes=regimes,
            hidden_size=config.baseline_hidden_size,
            layers=config.baseline_gru_layers,
            factor_rank=config.factor_rank,
            initial_gate=config.baseline_initial_gate,
            factor_scale_limit=config.factor_scale_limit,
            idio_scale_limit=config.idio_scale_limit,
            drift_limit=config.drift_limit,
            skew_limit=config.skew_limit,
            degrees_of_freedom=config.degrees_of_freedom,
        ).to(device)
    if config.generator_backbone == "ewma_factor_scale":
        if train_windows.baseline_cholesky is None or validation_windows.baseline_cholesky is None:
            raise ValueError("ewma_factor_scale backbone requires baseline Cholesky factors")
        if config.conditioning_mode != "none":
            raise ValueError("ewma_factor_scale currently supports conditioning_mode='none'")
        return EWMABackboneFactorScaleGenerator(
            latent_dim=config.latent_dim,
            horizon=train_windows.horizon,
            n_assets=train_windows.n_assets,
            n_regimes=regimes,
            hidden_size=config.baseline_hidden_size,
            layers=config.baseline_gru_layers,
            factor_rank=config.factor_rank,
            initial_gate=config.baseline_initial_gate,
            factor_scale_limit=config.factor_scale_limit,
            idio_scale_limit=config.idio_scale_limit,
            drift_limit=config.drift_limit,
            skew_limit=config.skew_limit,
            degrees_of_freedom=config.degrees_of_freedom,
        ).to(device)
    if config.generator_backbone == "ewma_gru":
        if train_windows.baseline_cholesky is None or validation_windows.baseline_cholesky is None:
            raise ValueError("ewma_gru backbone requires baseline Cholesky factors")
        if config.conditioning_mode == "riskgraph":
            if conditioner is None:
                raise ValueError("conditioning_mode='riskgraph' requires a conditioner")
            conditioner = conditioner.to(device)
            conditioner.freeze()
            return RiskGraphEWMABackboneScenarioGenerator(
                conditioner=conditioner,
                latent_dim=config.latent_dim,
                horizon=train_windows.horizon,
                n_assets=train_windows.n_assets,
                n_regimes=regimes,
                state_projection_dim=config.state_projection_dim,
                hidden_size=config.baseline_hidden_size,
                layers=config.baseline_gru_layers,
                initial_gate=config.baseline_initial_gate,
                degrees_of_freedom=config.degrees_of_freedom,
            ).to(device)
        if config.conditioning_mode != "none":
            raise ValueError("conditioning_mode must be 'none' or 'riskgraph'")
        return EWMABackboneTailGenerator(
            latent_dim=config.latent_dim,
            horizon=train_windows.horizon,
            n_assets=train_windows.n_assets,
            n_regimes=regimes,
            hidden_size=config.baseline_hidden_size,
            layers=config.baseline_gru_layers,
            initial_gate=config.baseline_initial_gate,
            degrees_of_freedom=config.degrees_of_freedom,
        ).to(device)
    if config.conditioning_mode == "riskgraph":
        if conditioner is None:
            raise ValueError("conditioning_mode='riskgraph' requires a RiskGraph conditioner")
        conditioner = conditioner.to(device)
        conditioner.freeze()
        return RiskGraphConditionedScenarioGenerator(
            conditioner=conditioner,
            latent_dim=config.latent_dim,
            horizon=train_windows.horizon,
            n_assets=train_windows.n_assets,
            n_regimes=regimes,
            state_projection_dim=config.state_projection_dim,
            hidden_sizes=tuple(int(value) for value in config.conditioned_hidden_sizes),
        ).to(device)
    if config.conditioning_mode != "none":
        raise ValueError("conditioning_mode must be 'none' or 'riskgraph'")
    return RegimeTailGenerator(
        latent_dim=config.latent_dim,
        horizon=train_windows.horizon,
        n_assets=train_windows.n_assets,
        n_regimes=regimes,
    ).to(device)


def train_tail_model(
    train_windows: TailWindowSet,
    validation_windows: TailWindowSet,
    strategy_bank: StrategyBank,
    alphas: list[float],
    output_dir: str | Path,
    seed: int,
    config: TailTrainingConfig,
    device_name: str = "auto",
    conditioner: RiskGraphConditioner | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = _device_from_string(device_name)
    if config.objective not in {"tailgan", "gom", "wgan_gp"}:
        raise ValueError("objective must be tailgan, gom or wgan_gp")

    generator = _build_generator(
        train_windows,
        validation_windows,
        config,
        conditioner,
        device,
    )
    trainable_generator_parameters = [
        parameter for parameter in generator.parameters() if parameter.requires_grad
    ]
    optimizer_g = torch.optim.Adam(
        trainable_generator_parameters,
        lr=config.learning_rate_generator,
        betas=config.betas,
    )

    discriminator: nn.Module | None
    regimes = int(max(train_windows.regimes.max(), validation_windows.regimes.max()) + 1)
    if config.objective == "tailgan":
        discriminator = TailRiskDiscriminator(
            sample_size=config.batch_size,
            n_strategies=len(strategy_bank.names),
            alphas=alphas,
            n_regimes=regimes,
            temperature=config.temperature,
            weight=config.score_weight,
            spectral_normalization=config.discriminator_spectral_normalization,
        ).to(device)
    elif config.objective == "wgan_gp":
        discriminator = ConditionalPathCritic(
            horizon=train_windows.horizon,
            n_assets=train_windows.n_assets,
            n_regimes=regimes,
        ).to(device)
    else:
        discriminator = None
    optimizer_d = (
        torch.optim.Adam(
            discriminator.parameters(),
            lr=config.learning_rate_discriminator,
            betas=config.betas,
        )
        if discriminator is not None
        else None
    )

    generator_ema = _GeneratorEMA(generator, config.generator_ema_decay)
    tail_score_scale = torch.tensor(1.0, device=device)
    smoothed_validation: float | None = None

    dataset = TailWindowDataset(train_windows)
    sampler = RegimeBatchSampler(
        train_windows.regimes,
        config.batch_size,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    best_metric = float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        sampler.set_epoch(epoch)
        generator.train()
        if discriminator is not None:
            discriminator.train()
        d_losses: list[float] = []
        g_losses: list[float] = []
        tail_losses: list[float] = []
        correlation_losses: list[float] = []
        autocorrelation_losses: list[float] = []
        marginal_losses: list[float] = []
        quantile_losses: list[float] = []
        energy_losses: list[float] = []

        for batch in loader:
            normalized_real = batch["normalized_path"].to(device)
            actual_real = batch["actual_path"].to(device)
            scale = batch["scale"].to(device)
            regime = batch["regime"].to(device)

            if config.objective == "tailgan":
                assert isinstance(discriminator, TailRiskDiscriminator)
                assert optimizer_d is not None
                for _ in range(config.discriminator_steps):
                    optimizer_d.zero_grad(set_to_none=True)
                    noise = sample_noise(
                        config.batch_size,
                        config.latent_dim,
                        device,
                        config.noise_distribution,
                        config.degrees_of_freedom,
                    )
                    fake_normalized, _, _ = _decode_batch(
                        generator, noise, regime, batch, device
                    )
                    fake_actual = fake_normalized.detach() * scale[:, None, :]
                    real_pnl = strategy_bank.pnl(actual_real)
                    fake_pnl = strategy_bank.pnl(fake_actual)
                    real_var, real_es = discriminator(real_pnl, regime)
                    fake_var, fake_es = discriminator(fake_pnl, regime)
                    real_score = multi_alpha_score(
                        real_var,
                        real_es,
                        real_pnl,
                        alphas,
                        weight=config.score_weight,
                    )
                    fake_score = multi_alpha_score(
                        fake_var,
                        fake_es,
                        real_pnl,
                        alphas,
                        weight=config.score_weight,
                    )
                    observed_scale = 0.5 * (real_score.detach().abs() + fake_score.detach().abs())
                    if config.normalize_tail_score:
                        tail_score_scale = (
                            config.tail_score_ema_decay * tail_score_scale
                            + (1.0 - config.tail_score_ema_decay)
                            * observed_scale.clamp_min(1e-6)
                        )
                    scale_value = tail_score_scale.detach().clamp_min(1e-6)
                    loss_d = (config.dual_lambda * real_score - fake_score) / scale_value
                    if config.discriminator_output_penalty > 0.0:
                        loss_d = loss_d + config.discriminator_output_penalty * (
                            real_var.square().mean()
                            + real_es.square().mean()
                            + fake_var.square().mean()
                            + fake_es.square().mean()
                        )
                    loss_d = torch.clamp(
                        loss_d, -float(config.tail_score_clip), float(config.tail_score_clip)
                    )
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), config.gradient_clip
                    )
                    optimizer_d.step()
                    d_losses.append(float(loss_d.detach().cpu()))

                for _ in range(config.generator_steps):
                    optimizer_g.zero_grad(set_to_none=True)
                    noise = sample_noise(
                        config.batch_size,
                        config.latent_dim,
                        device,
                        config.noise_distribution,
                        config.degrees_of_freedom,
                    )
                    fake_normalized, state, teacher = _decode_batch(
                        generator, noise, regime, batch, device
                    )
                    fake_actual = fake_normalized * scale[:, None, :]
                    fake_pnl = strategy_bank.pnl(fake_actual)
                    fake_var, fake_es = discriminator(fake_pnl, regime)
                    tail_loss = multi_alpha_score(
                        fake_var,
                        fake_es,
                        strategy_bank.pnl(actual_real),
                        alphas,
                        weight=config.score_weight,
                    )
                    if config.normalize_tail_score:
                        tail_loss = tail_loss / tail_score_scale.detach().clamp_min(1e-6)
                    tail_loss = torch.clamp(
                        tail_loss, -float(config.tail_score_clip), float(config.tail_score_clip)
                    )
                    auxiliary, components = _auxiliary_generator_losses(
                        generator,
                        normalized_real,
                        fake_normalized,
                        scale,
                        regime,
                        state,
                        teacher,
                        config,
                    )
                    loss_g = tail_loss + auxiliary
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(
                        trainable_generator_parameters, config.gradient_clip
                    )
                    optimizer_g.step()
                    generator_ema.update(generator)
                    g_losses.append(float(loss_g.detach().cpu()))
                    tail_losses.append(float(tail_loss.detach().cpu()))
                    correlation_losses.append(float(components["correlation"].detach().cpu()))
                    autocorrelation_losses.append(
                        float(components["autocorrelation"].detach().cpu())
                    )
                    marginal_losses.append(float(components["marginal"].detach().cpu()))
                    quantile_losses.append(
                        float(components["quantile_consistency"].detach().cpu())
                    )
                    energy_losses.append(float(components["energy"].detach().cpu()))

            elif config.objective == "gom":
                optimizer_g.zero_grad(set_to_none=True)
                noise = sample_noise(
                    config.batch_size,
                    config.latent_dim,
                    device,
                    config.noise_distribution,
                    config.degrees_of_freedom,
                )
                fake_normalized, state, teacher = _decode_batch(
                    generator, noise, regime, batch, device
                )
                fake_actual = fake_normalized * scale[:, None, :]
                fake_pnl = strategy_bank.pnl(fake_actual)
                fake_var, fake_es = soft_var_es(
                    fake_pnl,
                    alphas,
                    temperature=config.temperature,
                )
                tail_loss = multi_alpha_score(
                    fake_var,
                    fake_es,
                    strategy_bank.pnl(actual_real),
                    alphas,
                    weight=config.score_weight,
                )
                if config.normalize_tail_score:
                    tail_loss = tail_loss / tail_score_scale.detach().clamp_min(1e-6)
                tail_loss = torch.clamp(
                    tail_loss, -float(config.tail_score_clip), float(config.tail_score_clip)
                )
                auxiliary, components = _auxiliary_generator_losses(
                    generator,
                    normalized_real,
                    fake_normalized,
                    scale,
                    regime,
                    state,
                    teacher,
                    config,
                )
                loss_g = tail_loss + auxiliary
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_generator_parameters, config.gradient_clip
                )
                optimizer_g.step()
                generator_ema.update(generator)
                g_losses.append(float(loss_g.detach().cpu()))
                tail_losses.append(float(tail_loss.detach().cpu()))
                correlation_losses.append(float(components["correlation"].detach().cpu()))
                autocorrelation_losses.append(
                    float(components["autocorrelation"].detach().cpu())
                )
                marginal_losses.append(float(components["marginal"].detach().cpu()))
                quantile_losses.append(
                    float(components["quantile_consistency"].detach().cpu())
                )
                energy_losses.append(float(components["energy"].detach().cpu()))

            else:
                assert isinstance(discriminator, ConditionalPathCritic)
                assert optimizer_d is not None
                for _ in range(config.discriminator_steps):
                    optimizer_d.zero_grad(set_to_none=True)
                    noise = sample_noise(
                        config.batch_size,
                        config.latent_dim,
                        device,
                        config.noise_distribution,
                        config.degrees_of_freedom,
                    )
                    fake_normalized, _, _ = _decode_batch(
                        generator, noise, regime, batch, device
                    )
                    fake_normalized = fake_normalized.detach()
                    real_score = discriminator(normalized_real, regime).mean()
                    fake_score = discriminator(fake_normalized, regime).mean()
                    penalty = _gradient_penalty(
                        discriminator,
                        normalized_real,
                        fake_normalized,
                        regime,
                    )
                    loss_d = (
                        fake_score
                        - real_score
                        + config.wgan_gradient_penalty * penalty
                    )
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), config.gradient_clip
                    )
                    optimizer_d.step()
                    d_losses.append(float(loss_d.detach().cpu()))
                optimizer_g.zero_grad(set_to_none=True)
                noise = sample_noise(
                    config.batch_size,
                    config.latent_dim,
                    device,
                    config.noise_distribution,
                    config.degrees_of_freedom,
                )
                fake_normalized, state, teacher = _decode_batch(
                    generator, noise, regime, batch, device
                )
                adversarial = -discriminator(fake_normalized, regime).mean()
                auxiliary, components = _auxiliary_generator_losses(
                    generator,
                    normalized_real,
                    fake_normalized,
                    scale,
                    regime,
                    state,
                    teacher,
                    config,
                )
                loss_g = adversarial + auxiliary
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_generator_parameters, config.gradient_clip
                )
                optimizer_g.step()
                generator_ema.update(generator)
                g_losses.append(float(loss_g.detach().cpu()))
                tail_losses.append(float(adversarial.detach().cpu()))
                correlation_losses.append(float(components["correlation"].detach().cpu()))
                autocorrelation_losses.append(
                    float(components["autocorrelation"].detach().cpu())
                )
                marginal_losses.append(float(components["marginal"].detach().cpu()))
                quantile_losses.append(
                    float(components["quantile_consistency"].detach().cpu())
                )
                energy_losses.append(float(components["energy"].detach().cpu()))

        with generator_ema.average_parameters(generator):
            real_validation, generated_validation = _generate_validation_paths(
                generator,
                validation_windows,
                scenarios_per_regime=config.validation_scenarios,
                latent_dim=config.latent_dim,
                device=device,
                config=config,
            )
        validation = evaluate_scenario_generator(
            real_validation,
            generated_validation,
            strategy_bank,
            alphas,
            device,
            weight=config.score_weight,
        )
        validation_metric_raw = float(
            validation.structural_metrics["mean_tail_relative_error"]
            + config.validation_correlation_weight
            * validation.structural_metrics["correlation_l1_error"]
            + config.validation_autocorrelation_weight
            * validation.structural_metrics["autocorrelation_l1_error"]
            + config.validation_coverage_weight
            * validation.structural_metrics["coverage_rejection_rate_5pct"]
            + config.validation_score_rejection_weight
            * validation.structural_metrics["score_rejection_rate_5pct"]
        )
        if smoothed_validation is None or config.validation_smoothing <= 0.0:
            smoothed_validation = validation_metric_raw
        else:
            smoothed_validation = (
                config.validation_smoothing * smoothed_validation
                + (1.0 - config.validation_smoothing) * validation_metric_raw
            )
        validation_metric = float(smoothed_validation)
        row = {
            "epoch": epoch,
            "discriminator_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "generator_loss": float(np.mean(g_losses)) if g_losses else float("nan"),
            "tail_or_adversarial_loss": float(np.mean(tail_losses))
            if tail_losses
            else float("nan"),
            "correlation_loss": float(np.mean(correlation_losses))
            if correlation_losses
            else 0.0,
            "autocorrelation_loss": float(np.mean(autocorrelation_losses))
            if autocorrelation_losses
            else 0.0,
            "marginal_loss": float(np.mean(marginal_losses))
            if marginal_losses
            else 0.0,
            "quantile_consistency_loss": float(np.mean(quantile_losses))
            if quantile_losses
            else 0.0,
            "energy_distribution_loss": float(np.mean(energy_losses))
            if energy_losses
            else 0.0,
            "validation_metric_raw": float(validation_metric_raw),
            "validation_tail_relative_error": float(
                validation.structural_metrics["mean_tail_relative_error"]
            ),
            "validation_correlation_l1_error": float(
                validation.structural_metrics["correlation_l1_error"]
            ),
            "validation_autocorrelation_l1_error": float(
                validation.structural_metrics["autocorrelation_l1_error"]
            ),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} objective={config.objective} "
            f"conditioner={config.conditioning_mode} "
            f"d_loss={row['discriminator_loss']:.6f} "
            f"g_loss={row['generator_loss']:.6f} "
            f"validation_tail_re={validation_metric:.6f}"
        )

        if validation_metric < best_metric:
            best_metric = validation_metric
            best_epoch = epoch
            wait = 0
            checkpoint: dict[str, Any] = {
                "generator": (
                    {**generator.state_dict(), **generator_ema.shadow}
                    if generator_ema.decay > 0.0
                    else generator.state_dict()
                ),
                "generator_type": (
                    "riskgraph_ewma_backbone"
                    if isinstance(generator, RiskGraphEWMABackboneScenarioGenerator)
                    else "ewma_factor_scale_stable"
                    if isinstance(generator, StableEWMABackboneFactorScaleGenerator)
                    else "ewma_factor_scale"
                    if isinstance(generator, EWMABackboneFactorScaleGenerator)
                    else "ewma_backbone"
                    if isinstance(generator, EWMABackboneTailGenerator)
                    else "riskgraph_conditioned"
                    if isinstance(generator, RiskGraphConditionedScenarioGenerator)
                    else "regime_only"
                ),
                "discriminator": (
                    discriminator.state_dict() if discriminator is not None else None
                ),
                "training_config": asdict(config),
                "alphas": alphas,
                "strategy_names": strategy_bank.names,
                "strategy_static_weights": strategy_bank.static_weights,
                "strategy_static_names": strategy_bank.static_names,
                "strategy_include_mean_reversion": strategy_bank.include_mean_reversion,
                "strategy_include_trend_following": strategy_bank.include_trend_following,
                "strategy_signal_window": strategy_bank.signal_window,
                "regime_edges": train_windows.regime_edges,
                "horizon": train_windows.horizon,
                "n_assets": train_windows.n_assets,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_tail_relative_error": best_metric,
            }
            if isinstance(
                generator,
                (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
            ):
                checkpoint["conditioner_spec"] = generator.conditioner.export_spec()
            torch.save(checkpoint, output / "best_checkpoint.pt")
        else:
            wait += 1
            if wait >= config.patience and epoch >= config.minimum_epochs:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    metadata: dict[str, Any] = {
        "objective": config.objective,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_tail_relative_error": best_metric,
        "training_config": asdict(config),
        "regime_counts": {
            str(value): int((train_windows.regimes == value).sum())
            for value in np.unique(train_windows.regimes)
        },
        "strategy_count": len(strategy_bank.names),
        "strategies": strategy_bank.names,
        "conditioned": _is_conditioned(generator),
    }
    if isinstance(
        generator,
        (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
    ):
        metadata["riskgraph_conditioner"] = generator.conditioner.export_spec()
    write_json(output / "run_metadata.json", metadata)
    return {
        "checkpoint": str(output / "best_checkpoint.pt"),
        "best_epoch": best_epoch,
        "best_validation_tail_relative_error": best_metric,
        "history": history,
    }


def load_generator_from_checkpoint(
    path: str | Path,
    device_name: str = "auto",
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    device = _device_from_string(device_name)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    training = checkpoint["training_config"]
    generator_type = str(checkpoint.get("generator_type", "regime_only"))
    regimes = len(np.asarray(checkpoint["regime_edges"])) + 1
    if generator_type == "riskgraph_conditioned":
        conditioner = RiskGraphConditioner.from_spec(checkpoint["conditioner_spec"])
        generator: nn.Module = RiskGraphConditionedScenarioGenerator(
            conditioner=conditioner,
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
            state_projection_dim=int(training.get("state_projection_dim", 64)),
            hidden_sizes=tuple(
                int(value)
                for value in training.get("conditioned_hidden_sizes", (256, 512, 512))
            ),
        ).to(device)
    elif generator_type == "riskgraph_ewma_backbone":
        conditioner = RiskGraphConditioner.from_spec(checkpoint["conditioner_spec"])
        generator = RiskGraphEWMABackboneScenarioGenerator(
            conditioner=conditioner,
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
            state_projection_dim=int(training.get("state_projection_dim", 64)),
            hidden_size=int(training.get("baseline_hidden_size", 192)),
            layers=int(training.get("baseline_gru_layers", 2)),
            initial_gate=float(training.get("baseline_initial_gate", 0.08)),
            degrees_of_freedom=float(training.get("degrees_of_freedom", 5.0)),
        ).to(device)
    elif generator_type == "ewma_factor_scale_stable":
        generator = StableEWMABackboneFactorScaleGenerator(
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
            hidden_size=int(training.get("baseline_hidden_size", 160)),
            layers=int(training.get("baseline_gru_layers", 2)),
            factor_rank=int(training.get("factor_rank", 4)),
            initial_gate=float(training.get("baseline_initial_gate", 0.03)),
            factor_scale_limit=float(training.get("factor_scale_limit", 0.25)),
            idio_scale_limit=float(training.get("idio_scale_limit", 0.15)),
            drift_limit=float(training.get("drift_limit", 0.04)),
            skew_limit=float(training.get("skew_limit", 0.12)),
            degrees_of_freedom=float(training.get("degrees_of_freedom", 5.0)),
        ).to(device)
    elif generator_type == "ewma_factor_scale":
        generator = EWMABackboneFactorScaleGenerator(
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
            hidden_size=int(training.get("baseline_hidden_size", 160)),
            layers=int(training.get("baseline_gru_layers", 2)),
            factor_rank=int(training.get("factor_rank", 4)),
            initial_gate=float(training.get("baseline_initial_gate", 0.04)),
            factor_scale_limit=float(training.get("factor_scale_limit", 0.30)),
            idio_scale_limit=float(training.get("idio_scale_limit", 0.20)),
            drift_limit=float(training.get("drift_limit", 0.08)),
            skew_limit=float(training.get("skew_limit", 0.15)),
            degrees_of_freedom=float(training.get("degrees_of_freedom", 5.0)),
        ).to(device)
    elif generator_type == "ewma_backbone":
        generator = EWMABackboneTailGenerator(
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
            hidden_size=int(training.get("baseline_hidden_size", 192)),
            layers=int(training.get("baseline_gru_layers", 2)),
            initial_gate=float(training.get("baseline_initial_gate", 0.08)),
            degrees_of_freedom=float(training.get("degrees_of_freedom", 5.0)),
        ).to(device)
    else:
        generator = RegimeTailGenerator(
            latent_dim=int(training["latent_dim"]),
            horizon=int(checkpoint["horizon"]),
            n_assets=int(checkpoint["n_assets"]),
            n_regimes=regimes,
        ).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()
    if isinstance(
        generator,
        (RiskGraphConditionedScenarioGenerator, RiskGraphEWMABackboneScenarioGenerator),
    ):
        generator.conditioner.freeze()
    return generator, checkpoint, device
