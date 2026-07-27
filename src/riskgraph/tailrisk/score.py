from __future__ import annotations

import numpy as np
import torch


def joint_var_es_score(
    var: torch.Tensor,
    expected_shortfall: torch.Tensor,
    observations: torch.Tensor,
    alpha: float,
    weight: float = 10.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Acerbi-Szekely / Fissler-Ziegel joint VaR-ES score.

    The implementation follows the quadratic score used in Tail-GAN for the
    lower tail of a PnL distribution. `var` and `expected_shortfall` are
    broadcast against `observations`.
    """

    alpha = float(alpha)
    weight = float(weight)
    if not 0.0 < alpha < 0.5:
        raise ValueError("This implementation expects a lower-tail alpha in (0, 0.5)")
    indicator = (observations <= var).to(observations.dtype)
    score = (
        0.5 * weight * (indicator - alpha) * (observations.square() - var.square())
        + indicator * expected_shortfall * (var - observations)
        + alpha * expected_shortfall * (0.5 * expected_shortfall - var)
    )
    if reduction == "mean":
        return score.mean()
    if reduction == "none":
        return score
    raise ValueError(f"Unknown reduction: {reduction}")


def multi_alpha_score(
    var: torch.Tensor,
    expected_shortfall: torch.Tensor,
    observations: torch.Tensor,
    alphas: list[float] | tuple[float, ...],
    weight: float = 10.0,
) -> torch.Tensor:
    """Average the joint score across strategies, samples and risk levels.

    var/es: [strategies, alphas]
    observations: [samples, strategies]
    """

    if var.shape != expected_shortfall.shape:
        raise ValueError("var and expected_shortfall must have equal shapes")
    if observations.ndim != 2 or var.ndim != 2:
        raise ValueError("observations and risk estimates must be matrices")
    if observations.shape[1] != var.shape[0] or var.shape[1] != len(alphas):
        raise ValueError("Strategy or alpha dimensions do not match")
    losses = []
    obs = observations.transpose(0, 1)
    for index, alpha in enumerate(alphas):
        losses.append(
            joint_var_es_score(
                var[:, index, None],
                expected_shortfall[:, index, None],
                obs,
                alpha=float(alpha),
                weight=weight,
            )
        )
    return torch.stack(losses).mean()


def empirical_var_es(data: np.ndarray, alphas: list[float] | tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape [samples, strategies]")
    vars_: list[np.ndarray] = []
    es_: list[np.ndarray] = []
    for alpha in alphas:
        var = np.quantile(values, float(alpha), axis=0)
        tail = np.where(values <= var[None, :], values, np.nan)
        es = np.nanmean(tail, axis=0)
        vars_.append(var)
        es_.append(es)
    return np.stack(vars_, axis=-1), np.stack(es_, axis=-1)
