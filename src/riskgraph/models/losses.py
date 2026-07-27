from __future__ import annotations

import torch
from torch.nn import functional as F


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    q = quantiles.view(1, 1, -1).to(prediction)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def combined_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    direction_logit: torch.Tensor | None,
    direction_target: torch.Tensor,
    direction_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pinball = pinball_loss(prediction, target, quantiles)
    if direction_logit is None or direction_weight <= 0:
        return pinball, {"pinball": float(pinball.detach()), "direction": 0.0}
    direction = F.binary_cross_entropy_with_logits(direction_logit, direction_target.float())
    total = pinball + direction_weight * direction
    return total, {"pinball": float(pinball.detach()), "direction": float(direction.detach())}
