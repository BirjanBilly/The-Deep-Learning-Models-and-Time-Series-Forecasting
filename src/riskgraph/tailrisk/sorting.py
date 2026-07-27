from __future__ import annotations

import torch


def neural_sort(values: torch.Tensor, temperature: float = 0.1, descending: bool = True) -> torch.Tensor:
    """Differentiable relaxation of sorting from Grover et al. (2019).

    Parameters
    ----------
    values:
        Tensor with shape [..., n].
    temperature:
        Lower values approach a hard permutation but can make optimization
        unstable. Values in [0.05, 0.5] are practical starting points.
    descending:
        Sort largest-to-smallest when true.
    """

    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    n = values.shape[-1]
    if n < 2:
        return values
    tau = max(float(temperature), 1e-5)
    s = values.unsqueeze(-1)  # [..., n, 1]
    pairwise = torch.abs(s - s.transpose(-2, -1))
    row_sum = pairwise.sum(dim=-1, keepdim=True)
    ranks = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    scaling = n + 1 - 2 * ranks
    if not descending:
        scaling = -scaling
    score = s * scaling.view(*([1] * (s.ndim - 2)), 1, n) - row_sum
    permutation = torch.softmax(score.transpose(-2, -1) / tau, dim=-1)
    return torch.matmul(permutation, s).squeeze(-1)


def soft_var_es(
    pnl: torch.Tensor,
    alphas: list[float] | tuple[float, ...],
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate lower-tail VaR and ES from a differentiably sorted sample.

    pnl has shape [samples, strategies]. Outputs have shape
    [strategies, n_alphas].
    """

    if pnl.ndim != 2:
        raise ValueError("pnl must have shape [samples, strategies]")
    sorted_values = neural_sort(pnl.transpose(0, 1), temperature=temperature, descending=False)
    n = pnl.shape[0]
    var_columns = []
    es_columns = []
    for alpha in alphas:
        count = max(1, min(n, int(round(float(alpha) * n))))
        var_columns.append(sorted_values[:, count - 1])
        es_columns.append(sorted_values[:, :count].mean(dim=-1))
    return torch.stack(var_columns, dim=-1), torch.stack(es_columns, dim=-1)
