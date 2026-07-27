from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from riskgraph.llmtime.serialization import FinancialTokenizer


@dataclass(frozen=True)
class DecimalTransformerConfig:
    vocab_size: int
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    dropout: float = 0.10
    max_tokens: int = 1024
    condition_dim: int = 0


class DecimalCausalTransformer(nn.Module):
    """Compact GPT-style decoder for digit-tokenized financial returns.

    This is an efficient, locally trainable complement to the paper's zero-shot
    LLM pathway.  The same serializer and sampling semantics can also be used by
    the optional Hugging Face adapter.
    """

    def __init__(self, config: DecimalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_tokens, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight
        self.condition_projection = (
            nn.Sequential(
                nn.LayerNorm(config.condition_dim),
                nn.Linear(config.condition_dim, config.d_model),
                nn.Tanh(),
            )
            if config.condition_dim > 0
            else None
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def export_config(self) -> dict[str, int | float]:
        return asdict(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        condition: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        batch, length = input_ids.shape
        if length > self.config.max_tokens:
            raise ValueError(f"Sequence length {length} exceeds max_tokens={self.config.max_tokens}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        if self.condition_projection is not None:
            if condition is None:
                raise ValueError("condition tensor is required for a conditioned model")
            if condition.shape != (batch, self.config.condition_dim):
                raise ValueError(
                    f"condition shape {tuple(condition.shape)} does not match "
                    f"({batch}, {self.config.condition_dim})"
                )
            hidden = hidden + self.condition_projection(condition).unsqueeze(1)
        elif condition is not None:
            raise ValueError("condition supplied to an unconditioned model")

        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output_head(self.final_norm(hidden))

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        condition: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(input_ids[:, :-1], condition=condition, padding_mask=None if padding_mask is None else padding_mask[:, :-1])
        targets = input_ids[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid = torch.ones_like(losses, dtype=torch.bool)
        if padding_mask is not None:
            valid &= ~padding_mask[:, 1:]
        if loss_mask is not None:
            if loss_mask.shape != input_ids.shape:
                raise ValueError("loss_mask must have the same shape as input_ids")
            valid &= loss_mask[:, 1:]
        denominator = valid.sum().clamp_min(1)
        return (losses * valid).sum() / denominator

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        return filtered.scatter(-1, sorted_indices, sorted_logits)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        tokenizer: FinancialTokenizer,
        steps: int,
        num_samples: int,
        condition: torch.Tensor | None = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> torch.Tensor:
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, tokens]")
        if steps < 1 or num_samples < 1:
            raise ValueError("steps and num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = prompt_ids.device
        batch = prompt_ids.shape[0]
        sequence = prompt_ids.repeat_interleave(num_samples, dim=0)
        repeated_condition = (
            condition.repeat_interleave(num_samples, dim=0) if condition is not None else None
        )
        total_new = steps * tokenizer.tokens_per_value
        if sequence.shape[1] + total_new > self.config.max_tokens:
            raise ValueError("Prompt plus generated horizon exceeds model context")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        self.eval()
        for generated_count in range(total_new):
            logits = self(sequence, condition=repeated_condition)[:, -1, :]
            allowed = tokenizer.allowed_ids_for_generation_position(generated_count)
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed] = logits[:, allowed]
            mask = mask / temperature
            mask = self._top_p_filter(mask, top_p=top_p)
            probabilities = torch.softmax(mask, dim=-1)
            token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            sequence = torch.cat([sequence, token], dim=1)
        generated = sequence[:, -total_new:]
        return generated.reshape(batch, num_samples, total_new)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def sinusoidal_temperature_schedule(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return float(end)
    fraction = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    weight = 0.5 - 0.5 * math.cos(math.pi * fraction)
    return float(start + weight * (end - start))
