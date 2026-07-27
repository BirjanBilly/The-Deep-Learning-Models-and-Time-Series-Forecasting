from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from riskgraph.llmtime.serialization import FinancialScaler


@dataclass(frozen=True)
class TextSerializerConfig:
    precision: int = 3
    spaced_digits: bool = False
    separator: str = " , "
    missing_text: str = "NaN"


class LLMTimeTextSerializer:
    """Text serializer for zero-shot use with a modern Hugging Face causal LM."""

    def __init__(self, config: TextSerializerConfig) -> None:
        self.config = config

    def _number(self, value: float) -> str:
        if not np.isfinite(value):
            return self.config.missing_text
        sign = "-" if value < 0 else ""
        integer = int(round(abs(value) * (10**self.config.precision)))
        digits = str(integer)
        if self.config.spaced_digits:
            digits = " ".join(digits)
            if sign:
                return f"- {digits}"
        return f"{sign}{digits}"

    def serialize(self, values: Sequence[float]) -> str:
        return self.config.separator.join(self._number(float(value)) for value in values) + self.config.separator

    def parse(self, text: str, steps: int) -> np.ndarray | None:
        pieces = text.split(self.config.separator)
        values: list[float] = []
        for piece in pieces:
            cleaned = piece.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith("nan"):
                values.append(float("nan"))
            else:
                cleaned = cleaned.replace(" ", "")
                sign = -1.0 if cleaned.startswith("-") else 1.0
                cleaned = cleaned.lstrip("+-")
                digits = "".join(character for character in cleaned if character.isdigit())
                if not digits:
                    continue
                values.append(sign * int(digits) / (10**self.config.precision))
            if len(values) >= steps:
                return np.asarray(values[:steps], dtype=float)
        return None


class HuggingFaceLLMTimeAdapter:
    """Optional zero-shot LLMTIME adapter using current ``transformers`` APIs.

    The model is never downloaded implicitly by the core package.  The caller
    supplies a local model directory or a Hugging Face identifier and accepts
    the corresponding license and compute requirements.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the optional LLM dependencies with `pip install -e '.[llm]'`"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def forecast(
        self,
        history: Sequence[float],
        steps: int,
        samples: int,
        scaler: FinancialScaler,
        serializer: LLMTimeTextSerializer,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        import torch

        scaled = scaler.transform(history)
        prompt = serializer.serialize(scaled)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        token_budget = max_new_tokens or max(16, steps * (serializer.config.precision + 4))
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                num_return_sequences=int(samples),
                max_new_tokens=int(token_budget),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        texts = self.tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
        forecasts: list[np.ndarray] = []
        accepted_text: list[str] = []
        for text in texts:
            parsed = serializer.parse(text, steps=steps)
            if parsed is None or len(parsed) != steps or not np.isfinite(parsed).all():
                continue
            forecasts.append(scaler.inverse(parsed))
            accepted_text.append(text)
        if not forecasts:
            raise ValueError("The Hugging Face model did not produce any parseable numeric completions")
        return np.stack(forecasts), accepted_text
