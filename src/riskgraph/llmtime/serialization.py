from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<SEP>",
    "<POS>",
    "<NEG>",
    "<NAN>",
]
SIDE_TOKENS = [
    "<VOL_LOW>",
    "<VOL_MID>",
    "<VOL_HIGH>",
    "<TREND_DOWN>",
    "<TREND_FLAT>",
    "<TREND_UP>",
    "<CREDIT_TIGHT>",
    "<CREDIT_NORMAL>",
    "<CREDIT_WIDE>",
]


@dataclass(frozen=True)
class FinancialScaler:
    """History-only affine scaler inspired by LLMTIME's percentile rescaling.

    ``transform(x) = (x - offset) / scale``.  Both parameters are fitted from
    the historical context only, so future observations never influence the
    representation used at a forecast origin.
    """

    offset: float
    scale: float
    alpha: float
    beta: float
    basic: bool

    @classmethod
    def fit(
        cls,
        history: np.ndarray | Sequence[float],
        alpha: float = 0.99,
        beta: float = 0.30,
        basic: bool = True,
        minimum_scale: float = 1e-4,
    ) -> FinancialScaler:
        values = np.asarray(history, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return cls(offset=0.0, scale=1.0, alpha=float(alpha), beta=float(beta), basic=bool(basic))
        if basic:
            offset = 0.0
            scale = float(np.quantile(np.abs(finite), alpha))
        else:
            span = float(np.max(finite) - np.min(finite))
            offset = float(np.min(finite) - beta * span)
            scale = float(np.quantile(finite - offset, alpha))
        scale = max(abs(scale), float(minimum_scale))
        return cls(offset=offset, scale=scale, alpha=float(alpha), beta=float(beta), basic=bool(basic))

    def transform(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.offset) / self.scale

    def inverse(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.offset

    def export(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerConfig:
    mode: str = "digit"
    base: int = 10
    precision: int = 3
    integer_digits: int = 2
    clip_value: float = 25.0
    flat_bins: int = 401
    half_bin_correction: bool = True


class FinancialTokenizer:
    """Numeric tokenizer with a digit hierarchy or a one-token flat-bin ablation."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        if config.base != 10:
            raise ValueError("The financial implementation currently supports decimal base 10 only")
        if config.mode not in {"digit", "flat_bin"}:
            raise ValueError("mode must be 'digit' or 'flat_bin'")
        if config.precision < 0 or config.integer_digits < 1:
            raise ValueError("precision must be non-negative and integer_digits must be positive")
        if config.flat_bins < 11:
            raise ValueError("flat_bins must be at least 11")

        tokens = [*SPECIAL_TOKENS, *SIDE_TOKENS]
        if config.mode == "digit":
            tokens.extend([str(value) for value in range(10)])
        else:
            tokens.extend([f"<BIN_{index}>" for index in range(config.flat_bins)])
        self.id_to_token = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.sep_id = self.token_to_id["<SEP>"]
        self.pos_id = self.token_to_id["<POS>"]
        self.neg_id = self.token_to_id["<NEG>"]
        self.nan_id = self.token_to_id["<NAN>"]
        self.side_token_ids = {token: self.token_to_id[token] for token in SIDE_TOKENS}
        self.digit_start = self.token_to_id.get("0", -1)
        self.bin_start = self.token_to_id.get("<BIN_0>", -1)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def tokens_per_value(self) -> int:
        if self.config.mode == "flat_bin":
            return 1
        return 1 + self.config.integer_digits + self.config.precision + 1

    @property
    def numeric_bin_width(self) -> float:
        if self.config.mode == "flat_bin":
            return 2.0 * self.config.clip_value / self.config.flat_bins
        return 10.0 ** (-self.config.precision)

    def export(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "vocabulary": list(self.id_to_token),
            "tokens_per_value": self.tokens_per_value,
            "numeric_bin_width": self.numeric_bin_width,
        }

    def prefix_ids(self, labels: Iterable[str] | None = None) -> list[int]:
        if labels is None:
            return []
        result: list[int] = []
        for label in labels:
            if label not in self.side_token_ids:
                raise ValueError(f"Unknown side-information token: {label}")
            result.append(self.side_token_ids[label])
        return result

    def _digit_ids(self, value: float) -> tuple[list[int], bool]:
        clipped = bool(abs(value) > self.config.clip_value)
        value = float(np.clip(value, -self.config.clip_value, self.config.clip_value))
        sign = self.neg_id if value < 0 else self.pos_id
        magnitude = abs(value)
        factor = 10**self.config.precision
        integer = int(np.floor(magnitude * factor + 1e-9))
        width = self.config.integer_digits + self.config.precision
        max_integer = 10**width - 1
        integer = min(integer, max_integer)
        digits = f"{integer:0{width}d}"
        return [sign, *[self.digit_start + int(char) for char in digits], self.sep_id], clipped

    def _flat_bin_id(self, value: float) -> tuple[int, bool]:
        clipped = bool(abs(value) > self.config.clip_value)
        value = float(np.clip(value, -self.config.clip_value, self.config.clip_value))
        width = self.numeric_bin_width
        index = int(np.floor((value + self.config.clip_value) / width))
        index = min(max(index, 0), self.config.flat_bins - 1)
        return self.bin_start + index, clipped

    def encode_value(self, value: float) -> tuple[list[int], bool]:
        if not np.isfinite(value):
            if self.config.mode == "digit":
                width = self.config.integer_digits + self.config.precision
                return [self.nan_id, *([self.digit_start] * width), self.sep_id], False
            return [self.nan_id], False
        if self.config.mode == "digit":
            return self._digit_ids(float(value))
        token, clipped = self._flat_bin_id(float(value))
        return [token], clipped

    def encode_series(
        self,
        values: np.ndarray | Sequence[float],
        prefix_labels: Iterable[str] | None = None,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> tuple[list[int], int]:
        ids = [self.bos_id] if add_bos else []
        ids.extend(self.prefix_ids(prefix_labels))
        clipped = 0
        for value in np.asarray(values, dtype=float).reshape(-1):
            encoded, did_clip = self.encode_value(float(value))
            ids.extend(encoded)
            clipped += int(did_clip)
        if add_eos:
            ids.append(self.eos_id)
        return ids, clipped

    def decode_value_tokens(self, tokens: Sequence[int]) -> float:
        values = list(tokens)
        if self.config.mode == "flat_bin":
            if not values or values[0] == self.nan_id:
                return float("nan")
            index = int(values[0] - self.bin_start)
            if not 0 <= index < self.config.flat_bins:
                raise ValueError(f"Invalid flat-bin token {values[0]}")
            width = self.numeric_bin_width
            return -self.config.clip_value + (index + 0.5) * width

        if not values:
            raise ValueError("No tokens supplied")
        if values[0] == self.nan_id:
            return float("nan")
        expected = 1 + self.config.integer_digits + self.config.precision + 1
        if len(values) < expected:
            raise ValueError(f"Digit value requires {expected} tokens, got {len(values)}")
        sign = -1.0 if values[0] == self.neg_id else 1.0
        if values[0] not in {self.pos_id, self.neg_id}:
            raise ValueError("Digit value must start with a sign token")
        digit_ids = values[1 : 1 + self.config.integer_digits + self.config.precision]
        digits: list[int] = []
        for token in digit_ids:
            digit = int(token - self.digit_start)
            if not 0 <= digit <= 9:
                raise ValueError(f"Invalid digit token {token}")
            digits.append(digit)
        integer = 0
        for digit in digits:
            integer = integer * 10 + digit
        value = integer / (10**self.config.precision)
        if self.config.half_bin_correction:
            value += 0.5 * self.numeric_bin_width
        return sign * value

    def split_value_tokens(self, tokens: Sequence[int], steps: int | None = None) -> list[list[int]]:
        sequence = list(tokens)
        values: list[list[int]] = []
        if self.config.mode == "flat_bin":
            for token in sequence:
                if token in {self.eos_id, self.pad_id}:
                    break
                if token >= self.bin_start or token == self.nan_id:
                    values.append([token])
                    if steps is not None and len(values) >= steps:
                        break
            return values

        current: list[int] = []
        for token in sequence:
            if token in {self.eos_id, self.pad_id}:
                break
            current.append(token)
            if token == self.sep_id:
                values.append(current)
                current = []
                if steps is not None and len(values) >= steps:
                    break
        return values

    def decode_series(self, tokens: Sequence[int], steps: int | None = None) -> np.ndarray:
        groups = self.split_value_tokens(tokens, steps=steps)
        decoded = [self.decode_value_tokens(group) for group in groups]
        return np.asarray(decoded, dtype=float)

    def allowed_ids_for_generation_position(self, generated_token_count: int) -> list[int]:
        """Return grammar-constrained token IDs for one autoregressive position."""
        if self.config.mode == "flat_bin":
            return list(range(self.bin_start, self.bin_start + self.config.flat_bins))
        position = generated_token_count % self.tokens_per_value
        if position == 0:
            return [self.pos_id, self.neg_id]
        if position == self.tokens_per_value - 1:
            return [self.sep_id]
        return list(range(self.digit_start, self.digit_start + 10))


def financial_side_labels(
    volatility_value: float,
    volatility_thresholds: tuple[float, float],
    trend_value: float,
    trend_threshold: float,
    credit_value: float | None = None,
    credit_thresholds: tuple[float, float] | None = None,
) -> list[str]:
    low_vol, high_vol = volatility_thresholds
    if volatility_value <= low_vol:
        volatility = "<VOL_LOW>"
    elif volatility_value >= high_vol:
        volatility = "<VOL_HIGH>"
    else:
        volatility = "<VOL_MID>"

    if trend_value <= -abs(trend_threshold):
        trend = "<TREND_DOWN>"
    elif trend_value >= abs(trend_threshold):
        trend = "<TREND_UP>"
    else:
        trend = "<TREND_FLAT>"

    labels = [volatility, trend]
    if credit_value is not None and credit_thresholds is not None:
        tight, wide = credit_thresholds
        if credit_value <= tight:
            labels.append("<CREDIT_TIGHT>")
        elif credit_value >= wide:
            labels.append("<CREDIT_WIDE>")
        else:
            labels.append("<CREDIT_NORMAL>")
    return labels
