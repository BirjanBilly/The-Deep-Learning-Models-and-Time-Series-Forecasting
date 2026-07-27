from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from riskgraph.config import Fold
from riskgraph.data.dataset import MarketWindowDataset, Panel, fit_scalers, split_origins
from riskgraph.llmtime.serialization import (
    FinancialScaler,
    FinancialTokenizer,
    financial_side_labels,
)
from riskgraph.tailrisk.conditioning import RiskGraphConditioner


@dataclass(frozen=True)
class SideInfoThresholds:
    volatility: tuple[float, float]
    trend: float
    credit: tuple[float, float] | None

    def export(self) -> dict[str, Any]:
        return {
            "volatility": [float(value) for value in self.volatility],
            "trend": float(self.trend),
            "credit": None if self.credit is None else [float(value) for value in self.credit],
        }


@dataclass(frozen=True)
class TeacherOutputs:
    condition: np.ndarray
    state: np.ndarray
    quantiles: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LLMTimeExample:
    origin: int
    date: str
    prompt_ids: np.ndarray
    full_ids: np.ndarray
    loss_mask: np.ndarray
    history: np.ndarray
    future: np.ndarray
    scaler_offset: float
    scaler_scale: float
    prefix_labels: tuple[str, ...]
    clipped_history: int
    clipped_future: int
    condition: np.ndarray | None
    teacher_quantiles: np.ndarray | None


class LLMTimeDataset(Dataset):
    def __init__(self, examples: list[LLMTimeExample]) -> None:
        if not examples:
            raise ValueError("LLMTimeDataset requires at least one example")
        lengths = {len(example.full_ids) for example in examples}
        if len(lengths) != 1:
            raise ValueError(f"Examples must have a fixed token length, got {sorted(lengths)}")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        result: dict[str, torch.Tensor] = {
            "input_ids": torch.from_numpy(example.full_ids.astype(np.int64)),
            "loss_mask": torch.from_numpy(example.loss_mask.astype(bool)),
            "origin": torch.tensor(example.origin, dtype=torch.long),
        }
        if example.condition is not None:
            result["condition"] = torch.from_numpy(example.condition.astype(np.float32))
        return result


class LLMTimePromptDataset(Dataset):
    def __init__(self, examples: list[LLMTimeExample]) -> None:
        if not examples:
            raise ValueError("LLMTimePromptDataset requires at least one example")
        lengths = {len(example.prompt_ids) for example in examples}
        if len(lengths) != 1:
            raise ValueError(f"Prompts must have a fixed token length, got {sorted(lengths)}")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        result: dict[str, torch.Tensor] = {
            "prompt_ids": torch.from_numpy(example.prompt_ids.astype(np.int64)),
            "origin": torch.tensor(example.origin, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if example.condition is not None:
            result["condition"] = torch.from_numpy(example.condition.astype(np.float32))
        return result


def _feature_series(panel: Panel, feature: str, fallback: np.ndarray) -> np.ndarray:
    if feature in panel.macro_feature_names:
        return panel.macro_features[:, panel.macro_feature_names.index(feature)].astype(float)
    if feature in panel.asset_feature_names:
        return panel.asset_features[:, panel.target_index, panel.asset_feature_names.index(feature)].astype(float)
    return np.asarray(fallback, dtype=float)


def fit_side_info_thresholds(panel: Panel, train_origins: np.ndarray) -> SideInfoThresholds:
    origins = np.asarray(train_origins, dtype=np.int64)
    realized_vol = np.asarray(
        [np.std(panel.target_returns[max(0, origin - 20) : origin + 1]) for origin in origins],
        dtype=float,
    )
    volatility = _feature_series(panel, "vix", fallback=np.zeros(len(panel.dates)))[origins]
    if not np.isfinite(volatility).all() or np.nanstd(volatility) < 1e-8:
        volatility = realized_vol
    vol_thresholds = tuple(np.nanquantile(volatility, [0.33, 0.67]).astype(float).tolist())

    trend_series = _feature_series(panel, "ret_21d", fallback=np.zeros(len(panel.dates)))[origins]
    trend_threshold = float(np.nanquantile(np.abs(trend_series), 0.50))
    trend_threshold = max(trend_threshold, 1e-5)

    credit_series: np.ndarray | None = None
    if "baa_credit_spread" in panel.macro_feature_names:
        credit_series = panel.macro_features[
            origins, panel.macro_feature_names.index("baa_credit_spread")
        ].astype(float)
    credit_thresholds = None
    if credit_series is not None and np.isfinite(credit_series).any():
        credit_thresholds = tuple(
            np.nanquantile(credit_series, [0.33, 0.67]).astype(float).tolist()
        )
    return SideInfoThresholds(
        volatility=(float(vol_thresholds[0]), float(vol_thresholds[1])),
        trend=trend_threshold,
        credit=(float(credit_thresholds[0]), float(credit_thresholds[1]))
        if credit_thresholds is not None
        else None,
    )


def side_labels_for_origin(
    panel: Panel,
    origin: int,
    thresholds: SideInfoThresholds,
) -> list[str]:
    realized_vol = float(np.std(panel.target_returns[max(0, origin - 20) : origin + 1]))
    volatility = float(_feature_series(panel, "vix", np.zeros(len(panel.dates)))[origin])
    if not np.isfinite(volatility):
        volatility = realized_vol
    trend = float(_feature_series(panel, "ret_21d", np.zeros(len(panel.dates)))[origin])
    credit: float | None = None
    if "baa_credit_spread" in panel.macro_feature_names:
        value = float(panel.macro_features[origin, panel.macro_feature_names.index("baa_credit_spread")])
        credit = value if np.isfinite(value) else None
    return financial_side_labels(
        volatility_value=volatility,
        volatility_thresholds=thresholds.volatility,
        trend_value=trend,
        trend_threshold=thresholds.trend,
        credit_value=credit,
        credit_thresholds=thresholds.credit,
    )


def resolve_teacher_checkpoint(
    riskgraph_output_root: str | Path,
    fold: Fold,
    variant: str,
    seed: int,
) -> Path:
    run_directory = Path(riskgraph_output_root) / fold.name / variant / f"seed_{seed}"
    candidates = [
        run_directory / "best_checkpoint.pt",
        run_directory / "checkpoint.pt",
    ]
    for checkpoint in candidates:
        if checkpoint.is_file():
            return checkpoint
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "RiskGraph teacher checkpoint not found. Expected one of: "
        f"{expected}. Restore or train teachers first."
    )


def compute_teacher_outputs_from_conditioner(
    panel: Panel,
    origins: np.ndarray,
    fold: Fold,
    config: dict[str, Any],
    conditioner: RiskGraphConditioner,
    device: torch.device,
    batch_size: int = 256,
) -> TeacherOutputs:
    scalers = fit_scalers(panel, fold.train_end)
    dataset = MarketWindowDataset(
        panel=panel,
        origins=np.asarray(origins, dtype=np.int64),
        scalers=scalers,
        lookback=int(config["features"]["lookback"]),
        horizons=[int(value) for value in config["features"]["horizons"]],
        graph_mode="dynamic",
        macro_mode="enabled",
    )
    states: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    conditioner.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            items = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            asset = torch.stack([item["asset"] for item in items]).to(device)
            macro = torch.stack([item["macro"] for item in items]).to(device)
            adjacency = torch.stack([item["adjacency"] for item in items]).to(device)
            state, teacher = conditioner(asset, macro, adjacency)
            states.append(state.float().cpu().numpy())
            quantiles.append(teacher.float().cpu().numpy())
    state_array = np.concatenate(states, axis=0).astype(np.float32)
    quantile_array = np.concatenate(quantiles, axis=0).astype(np.float32)
    condition = np.concatenate(
        [state_array, quantile_array.reshape(len(quantile_array), -1)], axis=1
    ).astype(np.float32)
    return TeacherOutputs(
        condition=condition,
        state=state_array,
        quantiles=quantile_array,
        metadata={
            "teacher_spec": conditioner.export_spec(),
            "condition_dim": int(condition.shape[1]),
            "state_dim": int(state_array.shape[1]),
            "quantile_shape": list(quantile_array.shape[1:]),
        },
    )


def compute_teacher_outputs(
    panel: Panel,
    origins: np.ndarray,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int = 256,
) -> TeacherOutputs:
    conditioner = RiskGraphConditioner.from_checkpoint(checkpoint_path, panel, config, device)
    return compute_teacher_outputs_from_conditioner(
        panel=panel,
        origins=origins,
        fold=fold,
        config=config,
        conditioner=conditioner,
        device=device,
        batch_size=batch_size,
    )


def build_examples(
    panel: Panel,
    origins: np.ndarray,
    tokenizer: FinancialTokenizer,
    history_length: int,
    forecast_steps: int,
    alpha: float,
    beta: float,
    basic_scaler: bool,
    side_thresholds: SideInfoThresholds | None,
    use_side_info: bool,
    teacher_outputs: TeacherOutputs | None = None,
    missing_fraction: float = 0.0,
    seed: int = 42,
) -> list[LLMTimeExample]:
    origins = np.asarray(origins, dtype=np.int64)
    if teacher_outputs is not None and len(teacher_outputs.condition) != len(origins):
        raise ValueError("Teacher outputs and origins must have the same length")
    if not 0.0 <= missing_fraction < 1.0:
        raise ValueError("missing_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    examples: list[LLMTimeExample] = []
    for row, origin_value in enumerate(origins):
        origin = int(origin_value)
        start = origin - history_length + 1
        if start < 0 or origin + forecast_steps >= len(panel.target_returns):
            continue
        history = panel.target_returns[start : origin + 1].astype(float).copy()
        future = panel.target_returns[origin + 1 : origin + forecast_steps + 1].astype(float).copy()
        if missing_fraction > 0.0:
            mask = rng.random(history_length) < missing_fraction
            mask[-1] = False
            history[mask] = np.nan
        scaler = FinancialScaler.fit(history, alpha=alpha, beta=beta, basic=basic_scaler)
        scaled_history = scaler.transform(history)
        scaled_future = scaler.transform(future)
        labels: list[str] = []
        if use_side_info:
            if side_thresholds is None:
                raise ValueError("side_thresholds are required when use_side_info=True")
            labels = side_labels_for_origin(panel, origin, side_thresholds)
        prompt_ids, clipped_history = tokenizer.encode_series(
            scaled_history,
            prefix_labels=labels,
            add_bos=True,
            add_eos=False,
        )
        future_ids, clipped_future = tokenizer.encode_series(
            scaled_future,
            add_bos=False,
            add_eos=True,
        )
        full_ids = np.asarray([*prompt_ids, *future_ids], dtype=np.int64)
        loss_mask = np.zeros(len(full_ids), dtype=bool)
        loss_mask[len(prompt_ids):] = True
        condition = None if teacher_outputs is None else teacher_outputs.condition[row]
        teacher_quantiles = None if teacher_outputs is None else teacher_outputs.quantiles[row]
        examples.append(
            LLMTimeExample(
                origin=origin,
                date=str(panel.dates[origin].date()),
                prompt_ids=np.asarray(prompt_ids, dtype=np.int64),
                full_ids=full_ids,
                loss_mask=loss_mask,
                history=history.astype(np.float32),
                future=future.astype(np.float32),
                scaler_offset=float(scaler.offset),
                scaler_scale=float(scaler.scale),
                prefix_labels=tuple(labels),
                clipped_history=int(clipped_history),
                clipped_future=int(clipped_future),
                condition=None if condition is None else condition.astype(np.float32),
                teacher_quantiles=None
                if teacher_quantiles is None
                else teacher_quantiles.astype(np.float32),
            )
        )
    if not examples:
        raise ValueError("No valid LLMTIME examples were constructed")
    return examples


def fold_origins(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    forecast_steps: int,
) -> dict[str, np.ndarray]:
    horizons = sorted({int(value) for value in config["features"]["horizons"]} | {forecast_steps})
    return split_origins(
        panel=panel,
        fold=fold,
        lookback=max(int(config["features"]["lookback"]), int(config["llmtime"]["history_length"])),
        horizons=horizons,
        embargo_days=int(config["splits"]["embargo_days"]),
    )
