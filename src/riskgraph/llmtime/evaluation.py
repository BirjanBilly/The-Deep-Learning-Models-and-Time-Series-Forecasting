from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel, stress_mask_for_origins, targets_for_origins
from riskgraph.evaluation.metrics import evaluate_forecasts
from riskgraph.io import file_sha256, predictions_frame, write_json
from riskgraph.llmtime.data import (
    SideInfoThresholds,
    TeacherOutputs,
    build_examples,
    compute_teacher_outputs_from_conditioner,
    fold_origins,
)
from riskgraph.llmtime.training import load_llmtime_checkpoint, resolve_device
from riskgraph.tailrisk.conditioning import RiskGraphConditioner


@dataclass(frozen=True)
class SamplingChoice:
    temperature: float
    top_p: float
    validation_crps: float


def empirical_crps(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    draws = np.asarray(samples, dtype=float)
    truth = np.asarray(target, dtype=float)
    if draws.ndim != 2 or truth.shape != (draws.shape[0],):
        raise ValueError("samples must be [origins, scenarios] and target must be [origins]")
    first = np.mean(np.abs(draws - truth[:, None]), axis=1)
    ordered = np.sort(draws, axis=1)
    n = ordered.shape[1]
    weights = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n * n)
    pair_term = 2.0 * np.sum(ordered * weights[None, :], axis=1)
    return first - 0.5 * pair_term


def energy_score(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    draws = np.asarray(samples, dtype=float)
    truth = np.asarray(target, dtype=float)
    if draws.ndim != 3 or truth.shape != (draws.shape[0], draws.shape[2]):
        raise ValueError("samples must be [origins, scenarios, dimensions]")
    first = np.mean(np.linalg.norm(draws - truth[:, None, :], axis=-1), axis=1)
    # Pair adjacent independent permutations to avoid O(S^2) memory.
    rolled = np.roll(draws, shift=1, axis=1)
    second = np.mean(np.linalg.norm(draws - rolled, axis=-1), axis=1)
    return first - 0.5 * second


def _conditioner_from_llmtime_checkpoint(
    checkpoint: dict[str, Any],
    panel: Panel,
    config: dict[str, Any],
    device: torch.device,
) -> RiskGraphConditioner:
    metadata = checkpoint.get("teacher_metadata") or {}
    spec = metadata.get("teacher_spec")
    state = checkpoint.get("teacher_model_state_dict")
    if spec is not None and state is not None:
        conditioner = RiskGraphConditioner.from_spec(spec)
        conditioner.model.load_state_dict(state, strict=True)
        conditioner.to(device)
        conditioner.freeze()
        return conditioner

    teacher_path = Path(str(checkpoint.get("teacher_checkpoint", "")))
    if not teacher_path.is_file():
        raise FileNotFoundError(
            "The LLMTIME checkpoint has no embedded teacher and its external RiskGraph "
            f"checkpoint is unavailable: {teacher_path}"
        )
    expected_hash = checkpoint.get("teacher_checkpoint_sha256")
    if expected_hash is not None and file_sha256(teacher_path) != expected_hash:
        raise ValueError(f"RiskGraph teacher hash mismatch: {teacher_path}")
    return RiskGraphConditioner.from_checkpoint(teacher_path, panel, config, device)


def _normalise_teacher_from_checkpoint(
    value: TeacherOutputs,
    checkpoint: dict[str, Any],
) -> TeacherOutputs:
    metadata = checkpoint.get("teacher_metadata") or {}
    normalization = metadata.get("condition_normalization")
    if normalization is None:
        return value
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, -1)
    return TeacherOutputs(
        condition=((value.condition - mean) / std).astype(np.float32),
        state=value.state,
        quantiles=value.quantiles,
        metadata=value.metadata,
    )


def _thresholds(checkpoint: dict[str, Any]) -> SideInfoThresholds:
    record = checkpoint["side_thresholds"]
    credit = record.get("credit")
    return SideInfoThresholds(
        volatility=(float(record["volatility"][0]), float(record["volatility"][1])),
        trend=float(record["trend"]),
        credit=None if credit is None else (float(credit[0]), float(credit[1])),
    )


def _sample_examples(
    model,
    tokenizer,
    examples,
    device: torch.device,
    num_samples: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
    scenario_batch_size: int | None = None,
) -> np.ndarray:
    """Generate forecast paths without expanding every scenario in one GPU batch.

    ``model.generate`` repeats each origin ``num_samples`` times.  A large origin
    batch multiplied by a large scenario count can therefore exhaust GPU memory
    even though the model itself is small.  Scenario chunking preserves the total
    Monte Carlo sample count while bounding the expanded Transformer batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    scenario_chunk = int(scenario_batch_size or num_samples)
    if scenario_chunk < 1:
        raise ValueError("scenario_batch_size must be positive")

    outputs: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompt = torch.from_numpy(np.stack([example.prompt_ids for example in chunk])).to(device)
        condition = None
        if chunk[0].condition is not None:
            condition = torch.from_numpy(np.stack([example.condition for example in chunk])).to(device)
        steps = len(chunk[0].future)
        decoded_chunks: list[np.ndarray] = []

        for scenario_start in range(0, num_samples, scenario_chunk):
            current_samples = min(scenario_chunk, num_samples - scenario_start)
            generated = model.generate(
                prompt,
                tokenizer,
                steps=steps,
                num_samples=current_samples,
                condition=condition,
                temperature=temperature,
                top_p=top_p,
                seed=seed + 1_000_003 * start + scenario_start,
            ).cpu().numpy()
            decoded = np.empty((len(chunk), current_samples, steps), dtype=np.float32)
            for row, example in enumerate(chunk):
                for scenario in range(current_samples):
                    scaled = tokenizer.decode_series(generated[row, scenario], steps=steps)
                    if len(scaled) != steps or not np.isfinite(scaled).all():
                        raise ValueError("Grammar-constrained generation produced an invalid sample")
                    decoded[row, scenario] = (
                        scaled * example.scaler_scale + example.scaler_offset
                    ).astype(np.float32)
            decoded_chunks.append(decoded)

        outputs.append(np.concatenate(decoded_chunks, axis=1))
    return np.concatenate(outputs, axis=0)


def _cumulative_samples(paths: np.ndarray, horizons: list[int]) -> np.ndarray:
    cumulative = np.cumsum(paths, axis=-1)
    return np.stack([cumulative[:, :, horizon - 1] for horizon in horizons], axis=-1)


def _validation_crps(
    model,
    tokenizer,
    examples,
    horizons: list[int],
    temperature: float,
    top_p: float,
    samples: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    scenario_batch_size: int | None = None,
) -> float:
    paths = _sample_examples(
        model,
        tokenizer,
        examples,
        device=device,
        num_samples=samples,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        seed=seed,
        scenario_batch_size=scenario_batch_size,
    )
    draws = _cumulative_samples(paths, horizons)
    truth = np.stack(
        [np.asarray([example.future[:horizon].sum() for example in examples]) for horizon in horizons],
        axis=1,
    )
    values = [empirical_crps(draws[:, :, index], truth[:, index]) for index in range(len(horizons))]
    return float(np.mean(np.stack(values, axis=1)))


def choose_sampling_parameters(
    model,
    tokenizer,
    validation_examples,
    horizons: list[int],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[SamplingChoice, pd.DataFrame]:
    llm_config = config["llmtime"]
    limit = min(len(validation_examples), int(llm_config.get("tuning_origins", 64)))
    examples = validation_examples[-limit:]
    records: list[dict[str, float]] = []
    for temperature in llm_config["sampling_temperatures"]:
        for top_p in llm_config["sampling_top_p"]:
            score = _validation_crps(
                model,
                tokenizer,
                examples,
                horizons=horizons,
                temperature=float(temperature),
                top_p=float(top_p),
                samples=int(llm_config["tuning_samples"]),
                device=device,
                batch_size=int(llm_config["evaluation_batch_size"]),
                seed=seed,
                scenario_batch_size=int(
                    llm_config.get(
                        "tuning_scenario_batch_size", llm_config["tuning_samples"]
                    )
                ),
            )
            records.append(
                {
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "validation_crps": score,
                    "origins": len(examples),
                }
            )
    frame = pd.DataFrame(records).sort_values("validation_crps").reset_index(drop=True)
    best = frame.iloc[0]
    return (
        SamplingChoice(
            temperature=float(best["temperature"]),
            top_p=float(best["top_p"]),
            validation_crps=float(best["validation_crps"]),
        ),
        frame,
    )


def _continuous_nll(
    model,
    examples,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            ids = torch.from_numpy(np.stack([example.full_ids for example in chunk])).to(device)
            masks = torch.from_numpy(np.stack([example.loss_mask for example in chunk])).to(device)
            condition = None
            if chunk[0].condition is not None:
                condition = torch.from_numpy(np.stack([example.condition for example in chunk])).to(device)
            logits = model(ids[:, :-1], condition=condition)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(ids.shape[0], -1)
            valid = masks[:, 1:].clone()
            # The EOS token is a sequence terminator, not part of the continuous value density.
            valid &= ids[:, 1:] != tokenizer.eos_id
            token_nll = (losses * valid).sum(axis=1)
            for row, example in enumerate(chunk):
                token_value = float(token_nll[row].cpu()) / len(example.future)
                bin_width_original = tokenizer.numeric_bin_width * example.scaler_scale
                continuous = token_value + math.log(max(bin_width_original, 1e-12))
                rows.append(
                    {
                        "date": example.date,
                        "origin": example.origin,
                        "token_nll_per_value": token_value,
                        "continuous_nll_per_value": continuous,
                        "scaler_scale": example.scaler_scale,
                    }
                )
    frame = pd.DataFrame(rows)
    return float(frame["continuous_nll_per_value"].mean()), frame


def _plot_forecast_fan(predictions: pd.DataFrame, output: Path) -> None:
    frame = predictions[predictions["horizon"] == 1].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if {"q_0.05", "q_0.95"}.issubset(frame.columns):
        ax.fill_between(frame["date"], frame["q_0.05"], frame["q_0.95"], alpha=0.25, label="90% interval")
    ax.plot(frame["date"], frame["target"], linewidth=1.0, label="Realized")
    ax.plot(frame["date"], frame["q_0.5"], linewidth=1.0, label="Median")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("LLMTIME-inspired one-day return forecast")
    ax.set_ylabel("Cumulative log return")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _missingness_experiment(
    panel: Panel,
    origins: np.ndarray,
    tokenizer,
    model,
    checkpoint,
    thresholds,
    teacher_outputs,
    config,
    device,
    choice,
    seed,
) -> pd.DataFrame:
    llm_config = config["llmtime"]
    records: list[dict[str, float]] = []
    limit = min(len(origins), int(llm_config.get("missingness_origins", 64)))
    selected = origins[-limit:]
    selected_teacher = None
    if teacher_outputs is not None:
        selected_teacher = TeacherOutputs(
            condition=teacher_outputs.condition[-limit:],
            state=teacher_outputs.state[-limit:],
            quantiles=teacher_outputs.quantiles[-limit:],
            metadata=teacher_outputs.metadata,
        )
    for fraction in [0.0, *[float(value) for value in llm_config["missing_fractions"]]]:
        examples = build_examples(
            panel=panel,
            origins=selected,
            tokenizer=tokenizer,
            history_length=int(checkpoint["history_length"]),
            forecast_steps=int(checkpoint["forecast_steps"]),
            alpha=float(checkpoint["scale_alpha"]),
            beta=float(checkpoint["scale_beta"]),
            basic_scaler=bool(checkpoint["basic_scaler"]),
            side_thresholds=thresholds,
            use_side_info=bool(checkpoint["side_information"]),
            teacher_outputs=selected_teacher,
            missing_fraction=fraction,
            seed=seed + int(fraction * 1000),
        )
        paths = _sample_examples(
            model,
            tokenizer,
            examples,
            device=device,
            num_samples=int(llm_config["missingness_samples"]),
            temperature=choice.temperature,
            top_p=choice.top_p,
            batch_size=int(llm_config["evaluation_batch_size"]),
            seed=seed,
            scenario_batch_size=int(
                llm_config.get("missingness_scenario_batch_size", llm_config["missingness_samples"])
            ),
        )
        horizons = [int(value) for value in config["features"]["horizons"]]
        draws = _cumulative_samples(paths, horizons)
        truth = np.stack(
            [np.asarray([example.future[:h].sum() for example in examples]) for h in horizons],
            axis=1,
        )
        crps = np.mean(
            np.stack(
                [empirical_crps(draws[:, :, index], truth[:, index]) for index in range(len(horizons))],
                axis=1,
            )
        )
        median = np.median(draws, axis=1)
        records.append(
            {
                "missing_fraction": fraction,
                "origins": len(examples),
                "mean_crps": float(crps),
                "median_mae": float(np.mean(np.abs(median - truth))),
            }
        )
    return pd.DataFrame(records)


def evaluate_llmtime_fold(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    scenario_count: int | None = None,
) -> Path:
    device = resolve_device(device_name)
    model, tokenizer, checkpoint = load_llmtime_checkpoint(checkpoint_path, device)
    if checkpoint["fold"]["name"] != fold.name:
        raise ValueError("Checkpoint fold does not match requested fold")
    split = fold_origins(panel, fold, config, forecast_steps=int(checkpoint["forecast_steps"]))
    validation_origins = split["validation"]
    test_origins = split["test"]
    thresholds = _thresholds(checkpoint)

    teacher_validation = teacher_test = None
    if checkpoint.get("riskgraph_variant") is not None:
        conditioner = _conditioner_from_llmtime_checkpoint(
            checkpoint,
            panel,
            config,
            device,
        )
        teacher_batch_size = int(config["llmtime"].get("teacher_batch_size", 256))
        teacher_validation = compute_teacher_outputs_from_conditioner(
            panel,
            validation_origins,
            fold,
            config,
            conditioner,
            device,
            batch_size=teacher_batch_size,
        )
        teacher_test = compute_teacher_outputs_from_conditioner(
            panel,
            test_origins,
            fold,
            config,
            conditioner,
            device,
            batch_size=teacher_batch_size,
        )
        teacher_validation = _normalise_teacher_from_checkpoint(teacher_validation, checkpoint)
        teacher_test = _normalise_teacher_from_checkpoint(teacher_test, checkpoint)

    common = {
        "panel": panel,
        "tokenizer": tokenizer,
        "history_length": int(checkpoint["history_length"]),
        "forecast_steps": int(checkpoint["forecast_steps"]),
        "alpha": float(checkpoint["scale_alpha"]),
        "beta": float(checkpoint["scale_beta"]),
        "basic_scaler": bool(checkpoint["basic_scaler"]),
        "side_thresholds": thresholds,
        "use_side_info": bool(checkpoint["side_information"]),
        "missing_fraction": 0.0,
        "seed": int(checkpoint["seed"]),
    }
    validation_examples = build_examples(
        origins=validation_origins,
        teacher_outputs=teacher_validation,
        **common,
    )
    test_examples = build_examples(
        origins=test_origins,
        teacher_outputs=teacher_test,
        **common,
    )
    horizons = [int(value) for value in config["features"]["horizons"]]
    quantiles = [float(value) for value in config["features"]["quantiles"]]
    choice, tuning = choose_sampling_parameters(
        model,
        tokenizer,
        validation_examples,
        horizons=horizons,
        config=config,
        device=device,
        seed=int(checkpoint["seed"]),
    )
    samples = int(scenario_count or config["llmtime"]["evaluation_samples"])
    paths = _sample_examples(
        model,
        tokenizer,
        test_examples,
        device=device,
        num_samples=samples,
        temperature=choice.temperature,
        top_p=choice.top_p,
        batch_size=int(config["llmtime"]["evaluation_batch_size"]),
        seed=int(checkpoint["seed"]) + 1000,
        scenario_batch_size=int(
            config["llmtime"].get("evaluation_scenario_batch_size", samples)
        ),
    )
    cumulative = _cumulative_samples(paths, horizons)
    prediction = np.quantile(cumulative, quantiles, axis=1).transpose(1, 2, 0)
    prediction = np.sort(prediction, axis=-1)
    targets = targets_for_origins(panel, test_origins, horizons)
    stress = stress_mask_for_origins(panel, test_origins, fold.train_end)
    metrics, detail, backtests = evaluate_forecasts(
        prediction,
        targets,
        quantiles,
        horizons,
        stress_mask=stress,
    )

    crps_rows: list[dict[str, float | int]] = []
    crps_arrays: list[np.ndarray] = []
    for index, horizon in enumerate(horizons):
        values = empirical_crps(cumulative[:, :, index], targets[:, index])
        crps_arrays.append(values)
        crps_rows.append(
            {
                "horizon": horizon,
                "mean_crps": float(np.mean(values)),
                "median_crps": float(np.median(values)),
            }
        )
    joint_samples = cumulative
    joint_energy = energy_score(joint_samples, targets)
    mean_nll, nll_detail = _continuous_nll(
        model,
        test_examples,
        tokenizer,
        device=device,
        batch_size=int(config["llmtime"]["evaluation_batch_size"]),
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = predictions_frame(
        panel.dates[test_origins],
        targets,
        prediction,
        horizons,
        quantiles,
        stress_mask=stress,
    )
    frame.to_csv(output / "predictions.csv", index=False)
    detail.to_csv(output / "forecast_detail.csv", index=False)
    backtests.to_csv(output / "var_backtests.csv", index=False)
    tuning.to_csv(output / "sampling_selection.csv", index=False)
    pd.DataFrame(crps_rows).to_csv(output / "crps_detail.csv", index=False)
    nll_detail.to_csv(output / "continuous_nll_detail.csv", index=False)
    missingness = _missingness_experiment(
        panel,
        test_origins,
        tokenizer,
        model,
        checkpoint,
        thresholds,
        teacher_test,
        config,
        device,
        choice,
        int(checkpoint["seed"]),
    )
    missingness.to_csv(output / "missingness_robustness.csv", index=False)

    metrics.update(
        {
            "mean_crps": float(np.mean(np.stack(crps_arrays, axis=1))),
            "mean_energy_score": float(np.mean(joint_energy)),
            "continuous_nll_per_value": mean_nll,
            "sampling_temperature": choice.temperature,
            "sampling_top_p": choice.top_p,
            "validation_crps": choice.validation_crps,
            "scenario_count": samples,
        }
    )
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "probabilistic_metrics.json",
        {
            "mean_crps": metrics["mean_crps"],
            "mean_energy_score": metrics["mean_energy_score"],
            "continuous_nll_per_value": mean_nll,
            "horizons": crps_rows,
        },
    )
    teacher_consistency: dict[str, Any] = {}
    if teacher_test is not None:
        teacher_q = teacher_test.quantiles
        teacher_levels = np.asarray(checkpoint["teacher_metadata"]["teacher_spec"]["quantile_levels"])
        selected = [int(np.argmin(np.abs(teacher_levels - q))) for q in quantiles]
        teacher_selected = teacher_q[:, :, selected]
        teacher_consistency = {
            "teacher_quantile_mae": float(np.mean(np.abs(prediction - teacher_selected))),
            "teacher_median_mae": float(
                np.mean(
                    np.abs(
                        prediction[:, :, int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))]
                        - teacher_q[:, :, int(np.argmin(np.abs(teacher_levels - 0.5)))]
                    )
                )
            ),
        }
    write_json(output / "teacher_consistency.json", teacher_consistency)
    write_json(
        output / "evaluation_summary.json",
        {
            "fold": fold.name,
            "variant": Path(output_dir).parent.name,
            "seed": int(checkpoint["seed"]),
            "test_origins": len(test_origins),
            "forecast_mean_pinball": metrics["mean_pinball"],
            "mean_crps": metrics["mean_crps"],
            "continuous_nll_per_value": mean_nll,
            "mean_energy_score": metrics["mean_energy_score"],
            "sampling": {
                "temperature": choice.temperature,
                "top_p": choice.top_p,
                "validation_crps": choice.validation_crps,
                "scenarios": samples,
            },
            **teacher_consistency,
        },
    )
    write_json(
        output / "tokenization_audit.json",
        {
            **tokenizer.export(),
            "test_history_values_clipped": int(sum(item.clipped_history for item in test_examples)),
            "test_future_values_clipped": int(sum(item.clipped_future for item in test_examples)),
            "test_values": int(len(test_examples) * (checkpoint["history_length"] + checkpoint["forecast_steps"])),
            "prompt_tokens": int(len(test_examples[0].prompt_ids)),
            "full_tokens": int(len(test_examples[0].full_ids)),
        },
    )
    _plot_forecast_fan(frame, output / "forecast_fan.png")
    return output
