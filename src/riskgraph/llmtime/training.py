from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel
from riskgraph.io import file_sha256, runtime_metadata, write_json
from riskgraph.llmtime.data import (
    LLMTimeDataset,
    TeacherOutputs,
    build_examples,
    compute_teacher_outputs_from_conditioner,
    fit_side_info_thresholds,
    fold_origins,
    resolve_teacher_checkpoint,
)
from riskgraph.llmtime.model import (
    DecimalCausalTransformer,
    DecimalTransformerConfig,
    count_trainable_parameters,
)
from riskgraph.llmtime.serialization import FinancialTokenizer, TokenizerConfig
from riskgraph.tailrisk.conditioning import RiskGraphConditioner
from riskgraph.repro import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _normalise_teacher_outputs(
    train: TeacherOutputs,
    *others: TeacherOutputs,
) -> tuple[TeacherOutputs, ...]:
    mean = train.condition.mean(axis=0, keepdims=True)
    std = train.condition.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def convert(value: TeacherOutputs) -> TeacherOutputs:
        metadata = copy.deepcopy(value.metadata)
        metadata["condition_normalization"] = {
            "mean": mean.reshape(-1).astype(float).tolist(),
            "std": std.reshape(-1).astype(float).tolist(),
            "fitted_on": "training origins only",
        }
        return replace(
            value,
            condition=((value.condition - mean) / std).astype(np.float32),
            metadata=metadata,
        )

    return tuple(convert(item) for item in (train, *others))


def _epoch(
    model: DecimalCausalTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        condition = batch.get("condition")
        if condition is not None:
            condition = condition.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(input_ids, condition=condition, loss_mask=loss_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite LLMTIME loss: {float(loss.detach().cpu())}")
        if training:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_llmtime_fold(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    config_path: str | Path,
    seed: int,
    output_dir: str | Path,
    token_mode: str = "digit",
    use_side_info: bool = False,
    riskgraph_variant: str | None = None,
    device_name: str = "auto",
    max_epochs: int | None = None,
) -> Path:
    seed_everything(seed)
    device = resolve_device(device_name)
    llm_config = config["llmtime"]
    forecast_steps = int(llm_config["forecast_steps"])
    history_length = int(llm_config["history_length"])
    split = fold_origins(panel, fold, config, forecast_steps=forecast_steps)
    stride = max(1, int(llm_config.get("train_stride", 1)))
    train_origins = split["train"][::stride]
    validation_origins = split["validation"]
    test_origins = split["test"]

    tokenizer = FinancialTokenizer(
        TokenizerConfig(
            mode=token_mode,
            precision=int(llm_config["precision"]),
            integer_digits=int(llm_config["integer_digits"]),
            clip_value=float(llm_config["clip_value"]),
            flat_bins=int(llm_config["flat_bins"]),
            half_bin_correction=True,
        )
    )
    side_thresholds = fit_side_info_thresholds(panel, train_origins)

    teacher_train = teacher_validation = teacher_test = None
    teacher_checkpoint: Path | None = None
    conditioner: RiskGraphConditioner | None = None
    if riskgraph_variant is not None:
        teacher_checkpoint = resolve_teacher_checkpoint(
            riskgraph_output_root=Path(config_path).resolve().parent.parent
            / config["experiment"]["output_dir"],
            fold=fold,
            variant=riskgraph_variant,
            seed=seed,
        )
        conditioner = RiskGraphConditioner.from_checkpoint(
            teacher_checkpoint,
            panel,
            config,
            device,
        )
        teacher_batch_size = int(llm_config.get("teacher_batch_size", 256))
        teacher_train = compute_teacher_outputs_from_conditioner(
            panel,
            train_origins,
            fold,
            config,
            conditioner,
            device,
            batch_size=teacher_batch_size,
        )
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
        teacher_train, teacher_validation, teacher_test = _normalise_teacher_outputs(
            teacher_train, teacher_validation, teacher_test
        )

    common = {
        "panel": panel,
        "tokenizer": tokenizer,
        "history_length": history_length,
        "forecast_steps": forecast_steps,
        "alpha": float(llm_config["scale_alpha"]),
        "beta": float(llm_config["scale_beta"]),
        "basic_scaler": bool(llm_config["basic_scaler"]),
        "side_thresholds": side_thresholds,
        "use_side_info": use_side_info,
    }
    train_examples = build_examples(
        origins=train_origins,
        teacher_outputs=teacher_train,
        missing_fraction=0.0,
        seed=seed,
        **common,
    )
    validation_examples = build_examples(
        origins=validation_origins,
        teacher_outputs=teacher_validation,
        missing_fraction=0.0,
        seed=seed + 1,
        **common,
    )

    condition_dim = 0 if teacher_train is None else int(teacher_train.condition.shape[1])
    sequence_length = len(train_examples[0].full_ids)
    model_config = DecimalTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(llm_config["d_model"]),
        n_heads=int(llm_config["n_heads"]),
        n_layers=int(llm_config["n_layers"]),
        d_ff=int(llm_config["d_ff"]),
        dropout=float(llm_config["dropout"]),
        max_tokens=max(int(llm_config["max_tokens"]), sequence_length + 8),
        condition_dim=condition_dim,
    )
    model = DecimalCausalTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(llm_config["learning_rate"]),
        weight_decay=float(llm_config["weight_decay"]),
    )
    batch_size = int(llm_config["batch_size"])
    train_loader = DataLoader(
        LLMTimeDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(llm_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        LLMTimeDataset(validation_examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(llm_config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    epochs = int(max_epochs if max_epochs is not None else llm_config["epochs"])
    patience = int(llm_config["patience"])
    gradient_clip = float(llm_config["gradient_clip"])
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, train_loader, device, optimizer, gradient_clip)
        with torch.no_grad():
            validation_loss = _epoch(model, validation_loader, device, None, gradient_clip)
        history.append(
            {
                "epoch": epoch,
                "train_token_nll": train_loss,
                "validation_token_nll": validation_loss,
            }
        )
        print(
            f"{fold.name} llmtime seed={seed} epoch={epoch:03d} "
            f"train_nll={train_loss:.6f} validation_nll={validation_loss:.6f}"
        )
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No finite LLMTIME checkpoint was produced")
    model.load_state_dict(best_state)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    variant = output.parent.name
    checkpoint_payload: dict[str, Any] = {
        "package_version": "1.3.0",
        "fold": asdict(fold),
        "variant": variant,
        "seed": int(seed),
        "model_config": model.export_config(),
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.export(),
        "side_information": bool(use_side_info),
        "side_thresholds": side_thresholds.export(),
        "riskgraph_variant": riskgraph_variant,
        "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
        "teacher_checkpoint_sha256": None
        if teacher_checkpoint is None
        else file_sha256(teacher_checkpoint),
        "teacher_metadata": None if teacher_train is None else teacher_train.metadata,
        "teacher_model_state_dict": None
        if conditioner is None
        else {key: value.detach().cpu() for key, value in conditioner.model.state_dict().items()},
        "history_length": history_length,
        "forecast_steps": forecast_steps,
        "scale_alpha": float(llm_config["scale_alpha"]),
        "scale_beta": float(llm_config["scale_beta"]),
        "basic_scaler": bool(llm_config["basic_scaler"]),
        "best_validation_token_nll": best_validation,
        "config_path": str(Path(config_path).resolve()),
    }
    torch.save(checkpoint_payload, output / "best_checkpoint.pt")
    write_json(
        output / "run_metadata.json",
        {
            **runtime_metadata(seed, device),
            "fold": fold.name,
            "variant": variant,
            "token_mode": token_mode,
            "use_side_info": use_side_info,
            "riskgraph_variant": riskgraph_variant,
            "train_origins": len(train_origins),
            "validation_origins": len(validation_origins),
            "test_origins": len(test_origins),
            "train_stride": stride,
            "sequence_length": sequence_length,
            "vocab_size": tokenizer.vocab_size,
            "parameters": count_trainable_parameters(model),
            "best_validation_token_nll": best_validation,
            "elapsed_seconds": time.time() - started,
            "teacher_checkpoint": None if teacher_checkpoint is None else str(teacher_checkpoint),
            "teacher_checkpoint_sha256": None
            if teacher_checkpoint is None
            else file_sha256(teacher_checkpoint),
            "causality": (
                "origin-specific scaling uses history only; side thresholds and condition "
                "normalization are fitted on training origins only"
            ),
        },
    )
    write_json(
        output / "data_split.json",
        {
            "fold": asdict(fold),
            "history_length": history_length,
            "forecast_steps": forecast_steps,
            "train_origins": [int(value) for value in train_origins],
            "validation_origins": [int(value) for value in validation_origins],
            "test_origins": [int(value) for value in test_origins],
        },
    )
    write_json(output / "tokenizer_metadata.json", tokenizer.export())
    return output


def load_llmtime_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[DecimalCausalTransformer, FinancialTokenizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_config = TokenizerConfig(**checkpoint["tokenizer"]["config"])
    tokenizer = FinancialTokenizer(tokenizer_config)
    model = DecimalCausalTransformer(DecimalTransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, checkpoint
