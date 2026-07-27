from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from riskgraph.config import Fold, resolve_path
from riskgraph.data.dataset import Panel, split_origins
from riskgraph.tailrisk.data import TailWindowSet, attach_conditioning_state, build_tail_windows
from riskgraph.tailrisk.strategies import StrategyBank, build_strategy_bank
from riskgraph.tailrisk.trainer import TailTrainingConfig


def training_config_from_mapping(
    tail_config: dict[str, Any],
    objective: str,
    overrides: dict[str, Any] | None = None,
) -> TailTrainingConfig:
    accepted = {item.name for item in fields(TailTrainingConfig)}
    values = {key: value for key, value in tail_config.items() if key in accepted}
    values["objective"] = objective
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    if "betas" in values:
        values["betas"] = tuple(float(value) for value in values["betas"])
    if "conditioned_hidden_sizes" in values:
        values["conditioned_hidden_sizes"] = tuple(
            int(value) for value in values["conditioned_hidden_sizes"]
        )
    if "marginal_quantiles" in values:
        values["marginal_quantiles"] = tuple(
            float(value) for value in values["marginal_quantiles"]
        )
    return TailTrainingConfig(**values)


def prepare_fold_windows(
    panel: Panel,
    fold: Fold,
    full_config: dict[str, Any],
    conditioning_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], TailWindowSet, TailWindowSet, TailWindowSet]:
    tail = full_config["tailrisk"]
    scenario_horizon = int(tail["scenario_horizon"])
    forecasting_horizons = [int(value) for value in tail["forecast_horizons"]]
    groups = split_origins(
        panel,
        fold,
        lookback=max(int(full_config["features"]["lookback"]), int(tail["scale_lookback"])),
        horizons=[*forecasting_horizons, scenario_horizon],
        embargo_days=int(full_config["splits"]["embargo_days"]),
    )
    baseline_kwargs = {
        "baseline_cov_lookback": (
            int(tail.get("baseline_cov_lookback", 126))
            if str(tail.get("generator_backbone", "mlp")) in {"ewma_gru", "ewma_factor_scale", "ewma_factor_scale_stable"}
            else None
        ),
        "baseline_cov_decay": float(tail.get("baseline_cov_decay", 0.97)),
        "baseline_cov_shrinkage": float(tail.get("baseline_cov_shrinkage", 0.15)),
    }
    train = build_tail_windows(
        panel,
        groups["train"],
        horizon=scenario_horizon,
        scale_lookback=int(tail["scale_lookback"]),
        regime_fit_origins=groups["train"],
        **baseline_kwargs,
    )
    validation = build_tail_windows(
        panel,
        groups["validation"],
        horizon=scenario_horizon,
        scale_lookback=int(tail["scale_lookback"]),
        regime_edges=train.regime_edges,
        **baseline_kwargs,
    )
    test = build_tail_windows(
        panel,
        groups["test"],
        horizon=scenario_horizon,
        scale_lookback=int(tail["scale_lookback"]),
        regime_edges=train.regime_edges,
        **baseline_kwargs,
    )
    if conditioning_spec is not None:
        graph_mode = str(conditioning_spec.get("graph_mode", "dynamic"))
        macro_mode = str(conditioning_spec.get("macro_mode", "enabled"))
        lookback = int(conditioning_spec.get("lookback", full_config["features"]["lookback"]))
        train = attach_conditioning_state(
            panel, train, fold.train_end, groups["train"], lookback, graph_mode, macro_mode
        )
        validation = attach_conditioning_state(
            panel, validation, fold.train_end, groups["train"], lookback, graph_mode, macro_mode
        )
        test = attach_conditioning_state(
            panel, test, fold.train_end, groups["train"], lookback, graph_mode, macro_mode
        )
    return groups, train, validation, test


def strategy_bank_from_config(
    training_paths: np.ndarray,
    tail_config: dict[str, Any],
    mode: str,
    seed: int,
) -> StrategyBank:
    return build_strategy_bank(
        training_paths,
        mode=mode,
        num_eigenportfolios=int(tail_config["num_eigenportfolios"]),
        num_random_portfolios=int(tail_config["num_random_portfolios"]),
        seed=int(seed),
        signal_window=int(tail_config["signal_window"]),
    )


def strategy_bank_from_checkpoint(checkpoint: dict[str, Any]) -> StrategyBank:
    return StrategyBank(
        static_weights=np.asarray(checkpoint["strategy_static_weights"], dtype=np.float32),
        static_names=list(checkpoint["strategy_static_names"]),
        include_mean_reversion=bool(checkpoint["strategy_include_mean_reversion"]),
        include_trend_following=bool(checkpoint["strategy_include_trend_following"]),
        signal_window=int(checkpoint["strategy_signal_window"]),
    )


def tail_output_root(config_path: str | Path, config: dict[str, Any]) -> Path:
    return resolve_path(config_path, str(config["tailrisk"]["output_dir"]))
