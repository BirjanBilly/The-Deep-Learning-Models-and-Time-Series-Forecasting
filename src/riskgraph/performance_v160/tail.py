from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from riskgraph.config import Fold
from riskgraph.data.dataset import Panel
from riskgraph.io import write_json
from riskgraph.performance_v150.tail import export_tail_seed_predictions_v150


def refined_tail_config_v160(config: dict[str, Any]) -> dict[str, Any]:
    refined = copy.deepcopy(config)
    settings = refined["performance_v160"]["tail_model"]
    refined["tailrisk"].update(
        {
            "output_dir": refined["performance_v160"]["output_dir"],
            "generator_backbone": "ewma_factor_scale_stable",
            "strategy_mode": str(settings["strategy_mode"]),
            "epochs": int(settings["epochs"]),
            "patience": int(settings["patience"]),
            "minimum_epochs": int(settings["minimum_epochs"]),
            "batch_size": int(settings["batch_size"]),
            "latent_dim": int(settings["latent_dim"]),
            "learning_rate_generator": float(settings["learning_rate_generator"]),
            "learning_rate_discriminator": float(
                settings["learning_rate_discriminator"]
            ),
            "discriminator_steps": int(settings["discriminator_steps"]),
            "generator_steps": int(settings["generator_steps"]),
            "gradient_clip": float(settings["gradient_clip"]),
            "baseline_cov_lookback": int(settings["baseline_cov_lookback"]),
            "baseline_cov_decay": float(settings["baseline_cov_decay"]),
            "baseline_cov_shrinkage": float(settings["baseline_cov_shrinkage"]),
            "baseline_hidden_size": int(settings["hidden_size"]),
            "baseline_gru_layers": int(settings["gru_layers"]),
            "baseline_initial_gate": float(settings["initial_gate"]),
            "factor_rank": int(settings["factor_rank"]),
            "factor_scale_limit": float(settings["factor_scale_limit"]),
            "idio_scale_limit": float(settings["idio_scale_limit"]),
            "drift_limit": float(settings["drift_limit"]),
            "skew_limit": float(settings["skew_limit"]),
            "correlation_loss_weight": float(settings["correlation_loss_weight"]),
            "autocorrelation_loss_weight": float(
                settings["autocorrelation_loss_weight"]
            ),
            "marginal_loss_weight": float(settings["marginal_loss_weight"]),
            "energy_loss_weight": float(settings["energy_loss_weight"]),
            "validation_correlation_weight": float(
                settings["validation_correlation_weight"]
            ),
            "validation_autocorrelation_weight": float(
                settings["validation_autocorrelation_weight"]
            ),
            "validation_coverage_weight": float(
                settings["validation_coverage_weight"]
            ),
            "validation_score_rejection_weight": float(
                settings["validation_score_rejection_weight"]
            ),
            "validation_smoothing": float(settings["validation_smoothing"]),
            "validation_scenarios": int(settings["validation_scenarios"]),
            "discriminator_spectral_normalization": bool(
                settings["discriminator_spectral_normalization"]
            ),
            "discriminator_output_penalty": float(
                settings["discriminator_output_penalty"]
            ),
            "normalize_tail_score": bool(settings["normalize_tail_score"]),
            "tail_score_ema_decay": float(settings["tail_score_ema_decay"]),
            "tail_score_clip": float(settings["tail_score_clip"]),
            "generator_ema_decay": float(settings["generator_ema_decay"]),
            "conditioning_mode": "none",
            "quantile_consistency_weight": 0.0,
        }
    )
    return refined


def _align_canonical_baseline(
    canonical: pd.DataFrame,
    model: pd.DataFrame,
) -> pd.DataFrame:
    keys = model[["date", "horizon", "target"]].copy()
    keys["date"] = keys["date"].astype(str)
    source = canonical.copy()
    source["date"] = source["date"].astype(str)
    quantiles = [column for column in source if column.startswith("q_")]
    aligned = keys[["date", "horizon"]].merge(
        source[["date", "horizon", "target", *quantiles]],
        on=["date", "horizon"],
        how="left",
        validate="one_to_one",
    )
    if aligned.isna().any().any():
        raise ValueError("Canonical statistical baseline does not cover tail origins")
    if not (
        abs(aligned["target"].to_numpy(float) - keys["target"].to_numpy(float))
        <= 1e-10
    ).all():
        raise ValueError("Canonical statistical target mismatch")
    return aligned


def export_tail_seed_predictions_v160(
    panel: Panel,
    fold: Fold,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    statistical_dir: str | Path,
    seed: int,
    device_name: str = "auto",
) -> Path:
    # Reuse the fully audited v1.5 scenario export and diagnostics while replacing
    # its family-local baseline by the v1.6 statistical expert ensemble.
    adjusted = copy.deepcopy(config)
    adjusted["performance_v150"]["tail_factor_scale"] = copy.deepcopy(
        config["performance_v160"]["tail_model"]
    )
    output = export_tail_seed_predictions_v150(
        panel,
        fold,
        adjusted,
        checkpoint_path,
        output_dir,
        seed,
        device_name=device_name,
    )
    statistical = Path(statistical_dir)
    output = Path(output)
    for split_name in ("validation", "test"):
        raw = pd.read_csv(output / f"{split_name}_raw_predictions.csv")
        canonical = pd.read_csv(
            statistical / f"{split_name}_baseline_predictions.csv"
        )
        _align_canonical_baseline(canonical, raw).to_csv(
            output / f"{split_name}_baseline_predictions.csv", index=False
        )
    write_json(
        output / "v160_export_metadata.json",
        {
            "model_family": "stable_factor_scale_v160",
            "canonical_baseline": str(statistical),
            "generator_constraints": (
                "Origin-level factor and idiosyncratic scales, bounded drift and "
                "bounded downside asymmetry."
            ),
        },
    )
    return output
