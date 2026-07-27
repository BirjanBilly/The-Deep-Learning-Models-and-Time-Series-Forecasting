#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph.config import load_config, resolve_path
from riskgraph.io import write_json
from riskgraph.performance_v170.settings import get_probabilistic_gate_settings

FORMAL_VARIANTS = {
    "llmtime_target_ssl__statistical_champion__safe",
    "tailgan_stable_factor_scale__probabilistic_safe",
    "gom_stress_shrunk_factor_scale__probabilistic_safe",
}


def _qcols(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame if column.startswith("q_")],
        key=lambda value: float(value[2:]),
    )


def _loss_rows(frame: pd.DataFrame, qcols: list[str]) -> np.ndarray:
    q = np.asarray([float(value[2:]) for value in qcols], dtype=np.float64)
    prediction = frame[qcols].to_numpy(float)
    target = frame["target"].to_numpy(float)[:, None]
    error = target - prediction
    return np.maximum(q * error, (q - 1.0) * error).mean(axis=1)


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = frame["date"].astype(str)
    return frame.sort_values(["date", "horizon"]).reset_index(drop=True)


def _bootstrap_interval(values: np.ndarray, block: int, repetitions: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < max(20, 2 * block):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block = min(max(1, block), len(values))
    count = int(np.ceil(len(values) / block))
    offsets = np.arange(block)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        starts = rng.integers(0, len(values), size=count)
        selected = np.concatenate([(start + offsets) % len(values) for start in starts])[: len(values)]
        means[index] = values[selected].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _daily(frame: pd.DataFrame, loss: np.ndarray) -> pd.Series:
    return pd.DataFrame({"date": frame["date"], "loss": loss}).groupby("date", sort=True)["loss"].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare formal v1.7 probabilistic ensembles")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("ensembles/*/*/evaluation_summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("variant")) not in FORMAL_VARIANTS:
            continue
        summaries.append({**record, "run_dir": str(path.parent)})
    if not summaries:
        raise FileNotFoundError(f"No formal v1.7 ensemble summaries below {root}")
    frame = pd.DataFrame(summaries).sort_values(["fold", "family"])
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_performance_v170_ensembles.csv", index=False)

    settings = get_probabilistic_gate_settings(config)
    block = int(settings["block_length"])
    repetitions = min(2000, int(settings["bootstrap_repetitions"]))
    comparison_rows: list[dict[str, Any]] = []
    export_rows: list[pd.DataFrame] = []
    for fold_index, (fold, part) in enumerate(frame.groupby("fold", sort=True)):
        records: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, bool]] = {}
        common: set[tuple[str, int]] | None = None
        for row in part.itertuples(index=False):
            run = Path(row.run_dir)
            prediction = _read(run / "predictions.csv")
            champion = _read(run / "baseline_predictions.csv")
            anchor = _read(run / "anchor_predictions.csv")
            keys = set(zip(prediction["date"], prediction["horizon"].astype(int), strict=True))
            common = keys if common is None else common.intersection(keys)
            records[str(row.family)] = (
                prediction,
                champion,
                anchor,
                str(row.variant),
                bool(row.fallback_to_baseline),
            )
        if not common:
            raise ValueError(f"No common origins for {fold}")
        key_frame = pd.DataFrame(sorted(common), columns=["date", "horizon"])
        reference_family = "structured" if "structured" in records else sorted(records)[0]
        _, champion_source, anchor_source, _, _ = records[reference_family]
        champion = key_frame.merge(champion_source, on=["date", "horizon"], validate="one_to_one")
        anchor = key_frame.merge(anchor_source, on=["date", "horizon"], validate="one_to_one")
        qcols = _qcols(champion)
        champion_loss = _loss_rows(champion, qcols)
        anchor_loss = _loss_rows(anchor, qcols)
        champion_daily = _daily(champion, champion_loss)
        anchor_daily = _daily(anchor, anchor_loss)
        champion_mean = float(champion_loss.mean())
        anchor_mean = float(anchor_loss.mean())
        for family, variant, reference, mean in (
            ("baseline", "statistical_champion_v170", champion, champion_mean),
            ("anchor", "frozen_v154_ewma_student_t", anchor, anchor_mean),
        ):
            comparison_rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "variant": variant,
                    "common_rows": int(len(reference)),
                    "common_dates": int(reference["date"].nunique()),
                    "mean_pinball": mean,
                    "improvement_vs_statistical_champion_pct": 100.0 * (champion_mean - mean) / max(abs(champion_mean), 1e-12),
                    "improvement_vs_frozen_v154_anchor_pct": 100.0 * (anchor_mean - mean) / max(abs(anchor_mean), 1e-12),
                    "champion_loss_difference_ci_low": 0.0 if family == "baseline" else float("nan"),
                    "champion_loss_difference_ci_high": 0.0 if family == "baseline" else float("nan"),
                    "anchor_loss_difference_ci_low": 0.0 if family == "anchor" else float("nan"),
                    "anchor_loss_difference_ci_high": 0.0 if family == "anchor" else float("nan"),
                    "fallback_to_baseline": True,
                }
            )
            exported = reference.copy()
            exported["family"] = family
            exported["variant"] = variant
            export_rows.append(exported)

        for family, (prediction, _, _, variant, fallback) in records.items():
            aligned = key_frame.merge(prediction, on=["date", "horizon"], validate="one_to_one")
            if not np.allclose(aligned["target"].to_numpy(float), champion["target"].to_numpy(float), atol=1e-10, rtol=0.0):
                raise ValueError(f"Target mismatch for {fold}/{family}")
            model_loss = _loss_rows(aligned, qcols)
            model_daily = _daily(aligned, model_loss)
            champion_ci = _bootstrap_interval(
                champion_daily.to_numpy() - model_daily.to_numpy(), block, repetitions, 1700 + 20 * fold_index + len(comparison_rows)
            )
            anchor_ci = _bootstrap_interval(
                anchor_daily.to_numpy() - model_daily.to_numpy(), block, repetitions, 2700 + 20 * fold_index + len(comparison_rows)
            )
            mean = float(model_loss.mean())
            comparison_rows.append(
                {
                    "fold": fold,
                    "family": family,
                    "variant": variant,
                    "common_rows": int(len(aligned)),
                    "common_dates": int(aligned["date"].nunique()),
                    "mean_pinball": mean,
                    "improvement_vs_statistical_champion_pct": 100.0 * (champion_mean - mean) / max(abs(champion_mean), 1e-12),
                    "improvement_vs_frozen_v154_anchor_pct": 100.0 * (anchor_mean - mean) / max(abs(anchor_mean), 1e-12),
                    "champion_loss_difference_ci_low": champion_ci[0],
                    "champion_loss_difference_ci_high": champion_ci[1],
                    "anchor_loss_difference_ci_low": anchor_ci[0],
                    "anchor_loss_difference_ci_high": anchor_ci[1],
                    "fallback_to_baseline": fallback,
                }
            )
            exported = aligned.copy()
            exported["family"] = family
            exported["variant"] = variant
            export_rows.append(exported)

    common_frame = pd.DataFrame(comparison_rows).sort_values(["fold", "mean_pinball", "family"])
    common_frame.to_csv(output / "common_origin_comparison.csv", index=False)
    pd.concat(export_rows, ignore_index=True).to_csv(output / "common_origin_predictions.csv", index=False)
    frame.to_csv(output / "performance_v170_comparison.csv", index=False)
    winners = common_frame.loc[common_frame.groupby("fold")["mean_pinball"].idxmin()].copy()
    winners.to_csv(output / "performance_v170_fold_winners.csv", index=False)
    ml = common_frame[common_frame["family"].isin(["structured", "tailgan", "gom"])]
    ml.loc[ml.groupby("fold")["mean_pinball"].idxmin()].to_csv(output / "performance_v170_best_ml_challengers.csv", index=False)
    tolerance = 1e-8
    acceptance = frame.assign(
        test_win=(
            ~frame["fallback_to_baseline"].astype(bool)
            & (frame["improvement_vs_statistical_champion_pct"] > tolerance)
            & (frame["improvement_vs_frozen_v154_anchor_pct"] > tolerance)
        )
    )
    acceptance[
        [
            "family", "variant", "fold", "test_win",
            "improvement_vs_statistical_champion_pct",
            "improvement_vs_frozen_v154_anchor_pct",
            "fallback_to_baseline", "bootstrap_lower_bound",
            "effective_seed_count", "safety_reason",
        ]
    ].to_csv(output / "baseline_acceptance.csv", index=False)
    write_json(
        output / "comparison_manifest.json",
        {
            "version": "1.7.0",
            "seed_training_runs": 27,
            "formal_fold_ensembles": int(len(frame)),
            "formal_variants": sorted(FORMAL_VARIANTS),
            "folds": sorted(frame["fold"].unique().tolist()),
            "ranking_rule": (
                "Winners are ranked on identical origins. Every ML result is reported "
                "against both the train-only statistical champion and the frozen v1.5.4 EWMA anchor."
            ),
        },
    )
    print("\nFormal ensemble comparison")
    print(frame.to_string(index=False))
    print("\nCommon-origin comparison")
    print(common_frame.to_string(index=False))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
