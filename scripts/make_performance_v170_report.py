#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from riskgraph.config import load_config, resolve_path


def _markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write v1.7 statistical-champion research report")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    summary = root / "summary"
    common = pd.read_csv(summary / "common_origin_comparison.csv")
    winners = pd.read_csv(summary / "performance_v170_fold_winners.csv")
    challengers = pd.read_csv(summary / "performance_v170_best_ml_challengers.csv")
    acceptance = pd.read_csv(summary / "baseline_acceptance.csv")
    formal = pd.read_csv(summary / "all_performance_v170_ensembles.csv")

    columns = [
        "family", "fold", "variant", "mean_pinball", "raw_ensemble_pinball",
        "statistical_champion_pinball", "frozen_v154_anchor_pinball",
        "improvement_vs_statistical_champion_pct",
        "improvement_vs_frozen_v154_anchor_pct", "fallback_to_baseline",
        "effective_seed_count", "bootstrap_lower_bound", "safety_reason",
    ]
    columns = [column for column in columns if column in formal]
    report = f"""# Performance v1.7 statistical-champion and probabilistic-routing report

## Research question

Can a train-only statistical champion, target-relevant self-supervised Transformer,
constrained scenario generators, and uncertainty-aware probabilistic routing add
repeatable value while preserving the frozen v1.5.4 EWMA Student-t anchor?

## Design changes

1. The common fallback is a conservatively admitted statistical champion. The
   adaptive pool must pass disjoint train-only fitting, selection, calibration,
   confirmation, episode and bootstrap checks; otherwise the baseline is exactly
   the frozen v1.5.4 EWMA Student-t anchor.
2. The structured encoder adds future-distribution pretext tasks and a baseline-loss
   guard, while retaining a zero-initialized monotone residual head.
3. Regime routing uses causal smooth probabilities for calm, transition and stress,
   with confidence shrinkage and exact fallback under uncertain states.
4. Seed weights are shrunk, capped and required to have a minimum effective seed
   count, reducing single-seed promotion risk.
5. Neural gates apply only model residuals. Statistical calibration is completed
   before the ML gate, so calibration-only gains cannot be attributed to a neural model.
6. Every result is reported against both the statistical champion and the frozen
   v1.5.4 anchor. The reserved 2025 fold is excluded.

## Formal fold ensembles

{_markdown(formal[columns])}

## Common-origin comparison

{_markdown(common)}

## Actual fold winners

{_markdown(winners)}

## Best machine-learning challengers

{_markdown(challengers)}

## Gate decisions

{_markdown(acceptance)}

## Interpretation rules

- A rejected model is exactly the statistical champion on every quantile and origin.
- An adaptive statistical pool is not admitted merely because it beats the anchor in-sample.
- A neural test win requires a non-fallback forecast that beats both references above numerical tolerance.
- Confidence intervals are descriptive and do not fit or promote a model.
- No 2025 result belongs in this development report.
"""
    output = summary / "PERFORMANCE_V170_RESEARCH_REPORT.md"
    output.write_text(report, encoding="utf-8")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
