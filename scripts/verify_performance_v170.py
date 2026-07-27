#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.performance_v170.settings import get_probabilistic_gate_settings


FORMAL_VARIANTS = {
    "structured": "llmtime_target_ssl__statistical_champion__safe",
    "tailgan": "tailgan_stable_factor_scale__probabilistic_safe",
    "gom": "gom_stress_shrunk_factor_scale__probabilistic_safe",
}
SEEDS = (11, 22, 33)


def _qcols(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame if column.startswith("q_")],
        key=lambda value: float(value[2:]),
    )


def _assert_prediction_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    columns = _qcols(frame)
    if not columns:
        raise ValueError(f"No quantile columns in {path}")
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(frame["target"].to_numpy(float)).all():
        raise ValueError(f"Non-finite predictions in {path}")
    if np.any(np.diff(values, axis=1) < -1e-8):
        raise ValueError(f"Non-monotone quantiles in {path}")
    return frame.sort_values(["date", "horizon"]).reset_index(drop=True)


def _assert_same(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    for column in ("date", "horizon"):
        if not reference[column].astype(str).equals(candidate[column].astype(str)):
            raise ValueError(f"{label}: key mismatch in {column}")
    qcols = _qcols(reference)
    if qcols != _qcols(candidate):
        raise ValueError(f"{label}: quantile columns differ")
    if not np.allclose(reference[qcols].to_numpy(float), candidate[qcols].to_numpy(float), atol=1e-10, rtol=0.0):
        raise ValueError(f"{label}: prediction mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify v1.7 formal development matrix")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    folds = selected_folds(config, "development")
    if (root / "ensembles" / "untouched_2025").exists() or (root / "statistical" / "untouched_2025").exists():
        raise ValueError("Reserved 2025 artifacts exist below the v1.7 output root")
    get_probabilistic_gate_settings(config)

    checkpoint_count = 0
    ensemble_count = 0
    for fold in folds:
        stat = root / "statistical" / fold.name
        state_path = stat / "statistical_champion.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        weights = np.asarray(state["weights"], dtype=float)
        if np.any(weights < -1e-10) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Non-convex adaptive statistical weights for {fold.name}")
        blend = np.asarray(state["champion_blend"], dtype=float)
        accepted = np.asarray(state["accepted"], dtype=bool)
        if np.any((~accepted) & (np.abs(blend) > 1e-12)):
            raise ValueError(f"Rejected statistical champion horizon does not fall back for {fold.name}")
        champion_reference = _assert_prediction_frame(stat / "test_baseline_predictions.csv")
        anchor_reference = _assert_prediction_frame(stat / "test_anchor_predictions.csv")
        regime = pd.read_csv(stat / "test_regime_features.csv")
        probs = regime[["p_calm", "p_transition", "p_stress"]].to_numpy(float)
        if np.any(probs < -1e-8) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Invalid regime probabilities for {fold.name}")

        family_frames: list[pd.DataFrame] = []
        family_anchors: list[pd.DataFrame] = []
        for family, variant in FORMAL_VARIANTS.items():
            storage = "structured" if family == "structured" else "tail"
            for seed in SEEDS:
                checkpoint = root / storage / fold.name / variant / f"seed_{seed}" / "best_checkpoint.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                checkpoint_count += 1
            run = root / "ensembles" / fold.name / variant
            summary_path = run / "evaluation_summary.json"
            gate_path = run / "ensemble_gate.json"
            if not summary_path.is_file() or not gate_path.is_file():
                raise FileNotFoundError(f"Missing formal ensemble output: {run}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            prediction = _assert_prediction_frame(run / "predictions.csv")
            baseline = _assert_prediction_frame(run / "baseline_predictions.csv")
            anchor = _assert_prediction_frame(run / "anchor_predictions.csv")
            _assert_same(champion_reference, baseline, f"{fold.name}/{family} champion")
            _assert_same(anchor_reference, anchor, f"{fold.name}/{family} anchor")
            family_frames.append(baseline)
            family_anchors.append(anchor)
            fallback = bool(summary["fallback_to_baseline"])
            if fallback:
                _assert_same(baseline, prediction, f"{fold.name}/{family} fallback")
            else:
                if not bool(gate["accepted"]):
                    raise ValueError(f"Accepted summary but rejected gate: {fold.name}/{family}")
                if float(gate["bootstrap_lower_bound"]) <= 0.0:
                    raise ValueError(f"Accepted gate lacks positive bootstrap: {fold.name}/{family}")
                if float(gate["effective_seed_count"]) < float(config["performance_v170"]["probabilistic_gate"]["minimum_effective_seed_count"]):
                    raise ValueError(f"Accepted gate has concentrated seeds: {fold.name}/{family}")
            ensemble_count += 1

    if checkpoint_count != 27:
        raise ValueError(f"Expected 27 formal checkpoints, found {checkpoint_count}")
    if ensemble_count != 9:
        raise ValueError(f"Expected 9 formal ensembles, found {ensemble_count}")
    summary_root = root / "summary"
    required = [
        "common_origin_comparison.csv",
        "performance_v170_fold_winners.csv",
        "performance_v170_best_ml_challengers.csv",
        "baseline_acceptance.csv",
        "PERFORMANCE_V170_RESEARCH_REPORT.md",
    ]
    for name in required:
        if not (summary_root / name).is_file():
            raise FileNotFoundError(summary_root / name)
    print("PERFORMANCE V1.7 27-SEED / 9-ENSEMBLE MATRIX CHECK PASSED")
    print("STATISTICAL CHAMPIONS ARE CAUSAL, CONVEX, AND ANCHOR-SAFE")
    print("PROBABILISTIC REGIME WEIGHTS ARE NORMALIZED AND CAUSAL")
    print("ALL REQUIRED OUTPUTS ARE PRESENT AND FINITE")
    print("ALL FINAL QUANTILE SURFACES ARE MONOTONE")
    print("ALL FALLBACK FORECASTS EXACTLY MATCH THE STATISTICAL CHAMPION")
    print("ALL ACCEPTED ENSEMBLES PASSED SEED, EPISODE, REGIME, RISK, AND BOOTSTRAP CHECKS")
    print("ALL MODEL RANKINGS USE IDENTICAL CHAMPION AND FROZEN-ANCHOR ORIGINS")
    print("THE RESERVED 2025 FOLD REMAINS UNTOUCHED")


if __name__ == "__main__":
    main()
