#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.performance_v170.ensemble import evaluate_probabilistic_seed_ensemble
from riskgraph.performance_v170.settings import get_probabilistic_gate_settings


DEFAULT_VARIANTS = {
    "structured": "llmtime_target_ssl__statistical_champion__safe",
    "tailgan": "tailgan_stable_factor_scale__probabilistic_safe",
    "gom": "gom_stress_shrunk_factor_scale__probabilistic_safe",
}


def _seed_directories(
    root: Path,
    storage_family: str,
    fold: str,
    variant: str,
    seeds: list[int],
) -> list[Path]:
    paths = [
        root / storage_family / fold / variant / f"seed_{seed}" for seed in seeds
    ]
    missing = [
        str(path)
        for path in paths
        if not (path / "test_raw_predictions.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing v1.7 seed exports:\n" + "\n".join(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit probabilistic, validation-only seed ensembles and evaluate v1.7 "
            "forecasts against the statistical champion and frozen anchor"
        )
    )
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument(
        "--structured-variant", default=DEFAULT_VARIANTS["structured"]
    )
    parser.add_argument("--tailgan-variant", default=DEFAULT_VARIANTS["tailgan"])
    parser.add_argument("--gom-variant", default=DEFAULT_VARIANTS["gom"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    settings = get_probabilistic_gate_settings(config)
    horizons = [int(value) for value in config["features"]["horizons"]]
    variants = {
        "structured": ("structured", args.structured_variant),
        "tailgan": ("tail", args.tailgan_variant),
        "gom": ("tail", args.gom_variant),
    }

    for fold in selected_folds(config, args.folds):
        statistical = root / "statistical" / fold.name
        if not (statistical / "statistical_ensemble.json").is_file():
            raise FileNotFoundError(
                f"Missing statistical expert ensemble for {fold.name}: {statistical}"
            )
        for family, (storage_family, variant) in variants.items():
            output = root / "ensembles" / fold.name / variant
            if args.skip_existing and (output / "evaluation_summary.json").is_file():
                print(f"skipping existing ensemble -> {output}")
                continue
            seed_dirs = _seed_directories(
                root,
                storage_family,
                fold.name,
                variant,
                [int(seed) for seed in args.seeds],
            )
            evaluate_probabilistic_seed_ensemble(
                seed_dirs,
                output,
                horizons,
                settings,
                family=family,
                fold=fold.name,
                variant=variant,
                statistical_directory=statistical,
            )
            print(f"evaluated probabilistic ensemble -> {output}")


if __name__ == "__main__":
    main()
