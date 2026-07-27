#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.data.dataset import load_panel
from riskgraph.io import write_json
from riskgraph.performance_v170.tail import refined_tail_config_v170
from riskgraph.tailrisk.experiment import prepare_fold_windows, strategy_bank_from_config, training_config_from_mapping
from riskgraph.tailrisk.trainer import train_tail_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stabilized v1.7 factor-scale Tail-GAN/GOM")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--objective", choices=["tailgan", "gom"], default="tailgan")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = refined_tail_config_v170(load_config(config_path), args.objective)
    if args.epochs is not None:
        config["tailrisk"]["epochs"] = int(args.epochs)
    panel = load_panel(resolve_path(config_path, config["data"]["processed_path"]))
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    variant = args.variant or (
        "gom_stress_shrunk_factor_scale__probabilistic_safe"
        if args.objective == "gom"
        else "tailgan_stable_factor_scale__probabilistic_safe"
    )
    base_training = training_config_from_mapping(config["tailrisk"], args.objective)
    for fold in selected_folds(config, args.folds):
        groups, train_windows, validation_windows, _ = prepare_fold_windows(panel, fold, config)
        regime_counts = [int((train_windows.regimes == regime).sum()) for regime in np.unique(train_windows.regimes)]
        safe_batch = min(base_training.batch_size, min(regime_counts))
        if safe_batch < 8:
            raise ValueError(f"Too few observations per regime: {regime_counts}")
        training = replace(base_training, batch_size=safe_batch)
        for seed in args.seeds:
            output = root / "tail" / fold.name / variant / f"seed_{seed}"
            if args.skip_existing and (output / "best_checkpoint.pt").is_file():
                print(f"skipping existing checkpoint -> {output}")
                continue
            bank = strategy_bank_from_config(train_windows.actual_paths, config["tailrisk"], mode=str(config["tailrisk"]["strategy_mode"]), seed=seed)
            write_json(output / "data_split.json", {"fold": fold.name, "train_origins": len(groups["train"]), "validation_origins": len(groups["validation"]), "test_origins": len(groups["test"]), "regime_counts": regime_counts, "generator_backbone": "ewma_factor_scale_stable"})
            train_tail_model(train_windows, validation_windows, bank, [float(value) for value in config["tailrisk"]["alphas"]], output, seed, training, args.device)
            print(f"trained -> {output}")


if __name__ == "__main__":
    main()
