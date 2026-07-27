#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.performance_v150.data import load_long_history_panel
from riskgraph.performance_v170.structured import export_structured_seed_v170


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v1.7 structured seed forecasts")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant", default="llmtime_target_ssl__statistical_champion__safe")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    panel = load_long_history_panel(resolve_path(config_path, config["performance_v170"]["long_panel_path"]))
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    for fold in selected_folds(config, args.folds):
        for seed in args.seeds:
            output = root / "structured" / fold.name / args.variant / f"seed_{seed}"
            if args.skip_existing and (output / "test_raw_predictions.csv").is_file():
                print(f"skipping existing export -> {output}")
                continue
            export_structured_seed_v170(panel, fold, config, output / "best_checkpoint.pt", output, args.device)
            print(f"exported -> {output}")


if __name__ == "__main__":
    main()
