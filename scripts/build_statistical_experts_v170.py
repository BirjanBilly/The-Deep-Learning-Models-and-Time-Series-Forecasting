#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.performance_v150.data import load_long_history_panel
from riskgraph.performance_v170.statistical import save_statistical_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v1.7 robust statistical expert ensembles")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    panel = load_long_history_panel(
        resolve_path(config_path, config["performance_v170"]["long_panel_path"])
    )
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    for fold in selected_folds(config, args.folds):
        output = root / "statistical" / fold.name
        if args.skip_existing and (output / "statistical_ensemble.json").is_file():
            print(f"skipping existing statistical ensemble -> {output}")
            continue
        save_statistical_fold(panel, fold, config, output)
        print(f"saved statistical ensemble -> {output}")


if __name__ == "__main__":
    main()
