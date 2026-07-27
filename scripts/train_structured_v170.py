#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.performance_v150.data import load_long_history_panel
from riskgraph.performance_v170.statistical import StatisticalEnsembleState
from riskgraph.performance_v170.structured import train_structured_fold_v170


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune v1.7 SSL structured forecaster")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--variant", default="llmtime_target_ssl__statistical_champion__safe")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    panel = load_long_history_panel(resolve_path(config_path, config["performance_v170"]["long_panel_path"]))
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    for fold in selected_folds(config, args.folds):
        stat = StatisticalEnsembleState.from_record(json.loads((root / "statistical" / fold.name / "statistical_ensemble.json").read_text()))
        for seed in args.seeds:
            output = root / "structured" / fold.name / args.variant / f"seed_{seed}"
            if args.skip_existing and (output / "best_checkpoint.pt").is_file():
                print(f"skipping existing checkpoint -> {output}")
                continue
            train_structured_fold_v170(panel, fold, config, stat, seed, output / "pretrain_checkpoint.pt", output, args.device, args.epochs)
            print(f"trained -> {output}")


if __name__ == "__main__":
    main()
