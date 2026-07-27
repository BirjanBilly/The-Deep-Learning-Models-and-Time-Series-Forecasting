#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from riskgraph.config import load_config, resolve_path, selected_folds
from riskgraph.data.dataset import load_panel
from riskgraph.performance_v170.tail import export_tail_seed_predictions_v170


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stabilized v1.7 tail seed forecasts")
    parser.add_argument("--config", default="configs/financial_risk_graph_v170.yaml")
    parser.add_argument("--folds", default="development")
    parser.add_argument("--objective", choices=["tailgan", "gom"], default="tailgan")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    panel = load_panel(resolve_path(config_path, config["data"]["processed_path"]))
    root = resolve_path(config_path, config["performance_v170"]["output_dir"])
    variant = args.variant or (
        "gom_stress_shrunk_factor_scale__probabilistic_safe"
        if args.objective == "gom"
        else "tailgan_stable_factor_scale__probabilistic_safe"
    )
    for fold in selected_folds(config, args.folds):
        stat_dir = root / "statistical" / fold.name
        for seed in args.seeds:
            output = root / "tail" / fold.name / variant / f"seed_{seed}"
            if args.skip_existing and (output / "test_raw_predictions.csv").is_file():
                print(f"skipping existing export -> {output}")
                continue
            export_tail_seed_predictions_v170(panel, fold, config, output / "best_checkpoint.pt", output, stat_dir, seed, args.device)
            print(f"exported -> {output}")


if __name__ == "__main__":
    main()
