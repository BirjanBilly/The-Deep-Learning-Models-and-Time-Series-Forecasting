from __future__ import annotations

import argparse

from riskgraph.config import load_config, resolve_path
from riskgraph.data.features import build_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-aware market panel and dynamic graphs")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    features = config["features"]
    build_panel(
        tickers=data["tickers"],
        target_ticker=data["target_ticker"],
        fred_series=data["fred_series"],
        raw_dir=resolve_path(args.config, data["raw_dir"]),
        output_path=resolve_path(args.config, data["processed_path"]),
        metadata_path=resolve_path(args.config, data["metadata_path"]),
        correlation_window=int(features["correlation_window"]),
        graph_top_k=int(features["graph_top_k"]),
        macro_release_lag_days=int(data.get("macro_release_lag_days", 1)),
    )


if __name__ == "__main__":
    main()
