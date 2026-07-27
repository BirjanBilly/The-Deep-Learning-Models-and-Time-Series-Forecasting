from __future__ import annotations

import argparse
from pathlib import Path

from riskgraph.config import load_config, resolve_path
from riskgraph.data.download import download_fred_data, download_market_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public market and FRED data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    raw_dir = resolve_path(args.config, data["raw_dir"])
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    download_market_data(data["tickers"], data["start"], data["end"], raw_dir)
    download_fred_data(data["fred_series"], raw_dir, data["start"], data["end"])


if __name__ == "__main__":
    main()
