from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    validation_end: str
    test_end: str


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def selected_folds(config: dict[str, Any], selection: str) -> list[Fold]:
    splits = config["splits"]
    if selection == "development":
        records = splits["development"]
    elif selection == "final":
        records = [splits["final"]]
    elif selection == "all":
        records = [*splits["development"], splits["final"]]
    else:
        names = {item["name"]: item for item in splits["development"]}
        names[splits["final"]["name"]] = splits["final"]
        if selection not in names:
            raise ValueError(f"Unknown fold selection: {selection}")
        records = [names[selection]]
    return [Fold(**record) for record in records]


def resolve_path(config_path: str | Path, value: str) -> Path:
    base = Path(config_path).resolve().parent.parent
    return (base / value).resolve()
