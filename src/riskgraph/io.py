from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions_frame(
    dates: pd.DatetimeIndex | list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    horizons: list[int],
    quantiles: list[float],
    stress_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    rows: list[dict[str, Any]] = []
    for row_index, date in enumerate(dates):
        for horizon_index, horizon in enumerate(horizons):
            row: dict[str, Any] = {
                "date": date.strftime("%Y-%m-%d"),
                "horizon": int(horizon),
                "target": float(targets[row_index, horizon_index]),
            }
            if stress_mask is not None:
                row["stress_regime"] = bool(stress_mask[row_index])
            for q_index, quantile in enumerate(quantiles):
                row[f"q_{quantile:g}"] = float(predictions[row_index, horizon_index, q_index])
            rows.append(row)
    return pd.DataFrame(rows)


def runtime_metadata(seed: int, device: torch.device) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
