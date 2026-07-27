from __future__ import annotations

from typing import Any


def get_probabilistic_gate_settings(config: dict[str, Any]) -> dict[str, Any]:
    try:
        section = config["performance_v170"]
    except KeyError as exc:
        raise KeyError("Configuration is missing performance_v170") from exc
    try:
        settings = section["probabilistic_gate"]
    except KeyError as exc:
        raise KeyError(
            "performance_v170 must define probabilistic_gate; legacy regime_gate is not accepted"
        ) from exc
    if not isinstance(settings, dict):
        raise TypeError("performance_v170.probabilistic_gate must be a mapping")
    return settings
