from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def get_regime_gate_settings(
    config: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the v1.6 regime-gate settings with a precise diagnostic.

    ``regime_gate`` is the canonical v1.6 key.  ``ensemble_gate`` is accepted
    only as a compatibility alias for early development configurations.
    """

    performance = config.get("performance_v160")
    if not isinstance(performance, Mapping):
        raise KeyError("Missing mapping: performance_v160")

    settings = performance.get("regime_gate")
    source = "performance_v160.regime_gate"
    if not isinstance(settings, Mapping):
        settings = performance.get("ensemble_gate")
        source = "performance_v160.ensemble_gate (legacy alias)"

    if not isinstance(settings, Mapping):
        raise KeyError(
            "Missing mapping: performance_v160.regime_gate. "
            "The v1.6 comparison, evaluation, and verification stages must use "
            "the same regime-gate configuration."
        )

    missing = [name for name in required if name not in settings]
    if missing:
        raise KeyError(f"Missing {source} settings: {', '.join(missing)}")

    return dict(settings)
