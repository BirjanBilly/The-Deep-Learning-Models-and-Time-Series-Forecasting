"""Baseline-safe performance refinements for RiskGraph, Tail-GAN and LLMTIME."""

from riskgraph.performance_v140.baselines import (
    EWMATState,
    ewma_student_t_forecast,
    weighted_ewma_correlation,
)
from riskgraph.performance_v140.calibration import ForecastGate
from riskgraph.performance_v140.patch_model import (
    ResidualPatchConfig,
    ResidualPatchQuantileTransformer,
)

__all__ = [
    "EWMATState",
    "ForecastGate",
    "ResidualPatchConfig",
    "ResidualPatchQuantileTransformer",
    "ewma_student_t_forecast",
    "weighted_ewma_correlation",
]
