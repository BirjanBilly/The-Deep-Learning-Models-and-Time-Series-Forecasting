"""Tail-sensitive multi-asset scenario generation for RiskGraph.

This package is a clean-room implementation inspired by the methodology in
Cont, Cucuringu, Xu and Zhang (2025), "Tail-GAN: Learning to Simulate Tail
Risk Scenarios". It does not copy the original repository implementation.
"""

from riskgraph.tailrisk.conditioning import RiskGraphConditioner
from riskgraph.tailrisk.data import (
    TailWindowDataset,
    TailWindowSet,
    attach_conditioning_state,
    build_tail_windows,
)
from riskgraph.tailrisk.evaluation import evaluate_scenario_generator
from riskgraph.tailrisk.models import (
    RegimeTailGenerator,
    RiskGraphConditionedScenarioGenerator,
    RiskGraphConditionedTailGenerator,
    TailRiskDiscriminator,
)
from riskgraph.tailrisk.score import joint_var_es_score
from riskgraph.tailrisk.strategies import StrategyBank, build_strategy_bank
from riskgraph.tailrisk.synthetic import SyntheticTailSplit, make_synthetic_tail_split

__all__ = [
    "TailWindowDataset",
    "TailWindowSet",
    "build_tail_windows",
    "attach_conditioning_state",
    "evaluate_scenario_generator",
    "RiskGraphConditioner",
    "RegimeTailGenerator",
    "RiskGraphConditionedTailGenerator",
    "RiskGraphConditionedScenarioGenerator",
    "TailRiskDiscriminator",
    "joint_var_es_score",
    "StrategyBank",
    "build_strategy_bank",
    "SyntheticTailSplit",
    "make_synthetic_tail_split",
]
