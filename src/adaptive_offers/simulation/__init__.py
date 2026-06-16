"""Offline bandit simulation with regret, exploration and delayed rewards."""

from __future__ import annotations

from adaptive_offers.simulation.environment import (
    SimulationResult,
    build_arms,
    precompute_eligibility,
    run_simulation,
)
from adaptive_offers.simulation.metrics import compare_results, summarize

__all__ = [
    "SimulationResult",
    "run_simulation",
    "build_arms",
    "precompute_eligibility",
    "summarize",
    "compare_results",
]
