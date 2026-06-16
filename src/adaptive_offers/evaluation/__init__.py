"""Offline evaluation: golden set, metrics matrix, sensitivity and fairness."""

from __future__ import annotations

from adaptive_offers.evaluation.fairness import exposure_report
from adaptive_offers.evaluation.golden_set import (
    GoldenCase,
    evaluate_golden,
    load_cases,
)
from adaptive_offers.evaluation.offline_eval import (
    ips_estimate,
    metrics_matrix,
    sensitivity_analysis,
    train_frozen_policy,
)

__all__ = [
    "GoldenCase",
    "load_cases",
    "evaluate_golden",
    "train_frozen_policy",
    "metrics_matrix",
    "sensitivity_analysis",
    "ips_estimate",
    "exposure_report",
]
