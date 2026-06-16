"""Golden-set tests: file integrity, coverage and evaluation harness."""

from __future__ import annotations

import pytest

from adaptive_offers.config import get_settings
from adaptive_offers.evaluation.golden_set import evaluate_golden, load_cases

pytestmark = pytest.mark.unit

CASES_PATH = get_settings().paths.golden_set / "evaluation_cases.jsonl"


def test_golden_set_has_at_least_20_cases():
    cases = load_cases(CASES_PATH)
    assert len(cases) >= 20


def test_golden_set_covers_required_categories():
    cats = {c.category for c in load_cases(CASES_PATH)}
    assert {"typical", "edge", "adversarial"}.issubset(cats)


def test_every_case_has_criterion_and_justification():
    for c in load_cases(CASES_PATH):
        assert c.pass_fail, f"{c.case_id} missing pass/fail criterion"
        assert c.justification, f"{c.case_id} missing justification"


def test_linucb_passes_all_adversarial_guardrails(trained_service, small_layer):
    _, bundle = small_layer
    cases = load_cases(CASES_PATH)
    report = evaluate_golden(cases, trained_service.policy, bundle.catalog, bundle.rate_median)
    adversarial = report["by_category"].get("adversarial", {})
    assert adversarial["passed"] == adversarial["total"], "all guardrails must hold"
