"""Decision-service tests: guardrails, reason codes, audit logging."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_decision_has_reason_codes_and_version(trained_service):
    rec = trained_service.decide(context={"age": 66, "poutcome": "success", "euribor3m": 0.8})
    assert rec.arm_id.startswith("OFF_")
    assert rec.reason_codes
    assert "MARGIN_WEIGHTED" in rec.reason_codes
    assert rec.policy_version == "v1"
    assert rec.arm_id in rec.eligible_arms


def test_default_and_loan_client_never_offered_credit(trained_service):
    rec = trained_service.decide(
        context={"age": 30, "default": "yes", "loan": "yes", "poutcome": "nonexistent"}
    )
    assert "OFF_LOAN_PREAPP" not in rec.eligible_arms
    assert "OFF_CC_CASHBACK" not in rec.eligible_arms
    assert "ELIGIBILITY_FILTERED" in rec.reason_codes


def test_decision_is_audited(trained_service, tmp_path):
    trained_service.audit_path = tmp_path / "audit.jsonl"
    rec = trained_service.decide(context={"age": 40, "poutcome": "failure"})
    lines = trained_service.audit_path.read_text(encoding="utf-8").splitlines()
    assert lines, "decision must be appended to the audit log"
    logged = json.loads(lines[-1])
    assert logged["decision_id"] == rec.decision_id
    assert logged["arm_id"] == rec.arm_id


def test_decide_requires_context_or_id(trained_service):
    with pytest.raises(ValueError):
        trained_service.decide()
