"""Integration tests for the FastAPI decision service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(trained_service):
    # trained_service ensures an active policy + materialised feature store exist.
    from adaptive_offers.api.main import app

    return TestClient(app)


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["policy_loaded"] is True


def test_offers_listed(client):
    offers = client.get("/offers").json()
    assert len(offers) == 6
    assert any(o["offer_id"] == "OFF_NONE" for o in offers)


def test_decide_contract(client):
    r = client.post("/decide", json={"age": 24, "contact": "cellular", "euribor3m": 4.5})
    assert r.status_code == 200
    d = r.json()
    assert d["arm_id"] in d["eligible_arms"]
    assert d["reason_codes"]
    assert d["policy_version"]


def test_decide_adversarial_excludes_credit(client):
    r = client.post("/decide", json={"age": 30, "default": "yes", "loan": "yes"})
    elig = r.json()["eligible_arms"]
    assert "OFF_LOAN_PREAPP" not in elig
    assert "OFF_CC_CASHBACK" not in elig


def test_unknown_client_returns_404(client):
    r = client.post("/decide", json={"client_event_id": "ce_DOESNOTEXIST"})
    assert r.status_code == 404


def test_validation_error_returns_422(client):
    r = client.post("/decide", json={"age": 5})  # below ge=18
    assert r.status_code == 422


def test_assistant_explain(client):
    r = client.post("/assistant/explain", json={"age": 66, "poutcome": "success", "euribor3m": 0.8})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["provider"] in {"offline", "anthropic"}
