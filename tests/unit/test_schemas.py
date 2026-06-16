"""API contract tests (Pydantic validation)."""

from __future__ import annotations

import pytest

from adaptive_offers.api.schemas import ContextIn

pytestmark = pytest.mark.unit


def test_context_defaults_are_safe():
    feats = ContextIn(age=40).as_features()
    assert feats["contact"] == "cellular"
    assert feats["poutcome"] == "nonexistent"
    assert 0.0 <= feats["euribor3m"] <= 6.0


def test_invalid_age_rejected():
    with pytest.raises(Exception):
        ContextIn(age=5)


def test_invalid_contact_rejected():
    with pytest.raises(Exception):
        ContextIn(contact="carrier-pigeon")


def test_client_event_id_passthrough():
    ctx = ContextIn(client_event_id="ce_0000001")
    assert ctx.as_features()["client_event_id"] == "ce_0000001"
