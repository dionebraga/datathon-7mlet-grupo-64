"""Responsible AI — protected attributes are registered and audit-only."""

from __future__ import annotations

from adaptive_offers.responsible import (
    PROTECTED_ATTRIBUTES,
    age_band,
    protected_groups,
)


def test_protected_registry_covers_sensitive_proxies():
    assert {"age", "job", "marital", "education"} <= set(PROTECTED_ATTRIBUTES)


def test_age_band_buckets():
    assert age_band(25) == "<=30"
    assert age_band(40) == "31-45"
    assert age_band(55) == "46-60"
    assert age_band(72) == "60+"
    assert age_band(None) == "unknown"


def test_protected_groups_from_context():
    g = protected_groups(
        {"age": 67, "marital": "married", "education": "university.degree", "job": "retired"}
    )
    assert g["age_band"] == "60+"
    assert g["marital"] == "married"
    assert g["education"] == "university.degree"
    assert g["job"] == "retired"


def test_protected_groups_tolerant_to_missing():
    g = protected_groups({})
    assert g["age_band"] == "unknown"
    assert g["marital"] == "unknown"
