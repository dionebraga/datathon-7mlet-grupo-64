"""Data-contract tests: leakage removal, target, provenance, reproducibility."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_leakage_columns_removed(small_layer):
    processed, _ = small_layer
    assert "duration" not in processed.columns, "leakage column must be dropped"
    assert "pdays" not in processed.columns, "raw pdays must be transformed"
    assert "previously_contacted" in processed.columns
    assert "pdays_since" in processed.columns


def test_target_is_binary(small_layer):
    processed, _ = small_layer
    assert "subscribed" in processed.columns
    assert set(processed["subscribed"].unique()).issubset({0, 1})


def test_surrogate_key_unique(small_layer):
    processed, _ = small_layer
    assert processed["client_event_id"].is_unique


def test_provenance_records_source_and_license():
    from adaptive_offers.data.loader import load_raw

    _, prov = load_raw(n_rows=500, seed=1)
    assert prov.source in {"real", "facsimile"}
    assert "CC BY" in prov.license
    assert prov.kaggle_url.startswith("https://www.kaggle.com")


def test_facsimile_is_reproducible():
    from adaptive_offers.data.loader import load_raw

    a, _ = load_raw(n_rows=500, seed=7)
    b, _ = load_raw(n_rows=500, seed=7)
    assert a.equals(b), "same seed must yield identical data"
