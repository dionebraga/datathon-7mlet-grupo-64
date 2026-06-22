"""Behavioral segmentation — personas are deterministic, exhaustive and ordered."""

from __future__ import annotations

import pytest

from adaptive_offers.segmentation import SEGMENTS, label_of, segment_of


def test_all_personas_reachable():
    """Every declared persona is reached by at least one feature profile."""
    profiles = [
        {"age": 50, "loan": "yes", "default": "no", "poutcome": "nonexistent"},
        {"age": 70, "loan": "no", "default": "no", "poutcome": "success"},
        {"age": 24, "loan": "no", "default": "no", "contact": "cellular", "poutcome": "nonexistent"},
        {"age": 45, "loan": "no", "default": "no", "contact": "telephone", "poutcome": "success"},
        {"age": 38, "loan": "no", "default": "no", "poutcome": "nonexistent",
         "previously_contacted": 0},
        {"age": 40, "loan": "no", "default": "no", "contact": "cellular", "poutcome": "failure"},
    ]
    reached = {segment_of(p).seg_id for p in profiles}
    assert reached == {s.seg_id for s in SEGMENTS}


def test_priority_default_loan_over_age():
    """Risk (default/loan) takes precedence over age/engagement personas."""
    senior_with_loan = {"age": 72, "loan": "yes", "poutcome": "success"}
    assert segment_of(senior_with_loan).seg_id == "seg_renegociador"


def test_tolerant_to_missing_fields():
    """A sparse context still maps to a valid segment (catch-all)."""
    seg = segment_of({})
    assert seg.seg_id in {s.seg_id for s in SEGMENTS}


def test_label_lookup():
    assert label_of("seg_senior_conserv") == "Sênior conservador"
    assert label_of("unknown") == "unknown"


@pytest.mark.parametrize("row_type", [dict, "series"])
def test_accepts_dict_and_pandas_row(row_type):
    pd = pytest.importorskip("pandas")
    data = {"age": 70, "loan": "no", "default": "no", "poutcome": "success"}
    row = data if row_type is dict else pd.Series(data)
    assert segment_of(row).seg_id == "seg_senior_conserv"
