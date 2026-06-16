"""Synthetic-layer tests: catalog, eligibility gates, latent model bounds."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_offers.data.synthetic import (
    CONTEXT_FEATURES,
    build_context_vector,
    eligible_arms,
    is_eligible,
    latent_conversion_prob,
    offer_catalog,
)

pytestmark = pytest.mark.unit


def test_catalog_has_control_arm():
    ids = {a.offer_id for a in offer_catalog()}
    assert "OFF_NONE" in ids


def test_default_client_excluded_from_credit():
    catalog = offer_catalog()
    row = {"default": "yes", "loan": "no", "age": 40}
    ids = [a.offer_id for a in eligible_arms(row, catalog)]
    assert "OFF_CC_CASHBACK" not in ids
    assert "OFF_LOAN_PREAPP" not in ids
    assert "OFF_NONE" in ids  # control always eligible


def test_min_age_gate_for_fund():
    fund = next(a for a in offer_catalog() if a.offer_id == "OFF_FUND_INTRO")
    assert not is_eligible(fund, {"default": "no", "loan": "no", "age": 24})
    assert is_eligible(fund, {"default": "no", "loan": "no", "age": 25})


def test_context_vector_shape_and_bias():
    row = {"age": 40, "contact": "cellular", "poutcome": "success",
           "previously_contacted": 1, "euribor3m": 1.0}
    v = build_context_vector(row, rate_median=2.5)
    assert v.shape == (len(CONTEXT_FEATURES),)
    assert v[0] == 1.0  # bias term


def test_latent_prob_in_unit_interval():
    catalog = offer_catalog()
    v = build_context_vector({"age": 50, "contact": "cellular", "poutcome": "success",
                              "previously_contacted": 0, "euribor3m": 1.0}, 2.5)
    for arm in catalog:
        p = latent_conversion_prob(arm, v)
        assert 0.0 <= p <= 1.0


def test_control_arm_has_zero_reward_potential():
    control = next(a for a in offer_catalog() if a.offer_id == "OFF_NONE")
    assert control.margin == 0.0
