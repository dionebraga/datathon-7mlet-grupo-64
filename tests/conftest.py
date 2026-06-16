"""Shared pytest fixtures.

A small, fast platform is built once per session (1.5k rows, short horizon) so
unit and integration tests exercise the real code paths without being slow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def small_layer():
    """Build a small processed base + synthetic bundle once."""
    from adaptive_offers.data.preprocessing import build_processed
    from adaptive_offers.data.synthetic import generate

    processed, _, _ = build_processed(n_rows=1500, seed=42)
    bundle = generate(processed=processed, seed=42)
    return processed, bundle


@pytest.fixture(scope="session")
def trained_service(small_layer):
    """Train + register a small LinUCB and return a ready DecisionService."""
    from adaptive_offers.bootstrap import ensure_feature_store, train_and_register
    from adaptive_offers.policy.decision_service import DecisionService
    from adaptive_offers.policy.versioning import load_policy

    processed, bundle = small_layer
    ensure_feature_store(processed, bundle)
    train_and_register(policy_name="linucb", version="v1", horizon=1500, seed=42)
    policy, meta = load_policy("v1")
    return DecisionService(policy=policy, metadata=meta, catalog=bundle.catalog)
