"""Monitoring tests: PSI/KS drift detection and reward health alerting."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_offers.monitoring import drift_report, ks_drift, psi, reward_health
from adaptive_offers.monitoring.reward_monitor import RewardMonitor

pytestmark = pytest.mark.unit


def _rng():
    return np.random.default_rng(0)


def test_psi_low_for_same_distribution():
    rng = _rng()
    ref, cur = rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)
    assert psi(ref, cur) < 0.1


def test_psi_high_for_shift():
    rng = _rng()
    ref, cur = rng.normal(0, 1, 4000), rng.normal(1.0, 1.3, 4000)
    assert psi(ref, cur) > 0.25


def test_ks_detects_shift():
    rng = _rng()
    ref, cur = rng.normal(0, 1, 3000), rng.normal(0.7, 1, 3000)
    assert ks_drift(ref, cur)["drift"] is True


def test_drift_report_recommends_retrain_on_shift():
    rng = _rng()
    ref = rng.normal(0, 1, 3000)
    rep = drift_report({"f": ref}, {"f": rng.normal(1.0, 1.3, 3000)})
    assert rep["retrain_recommended"] is True


def test_reward_monitor_alerts_on_large_drop():
    # reference mean 11, low std -> a drop to ~2 is many sigma below.
    rewards = np.full(600, 2.0)
    status = reward_health(rewards, reference_mean=11.0, reference_std=8.0, window=500)
    assert status["alert"] is True
    assert status["action"] == "rollback/review"


def test_reward_monitor_not_ready_with_few_points():
    mon = RewardMonitor(reference_mean=10.0, reference_std=5.0, window=500)
    mon.observe(10.0)
    assert mon.status()["ready"] is False
