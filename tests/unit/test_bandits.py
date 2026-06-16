"""Policy tests: contract, eligibility, learning, cold-start, contextual dim."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_offers.bandits.base import Arm
from adaptive_offers.bandits.registry import POLICIES, build_policy

pytestmark = pytest.mark.unit

ARMS = [Arm("A", 100.0), Arm("B", 200.0), Arm("OFF_NONE", 0.0)]
DIM = 8


def _ctx() -> np.ndarray:
    return np.array([1, 1, 1, 0.5, 0, 1, 0, 0], dtype=float)


@pytest.mark.parametrize("name", POLICIES)
def test_select_returns_eligible_arm(name):
    pol = build_policy(name, ARMS, context_dim=DIM, seed=1)
    eligible = ["A", "OFF_NONE"]
    decision = pol.select(_ctx(), eligible)
    assert decision.arm_id in eligible
    assert decision.reason_codes, "decision must carry reason codes"


@pytest.mark.parametrize("name", POLICIES)
def test_update_does_not_crash(name):
    pol = build_policy(name, ARMS, context_dim=DIM, seed=1)
    d = pol.select(_ctx(), ["A", "B"])
    pol.update(d.arm_id, reward=1.0, context=_ctx())


def test_empty_eligible_raises():
    pol = build_policy("thompson", ARMS, context_dim=DIM, seed=1)
    with pytest.raises(ValueError):
        pol.select(_ctx(), [])


def test_thompson_posterior_moves_with_evidence():
    from adaptive_offers.bandits.thompson import ThompsonSampling

    pol = ThompsonSampling(ARMS, seed=1)
    before = pol.posterior_mean("A")
    for _ in range(20):
        pol.update("A", 1.0)
    assert pol.posterior_mean("A") > before


def test_nilos_ucb_cold_start_pulls_each_arm():
    from adaptive_offers.bandits.ucb import NilosUCB

    pol = NilosUCB(ARMS, seed=1)
    pulled = set()
    for _ in range(3):
        d = pol.select(_ctx(), ["A", "B", "OFF_NONE"])
        pulled.add(d.arm_id)
        pol.update(d.arm_id, 0.0)
    assert len(pulled) == 3, "each arm must get an initial cold-start pull"


def test_linucb_requires_context_on_update():
    from adaptive_offers.bandits.linucb import LinUCB

    pol = LinUCB(ARMS, dim=DIM, seed=1)
    with pytest.raises(ValueError):
        pol.update("A", 1.0, context=None)


def test_linucb_dim_mismatch_raises():
    from adaptive_offers.bandits.linucb import LinUCB

    pol = LinUCB(ARMS, dim=DIM, seed=1)
    with pytest.raises(ValueError):
        pol.select(np.zeros(3), ["A"])
