"""Neural bandit tests (skipped automatically when PyTorch is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # noqa: F841 - skip the module if torch absent

from adaptive_offers.bandits.base import Arm  # noqa: E402
from adaptive_offers.bandits.neural import NeuralBandit  # noqa: E402

pytestmark = pytest.mark.unit

ARMS = [Arm("A", 100.0), Arm("B", 200.0), Arm("OFF_NONE", 0.0)]
DIM = 8


def _ctx() -> np.ndarray:
    return np.array([1, 1, 1, 0.5, 0, 1, 0, 0], dtype=float)


def test_neural_selects_eligible_arm():
    pol = NeuralBandit(ARMS, dim=DIM, seed=1, warmup=4, train_every=2)
    d = pol.select(_ctx(), ["A", "B"])
    assert d.arm_id in ["A", "B"]
    assert "NEURAL" in d.reason_codes and "MC_DROPOUT" in d.reason_codes


def test_neural_dim_mismatch_raises():
    pol = NeuralBandit(ARMS, dim=DIM, seed=1)
    with pytest.raises(ValueError):
        pol.select(np.zeros(3), ["A"])


def test_neural_requires_context_on_update():
    pol = NeuralBandit(ARMS, dim=DIM, seed=1)
    with pytest.raises(ValueError):
        pol.update("A", 1.0, context=None)


def test_neural_learns_which_arm_converts():
    pol = NeuralBandit(ARMS, dim=DIM, seed=1, warmup=4, train_every=2, batch_size=8, lr=0.05)
    # B always converts, A never — the model should separate them.
    for _ in range(80):
        pol.update("B", 1.0, _ctx())
        pol.update("A", 0.0, _ctx())
    est_b = pol._probs(np.vstack([pol._features(_ctx(), "B")]), stochastic=False)[0]
    est_a = pol._probs(np.vstack([pol._features(_ctx(), "A")]), stochastic=False)[0]
    assert est_b > est_a
