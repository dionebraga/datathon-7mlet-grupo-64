"""Deterministic baseline policy (control).

Optimistic-greedy: tracks each arm's running conversion mean (seeded with an
optimistic prior so every arm is tried before being abandoned) and always
exploits the current best ``mean × margin``. It performs **no active
exploration**, which is exactly the limitation an adaptive policy must beat —
making it the reference control required by the Datathon.
"""

from __future__ import annotations

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy


class BaselineGreedy(Policy):
    name = "baseline"

    def __init__(self, arms: list[Arm], seed: int = 42, optimistic_prior: float = 0.15) -> None:
        super().__init__(arms, seed=seed)
        # Optimistic init: prior mean with a small pseudo-count.
        self._sum = {a: optimistic_prior for a in self.arm_ids}
        self._count = {a: 1.0 for a in self.arm_ids}

    def _mean(self, arm_id: str) -> float:
        return self._sum[arm_id] / self._count[arm_id]

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        self.t += 1
        estimates = {a: self._mean(a) for a in elig}
        scores = {a: estimates[a] * self.margin(a) for a in elig}
        best = max(scores, key=scores.get)
        return Decision(
            arm_id=best, score=scores[best], explored=False,
            reason_codes=["BASELINE_GREEDY", "NO_EXPLORATION"],
            scores=scores, estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        self._sum[arm_id] += float(reward)
        self._count[arm_id] += 1.0

    def state_dict(self) -> dict:
        return {**super().state_dict(), "means": {a: self._mean(a) for a in self.arm_ids}}
