"""LinUCB — contextual bandit with a disjoint linear model per arm.

Li, Chu, Langford & Schapire (2010). Each arm keeps a ridge-regression model of
``E[conversion | context]``:

    A_a = I_d + Σ x xᵀ          (d×d)
    b_a = Σ reward · x          (d,)
    θ_a = A_a⁻¹ b_a
    p_a = θ_aᵀ x                (predicted conversion)
    ucb = p_a + α · sqrt( xᵀ A_a⁻¹ x )   (uncertainty bonus)

Ranking is margin-weighted (``ucb × margin``). Because each offer's best segment
differs (Stage 2 latent model), the contextual policy can route the right offer
to the right context — outperforming context-free policies on heterogeneous
populations. ``α`` trades confidence for exploration.
"""

from __future__ import annotations

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy


class LinUCB(Policy):
    name = "linucb"

    def __init__(self, arms: list[Arm], dim: int, seed: int = 42, alpha: float = 0.6) -> None:
        super().__init__(arms, seed=seed)
        self.dim = dim
        self.alpha = alpha
        self.A = {a: np.identity(dim) for a in self.arm_ids}
        self.b = {a: np.zeros(dim) for a in self.arm_ids}

    def _predict(self, arm_id: str, x: np.ndarray) -> tuple[float, float]:
        A_inv = np.linalg.inv(self.A[arm_id])
        theta = A_inv @ self.b[arm_id]
        mean = float(theta @ x)
        bonus = float(self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))
        return mean, bonus

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        x = np.asarray(context, dtype=float)
        if x.shape[0] != self.dim:
            raise ValueError(f"context dim {x.shape[0]} != policy dim {self.dim}")
        self.t += 1
        estimates, scores, ucb = {}, {}, {}
        for a in elig:
            mean, bonus = self._predict(a, x)
            estimates[a] = mean
            ucb[a] = mean + bonus
            scores[a] = ucb[a] * self.margin(a)
        best = max(scores, key=scores.get)
        greedy = max(elig, key=lambda a: estimates[a] * self.margin(a))
        explored = best != greedy
        return Decision(
            arm_id=best, score=scores[best], explored=explored,
            reason_codes=["LINUCB", "CONTEXTUAL",
                          "EXPLORATION" if explored else "EXPLOITATION"],
            scores=scores, estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        if context is None:
            raise ValueError("LinUCB.update requires the decision context")
        x = np.asarray(context, dtype=float)
        self.A[arm_id] += np.outer(x, x)
        self.b[arm_id] += float(reward) * x

    def state_dict(self) -> dict:
        return {**super().state_dict(), "dim": self.dim, "alpha": self.alpha}
