"""LinThompson — Contextual Thompson Sampling for linear bandits.

Agrawal & Goyal (2013) "Thompson Sampling for Contextual Bandits with
Linear Payoffs", ICML. Posterior is Bayesian linear regression per arm:

    A_a  = λI + Σ x xᵀ          (d×d precision matrix, same as LinUCB)
    b_a  = Σ reward · x          (d,)
    θ_a  = A_a⁻¹ b_a             (posterior mean)
    Σ_a  = v² · A_a⁻¹            (posterior covariance, scaled)

At decision time, sample w̃_a ~ N(θ_a, Σ_a) for each arm and select
argmax_a (w̃_aᵀ x · margin_a). Unlike LinUCB, exploration requires no α
hyperparameter — uncertainty is captured directly in the posterior covariance.
"""

from __future__ import annotations

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy


class LinThompson(Policy):
    """Contextual Thompson Sampling — Bayesian linear regression per arm.

    The posterior is identical to LinUCB (A, b matrices) but selection is via
    posterior sampling rather than a deterministic UCB bonus. This makes
    exploration fully Bayesian and eliminates the α tuning hyperparameter.
    """

    name = "lin_thompson"

    def __init__(
        self,
        arms: list[Arm],
        dim: int,
        seed: int = 42,
        v: float = 1.0,   # posterior sampling scale (v=1 = theoretically correct)
        lam: float = 1.0,  # ridge regularisation (prior precision)
    ) -> None:
        super().__init__(arms, seed=seed)
        self.dim = dim
        self.v = v
        self.lam = lam
        # Posterior sufficient statistics, one per arm
        self.A: dict[str, np.ndarray] = {
            a: np.identity(dim) * lam for a in self.arm_ids
        }
        self.b: dict[str, np.ndarray] = {
            a: np.zeros(dim) for a in self.arm_ids
        }

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        x = np.asarray(context, dtype=float)
        if x.shape[0] != self.dim:
            raise ValueError(f"context dim {x.shape[0]} != policy dim {self.dim}")
        self.t += 1

        estimates: dict[str, float] = {}   # posterior-mean predictions (for greedy ref)
        scores: dict[str, float] = {}      # sampled-weight predictions (actual ranking)

        for a in elig:
            arm  = self.arms[a]
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]        # posterior mean

            # w̃ ~ N(θ, v² A⁻¹) — Bayesian posterior sample
            try:
                w_sample = self.rng.multivariate_normal(theta, (self.v ** 2) * A_inv)
            except np.linalg.LinAlgError:
                w_sample = theta             # degenerate fallback: use mean

            mean_reward  = max(0.0, float(theta @ x))
            sample_score = float(w_sample @ x)

            estimates[a] = round(mean_reward * arm.margin, 4)
            scores[a]    = sample_score * arm.margin

        best    = max(elig, key=lambda a: scores[a])
        greedy  = max(elig, key=lambda a: estimates[a])
        explored = best != greedy

        return Decision(
            arm_id=best,
            score=round(scores[best], 4),
            explored=explored,
            reason_codes=["LIN_THOMPSON", "CONTEXTUAL",
                          "EXPLORATION" if explored else "EXPLOITATION"],
            scores=scores,
            estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        if context is None:
            raise ValueError("LinThompson.update requires the decision context.")
        x = np.asarray(context, dtype=float)
        self.A[arm_id] += np.outer(x, x)
        self.b[arm_id] += float(reward) * x

    def state_dict(self) -> dict:
        return {**super().state_dict(), "dim": self.dim, "v": self.v, "lam": self.lam}
