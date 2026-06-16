"""Thompson Sampling (Beta-Bernoulli) with margin-weighted ranking.

For each arm we keep a Beta(α, β) posterior over its conversion probability.
At decision time we **sample** θ_a ~ Beta(α_a, β_a) for every eligible arm and
pick ``argmax θ_a × margin_a``. Sampling is the exploration mechanism: arms with
wide posteriors (little data) are occasionally drawn high and get explored — a
principled Bayesian balance of exploration vs exploitation.

Priors are documented and configurable (Stage 3 evidence: "priors documentados").
"""

from __future__ import annotations

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy


class ThompsonSampling(Policy):
    name = "thompson"

    def __init__(
        self,
        arms: list[Arm],
        seed: int = 42,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__(arms, seed=seed)
        # Beta(1,1) = uniform prior on conversion. Weakly-informative by design.
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = dict.fromkeys(self.arm_ids, prior_alpha)
        self.beta = dict.fromkeys(self.arm_ids, prior_beta)

    def posterior_mean(self, arm_id: str) -> float:
        a, b = self.alpha[arm_id], self.beta[arm_id]
        return a / (a + b)

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        self.t += 1
        samples = {a: float(self.rng.beta(self.alpha[a], self.beta[a])) for a in elig}
        scores = {a: samples[a] * self.margin(a) for a in elig}
        estimates = {a: self.posterior_mean(a) for a in elig}
        best = max(scores, key=scores.get)
        # Exploration flag: chosen arm is not the current posterior-mean leader.
        greedy = max(elig, key=lambda a: estimates[a] * self.margin(a))
        explored = best != greedy
        codes = ["THOMPSON_SAMPLE"]
        codes.append("EXPLORATION" if explored else "EXPLOITATION")
        return Decision(
            arm_id=best, score=scores[best], explored=explored,
            reason_codes=codes, scores=scores, estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        # Bernoulli conjugacy: success -> +alpha, failure -> +beta.
        r = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm_id] += r
        self.beta[arm_id] += 1.0 - r

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "prior": {"alpha": self.prior_alpha, "beta": self.prior_beta},
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
        }
