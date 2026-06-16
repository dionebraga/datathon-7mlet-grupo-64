"""Nilos-UCB — a variance-aware UCB-family policy (UCB-V style).

The Datathon asks for an explicit reference/justification of *Nilos-UCB* within
the UCB family. We implement it as a **variance-aware Upper Confidence Bound**
(Audibert, Munos & Szepesvári, 2009 — "UCB-V"), which tightens the classic UCB1
bonus using the empirically observed reward variance. This is justified because
conversion variance differs sharply across offers (margins and base rates vary),
so a variance-aware bonus explores high-uncertainty arms more efficiently than
the fixed UCB1 bonus.

Index for arm ``a`` at round ``t`` (n_a pulls, mean μ_a, variance v_a):

    UCB1   :  μ_a + c · sqrt( 2 · ln(t) / n_a )
    Nilos  :  μ_a + sqrt( 2 · v_a · ln(t) / n_a ) + c · 3 · ln(t) / n_a

Both are ranked margin-weighted: ``index_a × margin_a``. Unpulled arms get an
infinite index to force one initial pull each (cold-start handling).
"""

from __future__ import annotations

import math

import numpy as np

from adaptive_offers.bandits.base import Arm, Decision, Policy


class NilosUCB(Policy):
    name = "nilos_ucb"

    def __init__(
        self,
        arms: list[Arm],
        seed: int = 42,
        exploration_c: float = 1.0,
        variant: str = "nilos",  # "nilos" (UCB-V) | "ucb1"
    ) -> None:
        super().__init__(arms, seed=seed)
        self.c = exploration_c
        self.variant = variant
        self.counts = {a: 0 for a in self.arm_ids}
        self.mean = {a: 0.0 for a in self.arm_ids}
        self._m2 = {a: 0.0 for a in self.arm_ids}  # Welford sum of squares

    def variance(self, arm_id: str) -> float:
        n = self.counts[arm_id]
        return self._m2[arm_id] / n if n > 1 else 0.0

    def _index(self, arm_id: str, t: int) -> float:
        n = self.counts[arm_id]
        if n == 0:
            return math.inf  # force initial exploration of every arm
        ln_t = math.log(max(t, 2))
        mu = self.mean[arm_id]
        if self.variant == "ucb1":
            return mu + self.c * math.sqrt(2.0 * ln_t / n)
        # UCB-V (Nilos): variance-aware bonus + higher-order correction term.
        v = self.variance(arm_id)
        return mu + math.sqrt(2.0 * v * ln_t / n) + self.c * 3.0 * ln_t / n

    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        elig = self._check_eligible(eligible)
        self.t += 1
        indices = {a: self._index(a, self.t) for a in elig}
        scores = {a: (indices[a] if math.isfinite(indices[a]) else 1.0) * self.margin(a)
                  for a in elig}
        estimates = {a: self.mean[a] for a in elig}
        # Prefer an unpulled arm first (cold-start), else margin-weighted index.
        unpulled = [a for a in elig if self.counts[a] == 0]
        if unpulled:
            best = max(unpulled, key=self.margin)
            explored = True
            codes = ["NILOS_UCB", "COLD_START_PULL"]
        else:
            best = max(scores, key=scores.get)
            greedy = max(elig, key=lambda a: self.mean[a] * self.margin(a))
            explored = best != greedy
            codes = ["NILOS_UCB", "EXPLORATION" if explored else "EXPLOITATION"]
        return Decision(
            arm_id=best, score=scores[best], explored=explored,
            reason_codes=codes, scores=scores, estimates=estimates,
        )

    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        r = float(reward)
        self.counts[arm_id] += 1
        n = self.counts[arm_id]
        delta = r - self.mean[arm_id]
        self.mean[arm_id] += delta / n
        self._m2[arm_id] += delta * (r - self.mean[arm_id])

    def state_dict(self) -> dict:
        return {
            **super().state_dict(),
            "variant": self.variant, "c": self.c,
            "counts": dict(self.counts), "mean": dict(self.mean),
        }
