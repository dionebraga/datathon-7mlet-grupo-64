"""Shared bandit abstractions: arms, decisions and the policy contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Arm:
    """A selectable arm (offer) with its profit margin."""

    arm_id: str
    margin: float


@dataclass
class Decision:
    """Outcome of a policy selection, carrying explainability metadata."""

    arm_id: str
    score: float                       # margin-weighted score of the chosen arm
    explored: bool                     # True if the choice was exploratory
    reason_codes: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # per-arm scores
    estimates: dict[str, float] = field(default_factory=dict)  # conversion estimates


class Policy(ABC):
    """Common contract for every bandit policy.

    The learning signal passed to :meth:`update` is the **conversion** in
    ``[0, 1]`` (Bernoulli). Policies hold each arm's margin and rank by
    ``estimate × margin`` so they optimise expected business value, not just
    raw conversion.
    """

    name: str = "policy"

    def __init__(self, arms: list[Arm], seed: int = 42) -> None:
        self.arms: dict[str, Arm] = {a.arm_id: a for a in arms}
        self.arm_ids: list[str] = [a.arm_id for a in arms]
        self.rng = np.random.default_rng(seed)
        self.t: int = 0

    def margin(self, arm_id: str) -> float:
        return self.arms[arm_id].margin

    def _check_eligible(self, eligible: list[str]) -> list[str]:
        elig = [a for a in eligible if a in self.arms]
        if not elig:
            raise ValueError("no eligible arms provided to policy.select")
        return elig

    @abstractmethod
    def select(self, context: np.ndarray, eligible: list[str]) -> Decision:
        """Choose an arm among ``eligible`` given a context vector."""

    @abstractmethod
    def update(self, arm_id: str, reward: float, context: np.ndarray | None = None) -> None:
        """Update internal state with an observed conversion (in ``[0, 1]``)."""

    # --- versioning / persistence (used by Stage 5 & 7) ---------------------
    def state_dict(self) -> dict:
        return {"name": self.name, "t": self.t, "arm_ids": self.arm_ids}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(arms={len(self.arm_ids)}, t={self.t})"
