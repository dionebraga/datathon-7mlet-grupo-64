"""Factory for building policies by name (used by CLI, API and simulator)."""

from __future__ import annotations

from adaptive_offers.bandits.base import Arm, Policy
from adaptive_offers.bandits.baseline import BaselineGreedy
from adaptive_offers.bandits.linucb import LinUCB
from adaptive_offers.bandits.thompson import ThompsonSampling
from adaptive_offers.bandits.ucb import NilosUCB

POLICIES: tuple[str, ...] = ("baseline", "thompson", "nilos_ucb", "linucb")


def build_policy(
    name: str,
    arms: list[Arm],
    context_dim: int,
    seed: int = 42,
    **kwargs,
) -> Policy:
    """Instantiate a policy by name with consistent arms/seed/dim wiring."""
    name = name.lower()
    if name == "baseline":
        return BaselineGreedy(arms, seed=seed, **kwargs)
    if name == "thompson":
        return ThompsonSampling(arms, seed=seed, **kwargs)
    if name == "nilos_ucb":
        return NilosUCB(arms, seed=seed, **kwargs)
    if name == "linucb":
        return LinUCB(arms, dim=context_dim, seed=seed, **kwargs)
    raise ValueError(f"unknown policy '{name}'. Choose from {POLICIES}")
