"""Multi-armed bandit policies (Stage 3).

All policies implement the same :class:`~adaptive_offers.bandits.base.Policy`
contract so the simulator, evaluator and decision service can treat them
interchangeably:

* ``baseline``     — deterministic optimistic-greedy control (no active exploration).
* ``thompson``     — Beta-Bernoulli Thompson Sampling (Bayesian, context-free).
* ``nilos_ucb``    — variance-aware UCB family member (UCB-V style).
* ``linucb``       — contextual LinUCB (disjoint linear model per arm).
* ``lin_thompson`` — contextual Thompson Sampling (Bayesian linear, Agrawal & Goyal 2013).

Learning signal is the **conversion** (Bernoulli) per arm; business ranking is
margin-weighted (``estimate × margin``), so policies optimise expected *value*.
"""

from __future__ import annotations

from adaptive_offers.bandits.base import Arm, Decision, Policy
from adaptive_offers.bandits.baseline import BaselineGreedy
from adaptive_offers.bandits.linucb import LinUCB
from adaptive_offers.bandits.registry import POLICIES, build_policy
from adaptive_offers.bandits.thompson import ThompsonSampling
from adaptive_offers.bandits.thompson_linear import LinThompson
from adaptive_offers.bandits.ucb import NilosUCB

__all__ = [
    "Arm",
    "Decision",
    "Policy",
    "BaselineGreedy",
    "ThompsonSampling",
    "NilosUCB",
    "LinUCB",
    "LinThompson",
    "build_policy",
    "POLICIES",
]
