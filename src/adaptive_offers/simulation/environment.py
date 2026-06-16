"""Offline simulator that replays contexts against a policy.

Rewards come from the **same latent model** that generated the synthetic layer
(Stage 2), so the simulator knows the true expected reward of every arm and can
compute **pseudo-regret** against an oracle. It also models **delayed rewards**:
a fraction of feedback matures only after a delay, so the policy must decide with
partial information — and **cold-start** is reproduced naturally (no data at t=0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive_offers.bandits.base import Arm, Policy
from adaptive_offers.data.synthetic import (
    OfferArm,
    SyntheticBundle,
    build_context_vector,
    eligible_arms,
    expected_reward,
    latent_conversion_prob,
)
from adaptive_offers.logging_utils import get_logger

logger = get_logger("simulation.environment")


@dataclass
class SimulationResult:
    """Per-round trace and summary of a single policy run."""

    policy_name: str
    chosen_arms: list[str]
    realized_reward: np.ndarray
    expected_chosen: np.ndarray
    expected_oracle: np.ndarray
    explored: np.ndarray
    converted: np.ndarray
    seed: int
    horizon: int
    delayed_fraction: float
    arm_pulls: dict[str, int] = field(default_factory=dict)

    @property
    def instant_regret(self) -> np.ndarray:
        return self.expected_oracle - self.expected_chosen

    @property
    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum(self.realized_reward)

    @property
    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.instant_regret)


def build_arms(catalog: list[OfferArm]) -> list[Arm]:
    """Arm list (id + margin) consumed by every policy."""
    return [Arm(arm_id=a.offer_id, margin=a.margin) for a in catalog]


def precompute_eligibility(
    processed: pd.DataFrame, catalog: list[OfferArm]
) -> list[list[str]]:
    """Eligible arm-id list per processed row (eligibility is context-fixed)."""
    return [[a.offer_id for a in eligible_arms(row, catalog)] for _, row in processed.iterrows()]


def run_simulation(
    policy: Policy,
    processed: pd.DataFrame,
    bundle: SyntheticBundle,
    horizon: int | None = None,
    seed: int = 42,
    delayed_fraction: float = 0.4,
    max_delay: int = 30,
) -> SimulationResult:
    """Run one policy over ``horizon`` rounds and return its trace."""
    rng = np.random.default_rng(seed)
    catalog = bundle.catalog
    by_id: dict[str, OfferArm] = {a.offer_id: a for a in catalog}
    rate_median = bundle.rate_median

    n = len(processed)
    horizon = n if horizon is None else min(horizon, n)
    order = rng.permutation(n)[:horizon]

    # Precompute contexts + eligibility once (fast, vectorised inputs).
    contexts = (
        bundle.contexts
        if bundle.contexts.size
        else np.vstack([build_context_vector(processed.iloc[i], rate_median) for i in range(n)])
    )
    eligibility = precompute_eligibility(processed, catalog)

    chosen, realized, exp_chosen, exp_oracle, explored, converted = [], [], [], [], [], []
    pulls = dict.fromkeys(by_id, 0)
    pending: list[tuple[int, str, float, np.ndarray]] = []  # (mature_round, arm, reward, ctx)

    for round_idx, i in enumerate(order):
        # 1) Deliver matured (delayed) feedback before deciding.
        if pending:
            still: list[tuple[int, str, float, np.ndarray]] = []
            for mature_round, arm_id, rew, ctx in pending:
                if mature_round <= round_idx:
                    policy.update(arm_id, rew, ctx)
                else:
                    still.append((mature_round, arm_id, rew, ctx))
            pending = still

        x = contexts[i]
        elig = eligibility[i]

        # 2) Oracle (true expected best eligible arm) for pseudo-regret.
        oracle_arm = max(elig, key=lambda a: expected_reward(by_id[a], x))
        exp_oracle.append(expected_reward(by_id[oracle_arm], x))

        # 3) Policy decision.
        decision = policy.select(x, elig)
        arm_id = decision.arm_id
        pulls[arm_id] += 1
        chosen.append(arm_id)
        explored.append(bool(decision.explored))
        exp_chosen.append(expected_reward(by_id[arm_id], x))

        # 4) Sample realized outcome from the latent model.
        p_conv = latent_conversion_prob(by_id[arm_id], x)
        conv = int(rng.random() < p_conv)
        converted.append(conv)
        realized.append(by_id[arm_id].margin * conv)

        # 5) Schedule learning signal (immediate or delayed). Signal = conversion.
        if conv and rng.random() < delayed_fraction:
            delay = int(rng.integers(1, max_delay))
            pending.append((round_idx + delay, arm_id, float(conv), x))
        else:
            policy.update(arm_id, float(conv), x)

    # Flush remaining delayed feedback so final policy state is complete.
    for _, arm_id, rew, ctx in pending:
        policy.update(arm_id, rew, ctx)

    result = SimulationResult(
        policy_name=policy.name,
        chosen_arms=chosen,
        realized_reward=np.asarray(realized, dtype=float),
        expected_chosen=np.asarray(exp_chosen, dtype=float),
        expected_oracle=np.asarray(exp_oracle, dtype=float),
        explored=np.asarray(explored, dtype=bool),
        converted=np.asarray(converted, dtype=int),
        seed=seed,
        horizon=horizon,
        delayed_fraction=delayed_fraction,
        arm_pulls=pulls,
    )
    logger.info(
        '{"event": "simulation_done", "policy": "%s", "rounds": %d, '
        '"cum_reward": %.1f, "cum_regret": %.1f}',
        policy.name, horizon, float(result.cumulative_reward[-1]),
        float(result.cumulative_regret[-1]),
    )
    return result
