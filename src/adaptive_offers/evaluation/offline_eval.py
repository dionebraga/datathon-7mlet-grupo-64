"""Offline policy evaluation: metrics matrix, sensitivity and off-policy IPS.

* :func:`train_frozen_policy` — fit a policy on the synthetic stream and freeze it.
* :func:`metrics_matrix`      — reward/regret/exploration table across policies.
* :func:`sensitivity_analysis`— stability of results under seed perturbation.
* :func:`ips_estimate`        — Inverse-Propensity-Scoring off-policy value from
  the logged events (uses recorded propensities), with the self-normalised SNIPS
  variant for lower variance.
"""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np
import pandas as pd

from adaptive_offers.bandits.base import Policy
from adaptive_offers.bandits.registry import build_policy
from adaptive_offers.data.synthetic import (
    CONTEXT_FEATURES,
    SyntheticBundle,
    eligible_arms,
)
from adaptive_offers.simulation.environment import build_arms, run_simulation
from adaptive_offers.simulation.metrics import compare_results


def train_frozen_policy(
    name: str,
    processed: pd.DataFrame,
    bundle: SyntheticBundle,
    horizon: int | None = None,
    seed: int = 42,
) -> Policy:
    """Fit ``name`` on the synthetic stream and return the frozen policy."""
    arms = build_arms(bundle.catalog)
    policy = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
    run_simulation(policy, processed, bundle, horizon=horizon, seed=seed,
                   delayed_fraction=bundle.events["reward_is_delayed"].mean()
                   if not bundle.events.empty else 0.4)
    return policy


def metrics_matrix(
    processed: pd.DataFrame,
    bundle: SyntheticBundle,
    policy_names: tuple[str, ...] = ("baseline", "thompson", "nilos_ucb", "linucb"),
    horizon: int | None = None,
    seed: int = 123,
) -> list[dict[str, Any]]:
    """Run each policy once and return the sorted comparison table."""
    arms = build_arms(bundle.catalog)
    results = []
    for name in policy_names:
        pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
        results.append(run_simulation(pol, processed, bundle, horizon=horizon, seed=seed))
    return compare_results(results)


def sensitivity_analysis(
    processed: pd.DataFrame,
    bundle: SyntheticBundle,
    policy_name: str = "linucb",
    seeds: tuple[int, ...] = (1, 7, 21, 42, 123),
    horizon: int | None = None,
) -> dict[str, Any]:
    """Report mean/std of reward & regret over seeds (robustness check)."""
    arms = build_arms(bundle.catalog)
    rewards, regrets = [], []
    for s in seeds:
        pol = build_policy(policy_name, arms, context_dim=len(CONTEXT_FEATURES), seed=s)
        res = run_simulation(pol, processed, bundle, horizon=horizon, seed=s)
        rewards.append(float(res.cumulative_reward[-1]))
        regrets.append(float(res.cumulative_regret[-1]))
    return {
        "policy": policy_name,
        "seeds": list(seeds),
        "reward_mean": round(statistics.mean(rewards), 1),
        "reward_std": round(statistics.pstdev(rewards), 1),
        "reward_cv": round(statistics.pstdev(rewards) / statistics.mean(rewards), 4),
        "regret_mean": round(statistics.mean(regrets), 1),
        "regret_std": round(statistics.pstdev(regrets), 1),
    }


def ips_estimate(
    policy: Policy,
    processed: pd.DataFrame,
    bundle: SyntheticBundle,
) -> dict[str, float]:
    """Off-policy value of ``policy`` via IPS and SNIPS on the logged events.

    V_IPS  = mean_i [ 1{π(x_i)=a_i} · r_i / p_i ]
    V_SNIPS= Σ_i w_i r_i / Σ_i w_i ,  w_i = 1{π(x_i)=a_i}/p_i
    where (a_i, p_i, r_i) come from the logging policy in ``offer_events``.
    """
    events = bundle.events
    catalog = bundle.catalog
    contexts = bundle.contexts
    n = len(events)
    weighted_rewards, weights, matches = [], [], 0
    for i in range(n):
        ev = events.iloc[i]
        row = processed.iloc[i]
        elig = [a.offer_id for a in eligible_arms(row, catalog)]
        chosen = policy.select(contexts[i], elig).arm_id
        if chosen == ev["offer_id"]:
            matches += 1
            w = 1.0 / float(ev["propensity"])
            weights.append(w)
            weighted_rewards.append(w * float(ev["reward"]))
    v_ips = float(np.sum(weighted_rewards) / n) if n else 0.0
    v_snips = float(np.sum(weighted_rewards) / np.sum(weights)) if weights else 0.0
    return {
        "policy": policy.name,
        "v_ips_per_impression": round(v_ips, 3),
        "v_snips_per_impression": round(v_snips, 3),
        "match_rate": round(matches / n, 4) if n else 0.0,
        "effective_sample": matches,
    }
