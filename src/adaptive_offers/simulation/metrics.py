"""Bandit metrics: reward, regret, exploration and conversion.

These metrics are justified for the problem (Stage 3/4):

* **cumulative_reward** — total business value (margin-weighted) captured.
* **cumulative_regret** — expected value lost vs an oracle; the core bandit KPI.
* **conversion_rate**   — realized conversions / impressions.
* **exploration_rate**  — share of exploratory decisions (exploration/exploitation balance).
* **reward_per_1k**     — normalised value, comparable across horizons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from adaptive_offers.simulation.environment import SimulationResult


def summarize(result: "SimulationResult") -> dict[str, Any]:
    """Compact, JSON-serialisable summary of one policy run."""
    n = result.horizon
    cum_reward = float(result.cumulative_reward[-1])
    cum_regret = float(result.cumulative_regret[-1])
    oracle_total = float(result.expected_oracle.sum())
    return {
        "policy": result.policy_name,
        "rounds": n,
        "cumulative_reward": round(cum_reward, 2),
        "reward_per_1k": round(cum_reward / n * 1000, 2),
        "cumulative_regret": round(cum_regret, 2),
        "regret_per_1k": round(cum_regret / n * 1000, 2),
        "regret_ratio": round(cum_regret / oracle_total, 4) if oracle_total else None,
        "conversion_rate": round(float(result.converted.mean()), 4),
        "exploration_rate": round(float(result.explored.mean()), 4),
        "arm_pulls": dict(result.arm_pulls),
        "seed": result.seed,
        "delayed_fraction": result.delayed_fraction,
    }


def compare_results(results: list["SimulationResult"]) -> list[dict[str, Any]]:
    """Summaries for several policies, sorted by cumulative reward (desc)."""
    rows = [summarize(r) for r in results]
    rows.sort(key=lambda r: r["cumulative_reward"], reverse=True)
    # Relative lift vs the baseline policy, if present.
    base = next((r for r in rows if r["policy"] == "baseline"), None)
    if base and base["cumulative_reward"]:
        for r in rows:
            r["lift_vs_baseline_pct"] = round(
                (r["cumulative_reward"] - base["cumulative_reward"])
                / base["cumulative_reward"] * 100, 2
            )
    return rows


def regret_curve(result: "SimulationResult", points: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Down-sampled (round, cumulative_regret) curve for plotting."""
    cum = result.cumulative_regret
    idx = np.linspace(0, len(cum) - 1, min(points, len(cum))).astype(int)
    return idx, cum[idx]
