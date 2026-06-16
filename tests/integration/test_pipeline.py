"""End-to-end smoke test of the offline pipeline (small horizon)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_adaptive_beats_baseline():
    """At scale, adaptive policies must capture more value than the control.

    Built in-memory (no disk writes) at a horizon where the asymptotic advantage
    is robust — at very small horizons the optimistic-greedy baseline can win by
    chance, which is itself the point: exploration pays off over time.
    """
    from adaptive_offers.data.loader import load_raw
    from adaptive_offers.data.preprocessing import preprocess
    from adaptive_offers.data.synthetic import generate_events, offer_catalog
    from adaptive_offers.evaluation.offline_eval import metrics_matrix

    processed = preprocess(load_raw(n_rows=4000, seed=42)[0])
    bundle = generate_events(processed, offer_catalog(), seed=42)
    rows = metrics_matrix(processed, bundle, horizon=4000, seed=123)
    by_policy = {r["policy"]: r for r in rows}

    assert by_policy["linucb"]["cumulative_reward"] > by_policy["baseline"]["cumulative_reward"]
    assert rows[0]["policy"] != "baseline", "the best policy must be adaptive"


def test_golden_evaluation_runs_end_to_end(trained_service, small_layer):
    from adaptive_offers.config import get_settings
    from adaptive_offers.evaluation.golden_set import evaluate_golden, load_cases

    _, bundle = small_layer
    cases = load_cases(get_settings().paths.golden_set / "evaluation_cases.jsonl")
    report = evaluate_golden(cases, trained_service.policy, bundle.catalog, bundle.rate_median)
    assert report["n_cases"] >= 20
    assert 0.0 <= report["pass_rate"] <= 1.0


def test_ips_estimate_is_finite(trained_service, small_layer):
    from adaptive_offers.evaluation.offline_eval import ips_estimate

    processed, bundle = small_layer
    est = ips_estimate(trained_service.policy, processed.head(800),
                       type(bundle)(catalog=bundle.catalog, events=bundle.events.head(800),
                                    delayed=bundle.delayed, rate_median=bundle.rate_median,
                                    seed=bundle.seed, contexts=bundle.contexts[:800]))
    assert est["v_ips_per_impression"] >= 0.0
