"""Golden-set evaluation (Stage 4).

A golden set is a versioned list of decision cases — typical, edge, segment and
**adversarial** — each with an explicit context, an expected action, an expected
reward floor, a justification and a **pass/fail criterion**. Running a frozen
policy through them gives a reproducible, auditable quality gate.

Hard invariants checked on *every* case regardless of assertion:
* the chosen arm is **eligible** (suitability gate respected);
* the decision carries **reason codes** (explainability).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adaptive_offers.bandits.base import Policy
from adaptive_offers.data.synthetic import (
    OfferArm,
    build_context_vector,
    eligible_arms,
    expected_reward,
)


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    category: str
    description: str
    context: dict[str, Any]
    assertion: dict[str, Any]            # {"type": ..., "arm_ids": [...]}
    expected_reward_min: float
    justification: str
    pass_fail: str

    @staticmethod
    def from_dict(d: dict) -> GoldenCase:
        return GoldenCase(
            case_id=d["case_id"], category=d["category"], description=d["description"],
            context=d["context"], assertion=d["assertion"],
            expected_reward_min=float(d.get("expected_reward_min", 0.0)),
            justification=d.get("justification", ""), pass_fail=d.get("pass_fail", ""),
        )


def load_cases(path: Path) -> list[GoldenCase]:
    """Load cases from a JSON-Lines file (``evaluation_cases.jsonl``)."""
    cases: list[GoldenCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            cases.append(GoldenCase.from_dict(json.loads(line)))
    return cases


def _assertion_passed(assertion: dict, chosen: str, eligible: list[str]) -> bool:
    a_type = assertion["type"]
    arm_ids = assertion.get("arm_ids", [])
    if a_type == "choose_one_of":
        return chosen in arm_ids
    if a_type == "not_choose":
        return chosen not in arm_ids
    if a_type == "eligible_only":
        return chosen in eligible  # pure invariant case
    raise ValueError(f"unknown assertion type '{a_type}'")


def evaluate_case(
    case: GoldenCase,
    policy: Policy,
    catalog: list[OfferArm],
    rate_median: float,
) -> dict[str, Any]:
    """Evaluate one golden case against a (frozen) policy."""
    by_id = {a.offer_id: a for a in catalog}
    elig = [a.offer_id for a in eligible_arms(case.context, catalog)]
    ctx = build_context_vector(case.context, rate_median)
    decision = policy.select(ctx, elig)
    chosen = decision.arm_id

    eligibility_ok = chosen in elig
    reason_ok = len(decision.reason_codes) > 0
    assertion_ok = _assertion_passed(case.assertion, chosen, elig)
    exp_reward = expected_reward(by_id[chosen], ctx)
    reward_ok = exp_reward >= case.expected_reward_min

    passed = bool(eligibility_ok and reason_ok and assertion_ok and reward_ok)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "chosen": chosen,
        "eligible": elig,
        "expected_reward_chosen": round(float(exp_reward), 2),
        "expected_reward_min": case.expected_reward_min,
        "checks": {
            "eligibility_ok": eligibility_ok,
            "reason_codes_ok": reason_ok,
            "assertion_ok": assertion_ok,
            "reward_floor_ok": reward_ok,
        },
        "reason_codes": decision.reason_codes,
        "passed": passed,
    }


def evaluate_golden(
    cases: list[GoldenCase],
    policy: Policy,
    catalog: list[OfferArm],
    rate_median: float,
) -> dict[str, Any]:
    """Evaluate all cases and aggregate pass-rate overall and per category."""
    records = [evaluate_case(c, policy, catalog, rate_median) for c in cases]
    n = len(records)
    n_pass = sum(r["passed"] for r in records)
    by_cat: dict[str, dict[str, int]] = {}
    for r in records:
        c = r["category"]
        by_cat.setdefault(c, {"total": 0, "passed": 0})
        by_cat[c]["total"] += 1
        by_cat[c]["passed"] += int(r["passed"])
    return {
        "policy": policy.name,
        "n_cases": n,
        "n_passed": n_pass,
        "pass_rate": round(n_pass / n, 4) if n else 0.0,
        "by_category": by_cat,
        "failures": [r for r in records if not r["passed"]],
        "records": records,
    }
