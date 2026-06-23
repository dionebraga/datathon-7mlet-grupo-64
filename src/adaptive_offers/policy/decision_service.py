"""DecisionService — auditable, guardrailed policy serving (Stage 5).

Receives a context (or a ``client_event_id`` resolved via the feature store),
applies the eligibility/suitability gate, asks the active policy for an arm and
returns a fully auditable record: chosen arm, score, reason codes, policy
version and the eligible set. Every decision is appended to an audit log.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adaptive_offers.bandits.base import Policy
from adaptive_offers.channels import DEFAULT_CONTACT_POLICY
from adaptive_offers.config import get_settings
from adaptive_offers.data.synthetic import (
    OfferArm,
    build_context_vector,
    eligible_arms,
    expected_reward,
    offer_catalog,
)
from adaptive_offers.feature_store.store import FeatureStore
from adaptive_offers.logging_utils import get_logger, utc_now_iso
from adaptive_offers.policy.reason_codes import enrich
from adaptive_offers.policy.versioning import PolicyMetadata, load_policy
from adaptive_offers.segmentation import segment_of

logger = get_logger("policy.decision_service")

CONTROL_ARM = "OFF_NONE"


@dataclass
class DecisionRecord:
    """Auditable result of a single decision."""

    decision_id: str
    ts: str
    client_event_id: str | None
    arm_id: str
    arm_name: str
    score: float
    expected_reward: float
    explored: bool
    policy_name: str
    policy_version: str
    eligible_arms: list[str]
    reason_codes: list[str]
    reasons: list[dict[str, str]] = field(default_factory=list)
    estimates: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    segment_id: str = ""
    segment_label: str = ""
    channel_id: str = ""
    channel_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionService:
    def __init__(
        self,
        policy: Policy,
        metadata: PolicyMetadata,
        feature_store: FeatureStore | None = None,
        catalog: list[OfferArm] | None = None,
        audit_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.metadata = metadata
        self.fs = feature_store or FeatureStore()
        self.catalog = catalog or offer_catalog()
        self.by_id = {a.offer_id: a for a in self.catalog}
        settings = get_settings()
        self.exploration_floor = settings.exploration_floor
        self.audit_path = audit_path or (settings.paths.artifacts / "decisions" / "audit.jsonl")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    @classmethod
    def from_active(cls, version: str | None = None) -> DecisionService:
        """Build a service from the active (or given) saved policy version."""
        policy, meta = load_policy(version)
        return cls(policy=policy, metadata=meta)

    # --- core decision ------------------------------------------------------
    def decide(
        self,
        context: dict[str, Any] | None = None,
        client_event_id: str | None = None,
        log: bool = True,
    ) -> DecisionRecord:
        features, cid, rate_median = self._resolve_features(context, client_event_id)

        all_ids = [a.offer_id for a in self.catalog]
        elig = [a.offer_id for a in eligible_arms(features, self.catalog)]
        ctx_vec = build_context_vector(features, rate_median)

        decision = self.policy.select(ctx_vec, elig)
        codes = list(decision.reason_codes)
        codes.append("MARGIN_WEIGHTED")
        if len(elig) < len(all_ids):
            codes.append("ELIGIBILITY_FILTERED")
        codes.append("SUITABILITY_OK")
        if decision.arm_id == CONTROL_ARM:
            codes.append("CONTROL_FALLBACK")

        arm = self.by_id[decision.arm_id]
        exp_rew = expected_reward(arm, ctx_vec)
        seg = segment_of(features)

        # Channel orchestration: the bandit picked *what* to offer; the contact
        # policy decides *where/whether* to deliver it. The control arm (no offer)
        # has nothing to deliver, so contact is suppressed by construction.
        if arm.offer_id == CONTROL_ARM:
            channel, ch_codes = None, ["CONTACT_SUPPRESSED"]
        else:
            # High-value / complex products (margin >= R$150: e.g. Seguro Bundle,
            # Fundo, Crédito) warrant a rich-creative channel.
            channel, ch_codes = DEFAULT_CONTACT_POLICY.select(
                features, offer_rich=float(arm.margin) >= 150.0
            )
        codes.extend(ch_codes)

        self._counter += 1
        record = DecisionRecord(
            decision_id=f"dec_{self._counter:08d}",
            ts=utc_now_iso(),
            client_event_id=cid,
            arm_id=arm.offer_id,
            arm_name=arm.name,
            score=round(float(decision.score), 4),
            expected_reward=round(float(exp_rew), 3),
            explored=bool(decision.explored),
            policy_name=self.policy.name,
            policy_version=self.metadata.version,
            eligible_arms=elig,
            reason_codes=codes,
            reasons=enrich(codes),
            estimates={k: round(float(v), 4) for k, v in decision.estimates.items()},
            scores={k: round(float(v), 4) for k, v in (decision.scores or {}).items()},
            segment_id=seg.seg_id,
            segment_label=seg.label,
            channel_id=channel.channel_id if channel else "",
            channel_label=channel.label if channel else "—",
        )
        if log:
            self._audit(record)
        return record

    # --- helpers ------------------------------------------------------------
    def _resolve_features(
        self, context: dict[str, Any] | None, client_event_id: str | None
    ) -> tuple[dict[str, Any], str | None, float]:
        if context is not None:
            rate_median = float(self.fs.get_metadata("rate_median", 2.5)) if self.fs.is_materialized() else 2.5
            cid = context.get("client_event_id", client_event_id)
            return context, cid, rate_median
        if client_event_id is not None:
            if not self.fs.is_materialized():
                raise RuntimeError(
                    "feature store not materialised; run `adaptive-offers data build` "
                    "and materialisation, or pass an explicit context"
                )
            feats = self.fs.get_online_features("client_features", client_event_id)
            rate_median = float(self.fs.get_metadata("rate_median", 2.5))
            return feats, client_event_id, rate_median
        raise ValueError("decide() requires either `context` or `client_event_id`")

    def _audit(self, record: DecisionRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, default=str)
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info('{"event": "decision", "decision_id": "%s", "arm": "%s", '
                    '"policy": "%s@%s", "explored": %s}',
                    record.decision_id, record.arm_id, record.policy_name,
                    record.policy_version, str(record.explored).lower())
