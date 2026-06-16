"""Feature Store definitions: entities and feature views.

Mirrors the Feast-style mental model (entity → feature view → features) but with
a light, dependency-free implementation so it runs anywhere. Each feature view
declares its entity key, source layer, feature list and a **version**, enabling
feature versioning and train/serving parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    """A keyed business object features are attached to."""

    name: str
    join_key: str
    description: str


CLIENT_ENTITY = Entity(
    name="client_event",
    join_key="client_event_id",
    description="A contactable client-event row from the processed base.",
)

OFFER_ENTITY = Entity(
    name="offer",
    join_key="offer_id",
    description="A synthetic offer (bandit arm).",
)


@dataclass(frozen=True)
class FeatureView:
    """A named, versioned set of features for an entity."""

    name: str
    entity: Entity
    source: str               # "processed" | "offer_catalog"
    features: list[str]
    version: str = "v1"
    description: str = ""
    online: bool = True       # materialised to the online store for serving
    ttl_days: int = field(default=30)


def client_feature_view() -> FeatureView:
    """Leakage-free client features used to build the decision context."""
    return FeatureView(
        name="client_features",
        entity=CLIENT_ENTITY,
        source="processed",
        features=[
            "age", "contact", "poutcome", "previously_contacted",
            "euribor3m", "default", "loan", "job", "marital", "education",
        ],
        version="v1",
        description="Client/context features for the contextual bandit.",
    )


def offer_feature_view() -> FeatureView:
    """Offer/arm features used for eligibility and margin-weighted ranking."""
    return FeatureView(
        name="offer_features",
        entity=OFFER_ENTITY,
        source="offer_catalog",
        features=[
            "name", "category", "margin", "suitability_tier",
            "requires_no_default", "requires_no_loan", "min_age",
        ],
        version="v1",
        description="Offer catalog features (margins, eligibility, suitability).",
    )


def all_feature_views() -> list[FeatureView]:
    return [client_feature_view(), offer_feature_view()]
