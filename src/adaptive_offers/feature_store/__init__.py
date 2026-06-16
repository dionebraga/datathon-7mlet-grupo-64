"""Feature Store — offline (Parquet) + online (SQLite) feature serving.

Implements the course's Feature Store module: entities and feature views
(definitions), an offline store for training/backfills and a low-latency online
store for serving, plus materialisation that keeps them consistent. This is what
guarantees **train/serving parity**: the same context features used to train the
bandits are the ones served to the decision API.
"""

from __future__ import annotations

from adaptive_offers.feature_store.definitions import (
    CLIENT_ENTITY,
    OFFER_ENTITY,
    FeatureView,
    client_feature_view,
    offer_feature_view,
)
from adaptive_offers.feature_store.store import FeatureStore

__all__ = [
    "FeatureStore",
    "FeatureView",
    "CLIENT_ENTITY",
    "OFFER_ENTITY",
    "client_feature_view",
    "offer_feature_view",
]
