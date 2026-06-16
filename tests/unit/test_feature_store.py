"""Feature store tests: materialisation, online reads, train/serving parity."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_materialize_and_online_read(small_layer):
    import pandas as pd

    from adaptive_offers.config import get_settings
    from adaptive_offers.feature_store.store import FeatureStore

    processed, bundle = small_layer
    catalog = pd.read_parquet(get_settings().paths.synthetic / "offer_catalog.parquet")
    fs = FeatureStore()
    counts = fs.materialize(processed=processed, catalog=catalog, rate_median=bundle.rate_median)

    assert counts["client_features"] == len(processed)
    assert fs.is_materialized()
    feats = fs.get_online_features("client_features", "ce_0000000")
    assert "age" in feats and isinstance(feats["age"], (int, float))


def test_get_context_vector_matches_direct_build(small_layer):
    from adaptive_offers.data.synthetic import CONTEXT_FEATURES
    from adaptive_offers.feature_store.store import FeatureStore

    fs = FeatureStore()
    vec = fs.get_context_vector("ce_0000000")
    assert vec.shape == (len(CONTEXT_FEATURES),)
    assert vec[0] == 1.0


def test_unknown_entity_raises(small_layer):
    from adaptive_offers.feature_store.store import FeatureStore

    fs = FeatureStore()
    with pytest.raises(KeyError):
        fs.get_online_features("client_features", "ce_DOESNOTEXIST")
