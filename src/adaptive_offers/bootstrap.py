"""One-call bootstrap that wires the whole platform together.

Used by the CLI, the API and tests to guarantee the data layer, synthetic layer,
feature store and an active trained policy all exist — training one if needed.
This is the single source of truth for "is the system ready to serve?".
"""

from __future__ import annotations

import pandas as pd

from adaptive_offers.bandits.registry import build_policy
from adaptive_offers.config import get_settings
from adaptive_offers.data.preprocessing import build_processed, load_processed
from adaptive_offers.data.synthetic import CONTEXT_FEATURES, SyntheticBundle, generate
from adaptive_offers.feature_store.store import FeatureStore
from adaptive_offers.logging_utils import get_logger
from adaptive_offers.policy.decision_service import DecisionService
from adaptive_offers.policy.versioning import (
    PolicyMetadata,
    get_active_version,
    load_policy,
    save_policy,
)
from adaptive_offers.simulation.environment import build_arms, run_simulation
from adaptive_offers.simulation.metrics import summarize

logger = get_logger("bootstrap")


def ensure_data(n_rows: int = 20_000, seed: int | None = None) -> pd.DataFrame:
    """Ensure the processed base exists; build it if missing."""
    settings = get_settings()
    path = settings.paths.processed / "bank_marketing_processed.parquet"
    if not path.exists():
        build_processed(n_rows=n_rows, seed=seed)
    return load_processed()


def ensure_bundle(processed: pd.DataFrame | None = None, seed: int | None = None) -> SyntheticBundle:
    """Ensure (and return, in memory) the synthetic enrichment bundle."""
    processed = processed if processed is not None else ensure_data(seed=seed)
    return generate(processed=processed, seed=seed)


def ensure_feature_store(processed: pd.DataFrame, bundle: SyntheticBundle) -> FeatureStore:
    """Materialise the online feature store from the offline layers."""
    fs = FeatureStore()
    if not fs.is_materialized():
        catalog_df = pd.read_parquet(
            get_settings().paths.synthetic / "offer_catalog.parquet"
        )
        fs.materialize(processed=processed, catalog=catalog_df, rate_median=bundle.rate_median)
    return fs


def train_and_register(
    policy_name: str = "linucb",
    version: str = "v1",
    horizon: int = 12_000,
    seed: int | None = None,
) -> PolicyMetadata:
    """Train a policy on the synthetic stream and register it as active."""
    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    processed = ensure_data(seed=seed)
    bundle = ensure_bundle(processed, seed=seed)
    ensure_feature_store(processed, bundle)

    arms = build_arms(bundle.catalog)
    policy = build_policy(policy_name, arms, context_dim=len(CONTEXT_FEATURES), seed=seed)
    result = run_simulation(policy, processed, bundle, horizon=horizon, seed=seed,
                            delayed_fraction=0.4)
    meta = PolicyMetadata(
        name=policy.name, version=version,
        train_config={"horizon": horizon, "seed": seed, "policy": policy_name},
        metrics=summarize(result),
    )
    save_policy(policy, version=version, metadata=meta)
    logger.info('{"event": "policy_registered", "policy": "%s", "version": "%s"}',
                policy.name, version)
    return meta


def ensure_service(train_if_missing: bool = True, horizon: int = 12_000) -> DecisionService:
    """Return a ready DecisionService, training+registering a policy if needed."""
    if get_active_version() is None:
        if not train_if_missing:
            raise RuntimeError("no active policy; run `adaptive-offers train` first")
        train_and_register(horizon=horizon)
    # Ensure feature store is available for id-based decisions.
    processed = ensure_data()
    bundle = ensure_bundle(processed)
    ensure_feature_store(processed, bundle)
    policy, meta = load_policy()
    return DecisionService(policy=policy, metadata=meta, feature_store=FeatureStore(),
                           catalog=bundle.catalog)
