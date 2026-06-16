"""Leakage-free preprocessing of the factual base (Stage 1).

Transforms the raw Bank Marketing frame into a processed, modelling-ready table:

* drops temporal-leakage / post-contact columns (``duration``);
* turns the ``pdays == 999`` sentinel into an explicit ``previously_contacted``
  flag plus a clean numeric ``pdays_since`` (0 when never contacted);
* binarises the target ``y`` -> ``subscribed`` (1 = subscribed term deposit);
* writes the processed parquet and the provenance record.

The processed table is the single source of truth consumed by the feature store
and the bandit simulator.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adaptive_offers.config import get_settings
from adaptive_offers.data.loader import DatasetProvenance, load_raw
from adaptive_offers.data.schema import (
    LEAKAGE_COLUMNS,
    PDAYS_NOT_CONTACTED_SENTINEL,
    TARGET_COLUMN,
)
from adaptive_offers.logging_utils import get_logger, log_event

logger = get_logger("data.preprocessing")

PROCESSED_FILE = "bank_marketing_processed.parquet"
PROVENANCE_FILE = "provenance.json"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented leakage-free transformations to a raw frame."""
    out = df.copy()

    # 1) Drop leakage / post-contact columns.
    leakage_present = [c for c in LEAKAGE_COLUMNS if c in out.columns]
    out = out.drop(columns=leakage_present)

    # 2) pdays sentinel -> explicit flag + clean numeric.
    if "pdays" in out.columns:
        out["previously_contacted"] = (out["pdays"] != PDAYS_NOT_CONTACTED_SENTINEL).astype(int)
        out["pdays_since"] = out["pdays"].where(
            out["pdays"] != PDAYS_NOT_CONTACTED_SENTINEL, 0
        ).astype(int)
        out = out.drop(columns=["pdays"])

    # 3) Binarise target.
    if TARGET_COLUMN in out.columns:
        out["subscribed"] = (
            out[TARGET_COLUMN].astype(str).str.strip().str.lower().eq("yes").astype(int)
        )
        out = out.drop(columns=[TARGET_COLUMN])

    # 4) Normalise string categoricals (trim/lowercase) for stable joins.
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip().str.lower()

    # 5) Stable surrogate key for joining the synthetic enrichment layer.
    out = out.reset_index(drop=True)
    out.insert(0, "client_event_id", [f"ce_{i:07d}" for i in range(len(out))])
    return out


def build_processed(
    n_rows: int = 20_000,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, DatasetProvenance, Path]:
    """End-to-end Stage 1: load raw -> preprocess -> persist processed + provenance."""
    settings = get_settings()
    out_dir = out_dir or settings.paths.processed
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, provenance = load_raw(n_rows=n_rows, seed=seed)
    processed = preprocess(raw)

    processed_path = out_dir / PROCESSED_FILE
    processed.to_parquet(processed_path, index=False)
    provenance.to_json(out_dir / PROVENANCE_FILE)

    log_event(
        logger,
        "processed_built",
        source=provenance.source,
        rows=int(processed.shape[0]),
        cols=int(processed.shape[1]),
        target_rate=round(float(processed["subscribed"].mean()), 4),
        dropped_leakage=LEAKAGE_COLUMNS,
        path=str(processed_path),
    )
    return processed, provenance, processed_path


def load_processed(out_dir: Path | None = None) -> pd.DataFrame:
    """Load the persisted processed table (building it if missing)."""
    settings = get_settings()
    out_dir = out_dir or settings.paths.processed
    path = out_dir / PROCESSED_FILE
    if not path.exists():
        logger.info("processed table missing; building it now")
        build_processed(out_dir=out_dir)
    return pd.read_parquet(path)
