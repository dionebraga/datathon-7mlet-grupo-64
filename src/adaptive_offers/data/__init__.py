"""Data layer: source mapping, loading, leakage-free preprocessing.

This package implements Stage 1 of the Datathon: it turns a Kaggle-compatible
factual base (Bank Marketing) into a traceable, leakage-free processed dataset,
recording source / version / license provenance along the way.
"""

from __future__ import annotations

from adaptive_offers.data.loader import DatasetProvenance, load_raw
from adaptive_offers.data.preprocessing import build_processed
from adaptive_offers.data.schema import (
    CATEGORICAL_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

__all__ = [
    "DatasetProvenance",
    "load_raw",
    "build_processed",
    "CATEGORICAL_COLUMNS",
    "NUMERIC_COLUMNS",
    "LEAKAGE_COLUMNS",
    "TARGET_COLUMN",
]
