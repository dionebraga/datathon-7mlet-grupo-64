"""FeatureStore: offline (Parquet) reads + online (SQLite) low-latency serving.

* **Offline store** — the versioned Parquet layers (processed base, offer
  catalog). Used for training, backfills and historical (point-in-time) reads.
* **Online store** — a SQLite key-value projection of the latest feature values,
  materialised for millisecond reads at decision time.
* **Materialisation** — copies the latest features from offline to online and
  records run metadata (e.g. ``rate_median``) so served contexts match training.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adaptive_offers.config import get_settings
from adaptive_offers.feature_store.definitions import (
    FeatureView,
    all_feature_views,
    client_feature_view,
    offer_feature_view,
)
from adaptive_offers.logging_utils import get_logger, log_event

logger = get_logger("feature_store")


class FeatureStore:
    """Minimal offline+online feature store with train/serving parity."""

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self.paths = settings.paths
        self.db_path = db_path or (self.paths.artifacts / "feature_store" / "online.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.views = {v.name: v for v in all_feature_views()}

    # --- connection helper --------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # --- offline (historical) reads ----------------------------------------
    def get_historical_features(self, view_name: str, entity_ids: list[str] | None = None) -> pd.DataFrame:
        """Read a feature view from the offline (Parquet) layer."""
        view = self.views[view_name]
        if view.source == "processed":
            df = pd.read_parquet(self.paths.processed / "bank_marketing_processed.parquet")
            key = view.entity.join_key
        else:
            df = pd.read_parquet(self.paths.synthetic / "offer_catalog.parquet")
            key = view.entity.join_key
        cols = [key] + [f for f in view.features if f in df.columns]
        out = df[cols]
        if entity_ids is not None:
            out = out[out[key].isin(entity_ids)]
        return out.reset_index(drop=True)

    # --- materialisation ----------------------------------------------------
    def materialize(
        self,
        processed: pd.DataFrame | None = None,
        catalog: pd.DataFrame | None = None,
        rate_median: float | None = None,
    ) -> dict[str, int]:
        """Project the latest features from offline -> online (SQLite)."""
        if processed is None:
            processed = pd.read_parquet(
                self.paths.processed / "bank_marketing_processed.parquet"
            )
        if catalog is None:
            catalog = pd.read_parquet(self.paths.synthetic / "offer_catalog.parquet")
        if rate_median is None:
            rate_median = float(processed["euribor3m"].median())

        counts: dict[str, int] = {}
        with self._conn() as conn:
            counts["client_features"] = self._materialize_view(
                conn, client_feature_view(), processed
            )
            counts["offer_features"] = self._materialize_view(
                conn, offer_feature_view(), catalog
            )
            conn.execute("CREATE TABLE IF NOT EXISTS _metadata (k TEXT PRIMARY KEY, v TEXT)")
            conn.execute(
                "INSERT OR REPLACE INTO _metadata (k, v) VALUES (?, ?)",
                ("rate_median", json.dumps(rate_median)),
            )
            conn.commit()
        log_event(logger, "feature_store_materialized", counts=counts,
                  rate_median=round(rate_median, 4), db=str(self.db_path))
        return counts

    def _materialize_view(self, conn: sqlite3.Connection, view: FeatureView, df: pd.DataFrame) -> int:
        key = view.entity.join_key
        feats = [f for f in view.features if f in df.columns]
        cols = [key, *feats]
        table = view.name
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.execute(f'CREATE TABLE "{table}" ({col_defs}, PRIMARY KEY ("{key}"))')
        placeholders = ", ".join("?" for _ in cols)
        rows = [tuple(str(r[c]) for c in cols) for _, r in df[cols].iterrows()]
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table}" VALUES ({placeholders})', rows
        )
        return len(rows)

    # --- online (serving) reads --------------------------------------------
    def get_online_features(self, view_name: str, entity_id: str) -> dict[str, Any]:
        """Read the latest feature values for one entity (low-latency path)."""
        view = self.views[view_name]
        with self._conn() as conn:
            cur = conn.execute(
                f'SELECT * FROM "{view.name}" WHERE "{view.entity.join_key}" = ?',
                (entity_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(f"{entity_id!r} not found in online view '{view_name}'")
        return _coerce(dict(row))

    def get_metadata(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            try:
                cur = conn.execute("SELECT v FROM _metadata WHERE k = ?", (key,))
            except sqlite3.OperationalError:
                return default
            row = cur.fetchone()
        return json.loads(row["v"]) if row else default

    def is_materialized(self) -> bool:
        if not self.db_path.exists():
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='client_features'"
            )
            return cur.fetchone() is not None

    def get_context_vector(self, client_event_id: str) -> np.ndarray:
        """Build the bandit context vector for a client from online features."""
        from adaptive_offers.data.synthetic import build_context_vector

        feats = self.get_online_features("client_features", client_event_id)
        rate_median = float(self.get_metadata("rate_median", 2.5))
        return build_context_vector(feats, rate_median)


def _coerce(d: dict[str, Any]) -> dict[str, Any]:
    """SQLite stores TEXT; coerce numeric-looking values back to numbers."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                out[k] = float(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out
