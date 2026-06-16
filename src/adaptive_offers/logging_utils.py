"""Structured logging helpers.

Decision auditing (Stage 5) relies on consistent, machine-parseable logs, so we
expose a single :func:`get_logger` factory and a :func:`log_event` helper that
emits compact JSON lines suitable for ingestion by Azure Application Insights.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False


def _configure_root(level: str) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root = logging.getLogger("adaptive_offers")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers[:] = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a namespaced logger under the ``adaptive_offers`` root."""
    _configure_root(level)
    return logging.getLogger(f"adaptive_offers.{name}")


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp in ISO-8601 (used in audit records)."""
    return datetime.now(UTC).isoformat()


def log_event(logger: logging.Logger, event: str, **fields: Any) -> dict[str, Any]:
    """Emit a single structured JSON log line and return the record.

    The returned dict is reused as the audit payload by the decision service,
    so logging and auditing never drift apart.
    """
    record = {"event": event, "ts": utc_now_iso(), **fields}
    logger.info(json.dumps(record, ensure_ascii=False, default=str))
    return record
