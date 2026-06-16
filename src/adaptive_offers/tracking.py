"""MLflow experiment tracking wrapper (used by training & evaluation).

Thin, fail-safe helper so the pipeline always runs even if MLflow is misconfigured
(e.g. offline CI). When available, it records params, metrics and tags under the
configured experiment, satisfying the Datathon's experiment-tracking requirement.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from adaptive_offers.config import get_settings
from adaptive_offers.logging_utils import get_logger

logger = get_logger("tracking")


@contextmanager
def mlflow_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager yielding an MLflow run, or ``None`` if unavailable."""
    # MLflow 3.x puts the local file store in "maintenance mode" and raises by
    # default; opt back in so the ``file:./mlruns`` backend keeps working and
    # `mlflow ui` can read the runs. Set before importing mlflow.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except Exception:  # pragma: no cover
        logger.info('{"event": "mlflow_unavailable"}')
        yield None
        return

    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as exc:  # pragma: no cover - never break the pipeline
        logger.info('{"event": "mlflow_error", "err": "%s"}', exc)
        yield None


def log_params(params: dict[str, Any]) -> None:
    try:
        import mlflow

        mlflow.log_params(params)
    except Exception:  # pragma: no cover
        pass


def log_metrics(metrics: dict[str, Any]) -> None:
    try:
        import mlflow

        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            mlflow.log_metrics(numeric)
    except Exception:  # pragma: no cover
        pass
