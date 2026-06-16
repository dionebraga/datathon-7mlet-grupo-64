"""Central configuration and path management.

All tunable behaviour is funnelled through :class:`Settings` so the pipeline,
API and tests share one source of truth. Values are read from environment
variables (optionally loaded from a local ``.env``) with safe defaults, so the
project runs end-to-end with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv() -> None:
    """Load a local ``.env`` if python-dotenv is available. Never fails."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    env_path = _project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _project_root() -> Path:
    # src/adaptive_offers/config.py -> project root is three parents up.
    return Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the versioned data layer and run artifacts."""

    root: Path
    data: Path
    kaggle: Path
    processed: Path
    synthetic: Path
    golden_set: Path
    reports: Path
    artifacts: Path

    @staticmethod
    def build(root: Path, data_dirname: str, artifacts_dirname: str) -> Paths:
        data = root / data_dirname
        return Paths(
            root=root,
            data=data,
            kaggle=data / "kaggle",
            processed=data / "processed",
            synthetic=data / "synthetic_enrichment",
            golden_set=data / "golden_set",
            reports=root / "reports",
            artifacts=root / artifacts_dirname,
        )

    def ensure(self) -> Paths:
        """Create writable directories that are not version-controlled."""
        for p in (self.processed, self.synthetic, self.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    """Immutable, environment-driven application settings."""

    app_env: str = field(default_factory=lambda: _env("APP_ENV", "local"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))

    mlflow_tracking_uri: str = field(
        default_factory=lambda: _env("MLFLOW_TRACKING_URI", "file:./mlruns")
    )
    mlflow_experiment: str = field(
        default_factory=lambda: _env("MLFLOW_EXPERIMENT_NAME", "adaptive-offers")
    )

    policy_default: str = field(default_factory=lambda: _env("POLICY_DEFAULT", "thompson"))
    policy_version: str = field(default_factory=lambda: _env("POLICY_VERSION", "v1"))
    exploration_floor: float = field(
        default_factory=lambda: _env_float("EXPLORATION_FLOOR", 0.02)
    )

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "offline"))
    anthropic_model: str = field(
        default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-8")
    )

    api_host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    paths: Paths = field(init=False)

    def __post_init__(self) -> None:
        root = _project_root()
        paths = Paths.build(
            root=root,
            data_dirname=_env("DATA_DIR", "data"),
            artifacts_dirname=_env("ARTIFACTS_DIR", "artifacts"),
        )
        object.__setattr__(self, "paths", paths)

        # Resolve a relative file tracking URI (e.g. "file:./mlruns") to an
        # ABSOLUTE path under the project root, so MLflow always logs to the same
        # place regardless of the current working directory (avoids "no runs"
        # when training and `mlflow ui` are launched from different folders).
        uri = self.mlflow_tracking_uri
        if uri.startswith("file:") and not uri.startswith(("file:/", "file:\\")):
            rel = uri[len("file:") :].lstrip("./").lstrip(".\\").strip() or "mlruns"
            object.__setattr__(self, "mlflow_tracking_uri", (root / rel).as_uri())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    _load_dotenv()
    return Settings()
