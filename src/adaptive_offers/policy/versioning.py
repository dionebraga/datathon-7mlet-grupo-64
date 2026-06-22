"""Policy versioning and persistence.

A trained policy is saved with a metadata record (name, version, training config,
metrics, content hash). This underpins the MLOps promotion/rollback flow
(Stage 7): a new policy version can be trained, evaluated, approved and promoted
or rolled back by swapping the active version pointer.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptive_offers.bandits.base import Policy
from adaptive_offers.config import get_settings


@dataclass
class PolicyMetadata:
    name: str
    version: str
    trained_on: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    train_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


def _policy_dir(version: str) -> Path:
    return get_settings().paths.artifacts / "policies" / version


def save_policy(policy: Policy, version: str, metadata: PolicyMetadata | None = None) -> Path:
    """Persist a trained policy + metadata under ``artifacts/policies/<version>``."""
    pdir = _policy_dir(version)
    pdir.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(policy)
    (pdir / "policy.pkl").write_bytes(blob)

    meta = metadata or PolicyMetadata(name=policy.name, version=version)
    meta.content_hash = hashlib.sha256(blob).hexdigest()[:16]
    meta.train_config.setdefault("state", policy.state_dict())
    (pdir / "metadata.json").write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    _set_active(version)
    return pdir


def load_policy(version: str | None = None) -> tuple[Policy, PolicyMetadata]:
    """Load a policy by version (or the active version if ``None``)."""
    version = version or get_active_version()
    if version is None:
        raise FileNotFoundError("no active policy version; train and save one first")
    pdir = _policy_dir(version)
    policy: Policy = pickle.loads((pdir / "policy.pkl").read_bytes())
    meta_raw = json.loads((pdir / "metadata.json").read_text(encoding="utf-8"))
    return policy, PolicyMetadata(**meta_raw)


def _registry_path() -> Path:
    return get_settings().paths.artifacts / "policies" / "registry.json"


def _set_active(version: str) -> None:
    reg = _read_registry()
    reg["active"] = version
    reg.setdefault("history", [])
    if version not in reg["history"]:
        reg["history"].append(version)
    _registry_path().parent.mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _read_registry() -> dict[str, Any]:
    p = _registry_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def get_active_version() -> str | None:
    return _read_registry().get("active")


def promote(version: str) -> None:
    """Promote a version to active (used by the MLOps approval gate)."""
    _set_active(version)


def rollback() -> str | None:
    """Roll back to the previous version in history; returns the new active."""
    reg = _read_registry()
    history = reg.get("history", [])
    if len(history) < 2:
        return reg.get("active")
    history.pop()  # drop current
    previous = history[-1]
    reg["active"] = previous
    _registry_path().write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return previous


def list_versions() -> list[dict[str, Any]]:
    """Return metadata dicts for all versioned policies saved in artifacts/policies/."""
    base = get_settings().paths.artifacts / "policies"
    if not base.exists():
        return []
    result = []
    for meta_file in sorted(base.glob("*/metadata.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            result.append({
                "version": data.get("version", meta_file.parent.name),
                "policy": data.get("name", "unknown"),
                "trained_on": data.get("trained_on"),
                "metrics": data.get("metrics", {}),
            })
        except Exception:
            continue
    return result


def set_active_version(policy: str, version: str) -> None:
    """Switch the active policy to the given policy/version combination.

    Validates that the version directory exists before promoting.
    Raises ``FileNotFoundError`` if the version has not been saved yet.
    """
    pdir = _policy_dir(version)
    if not (pdir / "policy.pkl").exists():
        raise FileNotFoundError(
            f"Versão '{version}' não encontrada em artifacts/policies/. "
            "Treine e salve a política antes de ativá-la."
        )
    meta_path = pdir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_policy = meta.get("name", "")
        if stored_policy and stored_policy != policy:
            raise ValueError(
                f"Versão '{version}' corresponde à política '{stored_policy}', "
                f"não '{policy}'."
            )
    _set_active(version)
