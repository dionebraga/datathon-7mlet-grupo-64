"""Responsible AI — protected attributes registry + group mapping for fairness.

The challenge requires the LGPD plan to *map protected attributes* and the model
card to carry a *fairness analysis per group*. This module is the **single source
of truth** for which attributes are treated as sensitive, and derives the
**audit-only** group membership attached to each decision.

Design stance (documented in ``docs/lgpd-plan.md``):

* These attributes are **never the sole driver** of a decision and are **never
  used to exclude** anyone. ``age`` participates only as a *suitability* signal
  (e.g. a senior profile fits a conservative deposit), not as a gate.
* Group membership is recorded **only to audit fairness** — to verify that no
  protected group is systematically denied value (exposure parity). It is
  computed *after* the decision and does not feed the policy.
"""

from __future__ import annotations

from typing import Any

# Single source of truth: sensitive attributes monitored for fairness.
PROTECTED_ATTRIBUTES: dict[str, str] = {
    "age": "Idade — só entra como sinal de suitability (ex.: sênior→depósito), "
           "nunca como exclusão; auditada em fairness de exposição.",
    "job": "Profissão — proxy socioeconômico; monitorada, não é driver isolado.",
    "marital": "Estado civil — proxy sensível; monitorada em fairness.",
    "education": "Escolaridade — proxy socioeconômico; monitorada em fairness.",
}

# Age bands used consistently across the audit and the offline fairness report.
_AGE_BINS = [17, 30, 45, 60, 200]
_AGE_LABELS = ["<=30", "31-45", "46-60", "60+"]


def _get(row: Any, key: str, default: Any) -> Any:
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except Exception:
        return default
    return default if val is None else val


def age_band(age: Any) -> str:
    """Bucket an age into the canonical band (shared with the fairness report)."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "unknown"
    for hi, label in zip(_AGE_BINS[1:], _AGE_LABELS):
        if a <= hi:
            return label
    return _AGE_LABELS[-1]


def protected_groups(features: Any) -> dict[str, str]:
    """Audit-only protected-group membership for a context. NOT used to decide."""
    return {
        "age_band": age_band(_get(features, "age", None)),
        "job": str(_get(features, "job", "unknown")),
        "marital": str(_get(features, "marital", "unknown")),
        "education": str(_get(features, "education", "unknown")),
    }
