"""Behavioral segmentation — an explicit *persona* lens over the context.

The contextual bandit (LinUCB) already personalises decisions implicitly per
context vector. A fintech also needs an **explicit, human-readable segmentation**
to read the book of business: which behaviours exist, how offers distribute per
segment, and where fairness/risk concentrate. This module derives deterministic,
auditable personas from the **real** Bank Marketing features — no clustering
black box, no protected attribute used as the *sole* driver — so every customer
maps to a segment a risk/marketing analyst can reason about.

Personas are priority-ordered (first match wins) and cover the population
exhaustively (``seg_massa`` is the catch-all). Derived purely from features that
already exist in the processed table, so it works on the real base unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A behavioural persona with an id, label and one-line rationale."""

    seg_id: str
    label: str
    rationale: str


# Priority-ordered personas (first matching rule wins).
SEGMENTS: list[Segment] = [
    Segment("seg_renegociador", "Endividado / renegociador",
            "Tem empréstimo ativo ou default — foco em renegociação, não em crédito novo."),
    Segment("seg_senior_conserv", "Sênior conservador",
            "60+ anos — perfil de baixo risco, propenso a depósito/fundo."),
    Segment("seg_jovem_digital", "Jovem digital",
            "<30 anos em canal celular — engajamento digital, cashback/cartão."),
    Segment("seg_recorrente", "Recorrente engajado",
            "Conversão prévia bem-sucedida — alta propensão, cross-sell de valor."),
    Segment("seg_novo_cold", "Novo (cold-start)",
            "Sem contato anterior — incerteza alta, candidato a exploração."),
    Segment("seg_massa", "Massa padrão",
            "Perfil mediano sem sinal dominante — política contextual decide."),
]
_BY_ID = {s.seg_id: s for s in SEGMENTS}


def _get(row: Any, key: str, default: Any) -> Any:
    """Read a field from a dict or pandas-row tolerantly."""
    try:
        val = row.get(key, default) if hasattr(row, "get") else row[key]
    except Exception:
        return default
    return default if val is None else val


def segment_of(features: Any) -> Segment:
    """Map a feature row/context to its behavioural persona (first match wins)."""
    age = float(_get(features, "age", 40) or 40)
    poutcome = str(_get(features, "poutcome", "nonexistent")).lower()
    loan = str(_get(features, "loan", "no")).lower()
    default = str(_get(features, "default", "no")).lower()
    contact = str(_get(features, "contact", "cellular")).lower()
    prev_contacted = int(_get(features, "previously_contacted", 0) or 0)

    if default == "yes" or loan == "yes":
        return _BY_ID["seg_renegociador"]
    if age >= 60:
        return _BY_ID["seg_senior_conserv"]
    if age < 30 and contact == "cellular":
        return _BY_ID["seg_jovem_digital"]
    if poutcome == "success":
        return _BY_ID["seg_recorrente"]
    if poutcome == "nonexistent" and not prev_contacted:
        return _BY_ID["seg_novo_cold"]
    return _BY_ID["seg_massa"]


def label_of(seg_id: str) -> str:
    """Human label for a segment id (falls back to the id)."""
    seg = _BY_ID.get(seg_id)
    return seg.label if seg else seg_id
