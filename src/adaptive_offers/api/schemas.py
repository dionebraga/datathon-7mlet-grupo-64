"""API contracts (Pydantic) — documented input/output for the decision service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextIn(BaseModel):
    """Decision context. Provide raw features OR a ``client_event_id``.

    No protected attributes are accepted as decision inputs; ``age``/``job``
    etc. are treated as ordinary, non-protected features per the LGPD plan.
    """

    client_event_id: str | None = Field(
        default=None, description="Resolve features from the feature store by id."
    )
    age: int | None = Field(default=None, ge=18, le=100)
    contact: Literal["cellular", "telephone"] | None = None
    poutcome: Literal["success", "failure", "nonexistent"] | None = None
    previously_contacted: int | None = Field(default=None, ge=0, le=1)
    euribor3m: float | None = Field(default=None, ge=0.0, le=6.0)
    default: Literal["yes", "no", "unknown"] | None = None
    loan: Literal["yes", "no", "unknown"] | None = None
    job: str | None = None
    marital: str | None = None
    education: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 66, "contact": "cellular", "poutcome": "success",
                    "previously_contacted": 1, "euribor3m": 0.8,
                    "default": "no", "loan": "no",
                }
            ]
        }
    }

    def as_features(self) -> dict[str, Any]:
        """Feature dict with safe defaults for fields used by the context model."""
        return {
            "client_event_id": self.client_event_id,
            "age": self.age if self.age is not None else 40,
            "contact": self.contact or "cellular",
            "poutcome": self.poutcome or "nonexistent",
            "previously_contacted": self.previously_contacted or 0,
            "euribor3m": self.euribor3m if self.euribor3m is not None else 2.5,
            "default": self.default or "no",
            "loan": self.loan or "no",
            "job": self.job or "admin.",
            "marital": self.marital or "married",
            "education": self.education or "university.degree",
        }


class ReasonOut(BaseModel):
    code: str
    description: str


class DecisionOut(BaseModel):
    decision_id: str
    ts: str
    client_event_id: str | None
    arm_id: str
    arm_name: str
    score: float
    expected_reward: float
    explored: bool
    policy_name: str
    policy_version: str
    eligible_arms: list[str]
    reason_codes: list[str]
    reasons: list[ReasonOut]
    estimates: dict[str, float]


class OfferOut(BaseModel):
    offer_id: str
    name: str
    category: str
    margin: float
    suitability_tier: str


class PolicyOut(BaseModel):
    name: str
    version: str
    trained_on: str
    metrics: dict[str, Any]


class HealthOut(BaseModel):
    status: str
    policy_loaded: bool
    feature_store_materialized: bool
    version: str | None = None


class AssistantIn(BaseModel):
    question: str = Field(..., min_length=3, examples=["Por que o braço de depósito foi escolhido?"])
    decision_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=8)


class AssistantOut(BaseModel):
    answer: str
    provider: str
    citations: list[dict[str, Any]]


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
