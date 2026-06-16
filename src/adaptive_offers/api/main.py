"""FastAPI decision service.

Endpoints (documented contract):
* ``GET  /health``            — liveness + readiness.
* ``GET  /policy``            — active policy metadata/metrics.
* ``GET  /offers``            — offer catalog.
* ``POST /decide``            — context -> auditable decision.
* ``POST /assistant/explain`` — decide + LLM/RAG explanation.

Run: ``adaptive-offers serve`` or ``uvicorn adaptive_offers.api.main:app``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from adaptive_offers.api.schemas import (
    AssistantOut,
    ContextIn,
    DecisionOut,
    HealthOut,
    OfferOut,
    PolicyOut,
)
from adaptive_offers.data.synthetic import offer_catalog
from adaptive_offers.logging_utils import get_logger
from adaptive_offers.policy.versioning import get_active_version

logger = get_logger("api")


@lru_cache(maxsize=1)
def get_service():
    """Lazily build (training if needed) the decision service singleton."""
    from adaptive_offers.bootstrap import ensure_service

    return ensure_service(train_if_missing=True)


@lru_cache(maxsize=1)
def get_assistant():
    from adaptive_offers.assistant import Assistant

    return Assistant()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Adaptive Offers Platform",
        version="0.8.0",
        description="Multi-armed bandit decision service for financial offers "
        "(FIAP 7MLET — Grupo 64). Auditable, guardrailed, explainable.",
    )

    @app.get("/health", response_model=HealthOut, tags=["ops"])
    def health() -> HealthOut:
        from adaptive_offers.feature_store.store import FeatureStore

        return HealthOut(
            status="ok",
            policy_loaded=get_active_version() is not None,
            feature_store_materialized=FeatureStore().is_materialized(),
            version=get_active_version(),
        )

    @app.get("/policy", response_model=PolicyOut, tags=["ops"])
    def policy() -> PolicyOut:
        svc = get_service()
        m = svc.metadata
        return PolicyOut(name=m.name, version=m.version, trained_on=m.trained_on, metrics=m.metrics)

    @app.get("/offers", response_model=list[OfferOut], tags=["catalog"])
    def offers() -> list[OfferOut]:
        return [
            OfferOut(offer_id=a.offer_id, name=a.name, category=a.category,
                     margin=a.margin, suitability_tier=a.suitability_tier)
            for a in offer_catalog()
        ]

    @app.post("/decide", response_model=DecisionOut, tags=["decision"])
    def decide(ctx: ContextIn = Body(...)) -> DecisionOut:
        svc = get_service()
        try:
            if ctx.client_event_id and all(
                getattr(ctx, f) is None for f in ("age", "contact", "poutcome")
            ):
                record = svc.decide(client_event_id=ctx.client_event_id)
            else:
                record = svc.decide(context=ctx.as_features())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"client not found: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DecisionOut(**record.to_dict())

    @app.post("/assistant/explain", response_model=AssistantOut, tags=["assistant"])
    def explain(
        ctx: ContextIn = Body(...),
        question: str = Query(default="Por que esta oferta foi escolhida?"),
    ) -> AssistantOut:
        svc = get_service()
        record = svc.decide(context=ctx.as_features(), log=False)
        result = get_assistant().explain_decision(record.to_dict(), question=question)
        return AssistantOut(
            answer=result["answer"], provider=result["provider"], citations=result["citations"]
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_req, exc: RuntimeError) -> JSONResponse:  # pragma: no cover
        return JSONResponse(status_code=503, content={"error": "service_unavailable", "detail": str(exc)})

    return app


app = create_app()
