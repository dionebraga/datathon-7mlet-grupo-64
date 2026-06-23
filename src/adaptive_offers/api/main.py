"""FastAPI decision service — Adaptive Offers Platform.

Endpoints (documented contract):
* ``GET  /health``              — liveness + readiness.
* ``GET  /policy``              — active policy metadata/metrics.
* ``GET  /offers``              — offer catalog (6 arms).
* ``POST /decide``              — context → auditable decision.
* ``POST /assistant/explain``   — decide + LLM/RAG explanation.
* ``GET  /metrics``             — full policy comparison matrix.
* ``GET  /metrics/regret-curve``— regret curve data for plotting.

Run: ``adaptive-offers serve`` or ``uvicorn adaptive_offers.api.main:app``.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response

from adaptive_offers.api.schemas import (
    AssistantOut,
    AuditEntryOut,
    AuditSummaryOut,
    ContextIn,
    DecisionOut,
    DeleteOut,
    ErrorOut,
    HealthOut,
    MetricsOut,
    OfferOut,
    PolicyOut,
    PolicySwitchIn,
    PolicyVersionOut,
    RegretCurveOut,
    SimulateIn,
    SimulateOut,
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


_DESCRIPTION = """
Serviço de decisão contextual baseado em **Multi-Armed Bandits** para maximização
de receita em campanhas de ofertas financeiras. Cada request recebe o contexto do
cliente e retorna a oferta ótima segundo a política ativa, com rastreabilidade completa.

> 🎓 **FIAP 7MLET — Grupo 74** · Dione Braga

---

## Formulação do Problema

A cada impressão *t*, dado o vetor de contexto **x**ₜ ∈ ℝᵈ e o conjunto de braços
elegíveis 𝒜ₜ ⊆ 𝒜, o agente seleciona:

> **a**ₜ★ = argmax_{a ∈ 𝒜ₜ} 𝔼[Rₐ,ₜ | **x**ₜ]

onde o **reward composto** é definido como:

> Rₐ,ₜ = P(conversão | **x**ₜ, a) × margem_a

O **regret acumulado** mede a sub-optimalidade da política em relação ao oráculo (ótimo conhecido):

> Regret(T) = Σₜ₌₁ᵀ [ R★ₜ − Rₐₜ,ₜ ]

---

## Políticas Implementadas

| Política | Estratégia | Referência |
|----------|-----------|------------|
| **Thompson Sampling** | Amostragem do posterior Beta-Bernoulli: θₐ ~ Beta(αₐ + Sₐ, βₐ + Fₐ) | Thompson (1933) |
| **Nilos-UCB (UCB-V)** | UCB consciente de variância: μ̂ₐ + √(2Vₐ ln t/nₐ) + b ln t/nₐ | Audibert et al. (2009) |
| **LinUCB** | Bandit linear com Ridge: θ̂ = (AᵀA + λI)⁻¹Aᵀb; UCB = θ̂ᵀ**x** + α√(**x**ᵀA⁻¹**x**) | Li et al. (2010) |
| **Baseline** | Greedy puro: argmax μ̂ₐ (sem exploração) | Controle experimental |

---

## Guardrails de Elegibilidade

Antes de selecionar o braço, o serviço aplica filtros de elegibilidade:
- **Suitability tier** — clientes em `default=yes` não são elegíveis para ofertas `restricted`
- **Loan filter** — clientes com empréstimo ativo têm restrições de cross-sell
- **Minimum arms** — garante ≥ 1 braço elegível por decisão

---

## Auditoria

Toda decisão é registrada em `artifacts/decisions/audit.jsonl` com:
`decision_id`, `ts`, `arm_id`, `score`, `explored`, `reason_codes`, `estimates`

**FIAP 7MLET — Grupo 74** · Dione Braga · Licença MIT
"""

# Logo vetorial da marca (mesma arte do dashboard): 3 barras = braços do bandit,
# linha = aprendizado, ponto dourado = braço escolhido. Sem width/height fixos →
# escala via viewBox (serve como favicon e embute na landing).
_LOGO_SVG = (
    '<svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="aoG" x1="4" y1="4" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#0033CC"/><stop offset="1" stop-color="#1A6FFF"/></linearGradient>'
    '<filter id="aoF" x="-40%" y="-40%" width="180%" height="180%">'
    '<feGaussianBlur stdDeviation="1.4" result="b"/>'
    '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    '</defs>'
    '<rect x="2" y="2" width="44" height="44" rx="13" fill="#030D24" stroke="url(#aoG)" stroke-width="2"/>'
    '<rect x="11.5" y="28" width="5.2" height="9" rx="2" fill="#1A6FFF" opacity="0.50"/>'
    '<rect x="20.4" y="22" width="5.2" height="15" rx="2" fill="#1A6FFF" opacity="0.78"/>'
    '<rect x="29.3" y="15" width="5.2" height="22" rx="2" fill="url(#aoG)"/>'
    '<path d="M11 30.5 L23 24 L32 12.5" stroke="#FFFFFF" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round" opacity="0.9" fill="none"/>'
    '<circle cx="32" cy="12.5" r="3.7" fill="#FFC200" stroke="#FFFFFF" stroke-width="1.3" filter="url(#aoF)"/>'
    '</svg>'
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Adaptive Offers Platform — Decision API",
        version="0.8.0",
        description=_DESCRIPTION,
        docs_url=None,  # sobrescrito abaixo para usar o logo como favicon

        openapi_tags=[
            {
                "name": "admin",
                "description": "Endpoints administrativos: auditoria, simulação on-demand, "
                               "troca de política ativa e gestão de versões. Acesso restrito.",
            },
            {
                "name": "ops",
                "description": "Liveness, readiness e metadados da política ativa. "
                               "Use `/health` como probe de Kubernetes e `/policy` para "
                               "inspeção da versão em produção.",
            },
            {
                "name": "catalog",
                "description": "Catálogo dos braços disponíveis (6 ofertas financeiras). "
                               "Cada braço possui `margin` (R$), `category` e `suitability_tier`.",
            },
            {
                "name": "decision",
                "description": "**Endpoint central.** Recebe o contexto do cliente → "
                               "aplica guardrails de elegibilidade → executa a política bandit → "
                               "retorna decisão auditável com `reason_codes` e estimativas por braço.",
            },
            {
                "name": "metrics",
                "description": "Avaliação offline das políticas: matriz de comparação "
                               "(reward, regret, lift) e curvas de regret acumulado para visualização.",
            },
            {
                "name": "assistant",
                "description": "Explicação em linguagem natural da decisão via LLM + RAG. "
                               "Recupera chunks da política comercial e gera resposta contextualizada.",
            },
        ],
        contact={
            "name": "Dione Braga — Grupo 74",
            "url": "https://github.com/dionebraga/datathon-7mlet-grupo-74",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {"url": "http://localhost:8000", "description": "Desenvolvimento local"},
        ],
    )

    @app.get("/", include_in_schema=False)
    def root():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content="""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive Offers API v0.8.0</title>
<link rel="icon" type="image/svg+xml" href="/logo.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800;900&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'Inter',sans-serif;background:#000A18;color:#EDEDED;min-height:100vh;padding:40px 20px}
a{text-decoration:none}
.wrap{max-width:1000px;margin:0 auto}
.hero{text-align:center;padding:0 0 36px}
.badge{display:inline-block;background:rgba(0,51,204,.18);border:1px solid rgba(26,111,255,.45);
  color:#1A6FFF;border-radius:999px;padding:4px 16px;font-size:11px;font-weight:700;margin-bottom:18px}
h1{font-size:2.2rem;font-weight:900;letter-spacing:-.03em;margin-bottom:8px}
.sub{color:#8899BB;font-size:.88rem;line-height:1.6;margin-bottom:20px}
.stack-badges{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-top:18px 0 0}
.stack-badges img{height:26px;border-radius:3px;transition:transform .12s}
.stack-badges img:hover{transform:scale(1.07)}
.stats-bar{display:flex;gap:14px;justify-content:center;margin:28px 0 36px;flex-wrap:wrap}
.stat-pill{background:rgba(3,13,36,0.8);border:1px solid #0D1F42;border-radius:12px;
  padding:12px 22px;text-align:center;min-width:120px}
.stat-num{font-size:1.6rem;font-weight:900;color:#EDEDED;display:block;line-height:1}
.stat-lbl{font-size:10px;color:#8899BB;margin-top:4px;display:block;text-transform:uppercase;letter-spacing:.05em}
.sect-hdr{font-size:.7rem;font-weight:800;color:#8899BB;text-transform:uppercase;
  letter-spacing:.10em;margin:0 0 14px;padding-bottom:6px;border-bottom:1px solid #0D1F42}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin-bottom:36px}
.card{background:rgba(3,13,36,0.75);border:1px solid #0D1F42;border-radius:14px;
  padding:16px 18px;transition:border-color .15s,transform .12s;display:block}
.card:hover{border-color:rgba(26,111,255,.5);transform:translateY(-2px)}
.tag{display:inline-block;font-size:9.5px;font-weight:800;letter-spacing:.05em;
  padding:2px 8px;border-radius:4px;margin-bottom:9px;text-transform:uppercase}
.t-ops{background:rgba(26,158,26,.15);color:#1A9E1A;border:1px solid rgba(26,158,26,.3)}
.t-dec{background:rgba(26,111,255,.15);color:#1A6FFF;border:1px solid rgba(26,111,255,.3)}
.t-cat{background:rgba(255,194,0,.15);color:#FFC200;border:1px solid rgba(255,194,0,.3)}
.t-met{background:rgba(232,64,0,.15);color:#E84000;border:1px solid rgba(232,64,0,.3)}
.t-adm{background:rgba(160,200,48,.15);color:#A0C830;border:1px solid rgba(160,200,48,.3)}
.t-ass{background:rgba(255,154,0,.15);color:#FF9A00;border:1px solid rgba(255,154,0,.3)}
.m{display:inline-block;font-size:9px;font-weight:800;padding:1px 6px;border-radius:3px;
   margin-right:6px;font-family:monospace}
.get{background:rgba(26,158,26,.2);color:#1A9E1A}
.post{background:rgba(26,111,255,.2);color:#1A6FFF}
.del{background:rgba(232,64,0,.2);color:#E84000}
.put{background:rgba(255,194,0,.2);color:#FFC200}
.patch{background:rgba(255,154,0,.2);color:#FF9A00}
.path{font-size:13px;font-weight:700;color:#EDEDED;font-family:monospace}
.desc{font-size:11px;color:#8899BB;margin-top:6px;line-height:1.45}
.btns{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:36px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 22px;border-radius:10px;
  font-size:13px;font-weight:700;transition:filter .15s}
.bp{background:linear-gradient(135deg,#0033CC,#1A6FFF);color:#fff}
.bs{background:rgba(13,31,66,.9);border:1px solid #0D1F42;color:#EDEDED}
.btn:hover{filter:brightness(1.15)}
footer{text-align:center;color:#8899BB;font-size:11px;padding-top:24px;border-top:1px solid #0D1F42}
</style>
</head>
<body>
<div class="wrap">
<div class="hero">
  <div style="width:76px;height:76px;margin:0 auto 18px;filter:drop-shadow(0 6px 20px rgba(26,111,255,.45))">"""
        + _LOGO_SVG
        + """</div>
  <div class="badge">OAS 3.1 · FastAPI · v0.8.0</div>
  <h1>Adaptive Offers API</h1>
  <p class="sub">Multi-Armed Bandit contextual para decisão de ofertas financeiras em tempo real<br>
  Thompson Sampling · Nilos-UCB · LinUCB · Baseline Greedy · FIAP 7MLET Grupo 74</p>
  <div class="stack-badges">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white" alt="Pydantic">
    <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white" alt="Azure">
  </div>
</div>
<div class="stats-bar">
  <div class="stat-pill"><span class="stat-num">14</span><span class="stat-lbl">Endpoints</span></div>
  <div class="stat-pill"><span class="stat-num">4</span><span class="stat-lbl">Políticas bandit</span></div>
  <div class="stat-pill"><span class="stat-num">6</span><span class="stat-lbl">Ofertas (braços)</span></div>
  <div class="stat-pill"><span class="stat-num">v0.8.0</span><span class="stat-lbl">Versão</span></div>
  <div class="stat-pill"><span class="stat-num">OAS 3.1</span><span class="stat-lbl">OpenAPI</span></div>
</div>
<div class="sect-hdr">🔌 Endpoints disponíveis</div>
<div class="grid">
  <a class="card" href="/docs#/ops">
    <span class="tag t-ops">OPS</span><br>
    <span class="m get">GET</span><span class="path">/health</span>
    <div class="desc">Liveness &amp; readiness probe — verifica política carregada e feature store</div>
  </a>
  <a class="card" href="/docs#/ops">
    <span class="tag t-ops">OPS</span><br>
    <span class="m get">GET</span><span class="path">/policy</span>
    <div class="desc">Metadados da política ativa: nome, versão, data de treino e métricas offline</div>
  </a>
  <a class="card" href="/docs#/decision">
    <span class="tag t-dec">DECISION ⭐</span><br>
    <span class="m post">POST</span><span class="path">/decide</span>
    <div class="desc">Endpoint central — contexto do cliente → oferta ótima auditável com reason codes</div>
  </a>
  <a class="card" href="/docs#/catalog">
    <span class="tag t-cat">CATALOG</span><br>
    <span class="m get">GET</span><span class="path">/offers</span>
    <div class="desc">Lista os 6 braços (ofertas financeiras) com margem e suitability tier</div>
  </a>
  <a class="card" href="/docs#/catalog">
    <span class="tag t-cat">CATALOG</span><br>
    <span class="m get">GET</span><span class="path">/offers/{id}</span>
    <div class="desc">Detalhe de uma oferta específica pelo offer_id</div>
  </a>
  <a class="card" href="/docs#/metrics">
    <span class="tag t-met">METRICS</span><br>
    <span class="m get">GET</span><span class="path">/metrics</span>
    <div class="desc">Matriz completa de comparação: reward, regret, lift, conversão por política</div>
  </a>
  <a class="card" href="/docs#/metrics">
    <span class="tag t-met">METRICS</span><br>
    <span class="m get">GET</span><span class="path">/metrics/regret-curve</span>
    <div class="desc">Curvas de regret acumulado prontas para plotagem (downsampled)</div>
  </a>
  <a class="card" href="/docs#/assistant">
    <span class="tag t-ass">ASSISTANT</span><br>
    <span class="m post">POST</span><span class="path">/assistant/explain</span>
    <div class="desc">Explicação LLM + RAG da decisão em linguagem natural com citações</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m get">GET</span><span class="path">/audit</span>
    <div class="desc">Query no log auditável — últimas N decisões com filtros por política e modo</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m del">DELETE</span><span class="path">/audit</span>
    <div class="desc">Limpa o log de auditoria permanentemente (requer ?confirm=true)</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m get">GET</span><span class="path">/policy/versions</span>
    <div class="desc">Lista versões de política treinadas em artifacts/ com status ativo</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m put">PUT</span><span class="path">/policy/active</span>
    <div class="desc">Promove / faz rollback de versão de política sem reiniciar o serviço</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m patch">PATCH</span><span class="path">/policy/active</span>
    <div class="desc">Atualização parcial da política ativa — alias semântico do PUT</div>
  </a>
  <a class="card" href="/docs#/admin">
    <span class="tag t-adm">ADMIN</span><br>
    <span class="m post">POST</span><span class="path">/simulate</span>
    <div class="desc">Simulação on-demand multi-política com métricas completas por horizonte</div>
  </a>
</div>
<div class="btns">
  <a class="btn bp" href="/docs">📖 Swagger UI</a>
  <a class="btn bs" href="/redoc">📘 ReDoc</a>
  <a class="btn bs" href="/openapi.json">⚙️ OpenAPI JSON</a>
</div>
<footer>Adaptive Offers Platform v0.8.0 · © 2026 Dione Braga · FIAP Pós-Tech 7MLET · Grupo 74 · MIT</footer>
</div></body></html>""")

    @app.get("/logo.svg", include_in_schema=False)
    def logo() -> Response:
        """Logo vetorial da marca — usado como favicon e em páginas/branding."""
        return Response(content=_LOGO_SVG, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/docs", include_in_schema=False)
    def custom_swagger() -> HTMLResponse:
        """Swagger UI com o logo do projeto como favicon."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title="Adaptive Offers API — Swagger UI",
            swagger_favicon_url="/logo.svg",
        )

    @app.get(
        "/health",
        response_model=HealthOut,
        tags=["ops"],
        summary="Liveness & readiness probe",
        response_description="Status do serviço, política carregada e feature store materializado.",
        responses={503: {"model": ErrorOut, "description": "Serviço indisponível — bootstrap em progresso."}},
    )
    def health() -> HealthOut:
        """Verifica se o serviço está pronto para receber requisições de decisão.

        Realiza três checagens independentes:

        1. **`policy_loaded`** — há modelo treinado registrado em `artifacts/` e
           resolvível pelo versioning module (MLflow Model Registry ou arquivo local).
        2. **`feature_store_materialized`** — a lookup table `clientId → feature vector`
           está materializada em disco (necessário para requests via `client_event_id`).
        3. **`version`** — tag da versão ativa (`v1`, `v2`, …) ou `null` se nenhuma política
           foi ainda treinada.

        **Probe patterns:**
        - *Liveness*: qualquer resposta 200 indica o processo está vivo.
        - *Readiness*: verificar `policy_loaded = true` antes de rotear tráfego.

        > Retorna 200 mesmo que `feature_store_materialized = false` — o serviço ainda
        > consegue decidir usando features brutas do body da requisição.
        """
        from adaptive_offers.feature_store.store import FeatureStore

        return HealthOut(
            status="ok",
            policy_loaded=get_active_version() is not None,
            feature_store_materialized=FeatureStore().is_materialized(),
            version=get_active_version(),
        )

    @app.get(
        "/policy",
        response_model=PolicyOut,
        tags=["ops"],
        summary="Metadados da política ativa",
        response_description="Nome, versão, data de treino e métricas offline da política em produção.",
        responses={503: {"model": ErrorOut, "description": "Nenhuma política treinada encontrada."}},
    )
    def policy() -> PolicyOut:
        """Retorna os metadados e métricas offline da política bandit atualmente em produção.

        O campo `metrics` contém a avaliação offline completa:

        | Campo | Descrição |
        |-------|-----------|
        | `cumulative_reward` | Reward total (R$) acumulado na simulação de avaliação |
        | `reward_per_1k` | Reward normalizado por 1.000 impressões — proxy de CPM |
        | `regret_ratio` | Regret / Reward_ótimo — quanto da receita ótima foi perdida |
        | `conversion_rate` | Fração de rounds em que o braço converteu |
        | `exploration_rate` | Fração de rounds em que a política explorou (não explorou o melhor) |
        | `lift_vs_baseline_pct` | Ganho percentual de reward em relação à política baseline greedy |

        Use este endpoint para monitoramento de modelo: se `regret_ratio` subir significativamente
        entre deploys, é sinal de degradação ou data drift.
        """
        svc = get_service()
        m = svc.metadata
        return PolicyOut(name=m.name, version=m.version, trained_on=m.trained_on, metrics=m.metrics)

    @app.get(
        "/offers",
        response_model=list[OfferOut],
        tags=["catalog"],
        summary="Catálogo de ofertas (braços do bandit)",
        response_description="Lista das 6 ofertas financeiras disponíveis com margens e suitability.",
    )
    def offers() -> list[OfferOut]:
        """Retorna o catálogo completo de braços (arms) do bandit.

        Cada oferta representa um **braço** do multi-armed bandit. Os atributos relevantes
        para a política de decisão são:

        - **`margin`** (R$): valor financeiro capturado em caso de conversão — entra diretamente
          no cálculo do reward: `R = P(conv | x, a) × margin_a`
        - **`suitability_tier`**: `"standard"` ou `"restricted"`. Ofertas `"restricted"` aplicam
          guardrails de elegibilidade mais restritivos (ex.: excluem clientes em default).
        - **`category`**: `"credit"`, `"investment"`, `"insurance"`, `"deposit"` — usado no
          front-end para agrupamento e filtragem visual.

        O catálogo é estático em runtime (imutável por request). Alterações requerem re-treino
        da política pois mudam o espaço de ação 𝒜.
        """
        return [
            OfferOut(offer_id=a.offer_id, name=a.name, category=a.category,
                     margin=a.margin, suitability_tier=a.suitability_tier)
            for a in offer_catalog()
        ]

    @app.post(
        "/decide",
        response_model=DecisionOut,
        tags=["decision"],
        summary="Decisão contextual de oferta (endpoint principal)",
        response_description="Registro auditável: braço escolhido, score, estimates por oferta, reason codes.",
        responses={
            400: {"model": ErrorOut, "description": "Contexto inválido — nenhum braço elegível ou feature fora do domínio."},
            404: {"model": ErrorOut, "description": "client_event_id não encontrado no feature store."},
            503: {"model": ErrorOut, "description": "Serviço indisponível — política não carregada."},
        },
    )
    def decide(ctx: ContextIn = Body(...)) -> DecisionOut:
        """**Endpoint central** — recebe o contexto do cliente e retorna a decisão ótima.

        ## Pipeline interno

        ```
        ContextIn → Feature Extraction → Eligibility Guard → Bandit Policy → Audit Log → DecisionOut
        ```

        ### 1. Feature Extraction
        Se `client_event_id` for fornecido, resolve o vetor de features via feature store
        (lookup O(1) em memória). Caso contrário, usa os campos brutos do body.
        Features são normalizadas para o espaço aprendido pelo modelo.

        ### 2. Eligibility Guard
        Remove braços inelegíveis *antes* de passar para o bandit:
        - `default=yes` → exclui ofertas com `suitability_tier="restricted"`
        - `loan=yes` → aplica filtros de cross-sell
        - Garante `|𝒜ₜ| ≥ 1` (lança 400 se todos os braços foram filtrados)

        ### 3. Bandit Policy (LinUCB exemplo)
        ```
        score(a) = θ̂ᵀ · x  +  α · √(xᵀ · A⁻¹ · x)
               ─────────────    ──────────────────────
               exploit (média)  explore (incerteza)
        ```
        O braço com maior `score` é selecionado. `explored=true` indica que a escolha
        foi exploratória (o braço não era o de maior média estimada).

        ### 4. Auditoria
        Toda decisão é gravada em `artifacts/decisions/audit.jsonl` com timestamp ISO 8601,
        `decision_id` único e snapshot completo dos `estimates` por braço.

        ### Campos de resposta chave
        - **`estimates`**: mapa `{arm_id → P(conv)}` para **todos** os braços elegíveis —
          útil para análise de distribuição e visualizações de probabilidade.
        - **`reason_codes`**: lista de codes que explicam a decisão (ex.: `MARGIN_WEIGHTED`,
          `ELIGIBILITY_FILTERED`, `EXPLORED_UCB`).
        - **`explored`**: `true` se a política escolheu exploração (não-greedy).
        """
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

    @app.get(
        "/metrics",
        response_model=list[MetricsOut],
        tags=["metrics"],
        summary="Matriz de comparação de políticas (avaliação offline)",
        response_description="Lista ordenada por reward decrescente com todas as métricas por política.",
        responses={503: {"model": ErrorOut, "description": "Bootstrap incompleto — dados ou bundle ausentes."}},
    )
    def metrics() -> list[MetricsOut]:
        """Avaliação offline completa de todas as políticas disponíveis.

        Executa (ou lê do cache) uma simulação idêntica para cada política no mesmo
        conjunto de dados e horizonte, garantindo comparação justa (mesmo oráculo).

        ## Métricas retornadas

        | Campo | Fórmula | Interpretação |
        |-------|---------|---------------|
        | `cumulative_reward` | Σ Rₐₜ,ₜ | Receita total capturada |
        | `reward_per_1k` | cumulative_reward / T × 1000 | Proxy de RPM (revenue per mille) |
        | `cumulative_regret` | Σ (R★ₜ − Rₐₜ,ₜ) | Oportunidade perdida vs oráculo |
        | `regret_ratio` | cumulative_regret / Σ R★ₜ | Sub-optimalidade relativa |
        | `conversion_rate` | |{t: convertido}| / T | Taxa de conversão média |
        | `exploration_rate` | |{t: explored}| / T | Fração de rounds exploratórios |
        | `lift_vs_baseline_pct` | (reward − reward_base) / reward_base × 100 | Ganho vs greedy |

        **Uso típico:** alimenta o dashboard (lollipop charts, policy heatmap, lift curves)
        e o relatório de MLflow após cada ciclo de retreino.

        > ⚠️ Esta rota pode demorar 5-30s na primeira chamada (executa simulação).
        > Use o parâmetro `?cache=false` para forçar re-cálculo.
        """
        from adaptive_offers.bootstrap import ensure_bundle, ensure_data
        from adaptive_offers.evaluation.offline_eval import metrics_matrix

        proc = ensure_data()
        bundle = ensure_bundle(proc)
        return [MetricsOut(**r) for r in metrics_matrix(proc, bundle)]

    @app.get(
        "/metrics/regret-curve",
        response_model=list[RegretCurveOut],
        tags=["metrics"],
        summary="Curvas de regret acumulado por política",
        response_description="Arrays (steps, regret) downsampled para cada política — prontos para plotagem.",
        responses={503: {"model": ErrorOut, "description": "Simulação falhou — verifique os dados de entrada."}},
    )
    def regret_curve(
        points: int = Query(
            default=70, ge=10, le=200,
            description="Número de pontos de amostragem na curva (downsampling uniforme do horizonte total).",
        ),
    ) -> list[RegretCurveOut]:
        """Curvas de regret acumulado para visualização temporal do aprendizado.

        Roda uma simulação completa para cada política e aplica downsampling uniforme
        no eixo de tempo para reduzir o payload. Retorna uma série temporal por política.

        ## Estrutura da resposta

        ```json
        [
          {
            "policy": "linucb",
            "steps": [0, 86, 172, ...],   // índices no horizonte
            "regret": [0.0, 12.4, 18.1, ...]  // regret acumulado em R$
          },
          ...
        ]
        ```

        ## Interpretação visual

        - **Curva plana** no final → política convergiu (parou de cometer erros)
        - **Curva com inclinação constante** → exploração excessiva ou política subótima
        - **Área entre LinUCB e Baseline** → receita recuperada pelo bandit

        O parâmetro `points` controla a resolução do gráfico. Use `points=200` para
        análise detalhada de convergência e `points=30` para dashboards embarcados.
        """
        from adaptive_offers.bandits.registry import build_policy
        from adaptive_offers.bootstrap import ensure_bundle, ensure_data
        from adaptive_offers.data.synthetic import CONTEXT_FEATURES
        from adaptive_offers.simulation.environment import build_arms, run_simulation
        from adaptive_offers.simulation.metrics import regret_curve as _rc

        proc = ensure_data()
        bundle = ensure_bundle(proc)
        arms = build_arms(bundle.catalog)
        results = []
        for name in ("baseline", "thompson", "nilos_ucb", "linucb"):
            pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES))
            results.append(run_simulation(pol, proc, bundle, horizon=6000))
        curves = []
        for res in results:
            idx, cum = _rc(res, points=points)
            curves.append(RegretCurveOut(
                policy=res.policy_name,
                steps=idx.tolist(),
                regret=cum.tolist(),
            ))
        return curves

    @app.post(
        "/assistant/explain",
        response_model=AssistantOut,
        tags=["assistant"],
        summary="Explicação LLM+RAG de uma decisão contextual",
        response_description="Resposta em linguagem natural com citações das políticas comerciais (RAG).",
        responses={
            400: {"model": ErrorOut, "description": "Contexto inválido — não foi possível gerar decisão."},
            503: {"model": ErrorOut, "description": "LLM ou RAG indisponível."},
        },
    )
    def explain(
        ctx: ContextIn = Body(...),
        question: str = Query(
            default="Por que esta oferta foi escolhida?",
            description="Pergunta em linguagem natural sobre a decisão gerada.",
            min_length=5,
            max_length=300,
        ),
    ) -> AssistantOut:
        """Gera uma decisão de oferta e a explica em linguagem natural via LLM + RAG.

        ## Pipeline RAG

        ```
        ContextIn → /decide (log=False) → DecisionOut
                                              ↓
                                    Retrieval (top-k chunks da política comercial)
                                              ↓
                                    LLM (resposta contextualizada)
                                              ↓
                                    AssistantOut (answer + citations)
        ```

        ## Estratégia de Retrieval

        O assistente recupera chunks relevantes do documento de política comercial
        (ex.: regras de suitability, critérios de elegibilidade, estratégia de pricing)
        usando similaridade semântica com o contexto da decisão. Os chunks recuperados
        são injetados no prompt do LLM como contexto factual.

        ## Fallback Offline

        Se nenhum LLM externo estiver configurado, o assistente usa um gerador
        baseado em templates (provider `"offline"`) que produz respostas determinísticas
        mas estruturadas, citando os `reason_codes` da decisão.

        > Esta rota executa `/decide` internamente com `log=False` para não poluir
        > o audit log com chamadas de explicação. Use `/decide` diretamente se quiser
        > registrar a decisão no log auditável.
        """
        svc = get_service()
        record = svc.decide(context=ctx.as_features(), log=False)
        result = get_assistant().explain_decision(record.to_dict(), question=question)
        return AssistantOut(
            answer=result["answer"], provider=result["provider"], citations=result["citations"]
        )

    # ── AUDIT endpoints ────────────────────────────────────────────────────
    _AUDIT_PATH = (
        lambda: __import__("pathlib").Path(__file__).resolve().parents[3]
        / "artifacts" / "decisions" / "audit.jsonl"
    )

    @app.get(
        "/audit",
        response_model=AuditSummaryOut,
        tags=["admin"],
        summary="Query do log auditável de decisões",
        response_description="Últimas N decisões com metadados completos.",
        responses={503: {"model": ErrorOut}},
    )
    def get_audit(
        n: int = Query(default=50, ge=1, le=1000,
                       description="Número de decisões a retornar (mais recentes primeiro)."),
        policy: str | None = Query(default=None,
                                   description="Filtrar por política (ex: linucb)."),
        explored: bool | None = Query(default=None,
                                      description="Filtrar por modo: True=exploração, False=explotação."),
    ) -> AuditSummaryOut:
        """Lê o log auditável `audit.jsonl` e retorna as últimas N decisões com filtros opcionais.

        Útil para:
        - Monitorar distribuição de braços escolhidos em produção
        - Auditar proporção exploração/explotação
        - Verificar qualidade das decisões em tempo real
        """
        import json
        p = _AUDIT_PATH()
        if not p.exists():
            return AuditSummaryOut(total_decisions=0, entries=[])
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        total = len(lines)
        entries = []
        for ln in reversed(lines):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if policy and d.get("policy_name") != policy:
                continue
            if explored is not None and d.get("explored") != explored:
                continue
            entries.append(AuditEntryOut(
                decision_id=d.get("decision_id", ""),
                ts=d.get("ts", ""),
                arm_id=d.get("arm_id", ""),
                arm_name=d.get("arm_name"),
                score=d.get("score"),
                expected_reward=d.get("expected_reward"),
                explored=d.get("explored"),
                policy_name=d.get("policy_name"),
                policy_version=d.get("policy_version"),
                reason_codes=d.get("reason_codes", []),
            ))
            if len(entries) >= n:
                break
        return AuditSummaryOut(total_decisions=total, entries=entries)

    @app.delete(
        "/audit",
        response_model=DeleteOut,
        tags=["admin"],
        summary="Limpar log de auditoria",
        response_description="Número de decisões removidas.",
        responses={503: {"model": ErrorOut}},
    )
    def delete_audit(
        confirm: bool = Query(default=False,
                              description="Defina `confirm=true` para confirmar a limpeza permanente."),
    ) -> DeleteOut:
        """Remove todas as entradas do log auditável `audit.jsonl`.

        ⚠️ **Operação irreversível.** Requer `?confirm=true` para evitar deleções acidentais.
        """
        if not confirm:
            raise HTTPException(status_code=400,
                                detail="Passe ?confirm=true para confirmar a limpeza do log.")
        p = _AUDIT_PATH()
        if not p.exists():
            return DeleteOut(deleted=0, message="Log vazio — nada a remover.")
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        n_del = len(lines)
        p.write_text("", encoding="utf-8")
        logger.info('{"event": "audit_cleared", "rows_deleted": %d}', n_del)
        return DeleteOut(deleted=n_del, message=f"{n_del} decisões removidas do log auditável.")

    # ── POLICY MANAGEMENT ──────────────────────────────────────────────────
    @app.get(
        "/policy/versions",
        response_model=list[PolicyVersionOut],
        tags=["admin"],
        summary="Listar versões de política disponíveis",
        response_description="Versões treinadas em artifacts/ com status de ativação.",
    )
    def list_policy_versions() -> list[PolicyVersionOut]:
        """Lista todas as versões de política salvas em `artifacts/`.

        Cada versão contém metadados de treino (data, política, métricas) e
        indica qual está atualmente ativa como política de decisão do serviço.
        """
        from adaptive_offers.policy.versioning import get_active_version, list_versions
        active = get_active_version()
        try:
            versions = list_versions()
        except Exception:
            versions = []
        result = []
        for v in versions:
            result.append(PolicyVersionOut(
                version=v.get("version", "v1"),
                policy=v.get("policy", "unknown"),
                trained_on=v.get("trained_on"),
                active=(v.get("version") == active),
            ))
        if not result and active:
            result.append(PolicyVersionOut(version=active, policy="unknown",
                                           trained_on=None, active=True))
        return result

    @app.put(
        "/policy/active",
        response_model=PolicyOut,
        tags=["admin"],
        summary="Trocar política ativa (rollback / promoção)",
        response_description="Metadados da nova política ativa após a troca.",
        responses={
            400: {"model": ErrorOut, "description": "Versão inválida ou política não encontrada."},
            503: {"model": ErrorOut, "description": "Falha ao recarregar o serviço."},
        },
    )
    def switch_policy(body: PolicySwitchIn) -> PolicyOut:
        """Promove ou faz rollback de uma versão de política sem reiniciar o serviço.

        O serviço recarrega o modelo em memória (`lru_cache` é invalidado).
        Útil para:
        - Rollback imediato após detectar degradação via `/metrics`
        - Promoção A/B de uma nova versão treinada via MLflow
        - Troca entre estratégias de exploração

        Escreve `artifacts/.active_policy` com o nome/versão selecionados.
        """
        import json
        from adaptive_offers.policy.versioning import set_active_version
        try:
            set_active_version(body.policy, body.version)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        get_service.cache_clear()
        svc = get_service()
        m = svc.metadata
        return PolicyOut(name=m.name, version=m.version, trained_on=m.trained_on, metrics=m.metrics)

    @app.patch(
        "/policy/active",
        response_model=PolicyOut,
        tags=["admin"],
        summary="Atualizar parâmetros da política ativa (alias de PUT)",
        response_description="Mesma semântica que PUT /policy/active.",
        responses={
            400: {"model": ErrorOut},
            503: {"model": ErrorOut},
        },
    )
    def patch_policy(body: PolicySwitchIn) -> PolicyOut:
        """Alias semântico de `PUT /policy/active` — aceita atualização parcial."""
        return switch_policy(body)

    # ── SIMULATION ──────────────────────────────────────────────────────────
    @app.post(
        "/simulate",
        response_model=SimulateOut,
        tags=["admin"],
        summary="Rodar simulação on-demand e retornar métricas",
        response_description="Métricas de todas as políticas solicitadas para o horizonte dado.",
        responses={400: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )
    def simulate(body: SimulateIn) -> SimulateOut:
        """Executa uma simulação multi-política on-demand e retorna a matriz de métricas.

        Uso principal:
        - CI/CD: validar que uma nova versão supera a baseline antes de promover
        - Experimentos: comparar configurações de alpha/beta sem re-deploy

        ⚠️ Pode demorar de 5s a 120s dependendo do `horizon`. Recomendado ≤ 4000 para uso interativo.
        """
        from adaptive_offers.bandits.registry import build_policy
        from adaptive_offers.bootstrap import ensure_bundle, ensure_data
        from adaptive_offers.data.synthetic import CONTEXT_FEATURES
        from adaptive_offers.simulation.environment import build_arms, run_simulation
        from adaptive_offers.simulation.metrics import compare_results

        known = {"baseline", "thompson", "nilos_ucb", "linucb"}
        bad = [p for p in body.policies if p not in known]
        if bad:
            raise HTTPException(status_code=400,
                                detail=f"Políticas desconhecidas: {bad}. Válidas: {sorted(known)}")
        proc   = ensure_data()
        bundle = ensure_bundle(proc)
        arms   = build_arms(bundle.catalog)
        res_list = []
        for name in body.policies:
            pol = build_policy(name, arms, context_dim=len(CONTEXT_FEATURES), seed=body.seed)
            res_list.append(run_simulation(pol, proc, bundle, horizon=body.horizon, seed=body.seed))
        metrics = compare_results(res_list)
        return SimulateOut(horizon=body.horizon, seed=body.seed, results=metrics)

    # ── OFFERS MANAGEMENT ─────────────────────────────────────────────────
    @app.get(
        "/offers/{offer_id}",
        response_model=OfferOut,
        tags=["catalog"],
        summary="Detalhe de uma oferta específica",
        response_description="Dados completos da oferta pelo ID.",
        responses={404: {"model": ErrorOut, "description": "Oferta não encontrada."}},
    )
    def get_offer(offer_id: str) -> OfferOut:
        """Retorna os detalhes de uma oferta específica pelo `offer_id`."""
        cat = offer_catalog()
        arm = next((a for a in cat if a.offer_id == offer_id), None)
        if not arm:
            raise HTTPException(status_code=404, detail=f"Oferta '{offer_id}' não encontrada.")
        return OfferOut(offer_id=arm.offer_id, name=arm.name, category=arm.category,
                        margin=arm.margin, suitability_tier=arm.suitability_tier)

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(_req, exc: RuntimeError) -> JSONResponse:  # pragma: no cover
        return JSONResponse(status_code=503, content={"error": "service_unavailable", "detail": str(exc)})

    return app


app = create_app()
