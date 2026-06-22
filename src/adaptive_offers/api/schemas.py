"""API contracts (Pydantic) — documented input/output for the decision service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextIn(BaseModel):
    """Contexto de decisão do cliente.

    Forneça atributos brutos **OU** um ``client_event_id`` para resolução
    via feature store. Nenhum atributo protegido (LGPD) é aceito como
    input direto; ``age``, ``job`` etc. são tratados como features
    ordinárias e não protegidas conforme o plano LGPD do projeto.

    **Features disponíveis:**
    - ``age``: idade do cliente (18-100)
    - ``contact``: canal de contato (cellular/telephone)
    - ``poutcome``: resultado da campanha anterior
    - ``euribor3m``: taxa Euribor 3 meses (0.0-6.0)
    - ``default``: indicação de default
    - ``loan``: empréstimo ativo
    - ``previously_contacted``: contato prévio (0/1)
    """

    client_event_id: str | None = Field(
        default=None,
        description="Resolve features from the feature store by id.",
        examples=["evt_00000001"],
    )
    age: int | None = Field(
        default=None, ge=18, le=100,
        description="Idade do cliente (18-100).",
        examples=[66],
    )
    contact: Literal["cellular", "telephone"] | None = Field(
        default=None,
        description="Canal de contato.",
        examples=["cellular"],
    )
    poutcome: Literal["success", "failure", "nonexistent"] | None = Field(
        default=None,
        description="Resultado da campanha anterior.",
        examples=["success"],
    )
    previously_contacted: int | None = Field(
        default=None, ge=0, le=1,
        description="Cliente já foi contactado antes? (0/1).",
        examples=[1],
    )
    euribor3m: float | None = Field(
        default=None, ge=0.0, le=6.0,
        description="Taxa Euribor 3 meses (%) — proxy de condições macroeconômicas.",
        examples=[0.8],
    )
    default: Literal["yes", "no", "unknown"] | None = Field(
        default=None,
        description="Indica se o cliente está em situação de default.",
        examples=["no"],
    )
    loan: Literal["yes", "no", "unknown"] | None = Field(
        default=None,
        description="Indica se o cliente possui empréstimo ativo.",
        examples=["no"],
    )
    job: str | None = Field(
        default=None,
        description="Cargo/ocupação do cliente.",
        examples=["admin."],
    )
    marital: str | None = Field(
        default=None,
        description="Estado civil.",
        examples=["married"],
    )
    education: str | None = Field(
        default=None,
        description="Nível de escolaridade.",
        examples=["university.degree"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 66, "contact": "cellular", "poutcome": "success",
                    "previously_contacted": 1, "euribor3m": 0.8,
                    "default": "no", "loan": "no",
                },
                {
                    "age": 35, "contact": "telephone", "poutcome": "failure",
                    "previously_contacted": 0, "euribor3m": 3.2,
                    "default": "no", "loan": "yes",
                },
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
    """Código de razão com descrição legível."""

    code: str = Field(description="Identificador do reason code.", examples=["MARGIN_WEIGHTED"])
    description: str = Field(
        description="Descrição em português do reason code.",
        examples=["Oferta com maior valor esperado (margem × P conversão)."],
    )


class DecisionOut(BaseModel):
    """Registro auditável de uma decisão do bandit contextual."""

    decision_id: str = Field(description="Identificador único da decisão.", examples=["dec_00000001"])
    ts: str = Field(description="Timestamp ISO 8601 da decisão.", examples=["2026-06-18T10:30:00Z"])
    client_event_id: str | None = Field(
        description="ID do evento de cliente (se resolvido via feature store)."
    )
    arm_id: str = Field(description="ID da oferta escolhida.", examples=["OFF_LOAN_PREAPP"])
    arm_name: str = Field(description="Nome da oferta escolhida.", examples=["Empréstimo Pré-aprovado"])
    score: float = Field(description="Score bruto da política para o braço.", examples=[0.85])
    expected_reward: float = Field(
        description="Valor esperado: P(conversão) × margem.",
        examples=[25.5],
    )
    explored: bool = Field(
        description="True se foi uma decisão exploratória (vs. explotação)."
    )
    policy_name: str = Field(description="Nome da política ativa.", examples=["linucb"])
    policy_version: str = Field(description="Versão da política.", examples=["v1"])
    eligible_arms: list[str] = Field(
        description="Lista de IDs de ofertas elegíveis para este cliente."
    )
    reason_codes: list[str] = Field(
        description="Códigos de razão que explicam a decisão.",
        examples=[["MARGIN_WEIGHTED", "ELIGIBILITY_FILTERED", "SUITABILITY_OK"]],
    )
    reasons: list[ReasonOut] = Field(
        description="Reason codes com descrições completas."
    )
    estimates: dict[str, float] = Field(
        description="Mapa braço → estimativa de P(conversão) para todos os braços elegíveis.",
        examples=[{"OFF_LOAN_PREAPP": 0.085, "OFF_TD_PREMIUM": 0.043}],
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mapa braço → score ponderado por margem (ranqueamento da política).",
        examples=[{"OFF_LOAN_PREAPP": 25.5, "OFF_TD_PREMIUM": 8.6}],
    )
    segment_id: str = Field(
        default="", description="Persona comportamental do cliente.", examples=["seg_senior_conserv"]
    )
    segment_label: str = Field(
        default="", description="Rótulo legível da persona.", examples=["Sênior conservador"]
    )


class OfferOut(BaseModel):
    """Braço do catálogo de ofertas."""

    offer_id: str = Field(description="ID da oferta.", examples=["OFF_LOAN_PREAPP"])
    name: str = Field(description="Nome comercial da oferta.", examples=["Empréstimo Pré-aprovado"])
    category: str = Field(description="Categoria da oferta.", examples=["credit"])
    margin: float = Field(description="Margem financeira da oferta (R$).", examples=[300.0])
    suitability_tier: str = Field(
        description="Nível de suitability: 'standard' ou 'restricted'.",
        examples=["restricted"],
    )


class PolicyOut(BaseModel):
    """Política ativa no momento."""

    name: str = Field(description="Nome da política.", examples=["linucb"])
    version: str = Field(description="Versão da política.", examples=["v1"])
    trained_on: str = Field(description="Data ISO 8601 do treino.", examples=["2026-06-16T15:31:00"])
    metrics: dict[str, Any] = Field(
        description="Métricas da política: reward, regret, conversão, exploração.",
        examples=[{"cumulative_reward": 424820, "regret_ratio": 0.051}],
    )


class HealthOut(BaseModel):
    """Status de saúde do serviço."""

    status: str = Field(description="Status geral.", examples=["ok"])
    policy_loaded: bool = Field(description="Há uma política treinada carregada?")
    feature_store_materialized: bool = Field(
        description="O feature store online está materializado?"
    )
    version: str | None = Field(
        default=None, description="Versão da política ativa (se carregada).",
        examples=["v1"],
    )


class MetricsOut(BaseModel):
    """Matriz de métricas de uma política — usada para comparação."""

    policy: str = Field(description="Nome da política.", examples=["linucb"])
    cumulative_reward: float = Field(
        description="Reward acumulado (R$).", examples=[424820.0]
    )
    reward_per_1k: float = Field(
        description="Reward normalizado por 1k impressões.", examples=[35401.67]
    )
    cumulative_regret: float = Field(
        description="Regret acumulado.", examples=[21665.82]
    )
    regret_ratio: float = Field(
        description="Proporção do regret em relação ao ótimo.",
        examples=[0.051],
    )
    conversion_rate: float = Field(
        description="Taxa de conversão média.", examples=[0.091]
    )
    exploration_rate: float = Field(
        description="Taxa de exploração média.", examples=[0.15]
    )
    lift_vs_baseline_pct: float | None = Field(
        default=None,
        description="Lift percentual em reward vs. política baseline.",
        examples=[66.6],
    )


class RegretCurveOut(BaseModel):
    """Curva de regret para uma política — pontos (step, regret)."""

    policy: str = Field(description="Nome da política.", examples=["linucb"])
    steps: list[int] = Field(description="Índices dos steps.")
    regret: list[float] = Field(description="Regret acumulado em cada step.")


class AssistantIn(BaseModel):
    """Input para o assistente LLM+RAG."""

    question: str = Field(
        ..., min_length=3,
        description="Pergunta sobre a decisão.",
        examples=["Por que o braço de depósito foi escolhido?"],
    )
    decision_id: str | None = Field(
        default=None,
        description="ID da decisão (opcional — vincula ao log auditável).",
    )
    top_k: int = Field(
        default=3, ge=1, le=8,
        description="Nº de chunks do RAG a recuperar.",
    )


class AssistantOut(BaseModel):
    """Explicação do assistente LLM+RAG."""

    answer: str = Field(description="Resposta em linguagem natural.")
    provider: str = Field(description="Provedor LLM usado.", examples=["offline"])
    citations: list[dict[str, Any]] = Field(
        description="Citações das políticas comerciais (RAG)."
    )


class ErrorOut(BaseModel):
    """Resposta de erro padronizada."""

    error: str = Field(description="Código do erro.", examples=["service_unavailable"])
    detail: str | None = Field(default=None, description="Detalhe da falha.")


class AuditEntryOut(BaseModel):
    """Entrada do log auditável de decisões."""

    decision_id: str = Field(description="ID único da decisão.", examples=["dec_00000001"])
    ts: str = Field(description="Timestamp ISO 8601.", examples=["2026-06-20T10:30:00Z"])
    arm_id: str = Field(description="ID da oferta escolhida.", examples=["OFF_DEPOSIT"])
    arm_name: str | None = Field(default=None, description="Nome da oferta.")
    score: float | None = Field(default=None, description="Score bruto do bandit.")
    expected_reward: float | None = Field(default=None, description="Valor esperado (R$).")
    explored: bool | None = Field(default=None, description="Decisão exploratória?")
    policy_name: str | None = Field(default=None, description="Política usada.")
    policy_version: str | None = Field(default=None, description="Versão da política.")
    reason_codes: list[str] = Field(default_factory=list, description="Reason codes.")


class AuditSummaryOut(BaseModel):
    """Resumo do log de auditoria."""

    total_decisions: int = Field(description="Total de decisões registradas.")
    entries: list[AuditEntryOut] = Field(description="Entradas (últimas N).")


class DeleteOut(BaseModel):
    """Confirmação de deleção."""

    deleted: int = Field(description="Número de registros removidos.")
    message: str = Field(description="Mensagem de confirmação.")


class PolicyVersionOut(BaseModel):
    """Versão de política disponível em artifacts/."""

    version: str = Field(description="Tag de versão.", examples=["v1"])
    policy: str = Field(description="Nome da política.", examples=["linucb"])
    trained_on: str | None = Field(default=None, description="Data de treino.")
    active: bool = Field(description="Versão atualmente ativa?")


class PolicySwitchIn(BaseModel):
    """Corpo para troca de política ativa."""

    policy: str = Field(
        description="Nome da política a ativar.",
        examples=["linucb"],
    )
    version: str = Field(
        default="v1",
        description="Versão da política.",
        examples=["v1"],
    )


class SimulateIn(BaseModel):
    """Parâmetros de simulação on-demand."""

    horizon: int = Field(
        default=2000, ge=100, le=20000,
        description="Número de rounds.",
    )
    seed: int = Field(
        default=42, ge=0,
        description="Seed aleatória.",
    )
    policies: list[str] = Field(
        default_factory=lambda: ["linucb", "thompson", "nilos_ucb", "baseline"],
        description="Políticas a simular.",
    )


class SimulateOut(BaseModel):
    """Resultado de simulação on-demand."""

    horizon: int
    seed: int
    results: list[dict[str, Any]] = Field(description="Métricas por política simulada.")
