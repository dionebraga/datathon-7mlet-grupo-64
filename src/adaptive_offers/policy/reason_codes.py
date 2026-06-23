"""Reason-code catalog for auditable, explainable decisions.

Every served decision carries a list of reason codes. They make the audit log
human-readable and support governance (System Card) and incident review.
"""

from __future__ import annotations

REASON_CODES: dict[str, str] = {
    # policy family
    "BASELINE_GREEDY": "Decisão pela política de controle (greedy, sem exploração).",
    "THOMPSON_SAMPLE": "Braço escolhido por amostragem do posterior (Thompson).",
    "NILOS_UCB": "Braço escolhido por limite superior de confiança variance-aware.",
    "LINUCB": "Braço escolhido por modelo linear contextual (LinUCB).",
    "LIN_THOMPSON": "Braço escolhido por amostragem do posterior bayesiano linear (LinThompson).",
    "NEURAL": "Braço escolhido por bandit neural contextual (MLP em PyTorch).",
    "MC_DROPOUT": "Exploração bayesiana aproximada via MC-dropout (Thompson neural).",
    "CONTEXTUAL": "Decisão usou o vetor de contexto do cliente.",
    # exploration/exploitation
    "EXPLORATION": "Decisão exploratória (não é o líder atual de média).",
    "EXPLOITATION": "Decisão de explotação (melhor estimativa atual).",
    "NO_EXPLORATION": "Política de controle não explora.",
    "COLD_START_PULL": "Puxada inicial obrigatória de braço ainda não observado.",
    "EXPLORATION_FLOOR_APPLIED": "Exploração forçada pelo piso mínimo (guardrail).",
    # eligibility / suitability guardrails
    "SUITABILITY_OK": "Braço passou no gate de elegibilidade/suitability.",
    "ELIGIBILITY_FILTERED": "Braços inelegíveis foram removidos antes da seleção.",
    "CONTROL_FALLBACK": "Sem oferta de valor elegível; controle (no_offer) aplicado.",
    # value
    "MARGIN_WEIGHTED": "Ranqueamento ponderado pela margem da oferta.",
    "VALUE_FLOOR_OK": "Valor esperado acima do piso mínimo configurado.",
    # channel orchestration / contact policy
    "CHANNEL_SELECTED": "Canal de entrega escolhido pela política de contato.",
    "QUIET_HOURS": "Horário de silêncio — canais intrusivos (voz/SMS) suprimidos.",
    "FREQUENCY_CAPPED": "Limite de contatos na janela atingido — não recontatar.",
    "CONTACT_SUPPRESSED": "Contato suprimido pela política (sem canal elegível).",
    # next-best-action
    "NBA_GENERATED": "Mensagem e próximo passo (NBA) gerados por template governado.",
}


def describe(code: str) -> str:
    """Human-readable description for a reason code (or the code itself)."""
    return REASON_CODES.get(code, code)


def enrich(codes: list[str]) -> list[dict[str, str]]:
    """Expand reason codes into ``{code, description}`` pairs for responses."""
    return [{"code": c, "description": describe(c)} for c in codes]
