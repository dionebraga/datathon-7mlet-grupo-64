"""Next-Best-Action — turn the chosen *offer* into a *message* and a *next step*.

The challenge asks the platform to decide "qual **oferta, mensagem ou próximo
passo**". The bandit answers the first part (which arm); this module closes the
loop with the other two:

* a **message** (headline + body) whose *tone* adapts to the behavioural persona
  (``segmentation.py``) and whose *delivery hint* adapts to the chosen channel
  (``channels.py``); and
* a **next-best-action** — the concrete, machine-readable next step a customer
  (or an agent) should take (``SIMULATE_LOAN``, ``OPEN_DEPOSIT``, …) with its
  call-to-action label.

Everything here is **deterministic and template-based** (no generation at serving
time), so each decision carries a reproducible, auditable message + action. The
LLM assistant (``assistant/``) may *explain* a decision, but the served copy is
governed templates — never free-form model output — which keeps suitability and
compliance reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NextBestAction:
    """The concrete message + next step derived from a decision."""

    action_code: str  # machine-readable next step (e.g. SIMULATE_LOAN)
    headline: str     # message subject / headline
    message: str      # body copy (persona- and channel-aware)
    cta: str          # call-to-action label


# Per-offer next step: (action_code, headline, core body, CTA).
_OFFER_NBA: dict[str, tuple[str, str, str, str]] = {
    "OFF_CC_CASHBACK": (
        "ACTIVATE_CARD", "Seu Cartão Cashback está liberado",
        "um cartão sem anuidade que devolve parte de cada compra em dinheiro.",
        "Ativar cashback",
    ),
    "OFF_LOAN_PREAPP": (
        "SIMULATE_LOAN", "Você tem crédito pré-aprovado",
        "um empréstimo pré-aprovado com taxa personalizada e sem garantia.",
        "Ver valor pré-aprovado",
    ),
    "OFF_TD_PREMIUM": (
        "OPEN_DEPOSIT", "Depósito a Prazo Premium para o seu perfil",
        "um depósito a prazo com rendimento acima da poupança e liquidez programada.",
        "Aplicar agora",
    ),
    "OFF_FUND_INTRO": (
        "SIMULATE_FUND", "Comece a investir em fundos",
        "um fundo de entrada diversificado, ideal para dar o primeiro passo em investimentos.",
        "Simular rendimento",
    ),
    "OFF_INSURANCE": (
        "QUOTE_INSURANCE", "Proteção que cabe no seu dia a dia",
        "um seguro bundle com cobertura essencial e mensalidade enxuta.",
        "Ver cobertura",
    ),
    "OFF_NONE": (
        "NO_ACTION", "Sem oferta no momento",
        "nenhuma oferta é adequada agora; manter relacionamento sem contato comercial.",
        "",
    ),
}

# Persona-aware opener: sets the *tone* of the message per behavioural segment.
_SEGMENT_OPENER: dict[str, str] = {
    "seg_renegociador": "Pensando em organizar suas finanças, separamos para você",
    "seg_senior_conserv": "Com a segurança que o seu perfil pede, preparamos para você",
    "seg_jovem_digital": "Feito pro seu ritmo: dá pra resolver em segundos —",
    "seg_recorrente": "Porque você já é cliente e confia na gente, liberamos",
    "seg_novo_cold": "Para começar com o pé direito, preparamos para você",
    "seg_massa": "Selecionamos para você",
}

# Channel-aware delivery hint appended to the body.
_CHANNEL_HINT: dict[str, str] = {
    "app_push": "Toque na notificação para abrir no app.",
    "email": "Responda este e-mail ou clique no botão para continuar.",
    "sms": "Responda SIM para receber o link seguro.",
    "call": "Um especialista pode te explicar agora pelo telefone.",
}


def next_best_action(
    offer_id: str, segment_id: str = "seg_massa", channel_id: str = ""
) -> NextBestAction:
    """Compose the message + next step for a decision (deterministic).

    Tolerant to unknown ids: falls back to neutral persona / no channel hint, so
    it never fails on a partial decision record.
    """
    action_code, headline, core, cta = _OFFER_NBA.get(
        offer_id, _OFFER_NBA["OFF_NONE"]
    )

    if offer_id == "OFF_NONE":
        return NextBestAction(action_code, headline, core, cta)

    opener = _SEGMENT_OPENER.get(segment_id, _SEGMENT_OPENER["seg_massa"])
    hint = _CHANNEL_HINT.get(channel_id, "")
    body = f"{opener} {core}".strip()
    if hint:
        body = f"{body} {hint}"
    return NextBestAction(action_code, headline, body, cta)
