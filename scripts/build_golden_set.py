"""Generate the versioned golden set (data/golden_set/evaluation_cases.jsonl).

Each case's expected action is made **oracle-consistent**: we compute the true
expected-reward ranking from the Stage-2 latent model so the expectations are
correct by construction. Adversarial cases instead assert hard guardrail
invariants (forbidden arms must never be chosen).

Run: ``python scripts/build_golden_set.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_offers.data.synthetic import (  # noqa: E402
    build_context_vector,
    eligible_arms,
    expected_reward,
    offer_catalog,
)

CATALOG = offer_catalog()
BY_ID = {a.offer_id: a for a in CATALOG}
RATE_MEDIAN = 2.5  # reference; cases use euribor clearly above/below this


def oracle_ranking(ctx_row: dict) -> list[tuple[str, float]]:
    ctx = build_context_vector(ctx_row, RATE_MEDIAN)
    elig = eligible_arms(ctx_row, CATALOG)
    ranked = sorted(
        ((a.offer_id, expected_reward(a, ctx)) for a in elig),
        key=lambda kv: kv[1], reverse=True,
    )
    return ranked


def base_ctx(**over) -> dict:
    ctx = {
        "age": 40, "contact": "cellular", "poutcome": "nonexistent",
        "previously_contacted": 0, "euribor3m": 4.5, "default": "no",
        "loan": "no", "marital": "married", "job": "admin.",
        "education": "university.degree",
    }
    ctx.update(over)
    return ctx


# (category, description, context, assertion_strategy, justification)
# assertion_strategy: "top1" | "top2" | guardrail dict
SPECS: list[tuple[str, str, dict, object, str]] = [
    # ---- typical -------------------------------------------------------
    ("typical", "Sênior com sucesso prévio e juros baixos",
     base_ctx(age=66, poutcome="success", euribor3m=0.8, previously_contacted=1),
     "top2", "Sucesso prévio + sênior favorece depósito/fundo (alto valor)."),
    ("typical", "Jovem no canal celular",
     base_ctx(age=24, contact="cellular"),
     "top2", "Jovem + celular maximiza afinidade do cartão cashback."),
    ("typical", "Cliente já contatado, juros baixos, sem empréstimo",
     base_ctx(age=38, previously_contacted=1, euribor3m=0.8, loan="no"),
     "top2", "Empréstimo pré-aprovado tem maior margem para este perfil."),
    ("typical", "Sênior sem histórico",
     base_ctx(age=63, poutcome="nonexistent"),
     "top2", "Perfil sênior tende a depósito/seguro."),
    ("typical", "Meia-idade, canal telefone, juros altos",
     base_ctx(age=47, contact="telephone", euribor3m=4.8),
     "value", "Sem drivers fortes; exige oferta elegível de valor esperado acima do piso."),
    ("typical", "Sucesso prévio, meia-idade",
     base_ctx(age=44, poutcome="success", euribor3m=4.5),
     "top2", "Sucesso prévio eleva depósito premium."),
    ("typical", "Jovem com sucesso prévio",
     base_ctx(age=27, poutcome="success", contact="cellular"),
     "top2", "Combina afinidade jovem e sinal de sucesso."),
    ("typical", "Sênior, juros baixos, já contatado",
     base_ctx(age=68, euribor3m=0.8, previously_contacted=1, poutcome="success"),
     "top2", "Alto valor: depósito/fundo para sênior engajado."),
    # ---- segment -------------------------------------------------------
    ("segment", "Segmento jovem-celular puro",
     base_ctx(age=22, contact="cellular", poutcome="nonexistent"),
     "top2", "Valida roteamento do cartão para o segmento jovem."),
    ("segment", "Segmento sênior-investidor",
     base_ctx(age=70, poutcome="success", euribor3m=0.8),
     "top2", "Valida roteamento de produtos de investimento ao sênior."),
    ("segment", "Segmento reengajamento (previously_contacted)",
     base_ctx(age=35, previously_contacted=1, euribor3m=0.8, loan="no"),
     "top2", "Valida priorização de empréstimo no reengajamento."),
    ("segment", "Segmento canal telefone",
     base_ctx(age=50, contact="telephone"),
     "value", "Mesmo em telefone deve haver oferta elegível de valor acima do piso."),
    ("segment", "Segmento sem histórico, juros altos",
     base_ctx(age=41, poutcome="nonexistent", euribor3m=4.8),
     "value", "Contexto neutro; valida fallback de valor acima do piso."),
    ("segment", "Segmento jovem juros baixos",
     base_ctx(age=26, euribor3m=0.8, contact="cellular"),
     "top2", "Jovem + juros baixos; cartão ou empréstimo."),
    # ---- edge ----------------------------------------------------------
    ("edge", "Idade exatamente no limite do fundo (25)",
     base_ctx(age=25, poutcome="success"),
     "top2", "Idade mínima do fundo = 25; deve ser elegível."),
    ("edge", "Idade 24 (fundo inelegível)",
     base_ctx(age=24, poutcome="success"),
     {"type": "not_choose", "arm_ids": ["OFF_FUND_INTRO"]},
     "Abaixo de 25 o fundo é inelegível; não deve ser ofertado."),
    ("edge", "Todos sinais neutros",
     base_ctx(age=40, poutcome="nonexistent", euribor3m=2.6, previously_contacted=0),
     "value", "Contexto neutro; qualquer oferta elegível de valor acima do piso é aceitável."),
    ("edge", "Cliente muito idoso",
     base_ctx(age=92, poutcome="success", euribor3m=0.8),
     "top2", "Extremo etário não deve quebrar a decisão."),
    ("edge", "Cliente jovem no limite (18)",
     base_ctx(age=18, contact="cellular"),
     "top2", "Idade mínima legal; cartão exige sem default."),
    # ---- adversarial (guardrails) -------------------------------------
    ("adversarial", "Default=yes não pode receber cartão nem empréstimo",
     base_ctx(age=30, default="yes"),
     {"type": "not_choose", "arm_ids": ["OFF_CC_CASHBACK", "OFF_LOAN_PREAPP"]},
     "Suitability: clientes em default são inelegíveis a crédito."),
    ("adversarial", "Loan=yes não pode receber novo empréstimo",
     base_ctx(age=35, loan="yes", previously_contacted=1, euribor3m=0.8),
     {"type": "not_choose", "arm_ids": ["OFF_LOAN_PREAPP"]},
     "Já possui empréstimo; oferta de novo empréstimo é vetada."),
    ("adversarial", "Reward-hacking: contexto força margem alta porém inelegível",
     base_ctx(age=23, default="yes", loan="yes", previously_contacted=1, euribor3m=0.8),
     {"type": "not_choose", "arm_ids": ["OFF_LOAN_PREAPP", "OFF_CC_CASHBACK", "OFF_FUND_INTRO"]},
     "Mesmo com contexto 'atraente', gates de elegibilidade prevalecem."),
    ("adversarial", "Default=yes + idoso: apenas ofertas permitidas",
     base_ctx(age=67, default="yes", poutcome="success", euribor3m=0.8),
     {"type": "not_choose", "arm_ids": ["OFF_CC_CASHBACK", "OFF_LOAN_PREAPP"]},
     "Crédito vetado; depósito/seguro/controle permanecem elegíveis."),
    ("adversarial", "Cliente sem oferta elegível atraente aceita controle",
     base_ctx(age=24, default="yes", loan="yes"),
     {"type": "eligible_only", "arm_ids": []},
     "Quando crédito é vetado e nada converte bem, controle (no_offer) é aceitável."),
]


def build() -> list[dict]:
    cases: list[dict] = []
    for n, (cat, desc, ctx, strat, just) in enumerate(SPECS, start=1):
        ranked = oracle_ranking(ctx)
        if isinstance(strat, dict):
            assertion = strat
            exp_min = 0.0
            pf = "PASS se a oferta escolhida respeita o guardrail e é elegível."
        elif strat == "value":
            # Neutral/fallback context: assert eligibility + a value floor, not a
            # specific arm (multiple arms have near-equal expected reward).
            assertion = {"type": "eligible_only", "arm_ids": []}
            exp_min = round(0.45 * ranked[0][1], 2)
            pf = (f"PASS se a oferta é elegível, com reason codes e reward "
                  f"esperado ≥ {exp_min} (piso de valor).")
        else:
            k = 2 if strat == "top2" else 1
            top = [arm for arm, _ in ranked[:k]]
            assertion = {"type": "choose_one_of", "arm_ids": top}
            exp_min = round(0.55 * ranked[0][1], 2)  # value floor = 55% of oracle best
            pf = (f"PASS se oferta ∈ {top}, elegível, com reason codes e "
                  f"reward esperado ≥ {exp_min}.")
        cases.append({
            "case_id": f"GS-{n:03d}",
            "category": cat,
            "description": desc,
            "context": ctx,
            "assertion": assertion,
            "expected_reward_min": exp_min,
            "oracle_top3": [{"arm": a, "exp_reward": round(r, 2)} for a, r in ranked[:3]],
            "justification": just,
            "pass_fail": pf,
        })
    return cases


def main() -> None:
    out = ROOT / "data" / "golden_set" / "evaluation_cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = build()
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print(f"wrote {len(cases)} cases -> {out.relative_to(ROOT)}")
    print("by category:", by_cat)


if __name__ == "__main__":
    main()
