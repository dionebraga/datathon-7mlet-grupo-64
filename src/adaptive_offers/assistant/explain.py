"""Assistant: explain decisions and summarise experiments (grounded by RAG)."""

from __future__ import annotations

import re
from typing import Any

from adaptive_offers.assistant.llm import LLMClient
from adaptive_offers.assistant.rag import PolicyRAG
from adaptive_offers.policy.reason_codes import describe


def _strip_md(text: str) -> str:
    """Remove markdown noise so RAG snippets read as plain prose in the UI."""
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"^[#>\-\*\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*|\*|_", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_sentences(text: str, n: int = 1) -> str:
    """First ``n`` sentences of a cleaned text (keeps explanations short)."""
    parts = re.split(r"(?<=[.!?])\s+", _strip_md(text))
    out = " ".join(p for p in parts[:n] if p).strip()
    return out or "—"


class Assistant:
    """Combines policy RAG + an LLM client into grounded explanations."""

    def __init__(self, rag: PolicyRAG | None = None, llm: LLMClient | None = None) -> None:
        self.rag = rag or PolicyRAG()
        self.llm = llm or LLMClient()

    def explain_decision(
        self, decision: dict[str, Any], question: str = "", top_k: int = 3
    ) -> dict[str, Any]:
        arm_name = decision.get("arm_name", decision.get("arm_id", "?"))
        query = question or f"oferta {arm_name} elegibilidade margem suitability"
        chunks = self.rag.retrieve(query, top_k=top_k)

        reason_h = "; ".join(describe(c).rstrip(".") for c in decision.get("reason_codes", [])[:4])
        mode = ("exploração (testando uma alternativa promissora)"
                if decision.get("explored") else "explotação (melhor estimativa atual)")
        policy_line = _first_sentences(chunks[0].text, 1) if chunks else "—"

        # Clean, concise, formatted explanation — no raw markdown dump.
        clean = (
            f"A oferta **{arm_name}** foi recomendada pela política "
            f"`{decision.get('policy_name')}@{decision.get('policy_version')}` "
            f"em modo de **{mode}**, com valor esperado de "
            f"**R$ {decision.get('expected_reward')}**.\n\n"
            f"**Por quê:** {reason_h}.\n\n"
            f"**Base comercial (RAG):** {policy_line}"
        )

        if self.llm.provider == "offline":
            answer = clean
        else:
            prompt = (
                "Explique, para um público de negócio, a decisão de oferta abaixo de "
                "forma concisa (2-3 frases), usando SOMENTE os fatos fornecidos:\n\n" + clean
            )
            answer = self.llm.complete(prompt, grounding=clean)

        return {
            "answer": answer,
            "provider": self.llm.provider,
            "citations": [
                {"source": c.source, "score": c.score, "text": _first_sentences(c.text, 2)}
                for c in chunks
            ],
        }

    def summarize_experiment(self, metrics_rows: list[dict[str, Any]], top_k: int = 2) -> dict[str, Any]:
        if not metrics_rows:
            return {"answer": "Sem métricas para resumir.", "provider": self.llm.provider, "citations": []}
        best = max(metrics_rows, key=lambda r: r.get("cumulative_reward", 0))
        worst = min(metrics_rows, key=lambda r: r.get("cumulative_reward", 0))
        chunks = self.rag.retrieve("ranqueamento margem exploração bandit", top_k=top_k)
        lines = [
            f"- {r['policy']}: reward={r.get('cumulative_reward')}, "
            f"regret_ratio={r.get('regret_ratio')}, conversão={r.get('conversion_rate')}, "
            f"exploração={r.get('exploration_rate')}"
            for r in metrics_rows
        ]
        grounding = (
            "Comparação de políticas (experimento):\n" + "\n".join(lines)
            + f"\nMelhor por valor: {best['policy']} (reward {best.get('cumulative_reward')}). "
            f"Pior: {worst['policy']}."
        )
        prompt = (
            "Resuma o experimento de bandit a seguir para um público técnico e de "
            "negócio, destacando a melhor política por valor e o trade-off "
            "exploração/explotação. Use apenas os números fornecidos.\n\n" + grounding
        )
        answer = self.llm.complete(prompt, grounding=grounding)
        return {
            "answer": answer,
            "provider": self.llm.provider,
            "best_policy": best["policy"],
            "citations": [{"source": c.source, "score": c.score} for c in chunks],
        }


def explain_decision(decision: dict[str, Any], question: str = "", top_k: int = 3) -> dict[str, Any]:
    return Assistant().explain_decision(decision, question=question, top_k=top_k)


def summarize_experiment(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return Assistant().summarize_experiment(metrics_rows)
