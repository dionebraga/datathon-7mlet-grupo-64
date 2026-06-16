"""Assistant: explain decisions and summarise experiments (grounded by RAG)."""

from __future__ import annotations

from typing import Any

from adaptive_offers.assistant.llm import LLMClient
from adaptive_offers.assistant.rag import PolicyRAG
from adaptive_offers.policy.reason_codes import describe


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

        reasons = ", ".join(decision.get("reason_codes", []))
        reason_expl = "; ".join(describe(c) for c in decision.get("reason_codes", []))
        elig = ", ".join(decision.get("eligible_arms", []))
        snippet = chunks[0].text if chunks else "(sem política recuperada)"

        grounding = (
            f"Pergunta: {question or 'Por que esta oferta foi escolhida?'}\n"
            f"Decisão: oferta '{arm_name}' ({decision.get('arm_id')}) pela política "
            f"{decision.get('policy_name')}@{decision.get('policy_version')}.\n"
            f"Exploração: {decision.get('explored')}. "
            f"Score (margem-ponderado): {decision.get('score')}. "
            f"Valor esperado: {decision.get('expected_reward')}.\n"
            f"Reason codes: {reasons}.\n"
            f"Interpretação: {reason_expl}.\n"
            f"Braços elegíveis: {elig}.\n"
            f"Política comercial relevante (RAG): {snippet}"
        )
        prompt = (
            "Explique, para um público de negócio, a decisão de oferta abaixo, "
            "usando SOMENTE os fatos fornecidos. Seja conciso e cite a política.\n\n"
            + grounding
        )
        answer = self.llm.complete(prompt, grounding=grounding)
        return {
            "answer": answer,
            "provider": self.llm.provider,
            "citations": [
                {"source": c.source, "score": c.score, "text": c.text} for c in chunks
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
