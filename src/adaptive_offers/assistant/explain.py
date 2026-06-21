"""Assistant: explain decisions and summarise experiments (grounded by RAG)."""

from __future__ import annotations

import re
import time
from typing import Any

from adaptive_offers.assistant.llm import LLMClient
from adaptive_offers.assistant.rag import PolicyRAG
from adaptive_offers.logging_utils import get_logger
from adaptive_offers.policy.reason_codes import describe

logger = get_logger("assistant.explain")


def _strip_md(text: str) -> str:
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"^[#>\-\*\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*|\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_sentences(text: str, n: int = 1) -> str:
    parts = re.split(r"(?<=[.!?])\s+", _strip_md(text))
    out = " ".join(p for p in parts[:n] if p).strip()
    return out or "—"


_POLICY_EXPLAIN: dict[str, str] = {
    "linucb": (
        "LinUCB (Bandido Contextual Linear) aprende um vetor de pesos w por braço "
        "via regressão ridge — cada decisão usa o vetor de contexto do cliente para "
        "estimar P(conversão) com incerteza bayesiana pelo intervalo UCB."
    ),
    "lin_thompson": (
        "LinThompson (Thompson Sampling Linear Contextual, Agrawal & Goyal 2013) mantém "
        "um posterior Bayesiano N(θ_a, v²·A_a⁻¹) por braço e amostra w̃ ~ N(θ_a, Σ_a) "
        "a cada decisão — exploração puramente Bayesiana sem hiperparâmetro α, "
        "usando o mesmo modelo linear do LinUCB mas com seleção estocástica."
    ),
    "thompson": (
        "Thompson Sampling mantém distribuições Beta(α, β) para cada braço e amostra "
        "θ ~ Beta a cada decisão — exploração bayesiana natural que equilibra "
        "incerteza e valor esperado sem hiperparâmetros fixos."
    ),
    "nilos_ucb": (
        "Nilos-UCB combina média empírica com bônus de exploração variance-aware "
        "μ + c·√(ln t / n_a) — mais explorador que Thompson em cenários de cold-start "
        "e sensível à variância das recompensas por braço."
    ),
    "baseline": (
        "Política Baseline Greedy seleciona o braço de maior média histórica sem "
        "mecanismo de exploração — serve como benchmark determinístico de comparação."
    ),
}

_GUARDRAIL_LABELS: dict[str, str] = {
    "SUITABILITY_OK":            "✓ Gate de suitability/elegibilidade aprovado",
    "ELIGIBILITY_FILTERED":      "✓ Braços inelegíveis filtrados antes da seleção",
    "MARGIN_WEIGHTED":           "✓ Ranqueamento por P(conv) × margem da oferta",
    "VALUE_FLOOR_OK":            "✓ Valor esperado acima do piso mínimo configurado",
    "COLD_START_PULL":           "⚡ Cold-start: puxada inicial obrigatória de braço novo",
    "EXPLORATION_FLOOR_APPLIED": "⚡ Piso de exploração ativado (guardrail de diversidade)",
    "CONTROL_FALLBACK":          "⚠ Nenhuma oferta elegível — controle (no_offer) aplicado",
}


class Assistant:
    """Combines policy RAG + an LLM client into grounded explanations."""

    def __init__(self, rag: PolicyRAG | None = None, llm: LLMClient | None = None) -> None:
        self.rag = rag or PolicyRAG()
        self.llm = llm or LLMClient()

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def explain_decision(
        self, decision: dict[str, Any], question: str = "", top_k: int = 3
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        arm_name = decision.get("arm_name", decision.get("arm_id", "?"))
        rag_query = question or f"oferta {arm_name} elegibilidade margem suitability"

        # --- RAG retrieval ---------------------------------------------------
        t_rag0 = time.perf_counter()
        chunks = self.rag.retrieve(rag_query, top_k=top_k)
        t_rag = time.perf_counter() - t_rag0
        logger.info(
            '{"event":"rag_retrieve","query":"%s","chunks":%d,"top_score":%.3f,"ms":%.1f}',
            rag_query, len(chunks), chunks[0].score if chunks else 0, t_rag * 1000,
        )

        # --- LLM generation --------------------------------------------------
        t_llm0 = time.perf_counter()
        if self.llm.provider == "offline":
            answer = self._offline_explain(decision, arm_name, chunks)
            provider_display = "análise ML"
            prompt_used = "[offline — sem chamada externa]"
        else:
            # LLM path: build grounded prompt and call the model
            reason_h = "; ".join(
                describe(c).rstrip(".") for c in decision.get("reason_codes", [])[:4]
            )
            mode = (
                "exploração (testando uma alternativa promissora)"
                if decision.get("explored")
                else "explotação (melhor estimativa atual)"
            )
            rag_ctx = "\n".join(
                f"- {_first_sentences(c.text, 2)} (fonte: {c.source}, score {c.score:.2f})"
                for c in chunks[:3]
            ) or "—"
            grounding = (
                f"Oferta recomendada: **{arm_name}**\n"
                f"Política: `{decision.get('policy_name')}@{decision.get('policy_version')}` "
                f"(modo: {mode})\n"
                f"Valor esperado: R$ {decision.get('expected_reward')}\n"
                f"Reason codes: {reason_h}\n"
                f"Contexto comercial (RAG):\n{rag_ctx}"
            )
            sys_prompt = (
                "Você é um cientista de dados sênior (nível doutorado) especialista em "
                "multi-armed bandits contextuais e governança de ML. Seja técnico, preciso "
                "e claro. Responda em português e use SOMENTE os fatos fornecidos — "
                "jamais invente números ou afirmações."
            )
            prompt_used = (
                "Escreva um PARECER TÉCNICO da decisão de oferta abaixo, em markdown, "
                "com EXATAMENTE estas seções (cada uma começando com '### '):\n"
                "### Decisão — 1 frase: oferta escolhida e seu valor esperado.\n"
                "### Justificativa técnica — 2-3 frases: por que a política bandit selecionou "
                "este braço (estimativa contextual de P(conversão), trade-off "
                "exploração/explotação, papel da incerteza no score).\n"
                "### Risco & governança — 1-2 frases: guardrails de elegibilidade/suitability "
                "aplicados e o que monitorar (drift, fairness de exposição).\n"
                "### Leitura comercial — 1 frase conectando a decisão à política comercial (RAG).\n\n"
                "Use SOMENTE os fatos abaixo:\n\n" + grounding
            )
            raw = self.llm.complete(prompt_used, system=sys_prompt, grounding=grounding)
            if raw.startswith("Resumo (modo offline"):
                answer = self._offline_explain(decision, arm_name, chunks)
                provider_display = "análise ML"
            else:
                answer = raw
                provider_display = self.llm.provider

        t_llm = time.perf_counter() - t_llm0
        t_total = time.perf_counter() - t0
        logger.info(
            '{"event":"explain_done","provider":"%s","llm_ms":%.1f,"total_ms":%.1f}',
            provider_display, t_llm * 1000, t_total * 1000,
        )

        return {
            "answer": answer,
            "provider": provider_display,
            "citations": [
                {"source": c.source, "score": c.score, "text": _first_sentences(c.text, 2)}
                for c in chunks
            ],
            "_trace": {
                "rag_query": rag_query,
                "rag_ms": round(t_rag * 1000, 1),
                "llm_ms": round(t_llm * 1000, 1),
                "total_ms": round(t_total * 1000, 1),
                "chunks_retrieved": len(chunks),
                "model": getattr(self.llm, "model", "—"),
                "prompt_snippet": prompt_used[:300] + ("…" if len(prompt_used) > 300 else ""),
            },
        }

    def summarize_experiment(
        self, metrics_rows: list[dict[str, Any]], top_k: int = 2
    ) -> dict[str, Any]:
        if not metrics_rows:
            return {"answer": "Sem métricas para resumir.", "provider": self.llm.provider, "citations": []}
        best = max(metrics_rows, key=lambda r: r.get("cumulative_reward", 0))
        worst = min(metrics_rows, key=lambda r: r.get("cumulative_reward", 0))
        chunks = self.rag.retrieve("ranqueamento margem exploração bandit", top_k=top_k)

        if self.llm.provider == "offline":
            answer = self._offline_summarize(metrics_rows, best, worst)
            provider_display = "análise ML"
        else:
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
            provider_display = self.llm.provider

        return {
            "answer": answer,
            "provider": provider_display,
            "best_policy": best["policy"],
            "citations": [{"source": c.source, "score": c.score} for c in chunks],
        }

    # ---------------------------------------------------------------------- #
    # Offline deterministic generation
    # ---------------------------------------------------------------------- #

    def _offline_explain(
        self,
        decision: dict[str, Any],
        arm_name: str,
        chunks: list,
    ) -> str:
        arm_id   = decision.get("arm_id", arm_name)
        codes    = decision.get("reason_codes", [])
        explored = bool(decision.get("explored"))
        policy   = decision.get("policy_name", "?")
        version  = decision.get("policy_version", "v1")
        reward   = float(decision.get("expected_reward", 0))

        # Mode
        if explored:
            mode_label  = "EXPLORAÇÃO"
            mode_detail = (
                "O bandit selecionou este braço para reduzir incerteza epistêmica — "
                "a estimativa de valor ainda não é suficientemente confiante. "
                "UCB/Thompson regulam a taxa de exploração automaticamente, sem intervenção humana."
            )
        else:
            mode_label  = "EXPLOTAÇÃO"
            mode_detail = (
                "O modelo convergiu para este braço como melhor estimativa de valor esperado "
                "dado o vetor de contexto deste cliente. "
                "Após aprendizado suficiente, o algoritmo direciona a maior parte das decisões "
                "para o braço de máximo P(conv) × margem."
            )

        # Policy ML explanation
        ml_explain = _POLICY_EXPLAIN.get(policy, f"Política {policy} aplicada.")

        # Guardrails
        guardrails = [_GUARDRAIL_LABELS[c] for c in codes if c in _GUARDRAIL_LABELS]
        if not guardrails:
            guardrails = ["• Nenhum guardrail específico registrado para esta decisão."]
        else:
            guardrails = [f"• {g}" for g in guardrails]

        # RAG — smarter extraction: search for the selected arm specifically
        rag_lines = []
        arm_keywords = {arm_name.lower(), arm_id.lower()}
        for chunk in chunks:
            raw = _strip_md(chunk.text).strip()
            if not raw or len(raw) < 30:
                continue
            # Skip chunks starting mid-word (broken by character-limit chunking)
            if raw[0].islower():
                continue
            # Prefer chunks that mention the chosen arm
            mentions_arm = any(kw in raw.lower() for kw in arm_keywords)
            if mentions_arm:
                # Extract the sentence(s) about this arm specifically
                sentences = [s.strip() for s in re.split(r"\.\s+", raw) if s.strip()]
                arm_sentences = [
                    s for s in sentences
                    if any(kw in s.lower() for kw in arm_keywords)
                ]
                excerpt = ". ".join(arm_sentences[:2]) if arm_sentences else _first_sentences(raw, 2)
            else:
                excerpt = _first_sentences(raw, 2)
            if excerpt and len(excerpt) > 20:
                rag_lines.append(
                    f"• {excerpt} _(relevância: {chunk.score:.2f} · {chunk.source})_"
                )
            if len(rag_lines) >= 2:
                break
        if not rag_lines:
            rag_lines = ["• Nenhuma política comercial específica recuperada com score relevante."]

        # Financial interpretation
        if reward > 50:
            fin_comment = "Oferta de alto valor esperado — adequada para maximização de receita por contato."
        elif reward > 20:
            fin_comment = "Valor esperado moderado — equilibra volume e margem neste perfil de cliente."
        else:
            fin_comment = "Valor esperado baixo — possível perfil de difícil conversão ou baixa margem."

        # Compose professional structured response
        parts = [
            f"**{arm_name}** · `{policy}@{version}` · Modo: **{mode_label}**",
            f"Valor esperado: **R$ {reward:.1f}** = P(conversão | x) × margem_a",
            "---",
            f"### Análise financeira",
            fin_comment,
            "",
            f"### Algoritmo — {policy.upper()}",
            ml_explain,
            "",
            f"### Modo: {mode_label}",
            mode_detail,
            "",
            "### Governança aplicada",
            *guardrails,
            "",
            "### Política comercial (RAG)",
            *rag_lines,
        ]
        return "\n".join(parts)

    def _offline_summarize(
        self,
        rows: list[dict[str, Any]],
        best: dict[str, Any],
        worst: dict[str, Any],
    ) -> str:
        ranked = sorted(rows, key=lambda r: r.get("cumulative_reward", 0), reverse=True)
        best_policy = best.get("policy", "?")
        best_reward = best.get("cumulative_reward", 0)
        best_regret = best.get("regret_ratio", 0)
        best_conv   = best.get("conversion_rate", 0)
        worst_policy = worst.get("policy", "?")

        ml_explain = _POLICY_EXPLAIN.get(best_policy, f"Política {best_policy}.")

        lines = []
        for r in ranked:
            pol  = r.get("policy", "?")
            rew  = r.get("cumulative_reward", 0)
            reg  = r.get("regret_ratio", 0)
            conv = r.get("conversion_rate", 0)
            expl = r.get("exploration_rate", 0)
            lift = r.get("lift_vs_baseline_pct", 0)
            star = " ← **MELHOR**" if pol == best_policy else (" ← pior" if pol == worst_policy else "")
            lines.append(
                f"• **{pol}** — reward={rew:,.0f} · regret={reg:.1%} · "
                f"conv={conv:.1%} · exploração={expl:.1%} · lift={lift:+.0f}%{star}"
            )

        parts = [
            f"**Melhor política:** {best_policy.upper()} · "
            f"Reward={best_reward:,.0f} · Regret={best_regret:.1%} · Conv={best_conv:.1%}",
            "",
            f"**Modelo vencedor:** {ml_explain}",
            "",
            "**Ranking completo:**",
            *lines,
            "",
            "**Interpretação:** A política de melhor desempenho maximizou receita "
            "aprendendo a selecionar ofertas de maior P(conversão) × margem para cada "
            "perfil de cliente. O trade-off exploração/explotação é gerenciado "
            "automaticamente pelo algoritmo — sem intervenção humana por decisão.",
        ]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #

def explain_decision(decision: dict[str, Any], question: str = "", top_k: int = 3) -> dict[str, Any]:
    return Assistant().explain_decision(decision, question=question, top_k=top_k)


def summarize_experiment(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return Assistant().summarize_experiment(metrics_rows)
