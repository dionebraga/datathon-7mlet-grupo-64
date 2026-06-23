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

_ARM_LABELS: dict[str, str] = {
    "OFF_CC_CASHBACK": "Cartão Cashback", "OFF_LOAN_PREAPP": "Empréstimo Pré-aprovado",
    "OFF_TD_PREMIUM": "Depósito a Prazo Premium", "OFF_FUND_INTRO": "Fundo de Investimento Intro",
    "OFF_INSURANCE": "Seguro Bundle", "OFF_NONE": "Sem Oferta (controle)",
}


def _arm_label(arm_id: str) -> str:
    """Human-readable name for an offer id (falls back to a cleaned id)."""
    return _ARM_LABELS.get(arm_id, str(arm_id).replace("OFF_", "").replace("_", " ").title())


def _orchestration_grounding(decision: dict[str, Any]) -> str:
    """Facts about the persona / channel / next-step layers, for the LLM to cite.

    Returns an empty string when none of the orchestration fields are present, so
    older decision records stay compatible.
    """
    seg = decision.get("segment_label")
    chan = decision.get("channel_label")
    nba_action = decision.get("nba_action")
    nba_cta = decision.get("nba_cta")
    codes = decision.get("reason_codes", [])
    if not any((seg, chan, nba_action)):
        return ""
    why = []
    if "QUIET_HOURS" in codes:
        why.append("horário de silêncio derrubou canais intrusivos")
    if "FREQUENCY_CAPPED" in codes:
        why.append("limite de contato atingido")
    if "CHANNEL_SELECTED" in codes and not why:
        why.append("menor custo / preferência do cliente")
    why_txt = f" ({'; '.join(why)})" if why else ""
    lines = []
    if seg:
        lines.append(f"Persona comportamental: {seg}")
    if chan:
        lines.append(f"Canal de entrega escolhido: {chan}{why_txt}")
    if nba_action or nba_cta:
        step = " / ".join(x for x in (nba_action, nba_cta) if x)
        lines.append(f"Próximo passo (NBA): {step}")
    return "Orquestração (persona · canal · próximo passo):\n" + "\n".join(
        f"  {ln}" for ln in lines
    ) + "\n"


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
            # Concrete per-arm ranking (real margin-weighted scores) for the LLM to cite.
            scores = decision.get("scores") or {}
            ranking = "—"
            if isinstance(scores, dict) and len(scores) >= 2:
                ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
                ranking = "\n".join(
                    f"  {i + 1}º {_arm_label(k)}: score {float(v):.2f}"
                    + (" ← escolhido" if k == decision.get("arm_id") else "")
                    for i, (k, v) in enumerate(ranked)
                )
            estimates = decision.get("estimates") or {}
            p_chosen = estimates.get(decision.get("arm_id"))
            p_line = f"\nP(conversão | contexto) do braço escolhido: {p_chosen:.1%}" if p_chosen else ""
            orch = _orchestration_grounding(decision)
            grounding = (
                f"Oferta recomendada: **{arm_name}**\n"
                f"Política: `{decision.get('policy_name')}@{decision.get('policy_version')}` "
                f"(modo: {mode})\n"
                f"Valor esperado (P × margem): R$ {decision.get('expected_reward')}"
                f"{p_line}\n"
                f"Ranking dos braços elegíveis por score ponderado por margem:\n{ranking}\n"
                f"Reason codes: {reason_h}\n"
                f"{orch}"
                f"Contexto comercial (RAG):\n{rag_ctx}"
            )
            sys_prompt = (
                "Você é um cientista de dados sênior (nível doutorado) especialista em "
                "multi-armed bandits contextuais e governança de ML. Seja técnico, preciso "
                "e claro. Responda em português e use SOMENTE os fatos fornecidos — "
                "jamais invente números ou afirmações."
            )
            prompt_used = (
                "Escreva um PARECER TÉCNICO da decisão de oferta abaixo, em markdown. "
                "Use EXATAMENTE estes 5 títulos curtos (copie cada um literalmente, sem "
                "acrescentar a descrição ao título), cada um numa linha iniciando por '### ', "
                "seguido do conteúdo pedido:\n\n"
                "1) `### Decisão` — 1 frase: oferta escolhida e seu valor esperado.\n"
                "2) `### Justificativa técnica` — 2-3 frases: por que a política bandit "
                "selecionou este braço (P(conversão) contextual, trade-off "
                "exploração/explotação, papel da incerteza no score).\n"
                "3) `### Por que venceu os concorrentes` — 1-2 frases citando os NÚMEROS reais "
                "do ranking fornecido: nomeie o 2º colocado e a diferença de score ponderado "
                "para o escolhido. Jamais invente valores; use apenas o ranking dado.\n"
                "4) `### Risco & governança` — 1-2 frases: guardrails de "
                "elegibilidade/suitability e o que monitorar (drift, fairness de exposição).\n"
                "5) `### Orquestração & próximo passo` — 1-2 frases: a persona comportamental, "
                "o canal de entrega escolhido (e por quê: custo, horário de silêncio ou "
                "preferência) e o próximo passo (NBA) recomendado. Use só os fatos dados.\n"
                "6) `### Leitura comercial` — 1 frase conectando a decisão à política comercial "
                "(RAG).\n\n"
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
        expec_p  = decision.get("expected_p", None)
        margin   = decision.get("margin", None)

        # Mode with concrete explanation tied to algorithm mechanics
        if explored:
            mode_label  = "EXPLORAÇÃO"
            mode_detail = (
                "O bandit selecionou este braço para reduzir incerteza sobre seu "
                "potencial — a estimativa de P(conversão) ainda tem variância alta. "
                f"A penalidade UCB (termo α·√(xᵀA⁻¹x)) ou a amostragem da posterior "
                f"de Thompson superou o valor estimado dos braços consagrados. "
                "Sem essa exploração periódica, o modelo nunca descobriria ofertas "
                "potencialmente melhores para este perfil."
            )
        else:
            mode_label  = "EXPLOTAÇÃO"
            mode_detail = (
                "O modelo selecionou o braço de maior valor esperado dado o contexto "
                "deste cliente, após aprendizado acumulado das rodadas anteriores. "
                "A incerteza sobre este braço já é baixa o suficiente para que a "
                "penalidade UCB não altere o ranking — o algoritmo prioriza reward. "
                "A taxa de explotação aumenta naturalmente com o número de rodadas, "
                "conforme as estimativas convergem."
            )

        # Policy ML explanation
        ml_explain = _POLICY_EXPLAIN.get(policy, f"Política {policy} aplicada.")

        # Guardrails with concrete detail
        guardrails = [_GUARDRAIL_LABELS[c] for c in codes if c in _GUARDRAIL_LABELS]
        if not guardrails:
            guardrails = ["• Nenhum guardrail específico registrado para esta decisão."]
        else:
            guardrails = [f"• {g}" for g in guardrails]
        # Add policy version info
        guardrails.append(f"• Versão da política: `{policy}@{version}`")

        # RAG — smarter extraction: search for the selected arm specifically
        rag_lines = []
        arm_keywords = {arm_name.lower(), arm_id.lower()}
        for chunk in chunks:
            raw = _strip_md(chunk.text).strip()
            if not raw or len(raw) < 30:
                continue
            if raw[0].islower():
                continue
            mentions_arm = any(kw in raw.lower() for kw in arm_keywords)
            if mentions_arm:
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

        # Financial interpretation with concrete thresholds
        margin_val = margin if margin is not None else 0
        prob_val = expec_p if expec_p is not None else None
        if prob_val is not None:
            prob_line = f"P(conversão | contexto) ≈ **{prob_val:.1%}**"
        else:
            prob_line = f"P(conversão | contexto) ≈ **{reward / max(margin_val, 1):.1%}** (inferida)"
        
        if prob_val is not None and margin_val:
            value_math = f"**{prob_val:.1%} × R$ {margin_val:.0f} = R$ {reward:.1f}**"
        else:
            value_math = f"**R$ {reward:.1f}**"
        tier = ("alto valor esperado" if reward > 50
                else "valor esperado moderado" if reward > 20 else "valor esperado baixo")
        fin_comment = (
            f"Oferta de **{tier}**. O valor é a recompensa esperada por impressão, "
            f"computada como P(conversão) × margem = {value_math}. "
            "Não é necessariamente a oferta de maior conversão isolada, e sim a de maior "
            "**valor** — o ranqueamento pondera a probabilidade pela margem de cada produto."
        )

        # Concrete competitive comparison using the policy's own margin-weighted scores
        scores = decision.get("scores") or {}
        compare_lines: list[str] = []
        if isinstance(scores, dict) and len(scores) >= 2:
            ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
            chosen_score = float(scores.get(arm_id, ranked[0][1]))
            runner = next(((k, float(v)) for k, v in ranked if k != arm_id), None)
            if runner is not None:
                runner_id, runner_score = runner
                gap = chosen_score - runner_score
                rel = (gap / runner_score * 100.0) if runner_score else 0.0
                compare_lines.append(
                    f"• Entre **{len(scores)} braços elegíveis**, venceu o 2º colocado "
                    f"(**{_arm_label(runner_id)}**) por **{gap:+.2f}** no score ponderado "
                    f"(**{chosen_score:.2f}** vs {runner_score:.2f}, {rel:+.0f}%)."
                )
                last_id, last_score = ranked[-1]
                if last_id not in (arm_id, runner_id):
                    compare_lines.append(
                        f"• Último colocado: **{_arm_label(last_id)}** (score {float(last_score):.2f}) "
                        "— penalizado pela margem baixa ou pela menor probabilidade no contexto."
                    )
        if not compare_lines:
            compare_lines = ["• Comparação entre braços indisponível nesta decisão."]

        # Compose professional structured response
        parts = [
            f"**{arm_name}** · `{policy}@{version}` · Modo: **{mode_label}**",
            f"Valor esperado: **R$ {reward:.1f}**",
            f"Fórmula: **E[reward] = P(conv | x) × margem_a** · {prob_line}",
            "---",
            "### Análise financeira",
            fin_comment,
            "",
            "### Por que este braço venceu",
            *compare_lines,
            "",
            f"### Algoritmo — {policy.upper()}",
            ml_explain,
            "",
            f"### Modo: {mode_label}",
            mode_detail,
            "",
        ]

        # Orchestration: persona, channel and next step (NBA), if present.
        seg = decision.get("segment_label")
        chan = decision.get("channel_label")
        nba_cta = decision.get("nba_cta")
        nba_action = decision.get("nba_action")
        if any((seg, chan, nba_action)):
            orch_lines: list[str] = []
            if seg:
                orch_lines.append(f"• **Persona**: {seg} — segmentação comportamental sobre o contexto real.")
            if chan and chan != "—":
                why = ""
                if "QUIET_HOURS" in codes:
                    why = " (horário de silêncio derrubou canais intrusivos como voz/SMS)"
                elif "CHANNEL_SELECTED" in codes:
                    why = " (política de contato: menor custo / preferência do cliente)"
                orch_lines.append(f"• **Canal de entrega**: {chan}{why}.")
            if nba_action or nba_cta:
                step = " — ".join(x for x in (nba_action, nba_cta) if x)
                orch_lines.append(f"• **Próximo passo (NBA)**: {step}, com mensagem por template governado (sem texto livre do LLM).")
            parts += ["### Orquestração & próximo passo", *orch_lines, ""]

        parts += [
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
        best_lift   = best.get("lift_vs_baseline_pct", 0)
        best_expl   = best.get("exploration_rate", 0)
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

        # Compute concrete insights from actual metrics
        if best_reward > 0:
            lift_vs_worst = ((best_reward - worst.get('cumulative_reward', 0)) / best_reward) * 100
        else:
            lift_vs_worst = 0

        parts = [
            f"**Melhor política:** {best_policy.upper()} · "
            f"Reward=R$ {best_reward:,.0f} · Regret={best_regret:.1%} · "
            f"Conv={best_conv:.1%} · Lift=+{best_lift:.0f}% vs baseline · "
            f"Exploração={best_expl:.1%}",
            "",
            f"**Modelo vencedor:** {ml_explain}",
            "",
            f"**Vantagem quantitativa:** a {best_policy.upper()} superou a pior "
            f"política ({worst_policy}) em **{lift_vs_worst:.0f}%** de reward acumulado. "
            f"Comparada à baseline greedy, o ganho (lift) foi de **+{best_lift:.0f}%** — "
            f"comprovando que o aprendizado contextual agrega valor real vs um "
            f"selecionador determinístico sem exploração.",
            "",
            "**Ranking completo:**",
            *lines,
            "",
            "**Interpretação:** O bandit contextual aprendeu a associar perfis de "
            "cliente (idade, euribor3m, campanha anterior, canal de contato) às ofertas "
            "de maior probabilidade de conversão. Políticas com exploração estruturada "
            "(UCB, Thompson) superam a greedy porque descobrem braços melhores ao longo "
            "do tempo, enquanto a baseline fica presa em máximos locais. "
            "O trade-off exploração/explotação é gerenciado automaticamente pelo "
            f"algoritmo — a taxa de exploração de {best_expl:.1%} indica um "
            f"balanceamento {'agressivo' if best_expl > 0.15 else 'conservador'} "
            f"entre testar novas ofertas e explorar as melhores.",
        ]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #

def explain_decision(decision: dict[str, Any], question: str = "", top_k: int = 3) -> dict[str, Any]:
    return Assistant().explain_decision(decision, question=question, top_k=top_k)


def summarize_experiment(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return Assistant().summarize_experiment(metrics_rows)
