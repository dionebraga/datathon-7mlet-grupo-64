# Pitch — Roteiro (10 min + 5 min Q&A)

> Roteiro versionado dos slides para o Demo Day. Exportável para PDF
> (`docs/pitch.md` → slides). Cobre problema, abordagem, demonstração,
> evidências, riscos, governança e impacto, incluindo FinOps.

---

## Slide 1 — Capa
**Adaptive Offers Platform** · FIAP 7MLET · Grupo 64
Decisão de oferta em canais digitais com multi-armed bandits.

## Slide 2 — Problema (negócio)
- Regras fixas e A/B longos desperdiçam tráfego e reagem devagar.
- Pergunta: *qual oferta/mensagem/próximo passo* para cada cliente elegível?
- Oportunidade: personalização responsável que aprende em produção.

## Slide 3 — Abordagem
- Multi-armed bandit **contextual**: explora vs explota, aprende online.
- Recompensa **ponderada por margem** (valor, não só conversão).
- Assistente **LLM/RAG** resume experimentos e explica decisões.

## Slide 4 — Dados e enriquecimento
- Base factual **Bank Marketing** (Kaggle/UCI), sem vazamento (`duration` fora).
- Camada **sintética** separada: ofertas, eventos, **delayed rewards** (seeds).
- Oráculo conhecido ⇒ medimos **regret**.

## Slide 5 — Modelagem
- Baseline (controle) · Thompson · **Nilos-UCB** (UCB-V) · **LinUCB** (contextual).
- Cold-start e recompensas atrasadas tratados explicitamente.

## Slide 6 — Demonstração (ao vivo/gravada)
- `adaptive-offers pipeline` → dados→synth→treino→avaliação.
- `POST /decide`: contexto → oferta + **reason codes** + versão + log auditável.
- Dashboard BI: comparação, regret, mix, **explorador de decisão**.
- *Plano de contingência*: gravação versionada caso a demo ao vivo falhe.

## Slide 7 — Evidências
- **LinUCB +66,6%** de valor vs baseline; **regret 5,1%**.
- Golden set **100%** (inclui 5/5 adversariais); CV de reward **1,9%**.
- **IPS off-policy** concorda com a simulação (~21/impressão).
- Fairness: disparidade de exposição **0,00**.

## Slide 8 — Arquitetura técnica (Azure) + alternativas
- Diagrama (Mermaid): Container Apps, ADLS+Redis+PostgreSQL, Azure OpenAI+AI
  Search, Azure ML/MLflow, App Insights, Key Vault + Managed Identity.
- **Alternativas descartadas**: AKS (overhead), segredos em env (segurança),
  LLM externo (viola "Azure-only"). Fronteiras claras entre camadas.

## Slide 9 — FinOps (ROI · custo · TCO · escala)
- ROI: ganho de valor dilui custo de compute/LLM (LLM só p/ explicação).
- TCO dominado por Container Apps + Azure OpenAI; demais secundários.
- Escala ↑: réplicas + tier Redis no pico; ↓: **escala a zero** + cache de RAG.

## Slide 10 — Governança, riscos e impacto
- Model/System Card + Plano LGPD; humano no loop; rollback em 1 comando.
- Riscos: reward hacking, manipulação de contexto, abuso do assistente — mitigados.
- Impacto: mais valor por impressão, decisões **auditáveis e explicáveis**, sem
  alegar prontidão para produção regulada.

---

### Q&A — perguntas prováveis
- *Por que não A/B?* Bandit reduz regret e reage a contexto.
- *Conversão do baseline é alta — por quê perde?* Ignora margem/contexto.
- *E sem o LLM?* Decisão independe dele; *fallback* determinístico.
- *Como promover nova política?* Approval gate + canary + rollback (MLOps).
