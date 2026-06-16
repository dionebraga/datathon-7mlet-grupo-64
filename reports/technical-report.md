# Relatório Técnico — Adaptive Offers Platform

**FIAP Pós-Tech 7MLET · Fase 05 · Datathon · Grupo 64**
Plataforma de experimentação adaptativa para ofertas financeiras com multi-armed
bandits, feature store, assistente LLM/RAG, avaliação offline, ciclo MLOps e
arquitetura-alvo Azure.

---

## 1. Problema

Uma instituição financeira digital precisa decidir, em diferentes canais (app,
e-mail, push, SMS), **qual oferta apresentar** a cada cliente elegível. Regras
fixas e testes A/B longos desperdiçam tráfego, demoram a reagir a mudanças de
contexto e dificultam personalização responsável. Tratamos a decisão como um
**multi-armed bandit contextual**: cada braço é uma oferta; o contexto é o estado
anonimizado do cliente/canal; a recompensa é a conversão (frequentemente
**atrasada**), ponderada pela **margem** da oferta. O sistema equilibra
exploração e explotação, aprende com respostas observadas e inclui um
**assistente LLM/RAG** que resume experimentos, recupera políticas internas
(sintéticas) e explica decisões.

## 2. Base de dados escolhida

**Bank Marketing** (Kaggle: henriqueyamahata; origem UCI, Moro et al. 2014),
`bank-additional-full.csv` — campanhas de marketing bancário com atributos do
cliente, contato e contexto macroeconômico, e alvo `y` (assinou depósito a
prazo). Usamos `y` como **proxy de conversão**. Detalhes, licença (CC BY 4.0) e
download em `data/kaggle/README.md`.

**Decisão de vazamento**: removemos `duration` (conhecida só após o contato,
*target leakage*) e transformamos a sentinela `pdays==999` em
`previously_contacted` + `pdays_since`. EDA e qualidade em
`reports/eda-quality-report.md` (alvo desbalanceado ~5–11%).

> Para reprodutibilidade offline/CI, um **facsimile determinístico** replica o
> *schema* da base; a proveniência (`real`/`facsimile`, versão, licença) é
> registrada a cada build.

## 3. Enriquecimento sintético

Sobre a base factual construímos a camada de experimentação (separada
fisicamente em `data/synthetic_enrichment/`): **catálogo de 6 ofertas** com
modelo latente documentado (`P(convert)=σ(base_logit + w·contexto)`, recompensa =
`P×margem`), **eventos de impressão** logados por uma *logging policy*
ε-exploratória (com `propensity` para IPS) e **delayed rewards** (~40% das
conversões maturam em 1–30 dias). Processo, sementes, hipóteses e riscos em
`reports/data-generation.md`. Como o **oráculo** é conhecido, calculamos
**regret**.

## 4. Modelagem como multi-armed bandit

Quatro políticas sob um contrato único (`select`/`update`), ranqueando por
**margem×conversão**:

- **Baseline** — greedy otimista, **sem exploração** (controle).
- **Thompson Sampling** — Beta-Bernoulli, prior Beta(1,1) documentado.
- **Nilos-UCB** — UCB **variance-aware** (UCB-V; Audibert et al. 2009), justificada
  pela heterogeneidade de variância entre ofertas; cold-start com puxada inicial.
- **LinUCB** — contextual (modelo linear ridge por braço), roteia a oferta certa
  ao contexto certo.

**Cold-start**: priors/`A=I`/puxada inicial. **Delayed rewards**: fila de feedback
pendente; a política só aprende com recompensas maturadas. Detalhes em
`reports/algorithmic-strategy.md`.

## 5. Comparação quantitativa

Facsimile, 20.000 rounds, 40% delayed (seed=123):

| Política | Reward | Reward/1k | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| **LinUCB** | **424.820** | 21.241 | **5,1%** | 9,9% | 10,9% | **+66,6%** |
| Thompson | 389.180 | 19.459 | 12,2% | 7,5% | 4,7% | +52,6% |
| Nilos-UCB | 383.010 | 19.150 | 14,1% | 7,2% | 15,0% | +50,2% |
| Baseline | 255.060 | 12.753 | 42,7% | 10,7% | 0,0% | — |

**Evidências adicionais** (`reports/offline-evaluation.md`):
- **Golden set** (24 casos): LinUCB **100%** (8/8 típicos, 6/6 segmento, 5/5 borda,
  **5/5 adversariais**); não-contextuais ~54% → o gate **discrimina**.
- **Sensibilidade**: CV de reward do LinUCB **1,9%** (estável a seeds).
- **IPS/SNIPS** off-policy ≈ **21,0/impressão**, concordando com a simulação
  on-policy (~21,2) — validação independente.
- **Fairness**: disparidade de exposição **0,00** (sem negação de oferta).

**Métricas específicas e justificativa**: priorizamos **reward/regret ponderados
por margem** (valor de negócio) sobre acurácia/conversão crua — o baseline tem a
**maior conversão** mas o **menor valor**, ilustrando por que conversão isolada
engana.

## 6. Arquitetura-alvo Azure

Operação **exclusivamente Azure** (`docs/architecture-azure.md`): Container Apps
(API), Functions (jobs/delayed rewards), API Management + Front Door/WAF, ADLS
(offline) + Redis (online feature store) + PostgreSQL (auditoria), Azure OpenAI +
AI Search (assistente/RAG), Azure ML + MLflow + ACR (MLOps), App Insights
(observabilidade), **Key Vault + Managed Identity + Entra ID** (segredos/identidade
sem senha), Defender (postura). FinOps: escala a zero e cache de explicações como
maiores alavancas; ROI sustentado pelo +66,6% de valor sobre o volume.

## 7. Ciclo MLOps

`docs/mlops-lifecycle.md`: rastreio em MLflow; versionamento/registry de políticas
(`promote`/`rollback`); **approval gate** humano (golden set ≥0,95 e 100%
adversarial, lift ≥0, fairness ≤0,25, sensibilidade ≤5%); promoção canary;
monitoramento de **drift** (PSI/KS) e **reward** (control chart) com gatilho de
retreino/rollback. CI (lint+test+pipeline smoke) e CD (build/push imagem).

## 8. Limitações
- Dados sintéticos: comparam políticas, **não** estimam *lift* financeiro real.
- IPS tem variância alta sob baixa sobreposição (match ~17%; SNIPS mitiga).
- Modelo latente estacionário; *drift* injetado só em testes de monitoramento.
- **Não** pronto para produção financeira regulada.

## 9. Riscos e mitigação
Reward hacking, manipulação de contexto, abuso do assistente e violação de
suitability — mitigados por gate de elegibilidade, validação de schema, RAG com
*grounding* + *fallback* determinístico e humano no loop (`docs/system-card.md`).
Privacidade e atributos protegidos em `docs/lgpd-plan.md`.

## 10. Hipóteses
- O ganho relativo entre políticas (LinUCB > não-contextuais > baseline) se mantém
  na base real; magnitudes variam.
- A margem é conhecida por oferta e estável no horizonte de decisão.
- Recompensas atrasadas maturam dentro do horizonte de 30 dias.

## 11. Trabalhos futuros
- LinUCB com features cruzadas/kernels; Neural/Deep bandits (PyTorch).
- Off-policy evaluation mais robusta (Doubly Robust).
- *Drift* não-estacionário no simulador; *contextual* fairness constraints.
- Integração real com Azure OpenAI/AI Search e Feast no lugar do store próprio.

## 12. Referências
- Moro, Cortez & Rita (2014). *A data-driven approach to predict the success of
  bank telemarketing*. Decision Support Systems 62, 22–31.
- Li, Chu, Langford & Schapire (2010). *A contextual-bandit approach to
  personalized news article recommendation*. WWW.
- Audibert, Munos & Szepesvári (2009). *Exploration–exploitation tradeoff using
  variance estimates in multi-armed bandits*. TCS (UCB-V).
- Chapelle & Li (2011). *An empirical evaluation of Thompson Sampling*. NeurIPS.
- Documentação interna: `reports/` e `docs/` deste repositório.
