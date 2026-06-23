# Relatório Técnico — Adaptive Offers Platform

**FIAP Pós-Tech 7MLET · Fase 05 · Datathon · Grupo 74**
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

### 4.1 Camada de orquestração fintech (oferta → mensagem → canal → próximo passo)

O bandit decide *qual oferta*; quatro camadas determinísticas sobre ele fecham o
problema completo do desafio ("qual oferta, mensagem ou próximo passo, em
diferentes canais"):

- **Segmentação comportamental** (`segmentation.py`) — 6 personas determinísticas
  e prioritizadas (renegociador, sênior conservador, jovem digital, recorrente,
  novo/cold-start, massa) derivadas das features reais; leitura humana do book.
- **Orquestração multi-canal** (`channels.py`) — catálogo de canais (app push,
  e-mail, SMS, ligação) com custo/latência + `ContactPolicy` (frequency cap,
  horário de silêncio, escolha por custo/riqueza). Aplicada como guardrail após
  a seleção, emitindo reason codes (`CHANNEL_SELECTED`, `QUIET_HOURS`,
  `FREQUENCY_CAPPED`, `CONTACT_SUPPRESSED`).
- **Next-Best-Action** (`nba.py`) — mensagem por **template governado** (tom por
  persona, hint por canal) + próximo passo legível por máquina (`SIMULATE_LOAN`,
  `OPEN_DEPOSIT`, …). Copy nunca é texto livre do LLM, mantendo suitability
  auditável.
- **IA responsável** (`responsible.py`) — registro único de atributos protegidos
  (idade, profissão, estado civil, escolaridade) + fairness por grupo em duas
  dimensões (exposição **e valor**); o grupo protegido é gravado no log só para
  auditoria, nunca como variável de decisão (ver §6 do model card).

## 5. Comparação quantitativa

Base real (UCI Bank Marketing — **41.188 contatos**, `provenance="real"`), 6.000
rounds, 40% delayed (seed=123):

| Política | Reward | Reward/1k | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Thompson | **114.290** | 19.048 | 11,8% | 7,1% | 11,7% | **+9,2%** |
| **LinUCB** | 113.230 | 18.872 | **8,3%** | **9,1%** | 26,4% | +8,2% |
| Baseline | 104.700 | 17.450 | 10,9% | 6,2% | 0,0% | — |
| Nilos-UCB | 102.020 | 17.003 | 17,0% | 7,1% | 29,1% | **−2,6%** |

> Honestidade: na base real o ganho é **modesto (single digits)**, não os ~+60%
> que o fac-símile produzia. O resultado de **uma** seed é ruidoso.

**Robustez (5 seeds · 6.000 rounds)** — medindo a média, **o LinUCB lidera**:
reward médio **110.046**, vence **3/5**, e é **o mais estável (CV 2,97%)**, contra
Thompson (105.512; CV 7,07%; 2/5), Nilos-UCB (99.800; 0/5) e Baseline (87.616; CV
**20,2%** — instável, o que explica por que numa seed isolada ele parece forte).

**Evidências adicionais** (`reports/offline-evaluation.md`):
- **Golden set** (24 casos): **83,3%** (5/8 típicos, **6/6 segmento**, 4/5 borda,
  **5/5 adversariais**) — segmento e adversarial seguem em 100%.
- **Estabilidade**: CV de reward do LinUCB **2,97%** entre seeds (o mais estável).
- **Fairness**: disparidade de exposição **0,00** (sem negação de oferta).

**Métricas específicas e justificativa**: priorizamos **reward/regret ponderados
por margem** (valor de negócio). O **baseline colapsa ~85% das decisões no
Empréstimo** (maior margem) ignorando o contexto; o **LinUCB diversifica pelo
contexto** e entrega **menor regret (8,3%) e maior conversão (9,1%)** — por isso é
a política recomendada, mesmo com lift de valor modesto.

## 6. Arquitetura-alvo Azure

Operação **exclusivamente Azure** (`docs/architecture-azure.md`): Container Apps
(API), Functions (jobs/delayed rewards), API Management + Front Door/WAF, ADLS
(offline) + Redis (online feature store) + PostgreSQL (auditoria), Azure OpenAI +
AI Search (assistente/RAG), Azure ML + MLflow + ACR (MLOps), App Insights
(observabilidade), **Key Vault + Managed Identity + Entra ID** (segredos/identidade
sem senha), Defender (postura). FinOps: escala a zero e cache de explicações como
maiores alavancas; ROI sustentado pelo ganho de valor do aprendizado (modesto,
porém positivo e estável) diluído sobre o volume.

## 7. Ciclo MLOps

`docs/mlops-lifecycle.md`: rastreio em MLflow; versionamento/registry de políticas
(`promote`/`rollback`); **approval gate** humano (golden set ≥0,95 e 100%
adversarial, lift ≥0, fairness ≤0,25, sensibilidade ≤5%); promoção canary;
monitoramento de **drift** (PSI/KS) e **reward** (control chart) com gatilho de
retreino/rollback. CI (lint+test+pipeline smoke) e CD (build/push imagem).

## 8. Limitações
- **Base real (UCI), mas camada de ofertas/recompensas sintética**: comparamos
  políticas em simulação com contextos reais — **não** estimamos *lift* financeiro
  de produção.
- **Golden set real = 83,3%** (abaixo do gate de 0,95): adversarial e segmento
  seguem 100%, mas casos típicos/borda exigem ajuste antes de promover — limitação
  assumida, não mascarada.
- O *lift* sobre o baseline é **modesto e sensível à seed** (baseline com CV ~20%).
- Modelo latente de oferta estacionário; *drift* injetado só em testes de monitoramento.
- **Não** pronto para produção financeira regulada.

## 9. Riscos e mitigação
Reward hacking, manipulação de contexto, abuso do assistente e violação de
suitability — mitigados por gate de elegibilidade, validação de schema, RAG com
*grounding* + *fallback* determinístico e humano no loop (`docs/system-card.md`).
Privacidade e atributos protegidos em `docs/lgpd-plan.md`.

## 10. Hipóteses
- **Testado na base real**: o LinUCB lidera na média e tem o menor regret, mas as
  magnitudes caíram para single digits e o Nilos-UCB ficou **abaixo** do baseline —
  nem todo bandit vence.
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
