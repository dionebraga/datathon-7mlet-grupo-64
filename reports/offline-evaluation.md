# Avaliação Offline e Golden Set (Stage 4)

> Reprodutível com `adaptive-offers evaluate`. Implementação em
> `src/adaptive_offers/evaluation/`. Números do *facsimile* (seeds indicados).

## 1. Métricas e por que são justificadas

| Métrica | Definição | Por que importa |
|---|---|---|
| `cumulative_reward` | Σ recompensa realizada (margem×conversão) | Valor de negócio capturado |
| `cumulative_regret` | Σ (reward do oráculo − reward escolhido) | KPI central de bandit |
| `regret_ratio` | regret / reward total do oráculo | Quão perto do ótimo (0 = ótimo) |
| `conversion_rate` | conversões / impressões | Engajamento (enganoso isolado) |
| `exploration_rate` | fração de decisões exploratórias | Equilíbrio exploração/explotação |
| `V_IPS / V_SNIPS` | valor off-policy por impressão | Validação **sem** simular (dados logados) |

## 2. Golden set (24 casos versionados)

`data/golden_set/evaluation_cases.jsonl` — cobre **typical (8)**, **segment (6)**,
**edge (5)** e **adversarial (5)**. Cada caso traz contexto, ação esperada,
piso de recompensa, `oracle_top3`, justificativa e critério **pass/fail**.
Invariantes em todos: oferta **elegível** + **reason codes** presentes.

### Pass-rate na base real (`adaptive-offers evaluate`)

Política recomendada (LinUCB) na base real: **pass-rate 83,3% (20/24)**.

| Categoria | Resultado |
|---|:--:|
| typical | 5/8 |
| **segment** | **6/6** |
| edge | 4/5 |
| **adversarial** | **5/5** |

**Leitura**: os guardrails **adversariais (5/5) e de segmento (6/6) seguem 100%** —
o sistema respeita elegibilidade e suitability em todos os casos críticos. Os casos
**típicos/borda** caem em contextos reais (mais heterogêneos), o que mantém o golden
**abaixo do gate de 0,95** — limitação assumida (ver `reports/technical-report.md`).

## 3. Matriz de métricas — base real (UCI · 41.188 contatos · 6.000 rounds, seed=123)

| Política | Reward | Reward/1k | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| thompson | **114.290** | 19.048 | 11,8% | 7,1% | 11,7% | **+9,2%** |
| linucb | 113.230 | 18.872 | **8,3%** | **9,1%** | 26,4% | +8,2% |
| baseline | 104.700 | 17.450 | 10,9% | 6,2% | 0,0% | — |
| nilos_ucb | 102.020 | 17.003 | 17,0% | 7,1% | 29,1% | **−2,6%** |

Numa seed o Thompson ficou à frente por pouco; **na média de 5 seeds o LinUCB
lidera** (§4), com o menor regret e a maior conversão — a política recomendada. O
ganho de valor é **modesto e honesto** (single digits), não os ~+60% do fac-símile.

## 4. Análise de sensibilidade (robustez) — base real

Todas as políticas sobre seeds {123, 7, 42, 99, 2024}, horizon 6.000:

| Política | Reward médio | CV | Vitórias |
|---|---:|---:|:--:|
| **LinUCB** | **110.046** | **2,97%** | **3/5** |
| Thompson | 105.512 | 7,07% | 2/5 |
| Nilos-UCB | 99.800 | 5,29% | 0/5 |
| Baseline | 87.616 | 20,24% | 0/5 |

O **LinUCB** é o mais **estável** (menor CV) e vence na maioria das seeds; o
**baseline é instável** (CV ~20%) — por isso numa execução isolada ele às vezes
parece competitivo. A liderança do LinUCB **não é artefato de uma única seed**.

## 5. Validação off-policy (IPS / SNIPS)

A metodologia (estimadores *Inverse Propensity Scoring* sobre os eventos logados,
com as propensities registradas no Stage 2) permanece implementada e válida.

> ⚠️ **Pendente de recomputação na base real.** Os valores absolutos de IPS/SNIPS
> aqui exibidos eram do fac-símile e foram **removidos** para não induzir a erro.
> Rode `adaptive-offers evaluate` para os números atuais.

## 6. Fairness de exposição entre segmentos

Auditamos a **taxa de receber oferta** (qualquer oferta ≠ controle) por segmento
sintético (faixa etária, sucesso prévio, canal):

| Dimensão | Disparidade (max−min) | Flag |
|---|---:|:--:|
| age_band | 0,00 | ok |
| prior_success | 0,00 | ok |
| channel | 0,00 | ok |
| **máx. geral** | **0,00** | **ok** |

- **Sem negação de oferta** a nenhum segmento (paridade demográfica de exposição
  = perfeita). O `offer_mix` **varia** por segmento — isso é **personalização
  intencional** (a oferta certa para o contexto certo), não negação de valor.
- O mix por segmento é monitorado continuamente (Stage 7) para detectar
  *drift* que vire exclusão sistemática.

## 7. Limitações e quando NÃO usar a política

- Resultados em *facsimile*; magnitudes mudam na base real.
- Recompensa sintética: não estimar *lift* financeiro real a partir daqui.
- A política **não** deve decidir casos de *suitability* sensível sozinha —
  mantém-se **humano no loop** (ver `docs/system-card.md` e `docs/lgpd-plan.md`).
- IPS tem variância alta sob baixa sobreposição (match rate 16,9%); SNIPS mitiga,
  mas conclusões off-policy exigem cautela.
