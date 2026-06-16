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

### Pass-rate por política (treino horizon=12.000, seed=7)

| Política | Pass-rate | typical | segment | edge | adversarial |
|---|---:|:--:|:--:|:--:|:--:|
| **linucb** | **1.000** | 8/8 | 6/6 | 5/5 | **5/5** |
| thompson | 0.542 | 2/8 | 4/6 | 2/5 | 5/5 |
| nilos_ucb | 0.542 | 2/8 | 4/6 | 2/5 | 5/5 |
| baseline | 0.542 | 2/8 | 4/6 | 2/5 | 5/5 |

**Leitura**: o golden set **discrimina** — a política contextual (LinUCB) acerta
todos os casos típicos/segmento e **todos os guardrails adversariais**; políticas
não-contextuais passam só nos guardrails e em poucos típicos. Todos respeitam os
gates de elegibilidade (nenhuma oferta inelegível em casos adversariais).

## 3. Matriz de métricas (20.000 rounds, seed=123)

| Política | Reward | Reward/1k | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| linucb | 424.820 | 21.241 | 5,1% | 9,9% | 10,9% | **+66,6%** |
| thompson | 389.180 | 19.459 | 12,2% | 7,5% | 4,7% | +52,6% |
| nilos_ucb | 383.010 | 19.150 | 14,1% | 7,2% | 15,0% | +50,2% |
| baseline | 255.060 | 12.753 | 42,7% | 10,7% | 0,0% | — |

## 4. Análise de sensibilidade (robustez)

LinUCB sobre seeds {1, 7, 42}, horizon 12.000:

| Métrica | Média | Desvio | CV |
|---|---:|---:|---:|
| Reward acumulado | 255.477 | 4.963 | **1,9%** |
| Regret acumulado | 17.656 | 4.797 | — |

Coeficiente de variação de **1,9%** ⇒ resultado **estável** a perturbações de
semente; a vantagem do LinUCB não é artefato de uma única execução.

## 5. Validação off-policy (IPS / SNIPS)

Estimadores *Inverse Propensity Scoring* sobre os eventos logados (propensities
registradas no Stage 2), subamostra de 6.000 eventos:

| Estimador | Valor/impressão | Match rate | Amostra efetiva |
|---|---:|---:|---:|
| V_IPS | 21,05 | 16,9% | 1.015 |
| V_SNIPS | 21,68 | — | — |

**Convergência independente**: o IPS off-policy (~21,0/impressão) bate com o
`reward_per_1k`/1000 do simulador on-policy (~21,2/impressão). Duas metodologias
distintas concordam ⇒ confiança no ganho estimado.

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
