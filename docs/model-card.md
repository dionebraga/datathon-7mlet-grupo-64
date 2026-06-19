# Model Card — Adaptive Offers Policy

> Cartão de modelo da política de decisão de ofertas. Revisão periódica definida
> na Seção 10.

## 1. Identificação
- **Nome**: Adaptive Offers Policy (multi-armed bandit contextual)
- **Versão**: v1 (`artifacts/policies/v1/metadata.json` registra hash e métricas)
- **Tipo**: LinUCB (contextual) — alternativas: Thompson Sampling, Nilos-UCB, baseline
- **Owner**: Grupo 64 — FIAP 7MLET
- **Frameworks**: numpy/scipy/scikit-learn; tracking em MLflow

## 2. Dados de treino e avaliação
- **Base factual**: Bank Marketing (Kaggle/UCI) — propensão/conversão bancária.
- **Camada sintética**: catálogo de ofertas + eventos + *delayed rewards*
  (seeds controladas; ver `reports/data-generation.md`).
- **Sem vazamento**: `duration` e colunas pós-contato removidas.
- **Avaliação**: golden set (24 casos), métricas offline, sensibilidade, IPS,
  fairness de exposição (ver `reports/offline-evaluation.md`).

## 3. Métricas — base real (UCI Bank Marketing · 41.188 contatos · 6.000 rounds)
| Métrica | LinUCB | Baseline |
|---|---:|---:|
| Reward acumulado | 113.230 | 104.700 |
| Reward médio (5 seeds) | **110.046** | 87.616 |
| Regret ratio | **8,3%** | 10,9% |
| Conversão | **9,1%** | 6,2% |
| Lift de valor (seed 123) | +8,2% | — |
| Golden set pass-rate | **83,3%** (adversarial 5/5) | — |
| Estabilidade (CV reward, 5 seeds) | **2,97%** | 20,2% |
| Fairness (disparidade de exposição) | **0,00** | — |

> Numa seed isolada o Thompson capturou um pouco mais de valor (+9,2%); **na média
> de 5 seeds o LinUCB lidera e é o mais estável** — por isso é a política recomendada.
> O ganho é **modesto e honesto** (single digits), não os ~+60% do fac-símile.

## 4. Uso pretendido (intended use)
- Recomendar, **dentro do conjunto elegível**, a oferta de maior valor esperado
  (margem×conversão) em canais digitais, com exploração responsável.
- Sempre como **apoio à decisão**, com auditabilidade (reason codes, versão).

## 5. Uso fora de escopo (out-of-scope)
- **Não** decide concessão de crédito, limites, preços ou *suitability* sensível
  sozinho — isso exige humano no loop.
- **Não** é estimador de *lift* financeiro real (dados sintéticos).
- **Não** está pronto para produção financeira regulada.

## 6. Análise de fairness
- Disparidade de exposição (taxa de receber oferta) entre segmentos sintéticos:
  **0,00** (sem negação de oferta). Mix por segmento varia por **personalização
  intencional**, monitorado continuamente.
- Atributos potencialmente sensíveis (idade/profissão/estado civil) **não** são
  usados como atributos protegidos de decisão.

## 7. Vieses conhecidos
- Base factual é de telemarketing português (2008–2013); não representa o mercado
  atual/brasileiro.
- Forte desbalanceamento do alvo (~5–11% positivos).
- Modelo latente sintético é estacionário; realidade tem *drift*.

## 8. Limitações técnicas
- IPS off-policy tem variância alta sob baixa sobreposição (match ~17%).
- Cold-start: primeiras decisões exploram mais (regret inicial maior).
- *Delayed rewards* atrasam o aprendizado (até 30 dias de horizonte).

## 9. Guardrails
- Gate de elegibilidade/suitability **antes** da seleção.
- Piso mínimo de exploração; controle `no_offer` como *fallback*.
- Log auditável + monitor de drift/reward (ver `docs/system-card.md`).

## 10. Plano de revisão periódica
- **Cadência**: trimestral **ou** ao disparar alerta de drift/reward.
- **Responsável**: owner do modelo (Grupo 64) + revisor de risco.
- **Escopo da revisão**: métricas, golden set, fairness, vieses, incidentes;
  atualizar este card e o `system-card.md`; registrar aprovação.
