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

## 3. Métricas (facsimile, 20k rounds)
| Métrica | LinUCB | Baseline |
|---|---:|---:|
| Reward acumulado | 424.820 | 255.060 |
| Regret ratio | 5,1% | 42,7% |
| Lift de valor | **+66,6%** | — |
| Golden set pass-rate | 100% | 54% |
| Sensibilidade (CV reward) | 1,9% | — |

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
