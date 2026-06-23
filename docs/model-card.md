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
| Fairness — disparidade de exposição (grupos protegidos) | **0,00** | — |
| Fairness — disparidade de valor (margem média, grupos protegidos) | **0,24** (pior: ≤30 anos) | — |

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
A análise cobre os **atributos protegidos** registrados em
`adaptive_offers/responsible.py` (idade em faixas, estado civil, escolaridade —
ver `docs/lgpd-plan.md`), em duas dimensões:

- **Disparidade de exposição** (taxa de *receber alguma oferta*) entre grupos
  protegidos: **0,00** — nenhum grupo é negado contato/valor.
- **Disparidade de valor** (margem média da oferta recebida): pior caso **0,24**
  na faixa **≤30 anos** (margem média R$216 vs R$282 na faixa 46–60). É um
  **resultado de *suitability*, não discriminação**: clientes jovens são
  direcionados ao Cartão Cashback (margem R$120) em vez de crédito/depósito de
  alto valor. Estado civil (0,09) e escolaridade (0,14) ficam abaixo do limiar.
- **Flag automática**: `review` se a disparidade de exposição passar de 0,25 **ou**
  a de valor passar de 0,30; hoje **`ok`**. O grupo protegido de cada cliente é
  registrado no log de decisão **apenas para auditoria** (`protected_groups`),
  **nunca** como variável de decisão.

> Condição de não-uso documentada: se a disparidade de valor crescer com novas
> ofertas/segmentos, a política deve ser revisada antes de promover (gate de
> aprovação em `docs/mlops-lifecycle.md`).

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
