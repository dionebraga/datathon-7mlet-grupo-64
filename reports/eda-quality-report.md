# Relatório de EDA e Qualidade — Base processada

> Reprodutível via `notebooks/01_eda.ipynb` ou
> `python -c "from adaptive_offers.data.quality import quality_report; ..."`.
> Números abaixo referem-se ao **facsimile determinístico** (`seed=42`,
> `n=20.000`). Com a base Kaggle **real**, rode o build e regenere o notebook;
> as conclusões qualitativas (drivers de conversão) se mantêm.

## 1. Visão geral

| Métrica | Valor (facsimile) |
|---|---|
| Linhas | 20.000 |
| Colunas (processadas) | 22 |
| Duplicatas | 0 |
| Valores ausentes | 0 (categóricos usam nível `unknown`, como na base real) |
| Taxa do alvo `subscribed` | **5,41%** |
| Razão de desbalanceamento | ~17,5 : 1 |

> Na base **real**, a taxa positiva é ~11%. O desbalanceamento é uma
> característica central do problema e motiva o uso de **métricas além de
> acurácia** (uplift, recompensa, regret) na avaliação (Stage 4).

## 2. Qualidade dos dados

- **Sem vazamento**: `duration` removida; `pdays==999` convertido em
  `previously_contacted` + `pdays_since` (evita ler a sentinela como magnitude).
- **Cardinalidade categórica**: `job` (12), `education` (8), `month` (10) são as
  mais altas; todas com vocabulário fechado e nível `unknown`.
- **Consistência**: strings normalizadas (trim + lowercase) para junções
  estáveis com a camada sintética.
- **Chave**: `client_event_id` único por linha (cardinalidade = nº de linhas).

## 3. Sinais de conversão (drivers)

Taxas de assinatura por segmento (facsimile, ordenadas):

| Variável | Segmento | Taxa | Leitura |
|---|---|---|---|
| `poutcome` | success | **22,1%** | Sucesso prévio é o **maior** sinal positivo |
| `poutcome` | nonexistent / failure | ~4,9% | Sem histórico ou falha prévia convertem pouco |
| `contact` | cellular | 6,0% | Canal celular supera telefone fixo |
| `contact` | telephone | 4,4% | — |
| `euribor3m` | ↓ (juros baixos) | ↑ conversão | Contexto macro afeta propensão |
| `age` | extremos (jovens/idosos) | ↑ | `student`/`retired` convertem mais |

Esses drivers são coerentes com a literatura do Bank Marketing e orientam a
**definição de contexto** do bandit (Stage 2/3): `poutcome`, `contact`,
faixa etária, `previously_contacted` e indicadores macro entram no vetor de
contexto do LinUCB.

## 4. Riscos e limitações

- **Proxies sensíveis** (`age`, `job`, `marital`, `education`) não são usados
  como atributos protegidos de decisão; fairness é medida sobre **segmentos
  sintéticos** (Stage 4) e monitorada (`docs/lgpd-plan.md`).
- **Facsimile ≠ realidade**: relações foram desenhadas para serem plausíveis e
  reprodutíveis, não para estimar *lift* real. Conclusões de negócio exigem a
  base real.
- **Desbalanceamento severo**: exige cuidado com métricas e com *cold-start* de
  braços pouco expostos (tratado no Stage 3).
