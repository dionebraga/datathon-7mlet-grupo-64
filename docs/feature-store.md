# Feature Store — Arquitetura e Funcionalidades

> Documenta o módulo **Feature Store** do curso aplicado ao projeto.
> Implementação: `src/adaptive_offers/feature_store/`.

## 1. Requisitos e funcionalidades (Aula 1)

| Requisito | Como atendemos |
|---|---|
| Reuso de features treino↔serving | Mesma função `build_context_vector` no treino e no serving |
| Baixa latência no serving | **Online store** (SQLite local / Redis em Azure) |
| Backfill / histórico | **Offline store** (Parquet) com `get_historical_features` |
| Versionamento de features | `FeatureView.version` por view |
| Consistência | `rate_median` materializado junto às features |

## 2. Arquitetura de Data Storage (Aula 2)

```
Offline store (Parquet / ADLS)            Online store (SQLite / Redis)
  data/processed/*.parquet     --materialize-->  client_features (KV por client_event_id)
  data/synthetic/offer_catalog --materialize-->  offer_features  (KV por offer_id)
                                                  _metadata (rate_median, ...)
```

- **Offline**: fonte de verdade para treino e *backfill* (point-in-time simples).
- **Online**: projeção da última versão das features para leitura O(1) no serving.

## 3. Construindo o Feature Store (Aula 3)

Entidades e *feature views* declarativas (`definitions.py`):

- **Entidade `client_event`** (chave `client_event_id`) → view `client_features`
  (age, contact, poutcome, previously_contacted, euribor3m, default, loan, …).
- **Entidade `offer`** (chave `offer_id`) → view `offer_features`
  (margin, category, suitability_tier, regras de elegibilidade).

`FeatureStore.materialize()` cria as tabelas online a partir do offline e grava
metadados de run. `get_online_features(view, id)` lê em baixa latência.

## 4. Principais funcionalidades (Aula 4)

- `materialize()` — offline → online, idempotente.
- `get_online_features()` — leitura por entidade (serving).
- `get_historical_features()` — leitura em lote (treino/análise).
- `get_context_vector()` — monta o vetor de contexto do bandit a partir das
  features online + `rate_median`, garantindo **paridade treino/serving**.
- `is_materialized()` — readiness para o `/health` da API.

## 5. Mapeamento para Azure

| Componente local | Serviço Azure |
|---|---|
| Offline store (Parquet) | ADLS Gen2 |
| Online store (SQLite) | Azure Cache for Redis |
| Materialização | Job no Azure ML |
| Metadados/registry | MLflow (Azure ML) |

Ver `docs/architecture-azure.md`.
