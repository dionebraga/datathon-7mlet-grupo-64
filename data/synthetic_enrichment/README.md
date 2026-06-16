# `data/synthetic_enrichment/` (gerado — schema versionado)

Camada sintética de experimentação adaptativa, **separada** da base Kaggle.
Gerada por `adaptive-offers synth generate`.

- `offer_catalog.parquet` — braços/ofertas + modelo latente (gerado).
- `offer_events.parquet` — impressões logadas, contexto, recompensas (gerado).
- `delayed_rewards.parquet` — conversões atrasadas (gerado).
- **`schema.json`** — *schema*, chaves e horizonte temporal (**versionado**).

Parquets são gerados e ficam fora do versionamento (`.gitignore`); apenas o
`schema.json` e este README são versionados. Processo completo em
[`reports/data-generation.md`](../../reports/data-generation.md).
