# `data/processed/` (gerado — não versionado)

Saída do **Stage 1** (`adaptive-offers data build`). Contém a base factual
tratada **sem vazamento**:

- `bank_marketing_processed.parquet` — tabela modelável (ver `docs/data-dictionary.md`).
- `provenance.json` — fonte, versão, licença, nº de linhas, semente.

Estes arquivos são **gerados** e ficam fora do controle de versão (`.gitignore`).
Reconstrua com `adaptive-offers data build`.
