# Dicionário de dados

## Base processada — `data/processed/bank_marketing_processed.parquet`

Gerada por `adaptive-offers data build` a partir da base factual Bank Marketing,
**sem colunas de vazamento**. Chave de junção: `client_event_id`.

| Coluna | Tipo | Origem | Descrição |
|---|---|---|---|
| `client_event_id` | string | derivada | Chave surrogate estável (`ce_NNNNNNN`) para juntar com a camada sintética. |
| `age` | int | raw | Idade do cliente (anos). *Proxy sensível — não usado como atributo protegido.* |
| `job` | cat | raw | Tipo de ocupação. *Proxy sensível.* |
| `marital` | cat | raw | Estado civil. *Proxy sensível.* |
| `education` | cat | raw | Escolaridade. *Proxy sensível.* |
| `default` | cat | raw | Possui crédito em *default*? (yes/no/unknown) |
| `housing` | cat | raw | Possui financiamento imobiliário? |
| `loan` | cat | raw | Possui empréstimo pessoal? |
| `contact` | cat | raw | Canal do último contato (cellular/telephone). |
| `month` | cat | raw | Mês do último contato. |
| `day_of_week` | cat | raw | Dia da semana do último contato. |
| `campaign` | int | raw | Nº de contatos nesta campanha para o cliente. |
| `previous` | int | raw | Nº de contatos em campanhas anteriores. |
| `poutcome` | cat | raw | Resultado da campanha anterior (success/failure/nonexistent). |
| `emp_var_rate` | float | raw | Taxa de variação do emprego (indicador macro do período). |
| `cons_price_idx` | float | raw | Índice de preços ao consumidor. |
| `cons_conf_idx` | float | raw | Índice de confiança do consumidor. |
| `euribor3m` | float | raw | Euribor 3 meses. |
| `nr_employed` | float | raw | Nº de empregados (indicador macro). |
| `previously_contacted` | int (0/1) | derivada | 1 se o cliente já foi contatado antes (de `pdays != 999`). |
| `pdays_since` | int | derivada | Dias desde o último contato anterior (0 quando nunca contatado). |
| `subscribed` | int (0/1) | derivada (`y`) | **Alvo / proxy de conversão**: assinou depósito a prazo. |

### Colunas removidas

| Coluna | Motivo |
|---|---|
| `duration` | Vazamento pós-contato (ver `data/kaggle/README.md`). |
| `pdays` | Substituída por `previously_contacted` + `pdays_since`. |
| `y` | Renomeada/binarizada para `subscribed`. |

## Camadas sintéticas

Os *schemas* de `offer_catalog`, `offer_events` e `delayed_rewards` estão em
`data/synthetic_enrichment/schema.json` e documentados em
`reports/data-generation.md`.
