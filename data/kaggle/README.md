# Base factual Kaggle — Bank Marketing

> Esta camada documenta a **fonte factual** usada como referência. O pipeline
> **não versiona** o CSV bruto (ver `.gitignore`); ele é baixado sob demanda ou
> substituído por um **facsimile determinístico** com o mesmo *schema* para
> reprodutibilidade offline/CI.

## Identificação da fonte

| Campo | Valor |
|---|---|
| **Dataset** | Bank Marketing |
| **Autor (Kaggle)** | henriqueyamahata |
| **Link** | https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing |
| **Origem primária** | UCI Machine Learning Repository — *Bank Marketing* (Moro, Cortez & Rita, 2014) |
| **Arquivo usado** | `bank-additional-full.csv` (41.188 linhas, 21 colunas, separador `;`) |
| **Versão** | UCI `bank-additional-full` (2014) |
| **Licença** | CC BY 4.0 (UCI / Moro et al., 2014) — uso permitido com atribuição |
| **Citação** | Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing*. Decision Support Systems, 62, 22–31. |

## Como usar no desafio

A base é referência **factual de propensão/conversão bancária**: cada linha é um
contato de campanha com atributos do cliente, do contato e do contexto
econômico, e um alvo `y` (assinou depósito a prazo? sim/não). Usamos esse alvo
como **proxy de conversão** e construímos por cima a **camada sintética de
experimentação adaptativa** (ofertas/braços, impressões, contexto, recompensas
e *delayed rewards* — ver `data/synthetic_enrichment/` e `reports/data-generation.md`).

## Como baixar a base real (opcional)

```bash
# 1) Credenciais Kaggle: https://www.kaggle.com/settings -> "Create New Token"
#    Salve kaggle.json em ~/.kaggle/  (ou defina KAGGLE_USERNAME / KAGGLE_KEY no .env)
pip install kaggle

# 2) Baixar e extrair para a pasta esperada pelo loader:
mkdir -p data/kaggle/raw
kaggle datasets download -d henriqueyamahata/bank-marketing -p data/kaggle/raw --unzip

# 3) Rodar a camada de dados (o loader detecta o CSV real automaticamente):
adaptive-offers data build
```

Se o arquivo `data/kaggle/raw/bank-additional-full.csv` **existir**, o loader o
usa e marca a proveniência como `real`. Caso contrário, gera o **facsimile**
(`provenance.source = "facsimile"`).

## Decisão de vazamento temporal (leakage)

| Coluna | Decisão | Justificativa |
|---|---|---|
| `duration` | **DESCARTADA** | Duração do último contato em segundos — só é conhecida **após** o contato terminar e é altamente correlacionada com o alvo. Usá-la vaza informação pós-contato para uma política que decide **antes** de contatar. A própria documentação do UCI recomenda descartá-la para modelos realistas. |
| `pdays == 999` | **Transformada** | Sentinela de "não contatado antes" → vira flag `previously_contacted` + `pdays_since` (0 quando nunca contatado), evitando que 999 seja lido como magnitude numérica. |
| `emp_var_rate`, `euribor3m`, `cons_*`, `nr_employed` | Mantidas | Indicadores macroeconômicos do **período**, disponíveis no momento da decisão; não são pós-contato. |

A proveniência (fonte, versão, licença, nº de linhas, semente) é registrada em
`data/processed/provenance.json` a cada build, garantindo rastreabilidade
(evidência de aceite da Etapa 1).

## Limitações da fonte

- Campanhas de **telemarketing português (2008–2013)** — distribuição não
  representa necessariamente canais digitais atuais nem o mercado brasileiro.
- Forte **desbalanceamento** do alvo (~11% positivos na base real).
- Não contém identificadores diretos, renda, patrimônio, gênero ou raça; ainda
  assim `age`, `job`, `marital`, `education` podem atuar como **proxies
  sensíveis** e por isso **não** são usados como atributos protegidos de decisão
  (ver `docs/lgpd-plan.md`).
