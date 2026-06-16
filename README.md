# Adaptive Offers Platform — `datathon-7mlet-grupo-64`

> Plataforma de **experimentação adaptativa** para decidir, em canais digitais de
> uma instituição financeira, **qual oferta / mensagem / próximo passo** apresentar
> a cada cliente elegível — usando **multi-armed bandits** (Thompson Sampling,
> Nilos-UCB, LinUCB) em vez de regras fixas ou testes A/B longos.
>
> FIAP Pós-Tech **7MLET** — Fase 05 — **Datathon** — **Grupo 64**.

[![CI](https://github.com/dionebraga/datathon-7mlet-grupo-64/actions/workflows/ci.yml/badge.svg)](https://github.com/dionebraga/datathon-7mlet-grupo-64/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## 1. Visão do problema

Uma instituição financeira digital precisa decidir, em diferentes canais (app,
e-mail, push, SMS), **qual oferta apresentar** a cada cliente elegível. Regras
fixas e testes A/B longos **desperdiçam tráfego**, demoram a reagir a mudanças de
contexto e dificultam a personalização responsável.

Modelamos isso como um **multi-armed bandit contextual**: cada "braço" é uma
oferta; o "contexto" é o estado anonimizado do cliente/canal; a "recompensa" é a
conversão observada (muitas vezes **atrasada**). O sistema **equilibra exploração
e explotação**, aprende com respostas observadas e **nunca congela** a decisão em
regras estáticas. Um **assistente com LLM + RAG** resume experimentos, recupera
políticas comerciais internas (sintéticas) e explica cada decisão.

> ⚠️ **Não é um sistema bancário real.** Usamos uma base Kaggle factual como
> referência e construímos uma **camada sintética** por cima. Nenhum dado real de
> cliente, identificador, renda, patrimônio, gênero ou raça é utilizado. Decisões
> sensíveis mantêm **humano no loop** (ver [`docs/lgpd-plan.md`](docs/lgpd-plan.md)).

## 2. Escopo e escolhas de design

| Decisão | Escolha | Por quê |
|---|---|---|
| Formulação | Multi-armed bandit contextual | Equilibra exploração/explotação sem A/B longos |
| Algoritmos | Baseline determinístico · Thompson Sampling · Nilos-UCB · LinUCB | Cobre não-contextual (TS/UCB) e contextual (LinUCB) |
| Base factual | [Bank Marketing (Kaggle)](data/kaggle/README.md) | Propensão/conversão bancária, licença aberta |
| Vazamento | `duration` e colunas pós-contato **descartadas** | Evitar leakage temporal (ver Stage 1) |
| Recompensa atrasada | Modelada explicitamente no enriquecimento sintético | Realismo de canais digitais |
| Feature Store | Offline (Parquet) + Online (SQLite) com versionamento | Consistência treino/serving, baixa latência |
| Serving | FastAPI + CLI, log de decisão auditável | Contrato claro, reason codes, versão de política |
| Assistente | RAG sobre políticas sintéticas + LLM plugável (offline por padrão) | Roda sem chave de API; pronto p/ Azure OpenAI/Claude |
| Tracking | MLflow | Rastreio de experimentos e métricas |
| Nuvem-alvo | **Azure** (Key Vault, Managed Identity, App Insights…) | Requisito da Fase 05 |

## 3. Mapa de pastas

```
datathon-7mlet-grupo-64/
├── README.md                  # este arquivo
├── pyproject.toml             # dependências, Python, entrypoint, ferramentas
├── .env.example               # variáveis de ambiente (sem valores reais)
├── Dockerfile / docker-compose.yml
├── Makefile                   # atalhos de comandos
├── .github/workflows/         # CI (lint+test) e CD (build/publish)
├── data/
│   ├── kaggle/README.md       # fonte, link, versão, licença da base factual
│   ├── processed/             # base tratada SEM vazamento (gerada)
│   ├── synthetic_enrichment/  # offer_catalog, offer_events, delayed_rewards (gerada)
│   └── golden_set/evaluation_cases.jsonl  # >= 20 casos versionados
├── docs/                      # arquitetura Azure, model/system card, LGPD, feature store
├── reports/                   # data-generation, relatório técnico, EDA/qualidade
├── notebooks/                 # EDA executável
├── src/adaptive_offers/       # pacote Python (lib + API + CLI)
│   ├── data/                  # loader, preprocessing, synthetic
│   ├── feature_store/         # offline/online store, definitions, materialize
│   ├── bandits/               # baseline, thompson, ucb (nilos), linucb
│   ├── simulation/            # ambiente + métricas (reward, regret, exploração)
│   ├── evaluation/            # offline eval, golden set, fairness
│   ├── policy/                # decision service, reason codes, versionamento
│   ├── assistant/             # RAG + LLM + explicação
│   ├── monitoring/            # drift e reward monitoring
│   ├── api/                   # FastAPI (schemas/contratos + rotas)
│   └── cli.py                 # entrypoint único do pipeline
├── dashboard/                 # BI (Streamlit)
└── tests/                     # unit/ + integration/
```

## 4. Instalação local

Requisitos: **Python 3.11+** e `git`. (Opcional: Docker, conta Kaggle.)

```bash
git clone https://github.com/dionebraga/datathon-7mlet-grupo-64.git
cd datathon-7mlet-grupo-64

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[dev,bi]"      # instala o pacote + ferramentas de dev + dashboard
cp .env.example .env            # ajuste se necessário (roda sem editar)
```

> Sem credenciais Kaggle? O loader usa um **gerador determinístico** que reproduz
> o *schema* da base Bank Marketing para que **todo o pipeline rode offline e em
> CI**. Veja [`data/kaggle/README.md`](data/kaggle/README.md) para baixar a base real.

## 5. Comandos (pipeline ponta a ponta)

Tudo é exposto por um **entrypoint único** (`adaptive-offers`) e pelo `Makefile`:

```bash
# Pipeline completo: dados -> enriquecimento -> treino -> avaliação (1 comando)
adaptive-offers pipeline            # == make pipeline

# Etapas individuais
adaptive-offers data build          # Stage 1: base tratada sem vazamento
adaptive-offers synth generate      # Stage 2: catálogo + eventos + delayed rewards
adaptive-offers train --policy thompson   # Stage 3: simula e treina políticas
adaptive-offers evaluate            # Stage 4: avaliação offline + golden set + fairness
adaptive-offers decide --context examples/context_sample.json  # Stage 5: 1 decisão

# Serviço e UI
adaptive-offers serve               # API FastAPI em http://localhost:8000/docs
make dashboard                      # BI Streamlit em http://localhost:8501

# Qualidade
make test                           # pytest (unit + integration)
make lint                           # ruff + mypy
```

| Comando | Stage | O que entrega |
|---|---|---|
| `adaptive-offers data build` | 1 | Base processada, registro de fonte/versão/licença, decisão de vazamento |
| `adaptive-offers synth generate` | 2 | `offer_catalog`, `offer_events`, `delayed_rewards` + schema |
| `adaptive-offers train` | 3 | Baseline + Thompson + Nilos-UCB + LinUCB, métricas em MLflow |
| `adaptive-offers evaluate` | 4 | Métricas reproduzíveis, golden set, fairness de exposição |
| `adaptive-offers decide` | 5 | Decisão com braço, reason codes, versão da política, log auditável |
| `adaptive-offers serve` | 5 | API com contrato documentado e tratamento de erro |

## 6. Mapa Datathon → entregáveis (Etapas 0–8)

| Etapa | Onde está |
|---|---|
| 0 — Organização | este README, `pyproject.toml`, `.env.example`, `.gitignore`, histórico de commits |
| 1 — Kaggle + EDA | [`data/kaggle/README.md`](data/kaggle/README.md), [`notebooks/01_eda.ipynb`](notebooks/), [`reports/eda-quality-report.md`](reports/eda-quality-report.md), `src/adaptive_offers/data/` |
| 2 — Enriquecimento sintético | `src/adaptive_offers/data/synthetic.py`, [`reports/data-generation.md`](reports/data-generation.md), `data/synthetic_enrichment/schema.json` |
| 3 — Baseline + algoritmos | `src/adaptive_offers/bandits/`, `src/adaptive_offers/simulation/` |
| 4 — Avaliação + golden set | `src/adaptive_offers/evaluation/`, `data/golden_set/evaluation_cases.jsonl` |
| 5 — Serviço demonstrável | `src/adaptive_offers/api/`, `src/adaptive_offers/policy/`, `tests/` |
| 6 — Arquitetura Azure | [`docs/architecture-azure.md`](docs/architecture-azure.md) |
| 7 — Ciclo MLOps | [`docs/mlops-lifecycle.md`](docs/mlops-lifecycle.md), `src/adaptive_offers/monitoring/`, MLflow |
| 8 — Governança/Relatórios | [`docs/model-card.md`](docs/model-card.md), [`docs/system-card.md`](docs/system-card.md), [`docs/lgpd-plan.md`](docs/lgpd-plan.md), [`reports/technical-report.md`](reports/technical-report.md) |

## 7. Limitações conhecidas

- **Base sintética**: o gerador offline reproduz o *schema* do Bank Marketing para
  reprodutibilidade em CI, mas **não substitui** a base real para conclusões de
  negócio. Use a base Kaggle real para análises definitivas.
- **Recompensa simulada**: conversões e *delayed rewards* são gerados por um
  modelo probabilístico documentado — servem para comparar políticas, não para
  estimar lift real em produção.
- **LLM offline por padrão**: sem chave de API, o assistente usa um sumarizador
  determinístico. A qualidade das explicações melhora com Claude/Azure OpenAI.
- **Sem prontidão regulatória**: este é um protótipo acadêmico. **Não** está
  pronto para produção financeira regulada (ver `docs/system-card.md`).

## 8. Equipe e licença

Grupo 64 — FIAP Pós-Tech 7MLET. Licença [MIT](LICENSE).
