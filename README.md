<div align="center">

# 🎯 Adaptive Offers Platform

### Plataforma de experimentação adaptativa para ofertas financeiras com *multi-armed bandits*

*Decide, em canais digitais de uma instituição financeira, **qual oferta / mensagem / próximo passo** apresentar a cada cliente elegível — equilibrando exploração e explotação em vez de regras fixas ou testes A/B longos.*

**FIAP Pós-Tech `7MLET` · Fase 05 · Datathon · Grupo 64**

<br/>

[![CI](https://github.com/dionebraga/datathon-7mlet-grupo-64/actions/workflows/ci.yml/badge.svg)](https://github.com/dionebraga/datathon-7mlet-grupo-64/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen?style=flat&logo=pytest&logoColor=white)](tests/)
[![Ruff](https://img.shields.io/badge/lint-ruff%20clean-success?style=flat&logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.8.0-blue?style=flat)](pyproject.toml)

<br/>

**Stack**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)

</div>

---

## 📑 Índice

| | | |
|---|---|---|
| [1. Visão do problema](#1-visão-do-problema) | [4. 🚀 Como rodar (PowerShell)](#4--como-rodar-no-windows--powershell) | [7. 📚 Documentação](#7--documentação) |
| [2. Escopo e design](#2-escopo-e-escolhas-de-design) | [5. 👀 Como visualizar](#5--como-visualizar) | [8. ⚠️ Limitações](#8--limitações-conhecidas) |
| [3. 🗂️ Mapa de pastas](#3-️-mapa-de-pastas) | [6. 🧪 Comandos](#6--comandos-pipeline-ponta-a-ponta) | [9. 🏁 Resultados](#9--resultados-principais) |

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
| 🧠 Formulação | Multi-armed bandit contextual | Equilibra exploração/explotação sem A/B longos |
| 🎰 Algoritmos | Baseline · Thompson Sampling · Nilos-UCB · LinUCB | Cobre não-contextual (TS/UCB) e contextual (LinUCB) |
| 📊 Base factual | [Bank Marketing (Kaggle)](data/kaggle/README.md) | Propensão/conversão bancária, licença aberta |
| 🚫 Vazamento | `duration` e colunas pós-contato **descartadas** | Evitar *target leakage* (Stage 1) |
| ⏳ Recompensa atrasada | Modelada no enriquecimento sintético | Realismo de canais digitais |
| 🗄️ Feature Store | Offline (Parquet) + Online (SQLite) versionado | Consistência treino/serving, baixa latência |
| 🌐 Serving | FastAPI + CLI, log de decisão auditável | Contrato claro, reason codes, versão de política |
| 🤖 Assistente | RAG sobre políticas sintéticas + LLM plugável (offline por padrão) | Roda sem chave de API; pronto p/ Azure OpenAI/Claude |
| 📈 Tracking | MLflow | Rastreio de experimentos e métricas |
| ☁️ Nuvem-alvo | **Azure** (Key Vault, Managed Identity, App Insights…) | Requisito da Fase 05 |

## 3. 🗂️ Mapa de pastas

```
datathon-7mlet-grupo-64/
├── 📄 README.md · pyproject.toml · .env.example · .gitignore · LICENSE · Makefile
├── 🐳 Dockerfile · docker-compose.yml
├── ⚙️  .github/workflows/        # CI (lint+test) e CD (build/publish imagem)
├── 📦 data/
│   ├── kaggle/README.md          # fonte, link, versão, licença da base factual
│   ├── processed/                # base tratada SEM vazamento (gerada)
│   ├── synthetic_enrichment/      # offer_catalog, offer_events, delayed_rewards (gerada)
│   └── golden_set/evaluation_cases.jsonl  # >= 20 casos versionados
├── 📖 docs/                       # arquitetura Azure, model/system card, LGPD, feature store
├── 📝 reports/                    # data-generation, relatório técnico, EDA, avaliação
├── 📓 notebooks/                  # EDA executável
├── 🧩 src/adaptive_offers/        # pacote Python (lib + API + CLI)
│   ├── data/ · feature_store/ · bandits/ · simulation/ · evaluation/
│   ├── policy/ · assistant/ · monitoring/ · api/ · cli.py
├── 📊 dashboard/                  # BI (Streamlit)
└── ✅ tests/                      # unit/ + integration/ (55 testes)
```

## 4. 🚀 Como rodar no Windows / PowerShell

> Pré-requisitos: **Python 3.11+** e **git**. (Opcional: Docker, conta Kaggle.)
> Todos os comandos abaixo são **PowerShell** (testados no Windows 11).

```powershell
# 1) Clonar
git clone https://github.com/dionebraga/datathon-7mlet-grupo-64.git
cd datathon-7mlet-grupo-64

# 2) Ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#  ↳ se bloquear por política de execução, rode uma vez:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3) Instalar o pacote + ferramentas de dev + dashboard
pip install -e ".[dev,bi]"
Copy-Item .env.example .env      # roda sem editar

# 4) Pipeline completo (dados → enriquecimento → treino → avaliação)
adaptive-offers pipeline

# 5) Uma decisão de exemplo (saída JSON limpa, pipeável)
adaptive-offers decide --context examples\context_sample.json
adaptive-offers decide --context examples\context_sample.json | ConvertFrom-Json
```

> 💡 Sem credenciais Kaggle? O loader usa um **gerador determinístico** que
> reproduz o *schema* da base Bank Marketing para que **todo o pipeline rode
> offline**. Para baixar a base real, veja [`data/kaggle/README.md`](data/kaggle/README.md).

<details>
<summary>🐧 Linux / macOS (bash) e 🐳 Docker</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bi]"
cp .env.example .env
make pipeline        # atalhos do Makefile (Linux/macOS)
make test
```

```bash
# Stack completa (API + MLflow + dashboard) em containers
docker compose up --build
```
> No Windows, o `make` não existe por padrão — use os comandos `adaptive-offers ...`
> e `pytest`/`streamlit` diretamente (Seção 6). O Docker funciona igual.
</details>

## 5. 👀 Como visualizar

| O quê | Comando (PowerShell) | Abrir em |
|---|---|---|
| 🌐 **API + Swagger** (docs interativa) | `adaptive-offers serve` | http://localhost:8000/docs |
| 📊 **Dashboard BI** (comparação, regret, decisão) | `streamlit run dashboard\app.py` | http://localhost:8501 |
| 📈 **MLflow** (experimentos) | `mlflow ui` | http://localhost:5000 |
| 🧾 **Log auditável de decisões** | `Get-Content artifacts\decisions\audit.jsonl -Tail 5` | terminal |
| 📓 **Notebook de EDA** | `jupyter lab notebooks\01_eda.ipynb` | navegador |

**Testar a API com PowerShell** (com o `serve` rodando em outra janela):

```powershell
# Decisão para um cliente sênior com sucesso prévio
$body = @{ age = 66; contact = "cellular"; poutcome = "success"; euribor3m = 0.8 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/decide -Method Post -Body $body -ContentType "application/json"

# Explicação da decisão (assistente LLM/RAG)
Invoke-RestMethod -Uri "http://localhost:8000/assistant/explain?question=Por que essa oferta?" `
  -Method Post -Body $body -ContentType "application/json"
```

## 6. 🧪 Comandos (pipeline ponta a ponta)

| Comando | Stage | O que entrega |
|---|:---:|---|
| `adaptive-offers data build` | 1 | Base processada, registro de fonte/versão/licença, decisão de vazamento |
| `adaptive-offers synth generate` | 2 | `offer_catalog`, `offer_events`, `delayed_rewards` + schema |
| `adaptive-offers train` | 3 | Baseline + Thompson + Nilos-UCB + LinUCB, métricas em MLflow |
| `adaptive-offers evaluate` | 4 | Métricas reproduzíveis, golden set, fairness de exposição |
| `adaptive-offers decide` | 5 | Decisão com braço, reason codes, versão da política, log auditável |
| `adaptive-offers serve` | 5 | API com contrato documentado e tratamento de erro |
| `adaptive-offers pipeline` | 1–4 | **Tudo em um comando** |
| `pytest` | — | 55 testes (unit + integração) |

## 7. 📚 Documentação

| 📄 Documento | Conteúdo |
|---|---|
| [docs/architecture-azure.md](docs/architecture-azure.md) | ☁️ Arquitetura-alvo Azure (Mermaid, serviços, FinOps) |
| [docs/feature-store.md](docs/feature-store.md) | 🗄️ Feature Store offline/online |
| [docs/mlops-lifecycle.md](docs/mlops-lifecycle.md) | 🔁 Ciclo MLOps (MLflow, drift, promote/rollback) |
| [docs/model-card.md](docs/model-card.md) | 🪪 Model Card |
| [docs/system-card.md](docs/system-card.md) | 🛡️ System Card (riscos, guardrails) |
| [docs/lgpd-plan.md](docs/lgpd-plan.md) | 🔒 Plano LGPD |
| [docs/pitch.md](docs/pitch.md) | 🎤 Roteiro do pitch (Demo Day) |
| [reports/technical-report.md](reports/technical-report.md) | 📑 Relatório técnico (≤10 páginas) |
| [reports/algorithmic-strategy.md](reports/algorithmic-strategy.md) | 🎰 Estratégia algorítmica + comparação |
| [reports/offline-evaluation.md](reports/offline-evaluation.md) | 📏 Avaliação offline + golden set + fairness |
| [reports/data-generation.md](reports/data-generation.md) | 🧬 Geração de dados sintéticos |

### Mapa Datathon → entregáveis (Etapas 0–8)

| Etapa | Onde está |
|:---:|---|
| 0️⃣ Organização | README, `pyproject.toml`, `.env.example`, `.gitignore`, histórico de commits |
| 1️⃣ Kaggle + EDA | [`data/kaggle/`](data/kaggle/README.md), [`notebooks/`](notebooks/), `src/.../data/` |
| 2️⃣ Enriquecimento | `src/.../data/synthetic.py`, [`reports/data-generation.md`](reports/data-generation.md) |
| 3️⃣ Baseline + algoritmos | `src/.../bandits/`, `src/.../simulation/` |
| 4️⃣ Avaliação + golden set | `src/.../evaluation/`, `data/golden_set/` |
| 5️⃣ Serviço demonstrável | `src/.../api/`, `src/.../policy/`, `tests/` |
| 6️⃣ Arquitetura Azure | [`docs/architecture-azure.md`](docs/architecture-azure.md) |
| 7️⃣ Ciclo MLOps | [`docs/mlops-lifecycle.md`](docs/mlops-lifecycle.md), `src/.../monitoring/` |
| 8️⃣ Governança | [`docs/model-card.md`](docs/model-card.md), [`docs/system-card.md`](docs/system-card.md), [`docs/lgpd-plan.md`](docs/lgpd-plan.md) |

## 8. ⚠️ Limitações conhecidas

- **Base sintética**: o gerador offline reproduz o *schema* do Bank Marketing
  para reprodutibilidade em CI, mas **não substitui** a base real para conclusões
  de negócio.
- **Recompensa simulada**: conversões e *delayed rewards* são gerados por um
  modelo probabilístico documentado — servem para comparar políticas, não para
  estimar *lift* real.
- **LLM offline por padrão**: sem chave de API, o assistente usa um sumarizador
  determinístico (sem alucinação). A qualidade melhora com Claude/Azure OpenAI.
- **Sem prontidão regulatória**: protótipo acadêmico. **Não** está pronto para
  produção financeira regulada (ver [`docs/system-card.md`](docs/system-card.md)).

## 9. 🏁 Resultados principais

| Política | Reward acumulado | Regret ratio | Lift vs baseline | Golden set |
|---|---:|---:|---:|:---:|
| 🥇 **LinUCB** (contextual) | **424.820** | **5,1%** | **+66,6%** | **100%** |
| 🥈 Thompson Sampling | 389.180 | 12,2% | +52,6% | — |
| 🥉 Nilos-UCB (UCB-V) | 383.010 | 14,1% | +50,2% | — |
| Baseline (controle) | 255.060 | 42,7% | — | 54% |

- ✅ **55 testes** passando · **ruff** limpo · pipeline ponta-a-ponta em **1 comando**.
- 🔍 **IPS off-policy** concorda com a simulação on-policy (~21/impressão).
- ⚖️ **Fairness** de exposição: disparidade **0,00** entre segmentos.

---

<div align="center">

**Grupo 64 — FIAP Pós-Tech 7MLET** · Licença [MIT](LICENSE)

[⬆ Voltar ao topo](#-adaptive-offers-platform)

</div>
