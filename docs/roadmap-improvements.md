# Roadmap de Evoluções — Consultoria Técnica (MLOps & Bandits)

> Análise sênior da arquitetura atual do **Adaptive Offers Platform** com
> recomendações priorizadas de inovação, design de código e experimentação
> avançada. Cada item indica **esforço**, **impacto** e **como aplicar**.

## 0. Diagnóstico da arquitetura atual

| Camada | Hoje | Maturidade | Maior alavanca |
|---|---|:--:|---|
| CLI | `click` | 🟢 boa | Migrar para **Typer** (tipos + ajuda automática) |
| Versionamento de dados | `.gitignore` + provenance.json | 🟡 média | **DVC** para rastrear datasets/artefatos |
| Orquestração | CLI sequencial (`pipeline`) | 🟡 média | **Prefect/Dagster** (retries, agendamento, lineage) |
| Tracking | MLflow (file store) | 🟢 boa | Backend SQLite/Postgres + Model Registry |
| Monitoramento | PSI/KS + reward monitor caseiros | 🟡 média | **EvidentlyAI** (relatórios de drift/fairness) |
| Config | dataclass + env | 🟢 boa | **pydantic-settings** (validação tipada) |
| Logging | `logging` JSON (stderr) | 🟢 boa | **Rich/Loguru** para DX local |
| Bandits | TS, UCB-V, LinUCB (lineares) | 🟢 boa | **Embeddings** + redes neurais (deep bandits) |
| Assistente | RAG TF-IDF + LLM plugável | 🟡 média | Embeddings densos + LLM real (Azure OpenAI/Claude) |

---

## 1. Inovações modernas (Python/MLOps)

| Ferramenta | Substitui/Adiciona | Impacto | Esforço | Quando adotar |
|---|---|:--:|:--:|---|
| **Typer** | `click` | DX, tipos, `--help` rico | 🟢 baixo | Já |
| **DVC** | controle manual de dados | Reprodutibilidade, lineage de dados | 🟡 médio | Antes de dados grandes/reais |
| **Prefect 2** | `pipeline` sequencial | Retries, agendamento, observabilidade | 🟡 médio | Quando houver retraining recorrente |
| **Dagster** | idem (alternativa) | Asset lineage, catálogo | 🔴 alto | Plataforma multi-pipeline |
| **EvidentlyAI** | PSI/KS caseiros | Relatórios de drift/fairness prontos | 🟢 baixo | Já (alto valor p/ governança) |
| **pydantic-settings** | `Settings` (dataclass) | Validação de env tipada | 🟢 baixo | Já |
| **Ray/Vowpal Wabbit** | simulador próprio | Bandits em escala | 🔴 alto | Produção real |

### 1.1 EvidentlyAI (recomendação nº 1 — governança)
Gera *Data Drift* e *Fairness* reports com pouquíssimo código e complementa o
System Card. Mapeia diretamente para a Etapa 7/8.

```python
# src/adaptive_offers/monitoring/evidently_report.py  (opcional: pip install evidently)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def drift_html(reference_df, current_df, out="artifacts/drift_report.html"):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(out)
    return out
```
> Mantemos os monitores PSI/KS atuais como *gate* programático (rápido, sem dep)
> e usamos Evidently para o **relatório visual** de auditoria.

### 1.2 Prefect (orquestração com retries e agendamento)
```python
from prefect import flow, task

@task(retries=2, retry_delay_seconds=10)
def build(): from adaptive_offers.cli import data_build; ...
@flow(name="adaptive-offers-pipeline")
def pipeline_flow():
    build(); synth(); train(); evaluate()
```
Ganhos: *retries* idempotentes, *scheduling* (cron), UI de execução e lineage —
sem reescrever a lógica (apenas envolve os comandos existentes).

### 1.3 DVC (versionamento de dados)
```bash
dvc init && dvc add data/processed/bank_marketing_processed.parquet
git add data/processed/*.dvc .gitignore && dvc remote add -d azure azure://container/path
```
Liga cada experimento MLflow a um **hash de dados** rastreável (essencial com a
base Kaggle real).

---

## 2. Design de código & boas práticas

| Melhoria | Benefício | Aplicação |
|---|---|---|
| **pydantic-settings** | env tipado + validação | `Settings(BaseSettings)` |
| **Typer** | CLI declarativa, tipos | `app = typer.Typer()` |
| **Rich** (✅ implementado) | tabelas/console bonitos | comparação de políticas na CLI |
| **Loguru** | logging simples + sinks | alternativa ao `logging` |
| **Protocols/ABCs** | contrato de política | já temos `Policy(ABC)` ✅ |
| **Result/Either** | erros explícitos | retorno de `decide` |

### 2.1 pydantic-settings (migração do `Settings`)
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    random_seed: int = 42
    policy_default: str = "thompson"
    exploration_floor: float = 0.02
    mlflow_tracking_uri: str = "file:./mlruns"
```
Vantagem: validação de tipos automática, *fail-fast* em env inválido, e
documentação do schema de configuração "de graça".

### 2.2 Rich na CLI — **implementado neste repositório**
`adaptive-offers train --compare` agora imprime uma **tabela Rich** colorida
(com *fallback* para texto puro se Rich faltar). Ver `src/adaptive_offers/cli.py`
(`_print_comparison`). Exemplo de saída:

```
        Comparação de políticas
┏━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Política  ┃  Reward  ┃ Regret ┃ Lift ┃
┡━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ thompson  │  114.290 │ 11.8%  │  +9% │
│ linucb    │  113.230 │  8.3%  │  +8% │
└───────────┴──────────┴────────┴──────┘
```

---

## 3. Experimentação avançada

| Extensão | Ideia | Ganho esperado |
|---|---|---|
| **Deep/Neural bandit** (✅ implementado) | MLP PyTorch + MC-dropout (`bandits/neural.py`) | Captura não-linearidades; +31% vs baseline (data-hungry) |
| **Embeddings de contexto** | embeddar `job`/`education`/segmento → vetor denso | Generalização entre segmentos |
| **Embeddings no RAG** | trocar TF-IDF por `sentence-transformers`/Azure | Recuperação semântica melhor |
| **Doubly Robust OPE** | estimador off-policy DR (vs IPS) | Avaliação com menos variância |
| **Neural-LinUCB / NeuralTS** | bônus de incerteza sobre features neurais | Estado-da-arte contextual |
| **LLM real p/ explicação** | Azure OpenAI/Claude no assistente | Explicações naturais e auditáveis |

### 3.1 Embeddings de contexto + LinUCB
Hoje o contexto é um vetor de 8 features binárias/normalizadas. Um passo natural:
```python
# Esboço: embiddar categóricas e concatenar ao contexto numérico
from sklearn.preprocessing import OneHotEncoder
# (ou nn.Embedding em PyTorch para dimensões maiores)
ctx_dense = np.concatenate([numeric_ctx, onehot_job, onehot_education])
# LinUCB passa a operar em dimensão maior -> roteamento mais fino por segmento
```
Trade-off: dimensão maior → mais dados para convergir (cuidado com cold-start).

### 3.2 LLM real para explicabilidade (já plugável)
O assistente já abstrai o provedor (`LLM_PROVIDER=anthropic|azure_openai|offline`).
Basta definir a chave; o *fallback* offline garante CI. Próximo passo: cache de
explicações (custo) e *guardrails* anti-prompt-injection (já no System Card).

---

## 4. Priorização (o que eu faria primeiro)

| Prioridade | Item | Razão |
|:--:|---|---|
| 🥇 | **Rich na CLI** (✅ feito) + **pydantic-settings** | DX imediata, baixo risco |
| 🥈 | **EvidentlyAI** report de drift/fairness | Alto valor de governança (Etapa 7/8) |
| 🥉 | **Typer** + **Prefect** | Profissionaliza CLI e orquestração |
| 4 | **DVC** | Necessário ao adotar a base Kaggle real |
| 5 | **Deep/Neural bandit + embeddings** | Ganho de modelagem (requer mais dados) |

> **Princípio**: cada evolução deve preservar o *fallback* offline e a
> reprodutibilidade por semente — pilares do projeto. Nada de dependência dura
> que quebre CI sem rede.

## Referências
- Riquelme, Tucker & Snoek (2018). *Deep Bayesian Bandits Showdown*. ICLR.
- Dudík, Langford & Li (2011). *Doubly Robust Policy Evaluation*. ICML.
- EvidentlyAI docs · Prefect docs · DVC docs · Typer docs.
