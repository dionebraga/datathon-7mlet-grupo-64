<div align="center">

# 📚 Documentação — Adaptive Offers Platform

Índice da documentação técnica e de governança do projeto.

</div>

---

## ☁️ Arquitetura & Plataforma

| Documento | Descrição |
|---|---|
| [🏗️ architecture-azure.md](architecture-azure.md) | Arquitetura-alvo **Azure** — diagrama Mermaid, mapeamento de serviços (compute/API/dados/IA-RAG/observabilidade/segurança/identidade/governança), Key Vault + Managed Identity, plano de deploy e **FinOps** (ROI/TCO/escala). |
| [🗄️ feature-store.md](feature-store.md) | **Feature Store** — arquitetura offline (Parquet) + online (SQLite/Redis), entidades, *feature views*, materialização e paridade treino/serving. |
| [📖 data-dictionary.md](data-dictionary.md) | Dicionário de dados da base processada e camadas sintéticas. |

## 🔁 MLOps & Operação

| Documento | Descrição |
|---|---|
| [♻️ mlops-lifecycle.md](mlops-lifecycle.md) | Ciclo MLOps — MLflow, versionamento/registry de políticas, *approval gate*, canary, **rollback**, monitoramento de **drift** (PSI/KS) e **reward**, plano de retreino. |

## 🛡️ Governança & Conformidade

| Documento | Descrição |
|---|---|
| [🪪 model-card.md](model-card.md) | **Model Card** — uso pretendido/fora de escopo, métricas, fairness, vieses, limitações, plano de revisão. |
| [🧯 system-card.md](system-card.md) | **System Card** — fluxo de decisão, dependências, guardrails, cenários de risco (reward hacking, manipulação de contexto, abuso do assistente, suitability), monitoramento. |
| [🔒 lgpd-plan.md](lgpd-plan.md) | **Plano LGPD** — base legal, minimização, retenção, atributos protegidos, logs/telemetria, resposta a incidentes. |

## 🎤 Demo Day

| Documento | Descrição |
|---|---|
| [🎯 pitch.md](pitch.md) | Roteiro do pitch (10 min + 5 min Q&A) — problema, abordagem, demonstração, evidências, riscos, governança, impacto e FinOps. |

---

## 📝 Relatórios (em [`../reports/`](../reports/))

| Relatório | Descrição |
|---|---|
| [📑 technical-report.md](../reports/technical-report.md) | Relatório técnico completo (≤10 páginas). |
| [🎰 algorithmic-strategy.md](../reports/algorithmic-strategy.md) | Estratégia algorítmica e comparação quantitativa das políticas. |
| [📏 offline-evaluation.md](../reports/offline-evaluation.md) | Avaliação offline, golden set, sensibilidade, IPS e fairness. |
| [🧬 data-generation.md](../reports/data-generation.md) | Geração de dados sintéticos (processo, sementes, riscos). |
| [🔎 eda-quality-report.md](../reports/eda-quality-report.md) | EDA e qualidade da base processada. |

<div align="center">

[⬅ Voltar ao README principal](../README.md)

</div>
