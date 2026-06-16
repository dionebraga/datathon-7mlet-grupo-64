# Relatório de Geração de Dados Sintéticos (Stage 2)

> Documenta **processo, sementes, hipóteses, limitações e riscos** da camada de
> experimentação adaptativa. Reprodutível com `adaptive-offers synth generate`.
> Implementação: `src/adaptive_offers/data/synthetic.py`.

## 1. Por que uma camada sintética?

A base Kaggle é **factual** (propensão de conversão), mas **não contém braços
de oferta nem feedback de experimentação**. Construímos por cima uma camada
sintética que adiciona o que um multi-armed bandit precisa: **ofertas (braços),
contexto de decisão, recompensas e recompensas atrasadas** — mantendo-a
**fisicamente separada** da base original (`data/synthetic_enrichment/` vs
`data/processed/`).

## 2. Processo de geração

```
data/processed (sem vazamento)
        │  client_event_id, age, contact, poutcome, euribor3m, ...
        ▼
build_context_vector(row)         → vetor de contexto (8 dims, CONTEXT_FEATURES)
        ▼
logging policy (ε-exploração)     → escolhe braço elegível + propensity
        ▼
latent_conversion_prob(arm, ctx)  → P(convert) = sigmoid(base_logit + w·ctx)
        ▼
amostragem (seed)                 → converted, clicked, reward (= margem)
        ▼
delayed split (40% dos sucessos)  → delay ∈ [1, 30] dias  → delayed_rewards
```

### Artefatos

| Arquivo | Grão | Conteúdo |
|---|---|---|
| `offer_catalog.parquet` | 1 linha/braço | 6 ofertas + modelo latente (base_logit, pesos, margem, elegibilidade) |
| `offer_events.parquet` | 1 linha/impressão | contexto, braço logado, propensity, click, conversão, recompensa, flags de atraso |
| `delayed_rewards.parquet` | 1 linha/conversão atrasada | `conversion_ts`, `reward_delay_days`, `realized_reward` |
| `schema.json` | — | *schema* completo, chaves e horizonte temporal |

## 3. Sementes e reprodutibilidade

- Semente global: `RANDOM_SEED=42` (configurável no `.env`).
- Toda aleatoriedade usa `numpy.random.default_rng(seed)` — **sem estado global**.
- Mesma semente ⇒ mesmos eventos, conversões e atrasos (verificado em testes).

## 4. Modelo latente (hipóteses)

`P(convert | oferta, contexto) = σ(base_logit + Σ wᵢ·contextoᵢ)`, e a
**recompensa esperada = P(convert) × margem**. Cada oferta tem um *segmento
preferido* distinto, para que uma política **contextual** (LinUCB) possa superar
uma não-contextual:

| Oferta | Margem (R$) | Melhor para | Elegibilidade |
|---|---|---|---|
| Cartão Cashback | 120 | jovens, canal celular | sem *default* |
| Empréstimo Pré-aprovado | 300 | já contatado, juros baixos | sem *default* e sem empréstimo |
| Depósito a Prazo Premium | 200 | sucesso prévio, sênior | todos |
| Fundo de Investimento | 180 | sênior, sucesso prévio | idade ≥ 25 |
| Seguro Bundle | 90 | amplo | todos |
| Sem Oferta (controle) | 0 | — | sempre elegível |

Como o **oráculo** (recompensa esperada verdadeira por braço) é conhecido,
podemos calcular **regret** na avaliação (Stage 3/4).

## 5. Recompensas atrasadas e horizonte

- ~40% das conversões são marcadas como **atrasadas**, com atraso uniforme em
  `[1, 30]` dias (`REWARD_HORIZON_DAYS = 30`).
- No aprendizado online, recompensas atrasadas **não estão disponíveis** no
  instante da decisão → tratado por *reward maturation* (Stage 3): só recompensas
  já maturadas atualizam a política; pendentes entram quando "vencem".
- Modela o realismo de canais digitais (conversão de crédito/investimento leva dias).

## 6. Política de logging

Impressões são geradas por uma **logging policy** ε-exploratória (≈85% uniforme
sobre braços elegíveis + leve viés ao primeiro). A `propensity` registrada
habilita estimadores *off-policy* (IPS) na avaliação offline (Stage 4).

## 7. Limitações

- **Não é dado real**: relações são desenhadas; servem para comparar políticas,
  **não** para estimar *lift* de negócio.
- **Elegibilidade simplificada**: regras de *suitability* reais são mais ricas.
- **Estacionariedade parcial**: o modelo latente é estacionário; *drift* é
  injetado separadamente nos testes de monitoramento (Stage 7).

## 8. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| *Reward hacking* (braço explora margem alta ignorando elegibilidade) | Gate de elegibilidade/suitability **antes** da seleção; controle `no_offer` |
| Confundir sintético com real | Separação física + proveniência + avisos no README e nos cards |
| Viés de exposição entre segmentos | Análise de fairness de exposição na Etapa 4 |
| Vazamento da base factual para a sintética | Apenas colunas sem vazamento entram no contexto; `duration` já removida |
