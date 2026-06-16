# Estratégia Algorítmica e Comparação (Stage 3)

> Reprodutível com `adaptive-offers train` (treina e compara as 4 políticas) e
> registra métricas no MLflow. Implementação em `src/adaptive_offers/bandits/` e
> `src/adaptive_offers/simulation/`.

## 1. Por que multi-armed bandit?

Regras fixas e A/B longos desperdiçam tráfego e reagem devagar. Modelamos a
escolha de oferta como **bandit contextual**: equilibra exploração (descobrir
braços promissores) e explotação (usar o melhor conhecido), aprendendo online.

**Sinal de aprendizado** = conversão (Bernoulli). **Ranqueamento** = `estimativa
× margem`, de forma que as políticas otimizam **valor esperado**, não conversão
crua — distinção decisiva (ver baseline abaixo).

## 2. Políticas implementadas

| Política | Família | Exploração | Contextual? |
|---|---|---|---|
| `baseline` | greedy otimista | **nenhuma** | não |
| `thompson` | Bayesiana (Beta-Bernoulli) | amostragem do posterior | não |
| `nilos_ucb` | UCB-V (variance-aware) | bônus de incerteza | não |
| `linucb` | Linear contextual | bônus + contexto | **sim** |

### 2.1 Baseline determinístico (controle)

Greedy otimista: estima a média de conversão por braço (init otimista) e sempre
explora o melhor `média × margem`. **Sem exploração ativa** — fica preso no
primeiro braço que converte. É a limitação que a política adaptativa deve vencer.

### 2.2 Thompson Sampling (priors documentados)

Posterior Beta(α, β) por braço, prior **Beta(1,1)** (uniforme, fracamente
informativo). Em cada decisão amostra θ_a ~ Beta(α_a, β_a) e escolhe
`argmax θ_a × margem_a`. A amostragem **é** a exploração: braços com posterior
largo (poucos dados) ocasionalmente são amostrados alto. Conjugação Bernoulli:
sucesso → α+1, falha → β+1.

### 2.3 Nilos-UCB (justificativa explícita)

Implementado como **UCB-V** (Audibert, Munos & Szepesvári, 2009) — membro
*variance-aware* da família UCB:

```
UCB1   :  μ_a + c·sqrt( 2·ln(t)/n_a )
Nilos  :  μ_a + sqrt( 2·v_a·ln(t)/n_a ) + c·3·ln(t)/n_a
```

**Justificativa**: a variância de conversão difere muito entre ofertas (margens
e taxas-base distintas). O bônus *variance-aware* explora braços de alta
incerteza com mais eficiência que o bônus fixo do UCB1. Braços não puxados
recebem índice infinito → **uma puxada inicial garantida** (cold-start).

### 2.4 LinUCB (contextual)

Modelo linear ridge por braço sobre o vetor de contexto (8 dims):
`A_a = I + Σxxᵀ`, `b_a = Σr·x`, `θ_a = A_a⁻¹b_a`,
`ucb = θ_aᵀx + α·sqrt(xᵀA_a⁻¹x)`. Como cada oferta tem **segmento preferido**
distinto (Stage 2), o LinUCB roteia a oferta certa para o contexto certo.

## 3. Cold-start e recompensas atrasadas

- **Cold-start**: TS parte do prior; Nilos-UCB força uma puxada por braço; LinUCB
  parte de `A=I` (regularização ridge). Nenhuma política precisa de dados prévios.
- **Delayed rewards**: o simulador mantém uma fila de feedback pendente; ~40% das
  conversões maturam após 1–30 rounds e **só então** atualizam a política. Decisões
  são tomadas com informação parcial — realismo de canais digitais.

## 4. Resultados (facsimile, seed=123, 20.000 rounds, 40% delayed)

| Política | Reward acumulado | Reward/1k | Regret acum. | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| **linucb** | **424.820** | 21.241 | **22.890** | **5,1%** | 9,9% | 10,9% | **+66,6%** |
| thompson | 389.180 | 19.459 | 55.452 | 12,2% | 7,5% | 4,7% | +52,6% |
| nilos_ucb | 383.010 | 19.150 | 63.767 | 14,1% | 7,2% | 15,0% | +50,2% |
| baseline | 255.060 | 12.753 | 193.694 | 42,7% | 10,7% | 0,0% | — |

### Leitura

- **LinUCB vence** (+66,6% de valor vs baseline; regret de apenas 5,1% do ótimo):
  usar o **contexto** para personalizar a oferta supera políticas não-contextuais.
- **Thompson e Nilos-UCB** batem o baseline em ~50% — o valor da **exploração**.
- **Paradoxo do baseline**: tem a **maior taxa de conversão (10,7%)** mas o **menor
  reward**. Ele trava na oferta que converte mais (baixa margem) e ignora ofertas
  de alta margem e o contexto. Confirma que **conversão crua engana** — o KPI certo
  é **reward/regret ponderado por margem**.

> Números do *facsimile* (reprodutíveis). Conclusões qualitativas (ordem das
> políticas, valor de contexto/exploração) se mantêm na base real; magnitudes
> variam.
