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

### 2.5 Neural Bandit (PyTorch — 5ª política, opcional)

Bandit contextual **profundo** (`bandits/neural.py`, requer
`pip install "adaptive-offers[deep]"`): um MLP (PyTorch) prevê
`P(conversão | contexto, braço)` a partir do contexto concatenado ao *one-hot* do
braço; treina online de um *replay buffer* (SGD em mini-batch). A **exploração**
usa **MC-dropout** (Gal & Ghahramani, 2016; Riquelme et al., 2018): um *forward*
estocástico com dropout ligado equivale a uma amostra de Thompson neural.

**Resultado** (fac-símile, didático — o neural ainda não foi re-rodado na base
real): o neural bate o baseline mas fica **atrás do LinUCB linear**. Isso é
esperado e didático: redes neurais são mais flexíveis (capturam não-linearidades)
porém **data-hungry** — em horizontes curtos o LinUCB linear é mais eficiente em
amostra. Com mais dados/treino o neural tende a fechar o *gap*. Justifica o uso de
**PyTorch** e abre caminho para *deep bandits* (trabalhos futuros).

## 3. Cold-start e recompensas atrasadas

- **Cold-start**: TS parte do prior; Nilos-UCB força uma puxada por braço; LinUCB
  parte de `A=I` (regularização ridge). Nenhuma política precisa de dados prévios.
- **Delayed rewards**: o simulador mantém uma fila de feedback pendente; ~40% das
  conversões maturam após 1–30 rounds e **só então** atualizam a política. Decisões
  são tomadas com informação parcial — realismo de canais digitais.

## 4. Resultados (base real UCI · seed=123, 6.000 rounds, 40% delayed)

| Política | Reward acum. | Reward/1k | Regret acum. | Regret ratio | Conversão | Exploração | Lift vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| thompson | **114.290** | 19.048 | 14.070 | 11,8% | 7,1% | 11,7% | **+9,2%** |
| **linucb** | 113.230 | 18.872 | **9.967** | **8,3%** | **9,1%** | 26,4% | +8,2% |
| baseline | 104.700 | 17.450 | 13.029 | 10,9% | 6,2% | 0,0% | — |
| nilos_ucb | 102.020 | 17.003 | 20.347 | 17,0% | 7,1% | 29,1% | **−2,6%** |

**Robustez (5 seeds)**: o LinUCB lidera na média (reward **110.046**, vence **3/5**,
CV **2,97%** — o mais estável); Thompson 105.512 (2/5); baseline instável (CV 20,2%).

### Leitura

- **Ganho modesto e honesto** (~+8–9%) na base real — não os ~+60% do fac-símile. O
  **LinUCB** entrega o **menor regret (8,3%)** e a **maior conversão (9,1%)**, e
  **lidera na média de seeds** → política recomendada.
- **Nem todo bandit vence**: o **Nilos-UCB** sobre-explorou (29%) e ficou **abaixo**
  do baseline (−2,6%). Resultado negativo reportado sem maquiagem.
- **Baseline**: colapsa ~85% das decisões no Empréstimo (alta margem) e ignora o
  contexto. Confirma que **conversão crua engana** — o KPI certo é
  **reward/regret ponderado por margem**.

> Números **reais** (`provenance="real"`), reprodutíveis com `adaptive-offers evaluate`.
