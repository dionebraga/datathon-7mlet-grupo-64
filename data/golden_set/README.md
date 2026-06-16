# `data/golden_set/` (versionado)

`evaluation_cases.jsonl` — **24 casos** de avaliação versionados (1 JSON por linha),
cobrindo `typical` (8), `segment` (6), `edge` (5) e `adversarial` (5).

Cada caso traz: `context`, `assertion` (ação esperada), `expected_reward_min`
(piso de valor), `oracle_top3` (referência do modelo latente), `justification`
e `pass_fail` (critério explícito).

Regenerar (mantém consistência com o modelo latente):

```bash
python scripts/build_golden_set.py
```

Avaliar uma política contra o golden set:

```bash
adaptive-offers evaluate          # inclui o golden set + métricas + fairness
```

Invariantes checados em **todos** os casos: oferta **elegível** e **reason codes**
presentes. Ver `src/adaptive_offers/evaluation/golden_set.py`.
