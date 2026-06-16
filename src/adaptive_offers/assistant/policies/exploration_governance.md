# Governança de Exploração e Bandits (SINTÉTICO)

> Documento **sintético** para RAG.

## Exploração responsável
- A plataforma usa multi-armed bandits para equilibrar exploração e explotação.
- Um **piso mínimo de exploração** (`EXPLORATION_FLOOR`, default 2%) evita que a
  política congele cedo demais em um único braço.
- Exploração nunca pode violar gates de elegibilidade/suitability: explora-se
  apenas dentro do conjunto **elegível**.

## Recompensas atrasadas
- Conversões podem maturar após dias (horizonte de 30). A política só aprende com
  recompensas **maturadas**; pendências entram quando vencem. Decisões são
  tomadas com informação parcial.

## Riscos monitorados
- **Reward hacking**: braço que explora margem alta ignorando elegibilidade —
  mitigado pelo gate antes da seleção.
- **Manipulação de contexto**: variações artificiais de contexto para forçar
  ofertas restritas — mitigado por validação de schema e limites nos campos.
- **Drift**: mudança na relação contexto→conversão — monitorada (PSI/KS) com
  alerta e gatilho de retreino.
