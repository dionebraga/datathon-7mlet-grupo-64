# Resumo de Privacidade e LGPD (SINTÉTICO)

> Documento **sintético** para RAG. Plano completo em `docs/lgpd-plan.md`.

## Bases e princípios
- **Base legal**: legítimo interesse para personalização de oferta, com opção de
  *opt-out*; consentimento para canais específicos quando exigido.
- **Minimização**: apenas features sem vazamento e sem atributos protegidos
  entram na decisão. Idade/profissão/estado civil são tratados como features
  comuns, **não** como atributos de decisão protegidos.
- **Sem dados reais de cliente**: identificadores, renda, patrimônio, gênero e
  raça **não** são usados.

## Retenção e telemetria
- Logs de decisão retidos por período definido, com pseudonimização do
  `client_event_id`. Telemetria de modelo separada de dados pessoais.

## Resposta a incidentes
- Em caso de suspeita de viés ou vazamento, a política pode ser revertida
  (`rollback`) e a decisão volta a *baseline*/humano enquanto se investiga.
