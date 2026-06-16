# Plano LGPD

> Base legal, finalidade, minimização, retenção, mapeamento de identificadores e
> atributos protegidos, política de logs/telemetria e resposta a incidentes.
> Protótipo acadêmico com **dados sintéticos** — nenhum dado real de cliente.

## 1. Base legal e finalidade
- **Finalidade**: personalização responsável de oferta em canais digitais
  (decisão de qual oferta/mensagem/próximo passo apresentar).
- **Base legal (cenário de produção)**: legítimo interesse para personalização,
  com **opt-out** acessível; **consentimento** quando o canal/produto exigir.
- **Decisão automatizada**: mantém-se **humano no loop** em casos sensíveis e há
  direito a revisão (art. 20, LGPD).

## 2. Minimização de dados
- Entram na decisão **apenas** features sem vazamento e **sem atributos
  protegidos**. `duration` e colunas pós-contato são removidas (Stage 1).
- **Não** são usados: identificadores diretos, renda, patrimônio, gênero, raça,
  dados reais de cliente.
- Idade/profissão/estado civil/escolaridade são tratados como features comuns e
  **não** como atributos protegidos de decisão; seu efeito é auditado em fairness.

## 3. Mapeamento de identificadores e atributos protegidos
| Categoria | Campos | Tratamento |
|---|---|---|
| Identificador | `client_event_id` | Surrogate **pseudonimizado**; sem PII real |
| Proxies sensíveis | `age`, `job`, `marital`, `education` | Não protegidos de decisão; monitorados em fairness |
| Proibidos | renda, patrimônio, gênero, raça, ID real | **Não coletados / não usados** |
| Contexto macro | `euribor3m`, `emp_var_rate`, … | Públicos; sem PII |

## 4. Retenção
| Dado | Retenção | Observação |
|---|---|---|
| Log de decisão (auditoria) | período definido (ex.: 12 meses) | pseudonimizado; depois arquivar/anonimizar |
| Features online | TTL por feature view (ex.: 30 dias) | recomputáveis do offline |
| Telemetria de modelo | agregada | separada de dados pessoais |
| Eventos sintéticos | enquanto útil | não são dados pessoais reais |

## 5. Política de logs e telemetria
- Logs de decisão são **estruturados** (JSON) com reason codes e versão da
  política; **sem PII real** (apenas `client_event_id` pseudonimizado).
- Telemetria de modelo (drift, reward, latência) é **agregada** e separada de
  qualquer dado pessoal (Application Insights).
- Acesso a logs por **least privilege** via Entra ID/Managed Identity; segredos
  em Key Vault.

## 6. Direitos do titular (produção)
- Acesso, correção, eliminação e **oposição/opt-out** de personalização.
- Explicação da decisão via assistente (reason codes + política), preservando
  segredo de negócio mínimo.

## 7. Resposta a incidentes (privacidade)
1. Detecção (alerta de fairness/drift/vazamento) → conter e congelar promoção.
2. `rollback()` da política; decisões voltam a baseline/humano.
3. Avaliar impacto, notificar partes conforme exigência regulatória.
4. Registrar, corrigir causa-raiz e atualizar este plano + cards.

## 8. Revisão
Revisão deste plano alinhada à cadência do `model-card.md`/`system-card.md`
(trimestral ou por incidente), com responsáveis definidos.
