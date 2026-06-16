# Catálogo Comercial de Ofertas (SINTÉTICO)

> Documento **sintético** para RAG.

## Ofertas e racional comercial
- **OFF_CC_CASHBACK — Cartão Cashback** (margem 120): foco em público jovem e
  canal celular; aquisição de relacionamento.
- **OFF_LOAN_PREAPP — Empréstimo Pré-aprovado** (margem 300): maior margem; alvo
  reengajamento (cliente já contatado) em cenário de juros baixos; restrito por
  suitability.
- **OFF_TD_PREMIUM — Depósito a Prazo Premium** (margem 200): alvo clientes com
  sucesso prévio e perfil sênior; produto de captação.
- **OFF_FUND_INTRO — Fundo de Investimento Intro** (margem 180): alvo sênior com
  histórico positivo; idade mínima 25.
- **OFF_INSURANCE — Seguro Bundle** (margem 90): oferta ampla, baixo atrito.
- **OFF_NONE — Sem Oferta** (margem 0): controle; preserva experiência quando
  não há oferta de valor.

## Ranqueamento
A decisão é **ponderada por margem**: `valor esperado = P(conversão) × margem`.
Maior conversão **não** implica maior valor — ofertas de alta margem podem
superar ofertas de alta conversão e baixa margem.
