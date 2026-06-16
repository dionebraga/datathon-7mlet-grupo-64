# Política de Suitability e Elegibilidade (SINTÉTICA)

> Documento **sintético** para fins de RAG. Não representa política real de
> nenhuma instituição.

## Princípios
- Nenhuma oferta de crédito é apresentada a clientes com indicador de `default = yes`.
- Ofertas de empréstimo (`OFF_LOAN_PREAPP`) exigem ausência de empréstimo ativo
  (`loan = no`) e ausência de default.
- Produtos de investimento (`OFF_FUND_INTRO`) exigem idade mínima de 25 anos.
- A oferta de controle (`OFF_NONE`, "sem oferta") é sempre elegível e é o
  *fallback* quando nenhuma oferta elegível supera o piso de valor.

## Humano no loop
Decisões de *suitability* sensível (ex.: aumento de limite, realocação de
investimento de alto risco) não podem ser automatizadas: exigem aprovação humana
registrada. O bandit apenas **recomenda** dentro do conjunto elegível.

## Reason codes
Toda decisão deve registrar `SUITABILITY_OK` e, quando aplicável,
`ELIGIBILITY_FILTERED` e `CONTROL_FALLBACK`, garantindo auditabilidade.
