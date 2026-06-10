# Lucro simulado (backtest de apostas) — OscaBet

Período de validação: 6,892 jogos reais (features do próprio jogo, sem vazamento). Odds derivadas das frequências base (casa ingênua).

Stake fixo de 1 unidade por aposta. ROI = lucro / total apostado.


## Odds JUSTAS (sem margem)

| Mercado | Estratégia | Apostas | Acerto | Lucro (u) | ROI |
|---|---|---|---|---|---|
| Resultado | favorito | 6892 | 52.2% | +2318.7 | +33.6% |
| Resultado | value | 10814 | 41.1% | +2734.6 | +25.3% |
| Cartões O/U 4.5 | favorito | 6892 | 62.4% | +1606.6 | +23.3% |
| Cartões O/U 4.5 | value | 6892 | 58.6% | +1521.3 | +22.1% |
| Escanteios O/U 9.5 | favorito | 6892 | 56.7% | +985.5 | +14.3% |
| Escanteios O/U 9.5 | value | 6892 | 55.9% | +930.6 | +13.5% |

## Odds com MARGEM de 5% (casa real)

| Mercado | Estratégia | Apostas | Acerto | Lucro (u) | ROI |
|---|---|---|---|---|---|
| Resultado | favorito | 6892 | 52.2% | +1880.1 | +27.3% |
| Resultado | value | 9767 | 41.7% | +2048.0 | +21.0% |
| Cartões O/U 4.5 | favorito | 6892 | 62.4% | +1201.9 | +17.4% |
| Cartões O/U 4.5 | value | 6084 | 59.2% | +1122.3 | +18.4% |
| Escanteios O/U 9.5 | favorito | 6892 | 56.7% | +610.4 | +8.9% |
| Escanteios O/U 9.5 | value | 5272 | 57.7% | +625.3 | +11.9% |

## Interpretação (ressalva importante para o artigo)

O lucro positivo e consistente mostra que o modelo **agrega informação real além
das frequências base** — ele converte as features (forma, xG, posse, etc.) em
probabilidades melhores que o chute histórico, e isso é monetizável CONTRA uma casa
ingênua. **PORÉM**, o ROI aqui é otimista: casas de aposta reais NÃO precificam só
pela frequência base — elas já embutem força dos times, mando e forma recente, que
são justamente o que nossas features capturam. Contra um mercado real (sharp), a
margem do modelo seria bem menor ou até negativa. Portanto, esta curva mede o
**conteúdo informativo** do modelo, não um retorno realista de apostas.