# Backtest temporal — OscaBet (6,856 partidas reais)

Período: validação (`date >= 2024-03-01`). Cada jogo previsto só com features anteriores ao jogo (sem vazamento). Ensemble de 5 seeds.

## Métricas por mercado

| Mercado | Acurácia | Baseline (classe maj.) | Log-loss |
|---|---|---|---|
| Resultado (favorito/argmax) | 0.5181 | 0.4568 | 0.9925 |
| Resultado (com regra de empate) | 0.5063 | — | — |
| Cartões (O/U 4.5) | 0.6215 | 0.5871 | 0.6279 |
| Escanteios (O/U 9.5) | 0.5651 | 0.5210 | 0.6528 |

## Acurácia por temporada (mostra a variação ano a ano)

| Ano | Jogos | Resultado | Cartões | Escanteios |
|---|---|---|---|---|
| 2024 | 2473 | 0.531 | 0.599 | 0.582 |
| 2025 | 2942 | 0.517 | 0.625 | 0.541 |
| 2026 | 1441 | 0.498 | 0.652 | 0.584 |