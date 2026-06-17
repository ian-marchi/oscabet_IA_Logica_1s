# Comparação Rede Neural × CatBoost (backtest temporal)

Conjunto de validação: 6,965 partidas (`date >= 2024-03-01`), sem vazamento.

| Mercado | Métrica | Rede Neural | CatBoost | Δ |
|---|---|---|---|---|
| Resultado | Acurácia | 0.5219 | 0.5235 | +0.0016 |
| Resultado | Log-loss | 0.9910 | 0.9886 | -0.0025 |
| Cartões (O/U 4,5) | Acurácia | 0.6247 | 0.6296 | +0.0049 |
| Cartões (O/U 4,5) | Log-loss | 0.6265 | 0.6313 | +0.0048 |
| Escanteios (O/U 9,5) | Acurácia | 0.5661 | 0.5815 | +0.0154 |
| Escanteios (O/U 9,5) | Log-loss | 0.6535 | 0.6521 | -0.0014 |