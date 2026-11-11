| Modelo | Pedido exato | F1 de itens | JSON válido | Tokens de prompt | s/pedido |
|---|---:|---:|---:|---:|---:|
| Modelo base, sem ajuste (prompt longo) | 0.0% | 0.0% | 0.0% | 883 | 6.23 |
| **Treino com Soup CLI (lr 1e-3)** | 84.5% | 95.1% | 100.0% | 55 | 1.74 |
| Laço manual, ingrediente A (lr 2e-4) | 54.5% | 80.0% | 99.5% | 55 | 1.67 |
| Laço manual, ingrediente B (lr 5e-4) | 80.5% | 93.8% | 99.5% | 55 | 1.63 |
| Laço manual, ingrediente C (lr 1e-3) | 85.0% | 96.7% | 100.0% | 55 | 1.58 |
| Sopa uniforme (A+B+C) | 83.0% | 94.7% | 100.0% | 55 | 1.67 |
| Sopa gulosa (escolheu só C) | 85.0% | 96.7% | 100.0% | 55 | 1.71 |
| LLM grande via API (gpt-oss-120b, prompt longo) | 90.0% | 97.8% | 100.0% | 848 | 0.74 |
