"""Monta a tabela de resultados em Markdown a partir de results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

ORDEM = ["base_sem_ajuste", "soup_cli", "ing_a", "ing_b", "ing_c", "sopa_uniforme", "sopa_gulosa", "llm_grande_api"]
ROTULOS = {
    "base_sem_ajuste": "Modelo base, sem ajuste (prompt longo)",
    "soup_cli": "**Treino com Soup CLI (lr 1e-3)**",
    "ing_a": "Laço manual, ingrediente A (lr 2e-4)",
    "ing_b": "Laço manual, ingrediente B (lr 5e-4)",
    "ing_c": "Laço manual, ingrediente C (lr 1e-3)",
    "sopa_uniforme": "Sopa uniforme (A+B+C)",
    "sopa_gulosa": "Sopa gulosa (escolheu só C)",
    "llm_grande_api": "LLM grande via API (gpt-oss-120b, prompt longo)",
}


def main() -> None:
    linhas = ["| Modelo | Pedido exato | F1 de itens | JSON válido | Tokens de prompt | s/pedido |",
              "|---|---:|---:|---:|---:|---:|"]
    for nome in ORDEM:
        arq = Path(f"results/{nome}.json")
        if not arq.exists():
            continue
        r = json.loads(arq.read_text())
        linhas.append(f"| {ROTULOS[nome]} | {r['pedido_exato']}% | {r['itens_f1']}% | {r['json_valido']}% "
                      f"| {r['tokens_prompt_medio']} | {r['segundos_por_pedido']} |")
    tabela = "\n".join(linhas)
    Path("results/tabela.md").write_text(tabela + "\n")
    print(tabela)


if __name__ == "__main__":
    main()
