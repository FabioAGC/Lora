"""Monta a tabela de resultados em Markdown a partir de results/*.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROTULOS = {
    "qwen_0.5b": "Qwen2.5 0.5B + LoRA",
    "qwen_1.5b": "Qwen2.5 1.5B + LoRA",
    "qwen_3b": "Qwen2.5 3B + LoRA",
    "qwen_7b_qlora": "Qwen2.5 7B + QLoRA (4 bits)",
    "soup_cli": "Qwen2.5 0.5B + LoRA (CPU)",
    "llm_grande_api": "LLM grande via API, sem ajuste (gpt-oss-120b)",
}


def _celula(valor) -> str:
    return "—" if valor is None else f"{valor}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", type=Path, default=Path("results"))
    args = ap.parse_args()

    resultados = {arq.stem: json.loads(arq.read_text()) for arq in sorted(args.pasta.glob("*.json"))
                  if not arq.stem.startswith("_") and arq.stem != "sopa"}
    ordem = [n for n in ROTULOS if n in resultados] + sorted(set(resultados) - set(ROTULOS))
    linhas = ["| Modelo | Pedido exato | Casos normais | Casos de borda | F1 de itens | Tokens de prompt | Treino |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for nome in ordem:
        r = resultados[nome]
        grupos = r.get("por_tipo_de_caso", {})
        treino = f"{r['minutos_treino']:.0f} min" if r.get("minutos_treino") else "—"
        linhas.append(
            f"| {ROTULOS.get(nome, nome)} | {_celula(r['pedido_exato'])} "
            f"| {_celula(grupos.get('normal', {}).get('pedido_exato'))} "
            f"| {_celula(grupos.get('borda', {}).get('pedido_exato'))} "
            f"| {_celula(r['itens_f1'])} | {r['tokens_prompt_medio']} | {treino} |"
        )

    casos = sorted({g for r in resultados.values() for g in r.get("por_tipo_de_caso", {})}
                   - {"normal", "borda"})
    if casos:
        linhas += ["", "Pedido exato por tipo de caso de borda:", "",
                   "| Modelo | " + " | ".join(casos) + " |",
                   "|---|" + "---:|" * len(casos)]
        for nome in ordem:
            grupos = resultados[nome].get("por_tipo_de_caso", {})
            linhas.append(f"| {ROTULOS.get(nome, nome)} | "
                          + " | ".join(_celula(grupos.get(c, {}).get("pedido_exato")) for c in casos) + " |")

    tabela = "\n".join(linhas)
    (args.pasta / "tabela.md").write_text(tabela + "\n")
    print(tabela)


if __name__ == "__main__":
    main()
