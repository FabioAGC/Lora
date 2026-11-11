"""Avalia um modelo local no conjunto de teste.

Uso:
    python -m pedido_lora.avaliar --nome base_zero_shot --prompt longo
    python -m pedido_lora.avaliar --nome sopa_uniforme --adaptador outputs/sopa_uniforme
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .esquema import interpretar
from .metricas import agregar, comparar
from .modelo import MODELO_BASE, carregar_base, gerar, ler_jsonl, montar_prompt
from .prompts import PROMPT_CURTO, prompt_longo


def pontuar(saidas: list[str], exemplos: list[dict]) -> dict:
    comparacoes = [comparar(interpretar(s), json.loads(ex["pedido"])) for s, ex in zip(saidas, exemplos)]
    return agregar(comparacoes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--adaptador", default=None)
    ap.add_argument("--prompt", choices=["curto", "longo"], default="curto")
    ap.add_argument("--teste", default="data/teste.jsonl")
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--saida", type=Path, default=Path("results"))
    args = ap.parse_args()

    tok, modelo = carregar_base(args.modelo, usar_gpu=True)
    if args.adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, args.adaptador).merge_and_unload()

    exemplos = ler_jsonl(args.teste)
    if args.limite:
        exemplos = exemplos[: args.limite]
    sistema = PROMPT_CURTO if args.prompt == "curto" else prompt_longo()
    prompts = [montar_prompt(tok, sistema, ex["mensagem"]) for ex in exemplos]
    tokens_prompt = sum(len(tok(p)["input_ids"]) for p in prompts) / len(prompts)

    inicio = time.time()
    saidas = gerar(tok, modelo, prompts)
    segundos = time.time() - inicio

    resultado = {
        "nome": args.nome, "modelo": args.modelo, "adaptador": args.adaptador, "prompt": args.prompt,
        **pontuar(saidas, exemplos),
        "tokens_prompt_medio": round(tokens_prompt),
        "segundos_por_pedido": round(segundos / len(exemplos), 2),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / f"{args.nome}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    previsoes = Path("outputs") / "previsoes"
    previsoes.mkdir(parents=True, exist_ok=True)
    with (previsoes / f"{args.nome}.jsonl").open("w") as f:
        for ex, s in zip(exemplos, saidas):
            f.write(json.dumps({"mensagem": ex["mensagem"], "esperado": ex["pedido"], "saida": s},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
