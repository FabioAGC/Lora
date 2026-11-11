"""Demonstração: extrai o pedido de uma mensagem digitada.

Uso:
    python -m pedido_lora.inferir --adaptador outputs/soup_cli "2 xburguer sem cebola e uma coca zero, pago no pix"
"""
from __future__ import annotations

import argparse
import json
import time

from .esquema import interpretar
from .modelo import MODELO_BASE, carregar_base, gerar, montar_prompt
from .prompts import PROMPT_CURTO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mensagem")
    ap.add_argument("--adaptador", default="outputs/soup_cli")
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--quant4", action="store_true", help="carrega o modelo base em 4 bits (GPU)")
    args = ap.parse_args()

    from peft import PeftModel

    tok, modelo = carregar_base(args.modelo, usar_gpu=True, quant4=args.quant4)
    modelo = PeftModel.from_pretrained(modelo, args.adaptador)
    if not args.quant4:  # mesclar num modelo em 4 bits perderia precisão
        modelo = modelo.merge_and_unload()
    inicio = time.time()
    saida = gerar(tok, modelo, [montar_prompt(tok, PROMPT_CURTO, args.mensagem)])[0]
    pedido = interpretar(saida)
    print(json.dumps(pedido, ensure_ascii=False, indent=2) if pedido else f"saída inválida: {saida}")
    print(f"({time.time() - inicio:.1f}s)")


if __name__ == "__main__":
    main()
