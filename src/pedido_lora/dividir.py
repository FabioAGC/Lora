"""Junta os arquivos verificados, remove duplicatas e separa treino/validação/teste.

A divisão é estratificada pelo tipo de caso (normal ou cada caso de borda), para que
o teste tenha a mesma proporção de casos difíceis que o treino.

Uso:
    python -m pedido_lora.dividir data/v1_verificados.jsonl data/bruto_v2.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .dados import ler_jsonl

N_TESTE, N_VAL = 400, 150


def main() -> None:
    vistos, grupos = set(), defaultdict(list)
    for caminho in sys.argv[1:]:
        for ex in ler_jsonl(caminho):
            chave = ex["mensagem"].lower()
            if chave in vistos:  # mensagens repetidas vazariam do treino para o teste
                continue
            vistos.add(chave)
            ex.pop("motivo", None)
            ex["complicacao"] = ex.get("complicacao")
            grupos[ex["complicacao"] or "normal"].append(ex)

    total = sum(len(g) for g in grupos.values())
    rng = random.Random(0)
    partes = {"teste": [], "validacao": [], "treino": []}
    for nome in sorted(grupos):
        grupo = grupos[nome]
        rng.shuffle(grupo)
        n_teste = round(len(grupo) * N_TESTE / total)
        n_val = round(len(grupo) * N_VAL / total)
        partes["teste"] += grupo[:n_teste]
        partes["validacao"] += grupo[n_teste : n_teste + n_val]
        partes["treino"] += grupo[n_teste + n_val :]

    for nome, parte in partes.items():
        rng.shuffle(parte)
        with Path(f"data/{nome}.jsonl").open("w", encoding="utf-8") as f:
            for ex in parte:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        bordas = sum(1 for ex in parte if ex["complicacao"])
        print(f"{nome}: {len(parte)} exemplos, {bordas} casos de borda")


if __name__ == "__main__":
    main()
