"""Junta os arquivos brutos, remove duplicatas e separa treino/validação/teste.

Uso:
    python -m pedido_lora.dividir data/bruto.jsonl data/bruto_qwen.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from .dados import ler_jsonl

N_TESTE, N_VAL = 200, 140


def main() -> None:
    vistos, exemplos = set(), []
    for caminho in sys.argv[1:]:
        for ex in ler_jsonl(caminho):
            chave = ex["mensagem"].lower()
            if chave not in vistos:  # mensagens repetidas vazariam do treino para o teste
                vistos.add(chave)
                exemplos.append(ex)
    random.Random(0).shuffle(exemplos)
    partes = {
        "teste": exemplos[:N_TESTE],
        "validacao": exemplos[N_TESTE : N_TESTE + N_VAL],
        "treino": exemplos[N_TESTE + N_VAL :],
    }
    for nome, parte in partes.items():
        with Path(f"data/{nome}.jsonl").open("w", encoding="utf-8") as f:
            for ex in parte:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(nome, len(parte))


if __name__ == "__main__":
    main()
