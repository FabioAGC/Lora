"""Converte os dados para o formato de conversa (ChatML) usado pelo Soup CLI.

Uso:
    python -m pedido_lora.exportar_chat
"""
from __future__ import annotations

import json
from pathlib import Path

from .dados import ler_jsonl
from .prompts import PROMPT_CURTO


def main() -> None:
    saida = Path("data/chat")
    saida.mkdir(exist_ok=True)
    for parte in ("treino", "validacao"):
        with (saida / f"{parte}.jsonl").open("w", encoding="utf-8") as f:
            for ex in ler_jsonl(f"data/{parte}.jsonl"):
                conversa = {"messages": [
                    {"role": "system", "content": PROMPT_CURTO},
                    {"role": "user", "content": ex["mensagem"]},
                    {"role": "assistant", "content": ex["pedido"]},
                ]}
                f.write(json.dumps(conversa, ensure_ascii=False) + "\n")
        print(parte, "->", saida / f"{parte}.jsonl")


if __name__ == "__main__":
    main()
