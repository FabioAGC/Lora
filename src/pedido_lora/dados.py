"""Leitura dos arquivos de dados (sem dependências pesadas)."""
from __future__ import annotations

import json
from pathlib import Path


def ler_jsonl(caminho: str | Path) -> list[dict]:
    return [json.loads(linha) for linha in Path(caminho).open(encoding="utf-8")]
