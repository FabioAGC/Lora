"""Gera o dataset: sorteia pedidos e pede a um LLM grande que escreva as mensagens.

Uso:
    export GROQ_API_KEY=...
    python -m pedido_lora.gerar_dados --total 1500 --saida data/bruto.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

from .amostrador import sortear_estilo, sortear_pedido
from .esquema import serializar
from .prompts import PROMPT_GERADOR

URL = "https://api.groq.com/openai/v1/chat/completions"


def descrever(pedido: dict, estilo: str) -> str:
    """Forma compacta do pedido para o prompt (gasta menos tokens que JSON)."""
    itens = "; ".join(
        f"{i['qtd']}x {i['item']}" + (f" ({', '.join(i['obs'])})" if i["obs"] else "")
        for i in pedido["itens"]
    )
    extras = "".join(f" | {c}: {pedido[c]}" for c in ("pagamento", "tipo", "orcamento", "troco_para")
                     if pedido[c] is not None)
    return f"{itens}{extras} | estilo: {estilo}"


def _espera(resp: httpx.Response) -> float:
    """Lê quanto tempo a API pede para esperar depois de um 429."""
    if "retry-after" in resp.headers:
        return float(resp.headers["retry-after"]) + 0.5
    achado = re.search(r"try again in ([\d.]+)s", resp.text)
    return float(achado.group(1)) + 0.5 if achado else 10.0


def escrever_mensagens(cliente: httpx.Client, modelo: str, lote: list[str],
                       max_tokens: int, raciocinio: str) -> list[str] | None:
    """Pede as mensagens de um lote. Devolve None se a resposta vier malformada."""
    corpo = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": PROMPT_GERADOR},
            {"role": "user", "content": "\n".join(f"{n}. {d}" for n, d in enumerate(lote, 1))},
        ],
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "reasoning_effort": raciocinio,
        "max_completion_tokens": max_tokens,
    }
    for _ in range(6):
        resp = cliente.post(URL, json=corpo)
        if resp.status_code == 429:
            if "per day" in resp.text or "too large" in resp.text:
                raise RuntimeError(f"limite atingido em {modelo}: {resp.text[:200]}")
            time.sleep(_espera(resp))
            continue
        if resp.status_code >= 500:
            time.sleep(5)
            continue
        if resp.status_code != 200:
            return None
        try:
            mensagens = json.loads(resp.json()["choices"][0]["message"]["content"])["mensagens"]
        except (KeyError, json.JSONDecodeError, TypeError):
            return None
        ok = isinstance(mensagens, list) and len(mensagens) == len(lote)
        return mensagens if ok and all(isinstance(m, str) and m.strip() for m in mensagens) else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1500)
    ap.add_argument("--modelo", default="openai/gpt-oss-120b")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--raciocinio", default="low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", type=Path, default=Path("data/bruto.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    feitos = sum(1 for _ in args.saida.open()) if args.saida.exists() else 0
    # Avança o gerador aleatório para que retomar não repita pedidos já gerados.
    for _ in range(feitos):
        sortear_pedido(rng), sortear_estilo(rng)

    cabecalho = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    with httpx.Client(headers=cabecalho, timeout=120) as cliente, args.saida.open("a") as saida:
        while feitos < args.total:
            tamanho = min(args.lote, args.total - feitos)
            estado = rng.getstate()
            pedidos = [(sortear_pedido(rng), sortear_estilo(rng)) for _ in range(tamanho)]
            lote = [descrever(p, e) for p, e in pedidos]
            mensagens = escrever_mensagens(cliente, args.modelo, lote, args.max_tokens, args.raciocinio)
            if mensagens is None:
                rng.setstate(estado)  # tenta de novo o mesmo lote
                continue
            for (pedido, estilo), msg in zip(pedidos, mensagens):
                linha = {"mensagem": msg.replace("\u2011", "-").strip(), "pedido": serializar(pedido),
                         "estilo": estilo, "gerador": args.modelo}
                saida.write(json.dumps(linha, ensure_ascii=False) + "\n")
            saida.flush()
            feitos += tamanho
            print(f"{feitos}/{args.total}", flush=True)


if __name__ == "__main__":
    main()
