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

from .amostrador import sortear_complicacao, sortear_estilo, sortear_pedido
from .esquema import serializar
from .prompts import PROMPT_GERADOR
from .verificar import verificar

URL = "https://api.groq.com/openai/v1/chat/completions"


def descrever(pedido: dict, estilo: str, complicacao: str | None = None) -> str:
    """Forma compacta do pedido para o prompt (gasta menos tokens que JSON)."""
    itens = "; ".join(
        f"{i['qtd']}x {i['item']}" + (f" ({', '.join(i['obs'])})" if i["obs"] else "")
        for i in pedido["itens"]
    )
    extras = "".join(f" | {c}: {pedido[c]}" for c in ("pagamento", "tipo", "orcamento", "troco_para")
                     if pedido[c] is not None)
    texto = f"{itens}{extras} | estilo: {estilo}"
    return texto + (f" | complicação: {complicacao}" if complicacao else "")


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


def sortear_exemplo(rng: random.Random, fracao_borda: float) -> tuple[dict, str, tuple[str, str] | None]:
    """Sorteia pedido, estilo e, com probabilidade `fracao_borda`, um caso de borda."""
    if rng.random() >= fracao_borda:
        return sortear_pedido(rng), sortear_estilo(rng), None
    grande = rng.random() < 1 / 8
    pedido = sortear_pedido(rng, grande=grande)
    complicacao = ("grande", "") if grande else sortear_complicacao(rng, pedido)
    return pedido, sortear_estilo(rng), complicacao


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1500, help="exemplos aprovados desejados no arquivo")
    ap.add_argument("--modelo", default="openai/gpt-oss-120b")
    ap.add_argument("--lote", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--raciocinio", default="low")
    ap.add_argument("--borda", type=float, default=0.5, help="fração de casos de borda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", type=Path, default=Path("data/bruto.jsonl"))
    ap.add_argument("--rejeitados", type=Path, default=Path("data/rejeitados.jsonl"))
    ap.add_argument("--sem-verificar", action="store_true")
    args = ap.parse_args()

    feitos = sum(1 for _ in args.saida.open()) if args.saida.exists() else 0
    # Ao retomar, usa outra semente para não repetir os pedidos já sorteados.
    rng = random.Random(args.seed * 1_000_003 + feitos)
    cabecalho = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    aprovados = rejeitados = 0
    with httpx.Client(headers=cabecalho, timeout=180) as cliente, \
            args.saida.open("a") as saida, args.rejeitados.open("a") as ruins:
        while feitos < args.total:
            sorteados = [sortear_exemplo(rng, args.borda) for _ in range(args.lote)]
            lote = [descrever(p, e, c[1] if c and c[1] else None) for p, e, c in sorteados]
            mensagens = escrever_mensagens(cliente, args.modelo, lote, args.max_tokens, args.raciocinio)
            if mensagens is None:
                continue
            exemplos = [
                {"mensagem": msg.replace("\u2011", "-").strip(), "pedido": serializar(p),
                 "estilo": e, "complicacao": c[0] if c else None, "gerador": args.modelo}
                for (p, e, c), msg in zip(sorteados, mensagens)
            ]
            veredictos = [(True, "")] * len(exemplos) if args.sem_verificar else verificar(cliente, exemplos)
            for ex, (passou, motivo) in zip(exemplos, veredictos):
                if passou and feitos < args.total:
                    saida.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    feitos += 1
                    aprovados += 1
                elif not passou:
                    ruins.write(json.dumps({**ex, "motivo": motivo}, ensure_ascii=False) + "\n")
                    rejeitados += 1
            saida.flush(); ruins.flush()
            print(f"{feitos}/{args.total} (rejeitados até agora: {rejeitados})", flush=True)
    total = aprovados + rejeitados
    if total:
        print(f"taxa de rejeição: {100 * rejeitados / total:.1f}%")


if __name__ == "__main__":
    main()
