"""Avalia um LLM grande (via Groq) no conjunto de teste, como referência.

O modelo grande não foi ajustado: recebe o esquema e o catálogo no prompt.

Uso:
    export GROQ_API_KEY=...
    python -m pedido_lora.avaliar_api --modelo openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

from .dados import ler_jsonl
from .esquema import interpretar
from .gerar_dados import URL, _espera
from .metricas import agregar, comparar
from .prompts import prompt_longo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="openai/gpt-oss-120b")
    ap.add_argument("--nome", default="llm_grande_api")
    ap.add_argument("--teste", default="data/teste.jsonl")
    ap.add_argument("--limite", type=int, default=0)
    args = ap.parse_args()

    exemplos = ler_jsonl(args.teste)
    if args.limite:
        exemplos = exemplos[: args.limite]
    sistema = prompt_longo()
    comparacoes, latencias, tokens_prompt, tokens_saida = [], [], 0, 0
    cabecalho = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    with httpx.Client(headers=cabecalho, timeout=120) as cliente:
        for n, ex in enumerate(exemplos, 1):
            corpo = {"model": args.modelo, "temperature": 0, "reasoning_effort": "low",
                     "messages": [{"role": "system", "content": sistema},
                                  {"role": "user", "content": ex["mensagem"]}]}
            texto = ""
            for _ in range(8):
                inicio = time.time()
                resp = cliente.post(URL, json=corpo)
                if resp.status_code == 429:
                    time.sleep(_espera(resp))
                    continue
                if resp.status_code == 200:
                    dados = resp.json()
                    texto = dados["choices"][0]["message"]["content"] or ""
                    latencias.append(time.time() - inicio)
                    tokens_prompt += dados["usage"]["prompt_tokens"]
                    tokens_saida += dados["usage"]["completion_tokens"]
                    break
                time.sleep(3)
            comparacoes.append(comparar(interpretar(texto), json.loads(ex["pedido"])))
            if n % 20 == 0:
                print(n, agregar(comparacoes)["pedido_exato"], flush=True)

    resultado = {
        "nome": args.nome, "modelo": args.modelo, "adaptador": None, "prompt": "longo",
        **agregar(comparacoes),
        "tokens_prompt_medio": round(tokens_prompt / len(exemplos)),
        "tokens_saida_medio": round(tokens_saida / len(exemplos)),
        "segundos_por_pedido": round(sum(latencias) / max(1, len(latencias)), 2),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.nome}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
