"""Confere se cada mensagem leva ao pedido do rótulo, e descarta as que não levam.

Duas camadas:
1. Regras de palavras-chave, baratas e determinísticas, para os erros mais comuns do
   gerador: citar um pagamento ou um tipo de entrega que não está no pedido, ou
   esquecer um que está.
2. Um LLM revisor, que recebe a mensagem E o rótulo e responde se batem. Como ele vê
   o rótulo, não é uma segunda extração: é uma checagem de consistência. Isso evita
   descartar justamente os casos difíceis em que um extrator erraria.

Uso, para filtrar um arquivo existente:
    python -m pedido_lora.verificar data/treino.jsonl --saida data/treino_ok.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import httpx

from .dados import ler_jsonl
from .prompts import PROMPT_VERIFICADOR

URL = "https://api.groq.com/openai/v1/chat/completions"
# Medido em 31 erros conhecidos do conjunto de teste v1 (12 com rótulo errado):
#   só regras: 7/12 detectados, 0/19 rótulos bons rejeitados
#   gpt-oss-120b, raciocínio low: 12/12 e 0/19, mas disputa cota com o gerador
#   qwen3.8-27b, sem raciocínio, listando só os rejeitados: 12/12 e 0/19,
#   rápido e com cota própria (o gerador usa o gpt-oss-120b), mas limite de
#   200 mil tokens por dia na conta gratuita
#   gpt-oss-120b, raciocínio low, listando só os rejeitados: 12/12 e 0/19
#   gpt-oss-20b, raciocínio low: 10/12 e 1/19 (descartado)
# Para trocar: MODELO_VERIFICADOR=openai/gpt-oss-120b ESFORCO_VERIFICADOR=low
MODELO_VERIFICADOR = os.environ.get("MODELO_VERIFICADOR", "qwen/qwen3.8-27b")
ESFORCO_VERIFICADOR = os.environ.get("ESFORCO_VERIFICADOR", "none")

PALAVRAS_PAGAMENTO = {
    "pix": r"\bpix\b",
    "cartao": r"cartao|credito|debito|maquininha",
    "dinheiro": r"dinheiro|especie|troco|\bnota de\b|cash",
}
PALAVRAS_TIPO = {
    "entrega": r"entreg|delivery|\bmanda\b|\bmandar\b|\btraz\b|\btrazer\b|\benvi",
    "retirada": r"retir|\bbusc|\bpego\b|\bpegar\b|balcao|\bpasso ai\b|\bpasso la\b",
}


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


# Trechos da instrução de estilo que o gerador às vezes copia para dentro da mensagem.
VAZAMENTO_DE_INSTRUCAO = r"sem acentos? e (com )?erros|erros? de digitacao|\bestilo\b|transcricao de audio|por extenso"


def problemas_por_regra(mensagem: str, pedido: dict, complicacao: str | None = None) -> list[str]:
    """Devolve a lista de inconsistências óbvias entre mensagem e rótulo."""
    msg = _normalizar(mensagem)
    achados = []
    if re.search(VAZAMENTO_DE_INSTRUCAO, msg):
        achados.append("a mensagem copia a instrução de estilo")
    for valor, padrao in PALAVRAS_PAGAMENTO.items():
        presente = re.search(padrao, msg) is not None
        if pedido["pagamento"] == valor and not presente:
            achados.append(f"pagamento {valor} não aparece na mensagem")
    if pedido["pagamento"] is None and any(
        re.search(p, msg) for v, p in PALAVRAS_PAGAMENTO.items() if v != "dinheiro"
    ):
        achados.append("a mensagem cita pagamento, mas o rótulo não tem")
    if complicacao != "pergunta_entrega":
        for valor, padrao in PALAVRAS_TIPO.items():
            if pedido["tipo"] == valor and not re.search(padrao, msg):
                achados.append(f"tipo {valor} não aparece na mensagem")
        if pedido["tipo"] is None and re.search(PALAVRAS_TIPO["retirada"], msg):
            achados.append("a mensagem cita retirada, mas o rótulo não tem tipo")
    return achados


def descrever_para_revisao(pedido: dict) -> str:
    itens = "; ".join(
        f"{i['qtd']}x {i['item']}" + (f" ({', '.join(i['obs'])})" if i["obs"] else "")
        for i in pedido["itens"]
    )
    campos = " | ".join(
        f"{c}: {pedido[c] if pedido[c] is not None else 'não informado'}"
        for c in ("pagamento", "tipo", "orcamento", "troco_para")
    )
    return f"{itens} | {campos}"


def _espera(resp: httpx.Response) -> float:
    if "retry-after" in resp.headers:
        return float(resp.headers["retry-after"]) + 0.5
    achado = re.search(r"try again in ([\d.]+)s", resp.text)
    return float(achado.group(1)) + 0.5 if achado else 10.0


def revisar_com_llm(cliente: httpx.Client, pares: list[tuple[str, dict]]) -> list[tuple[bool, str]] | None:
    """Pergunta ao LLM revisor se cada (mensagem, pedido) bate. None se a resposta vier malformada."""
    linhas = "\n".join(
        f"{n}. MENSAGEM: {msg!r}\n   PEDIDO FINAL: {descrever_para_revisao(ped)}"
        for n, (msg, ped) in enumerate(pares, 1)
    )
    corpo = {
        "model": MODELO_VERIFICADOR,
        "messages": [{"role": "system", "content": PROMPT_VERIFICADOR},
                     {"role": "user", "content": linhas}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "reasoning_effort": ESFORCO_VERIFICADOR,
        "max_completion_tokens": 900,
    }
    for _ in range(8):
        resp = cliente.post(URL, json=corpo)
        if resp.status_code == 429:
            if "per day" in resp.text:
                raise RuntimeError(f"limite diário atingido: {resp.text[:200]}")
            time.sleep(_espera(resp))
            continue
        if resp.status_code != 200:
            time.sleep(5)
            continue
        try:
            dados = json.loads(resp.json()["choices"][0]["message"]["content"])["rejeitados"]
            ruins = {int(r["n"]): str(r.get("motivo", "")) for r in dados}
            if not set(ruins) <= set(range(1, len(pares) + 1)):
                return None
            return [(n not in ruins, ruins.get(n, "")) for n in range(1, len(pares) + 1)]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
    return None


def verificar(cliente: httpx.Client, exemplos: list[dict]) -> list[tuple[bool, str]]:
    """Aplica as duas camadas. Um exemplo passa só se passar nas duas."""
    pares = [(ex["mensagem"], json.loads(ex["pedido"])) for ex in exemplos]
    resultado = None
    for _ in range(3):
        resultado = revisar_com_llm(cliente, pares)
        if resultado is not None:
            break
    if resultado is None:
        return [(False, "revisor não respondeu")] * len(exemplos)
    finais = []
    for ex, (msg, ped), (ok_llm, motivo) in zip(exemplos, pares, resultado):
        regras = problemas_por_regra(msg, ped, ex.get("complicacao"))
        if regras:
            finais.append((False, "regra: " + "; ".join(regras)))
        elif not ok_llm:
            finais.append((False, "revisor: " + motivo))
        else:
            finais.append((True, ""))
    return finais


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", type=Path)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--rejeitados", type=Path, default=Path("data/rejeitados.jsonl"))
    ap.add_argument("--lote", type=int, default=20)
    args = ap.parse_args()

    exemplos = ler_jsonl(args.entrada)
    feitos = 0
    if args.saida.exists() and args.rejeitados.exists():  # retoma de onde parou
        vistos = {json.loads(l)["mensagem"] for arq in (args.saida, args.rejeitados) for l in arq.open()}
        exemplos = [ex for ex in exemplos if ex["mensagem"] not in vistos]
    cabecalho = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    with httpx.Client(headers=cabecalho, timeout=180) as cliente, \
            args.saida.open("a") as ok, args.rejeitados.open("a") as ruins:
        for ini in range(0, len(exemplos), args.lote):
            lote = exemplos[ini : ini + args.lote]
            for ex, (passou, motivo) in zip(lote, verificar(cliente, lote)):
                if passou:
                    ok.write(json.dumps(ex, ensure_ascii=False) + "\n")
                else:
                    ruins.write(json.dumps({**ex, "motivo": motivo}, ensure_ascii=False) + "\n")
            ok.flush(); ruins.flush()
            feitos += len(lote)
            print(f"{feitos}/{len(exemplos)}", flush=True)


if __name__ == "__main__":
    main()
