"""Sorteia pedidos estruturados. Os rótulos nascem aqui, corretos por construção.

O LLM grande só escreve a mensagem informal que corresponde a cada pedido; ele
nunca inventa o rótulo. Isso evita o principal problema de datasets sintéticos:
rótulos errados gerados pelo próprio modelo.
"""
from __future__ import annotations

import random
from typing import Any

from .catalogo import PAGAMENTOS, TIPOS, TODOS_ITENS, obs_do_item

ESTILOS = [
    "mensagem curta de WhatsApp, tudo em minúsculas e sem pontuação",
    "informal com gírias e abreviações (vc, pfv, blz, qro)",
    "com alguns erros de digitação e sem acentos",
    "educado e completo, com cumprimento no começo",
    "transcrição de áudio, com hesitações (é..., deixa eu ver, ah e também)",
    "direto, em formato de lista separada por vírgulas",
    "apressado, frases cortadas",
    "simpático, com um ou dois emojis",
]


def _escolher(rng: random.Random, opcoes: list, pesos: list[float]):
    return rng.choices(opcoes, weights=pesos, k=1)[0]


def sortear_pedido(rng: random.Random) -> dict[str, Any]:
    n_itens = _escolher(rng, [1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])
    itens = []
    for nome in rng.sample(TODOS_ITENS, n_itens):
        disponiveis = obs_do_item(nome)
        n_obs = min(_escolher(rng, [0, 1, 2], [0.6, 0.3, 0.1]), len(disponiveis))
        obs = rng.sample(disponiveis, n_obs)
        # "bem passado" e "mal passado" juntos não fazem sentido
        if {"bem passado", "mal passado"} <= set(obs):
            obs.remove("mal passado")
        itens.append({
            "item": nome,
            "qtd": _escolher(rng, [1, 2, 3, 4, 5, 6], [0.58, 0.25, 0.1, 0.04, 0.02, 0.01]),
            "obs": sorted(obs),
        })
    pagamento = _escolher(rng, [None, *PAGAMENTOS], [0.35, 0.3, 0.2, 0.15])
    troco = None
    if pagamento == "dinheiro" and rng.random() < 0.5:
        troco = rng.choice([20, 50, 100, 200])
    return {
        "itens": itens,
        "pagamento": pagamento,
        "tipo": _escolher(rng, [None, *TIPOS], [0.4, 0.4, 0.2]),
        "orcamento": rng.choice([30, 40, 50, 60, 80, 100, 120, 150]) if rng.random() < 0.2 else None,
        "troco_para": troco,
    }


def sortear_estilo(rng: random.Random) -> str:
    return rng.choice(ESTILOS)
