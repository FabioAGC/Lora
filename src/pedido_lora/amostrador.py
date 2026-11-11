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


def sortear_pedido(rng: random.Random, grande: bool = False) -> dict[str, Any]:
    """Sorteia um pedido. Com `grande=True`, sorteia 5 ou 6 itens ou uma quantidade alta."""
    n_itens = _escolher(rng, [1, 2, 3, 4], [0.3, 0.35, 0.25, 0.1])
    if grande:
        n_itens = rng.choice([1, 2, 5, 6])
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
    if grande and n_itens <= 2:
        itens[0]["qtd"] = rng.choice([10, 12, 15, 20, 25, 30])
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


# Casos de borda. Cada um muda a forma como o cliente escreve, mas o pedido final
# (o rótulo) continua sendo exatamente o sorteado. A instrução vai para o LLM que
# escreve a mensagem, e o verificador confere depois se ele seguiu.
COMPLICACOES = [
    "corrige_qtd", "desiste_item", "divide_item", "pergunta_entrega",
    "por_extenso", "troco_implicito", "papo", "grande",
]


def sortear_complicacao(rng: random.Random, pedido: dict[str, Any]) -> tuple[str, str] | None:
    """Escolhe um caso de borda compatível com o pedido.

    Devolve (nome, instrução para o gerador), ou None se nenhum couber.
    O caso "grande" é decidido no sorteio do pedido, não aqui.
    """
    itens = pedido["itens"]
    possiveis = ["corrige_qtd", "desiste_item", "por_extenso", "papo"]
    if any(i["qtd"] >= 2 for i in itens):
        possiveis.append("divide_item")
    if pedido["tipo"] is None:
        possiveis.append("pergunta_entrega")
    if pedido["pagamento"] == "dinheiro" and pedido["troco_para"]:
        possiveis.append("troco_implicito")
    nome = rng.choice(possiveis)

    if nome == "corrige_qtd":
        item = rng.choice(itens)
        errada = rng.choice([q for q in range(1, item["qtd"] + 3) if q != item["qtd"]])
        return nome, (f"o cliente primeiro pede {errada} {item['item']} e depois, na mesma mensagem, "
                      f"corrige para {item['qtd']}")
    if nome == "desiste_item":
        extra = rng.choice([n for n in TODOS_ITENS if n not in {i["item"] for i in itens}])
        return nome, f"o cliente também pede {extra} e depois desiste dele na mesma mensagem"
    if nome == "divide_item":
        item = rng.choice([i for i in itens if i["qtd"] >= 2])
        a = rng.randint(1, item["qtd"] - 1)
        return nome, (f"o cliente pede {a} {item['item']} no começo e mais {item['qtd'] - a} no fim "
                      f"da mensagem, como se tivesse lembrado depois")
    if nome == "pergunta_entrega":
        return nome, ("o cliente pergunta sobre prazo ou taxa de entrega, mas não diz se quer "
                      "entrega ou retirada")
    if nome == "por_extenso":
        return nome, "todas as quantidades e valores escritos por extenso, sem algarismos"
    if nome == "troco_implicito":
        return nome, ("o cliente fala do troco, mas não usa a palavra dinheiro nem diz a forma "
                      "de pagamento com outras palavras")
    return nome, ("a mensagem tem bastante conversa que não tem a ver com o pedido "
                  "(o dia, o trânsito, o jogo de ontem), com o pedido no meio")
