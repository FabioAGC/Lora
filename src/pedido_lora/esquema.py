"""Formato canônico do pedido e validação da saída do modelo.

Exemplo:
{"itens":[{"item":"x-burguer","qtd":2,"obs":["sem cebola"]}],
 "pagamento":"pix","tipo":"entrega","orcamento":60,"troco_para":null}
"""
from __future__ import annotations

import json
from typing import Any

from .catalogo import PAGAMENTOS, TIPOS

CAMPOS = ("itens", "pagamento", "tipo", "orcamento", "troco_para")


def serializar(pedido: dict[str, Any]) -> str:
    """JSON compacto, com chaves em ordem fixa e observações ordenadas.

    Uma saída determinística importa no treino: se o mesmo pedido pudesse ser
    escrito de várias formas, o modelo seria penalizado por respostas corretas.
    """
    itens = [
        {"item": i["item"], "qtd": int(i["qtd"]), "obs": sorted(i.get("obs", []))}
        for i in pedido["itens"]
    ]
    canonico = {
        "itens": itens,
        "pagamento": pedido.get("pagamento"),
        "tipo": pedido.get("tipo"),
        "orcamento": pedido.get("orcamento"),
        "troco_para": pedido.get("troco_para"),
    }
    return json.dumps(canonico, ensure_ascii=False, separators=(",", ":"))


def _inteiro_ou_none(v: Any) -> bool:
    return v is None or (isinstance(v, int) and not isinstance(v, bool))


def interpretar(texto: str) -> dict[str, Any] | None:
    """Converte a saída bruta do modelo em pedido. Devolve None se for inválida.

    Aceita texto antes ou depois do JSON (modelos sem fine-tuning costumam
    embrulhar a resposta em ```json ... ```), mas exige o esquema exato.
    """
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini == -1 or fim <= ini:
        return None
    try:
        d = json.loads(texto[ini : fim + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("itens"), list) or not d["itens"]:
        return None
    for i in d["itens"]:
        if not isinstance(i, dict) or not isinstance(i.get("item"), str):
            return None
        if not isinstance(i.get("qtd"), int) or isinstance(i["qtd"], bool):
            return None
        obs = i.get("obs", [])
        if not isinstance(obs, list) or not all(isinstance(o, str) for o in obs):
            return None
    if d.get("pagamento") not in (None, *PAGAMENTOS) or d.get("tipo") not in (None, *TIPOS):
        return None
    if not _inteiro_ou_none(d.get("orcamento")) or not _inteiro_ou_none(d.get("troco_para")):
        return None
    return json.loads(serializar(d))
