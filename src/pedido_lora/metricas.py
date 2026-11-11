"""Métricas de extração. Funções puras, sem dependência de modelo."""
from __future__ import annotations

from collections import Counter
from typing import Any

CAMPOS_SIMPLES = ("pagamento", "tipo", "orcamento", "troco_para")


def _chaves(pedido: dict[str, Any]) -> Counter:
    """Cada item vira (nome, quantidade, observações); a ordem não importa."""
    return Counter((i["item"], i["qtd"], tuple(sorted(i["obs"]))) for i in pedido["itens"])


def comparar(previsto: dict[str, Any] | None, esperado: dict[str, Any]) -> dict[str, Any]:
    """Compara uma previsão com o gabarito. `previsto` é None se o JSON foi inválido."""
    ouro = _chaves(esperado)
    if previsto is None:
        return {"valido": False, "exato": False, "tp": 0, "fp": 0, "fn": sum(ouro.values()),
                **{c: False for c in CAMPOS_SIMPLES}}
    pred = _chaves(previsto)
    tp = sum((pred & ouro).values())
    campos = {c: previsto.get(c) == esperado.get(c) for c in CAMPOS_SIMPLES}
    return {
        "valido": True,
        "exato": pred == ouro and all(campos.values()),
        "tp": tp,
        "fp": sum(pred.values()) - tp,
        "fn": sum(ouro.values()) - tp,
        **campos,
    }


def agregar(comparacoes: list[dict[str, Any]]) -> dict[str, float]:
    n = len(comparacoes)
    tp, fp, fn = (sum(c[k] for c in comparacoes) for k in ("tp", "fp", "fn"))
    precisao = tp / (tp + fp) if tp + fp else 0.0
    revocacao = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    taxa = lambda k: round(100 * sum(c[k] for c in comparacoes) / n, 1)  # noqa: E731
    return {
        "n": n,
        "json_valido": taxa("valido"),
        "pedido_exato": taxa("exato"),
        "itens_f1": round(100 * f1, 1),
        **{f"acc_{c}": taxa(c) for c in CAMPOS_SIMPLES},
    }
