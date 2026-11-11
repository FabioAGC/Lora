"""Model soup de adaptadores LoRA.

Ideia (Wortsman et al., 2022, "Model soups"): em vez de escolher o melhor entre
vários modelos ajustados com hiperparâmetros diferentes, tira-se a MÉDIA dos
pesos deles. O resultado é um único modelo, com custo de inferência de um só,
que costuma ser tão bom quanto o melhor ingrediente, ou melhor.

Com LoRA, cada ingrediente i altera o peso W de uma camada em
    dW_i = (alpha_i / r_i) * B_i @ A_i
A sopa é   dW = soma_i  p_i * dW_i   (p_i = peso do ingrediente; 1/N na uniforme).

Atenção: tirar a média de A e de B separadamente NÃO dá esse resultado, porque
média(B) @ média(A) != média(B @ A). O jeito exato é empilhar as matrizes:
    A_sopa = [A_1; A_2; ...]                (concatena nas linhas)
    B_sopa = [c_1*B_1, c_2*B_2, ...]        (concatena nas colunas, c_i = p_i*alpha_i/r_i)
e então B_sopa @ A_sopa = soma_i c_i * B_i @ A_i. A sopa continua sendo um
adaptador LoRA pequeno, só que de rank r_1 + r_2 + ...

Uso:
    python -m pedido_lora.sopa --ingredientes outputs/ing_a outputs/ing_b outputs/ing_c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file


def carregar_adaptador(pasta: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    pasta = Path(pasta)
    pesos = load_file(pasta / "adapter_model.safetensors")
    config = json.loads((pasta / "adapter_config.json").read_text())
    return pesos, config


def misturar(adaptadores: list[tuple[dict[str, torch.Tensor], dict]],
             proporcoes: list[float] | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    """Combina adaptadores LoRA em um só, equivalente à média ponderada dos dW."""
    n = len(adaptadores)
    proporcoes = proporcoes or [1 / n] * n
    assert len(proporcoes) == n and abs(sum(proporcoes) - 1) < 1e-6
    chaves_a = [k for k in adaptadores[0][0] if "lora_A" in k]
    sopa: dict[str, torch.Tensor] = {}
    for chave_a in chaves_a:
        chave_b = chave_a.replace("lora_A", "lora_B")
        blocos_a, blocos_b = [], []
        for (pesos, config), p in zip(adaptadores, proporcoes):
            escala = config["lora_alpha"] / config["r"]
            blocos_a.append(pesos[chave_a])
            blocos_b.append(p * escala * pesos[chave_b])
        sopa[chave_a] = torch.cat(blocos_a, dim=0)
        sopa[chave_b] = torch.cat(blocos_b, dim=1)
    rank_total = sum(config["r"] for _, config in adaptadores)
    # lora_alpha = r faz a escala do adaptador final ser 1: ela já foi embutida em B.
    config = {**adaptadores[0][1], "r": rank_total, "lora_alpha": rank_total, "lora_dropout": 0.0}
    return sopa, config


def salvar_adaptador(pesos: dict[str, torch.Tensor], config: dict, pasta: str | Path) -> None:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in pesos.items()}, pasta / "adapter_model.safetensors")
    (pasta / "adapter_config.json").write_text(json.dumps(config, indent=2))


def sopa_gulosa(nomes: list[str], notas: dict[str, float],
                avaliar: Callable[[list[str]], float]) -> tuple[list[str], list[dict]]:
    """Sopa gulosa: testa os ingredientes do melhor para o pior e só mantém
    os que não pioram a nota de validação (aqui, perda: menor é melhor).

    `avaliar(lista_de_nomes)` devolve a perda de validação da sopa uniforme
    daqueles ingredientes.
    """
    ordenados = sorted(nomes, key=lambda n: notas[n])
    escolhidos, melhor = [ordenados[0]], notas[ordenados[0]]
    registro = [{"ingredientes": list(escolhidos), "perda": melhor, "aceito": True}]
    for candidato in ordenados[1:]:
        perda = avaliar(escolhidos + [candidato])
        aceito = perda <= melhor
        registro.append({"ingredientes": escolhidos + [candidato], "perda": perda, "aceito": aceito})
        if aceito:
            escolhidos, melhor = escolhidos + [candidato], perda
    return escolhidos, registro


def main() -> None:
    from peft import PeftModel

    from .modelo import MODELO_BASE, carregar_base, ler_jsonl
    from .treinar import perda_media, tokenizar

    ap = argparse.ArgumentParser()
    ap.add_argument("--ingredientes", nargs="+", required=True)
    ap.add_argument("--val", default="data/validacao.jsonl")
    ap.add_argument("--saida", type=Path, default=Path("outputs"))
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--registro", type=Path, default=Path("results/sopa.json"))
    args = ap.parse_args()

    tok, base = carregar_base(args.modelo)
    val = tokenizar(tok, ler_jsonl(args.val))
    adaptadores = {nome: carregar_adaptador(nome) for nome in args.ingredientes}
    tmp = args.saida / "_sopa_tmp"

    def perda_da_sopa(nomes: list[str]) -> float:
        salvar_adaptador(*misturar([adaptadores[n] for n in nomes]), tmp)
        modelo = PeftModel.from_pretrained(base, tmp)
        perda = perda_media(modelo, val, tok.pad_token_id)
        modelo.unload()  # devolve o modelo base limpo para a próxima avaliação
        return round(perda, 4)

    notas = {n: perda_da_sopa([n]) for n in args.ingredientes}
    print("perda de validação por ingrediente:", notas)

    salvar_adaptador(*misturar(list(adaptadores.values())), args.saida / "sopa_uniforme")
    perda_uniforme = perda_da_sopa(args.ingredientes)
    print("sopa uniforme:", perda_uniforme)

    escolhidos, registro = sopa_gulosa(args.ingredientes, notas, perda_da_sopa)
    salvar_adaptador(*misturar([adaptadores[n] for n in escolhidos]), args.saida / "sopa_gulosa")
    print("sopa gulosa:", escolhidos)

    args.registro.parent.mkdir(parents=True, exist_ok=True)
    args.registro.write_text(json.dumps({
        "perda_validacao_ingredientes": notas,
        "perda_validacao_sopa_uniforme": perda_uniforme,
        "sopa_gulosa": {"escolhidos": escolhidos, "passos": registro},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
