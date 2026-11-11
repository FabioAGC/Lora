"""Fine-tuning com LoRA em CPU.

Uso:
    python -m pedido_lora.treinar --saida outputs/ing_a --lr 3e-4 --seed 1

O laço de treino é escrito à mão (sem Trainer) para deixar cada passo visível:
tokenizar, mascarar o prompt, calcular a perda, atualizar só as matrizes LoRA.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from .modelo import MODELO_BASE, carregar_base, ler_jsonl, montar_exemplo_treino
from .prompts import PROMPT_CURTO

IGNORAR = -100  # rótulo que a função de perda ignora


def tokenizar(tok, exemplos: list[dict], max_len: int = 320) -> list[dict]:
    """Transforma cada exemplo em ids + rótulos, com o prompt mascarado.

    O modelo só é cobrado pelo JSON da resposta. Sem a máscara, ele gastaria
    capacidade aprendendo a prever a mensagem do cliente, o que não interessa.
    """
    dados = []
    for ex in exemplos:
        prompt, resposta = montar_exemplo_treino(tok, PROMPT_CURTO, ex["mensagem"], ex["pedido"])
        ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
        ids_r = tok(resposta, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_r)[:max_len]
        rotulos = ([IGNORAR] * len(ids_p) + ids_r)[:max_len]
        dados.append({"ids": ids, "rotulos": rotulos})
    return dados


def empacotar(lote: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Junta exemplos de tamanhos diferentes em um tensor, com padding à direita."""
    maior = max(len(x["ids"]) for x in lote)
    ids = torch.full((len(lote), maior), pad_id)
    rotulos = torch.full((len(lote), maior), IGNORAR)
    mascara = torch.zeros((len(lote), maior), dtype=torch.long)
    for i, x in enumerate(lote):
        n = len(x["ids"])
        ids[i, :n] = torch.tensor(x["ids"])
        rotulos[i, :n] = torch.tensor(x["rotulos"])
        mascara[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mascara, "labels": rotulos}


@torch.no_grad()
def perda_media(modelo, dados: list[dict], pad_id: int, lote: int = 16) -> float:
    """Perda média por token de resposta no conjunto dado."""
    modelo.eval()
    dados = sorted(dados, key=lambda x: len(x["ids"]))
    soma, tokens = 0.0, 0
    for ini in range(0, len(dados), lote):
        pacote = empacotar(dados[ini : ini + lote], pad_id)
        n = int((pacote["labels"] != IGNORAR).sum())
        soma += modelo(**pacote).loss.item() * n
        tokens += n
    return soma / tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO_BASE)
    ap.add_argument("--treino", default="data/treino.jsonl")
    ap.add_argument("--val", default="data/validacao.jsonl")
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--max-exemplos", type=int, default=0, help="0 = usa tudo")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok, base = carregar_base(args.modelo)

    # LoRA: congela o modelo e treina duas matrizes pequenas (A e B) por camada.
    config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    modelo = get_peft_model(base, config)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"parâmetros treináveis: {treinaveis:,} de {total:,} ({100 * treinaveis / total:.2f}%)")

    treino = ler_jsonl(args.treino)
    if args.max_exemplos:
        treino = treino[: args.max_exemplos]
    treino, val = tokenizar(tok, treino), tokenizar(tok, ler_jsonl(args.val))

    passos_por_epoca = math.ceil(len(treino) / args.lote)
    total_passos = int(passos_por_epoca * args.epocas)
    aquecimento = max(1, int(0.06 * total_passos))
    otimizador = torch.optim.AdamW((p for p in modelo.parameters() if p.requires_grad),
                                   lr=args.lr, weight_decay=0.0)
    # Taxa de aprendizado: sobe linearmente no aquecimento e depois cai até zero.
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otimizador,
        lambda p: (p + 1) / aquecimento if p < aquecimento
        else max(0.0, (total_passos - p) / max(1, total_passos - aquecimento)),
    )

    inicio, passo, historico = time.time(), 0, []
    while passo < total_passos:
        random.shuffle(treino)
        for ini in range(0, len(treino), args.lote):
            if passo >= total_passos:
                break
            modelo.train()
            perda = modelo(**empacotar(treino[ini : ini + args.lote], tok.pad_token_id)).loss
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            otimizador.zero_grad()
            passo += 1
            if passo % 20 == 0 or passo == total_passos:
                historico.append({"passo": passo, "perda": round(perda.item(), 4)})
                print(f"passo {passo}/{total_passos} perda {perda.item():.4f} "
                      f"({time.time() - inicio:.0f}s)", flush=True)

    perda_val = perda_media(modelo, val, tok.pad_token_id)
    print(f"perda de validação: {perda_val:.4f}")
    args.saida.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(args.saida)
    meta = {"modelo_base": args.modelo, "lr": args.lr, "seed": args.seed, "epocas": args.epocas,
            "rank": args.rank, "dropout": args.dropout, "exemplos_treino": len(treino),
            "perda_validacao": round(perda_val, 4), "minutos_treino": round((time.time() - inicio) / 60, 1),
            "historico": historico}
    (args.saida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
