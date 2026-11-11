"""Carregamento do modelo base e montagem dos prompts no formato de chat."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dados import ler_jsonl  # noqa: F401  (reexportado)

MODELO_BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def carregar_base(nome: str = MODELO_BASE):
    tok = AutoTokenizer.from_pretrained(nome)
    tok.padding_side = "left"  # necessário para gerar em lote com modelos causais
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.float32)
    return tok, modelo


def montar_prompt(tok, sistema: str, mensagem: str) -> str:
    conversa = [{"role": "system", "content": sistema}, {"role": "user", "content": mensagem}]
    return tok.apply_chat_template(conversa, tokenize=False, add_generation_prompt=True)


def montar_exemplo_treino(tok, sistema: str, mensagem: str, resposta: str) -> tuple[str, str]:
    """Devolve (prompt, continuação). A perda é calculada só sobre a continuação."""
    prompt = montar_prompt(tok, sistema, mensagem)
    conversa = [
        {"role": "system", "content": sistema},
        {"role": "user", "content": mensagem},
        {"role": "assistant", "content": resposta},
    ]
    completo = tok.apply_chat_template(conversa, tokenize=False).rstrip("\n")
    assert completo.startswith(prompt), "template de chat inesperado"
    return prompt, completo[len(prompt):]


@torch.no_grad()
def gerar(tok, modelo, prompts: list[str], max_tokens: int = 200, lote: int = 8) -> list[str]:
    """Geração gulosa em lote. Ordena por tamanho para desperdiçar menos padding."""
    modelo.eval()
    ordem = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    saidas = [""] * len(prompts)
    for ini in range(0, len(ordem), lote):
        idx = ordem[ini : ini + lote]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True)
        out = modelo.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id)
        novos = out[:, enc["input_ids"].shape[1]:]
        for i, texto in zip(idx, tok.batch_decode(novos, skip_special_tokens=True)):
            saidas[i] = texto.strip()
    return saidas
