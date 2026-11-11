"""Treina e avalia vários tamanhos de modelo em sequência, numa GPU T4 do Kaggle.

Para cada modelo: escreve a configuração do Soup CLI, treina, avalia no conjunto de
teste e salva o resultado. Modelos já avaliados são pulados, então dá para rodar de
novo depois de uma interrupção.

Uso:
    python -m pedido_lora.kaggle                       # todos os tamanhos
    python -m pedido_lora.kaggle --modelos 0.5b 7b     # só alguns
    python -m pedido_lora.kaggle --teste-rapido        # confere o encanamento em CPU

Os tempos estimados são para uma T4 e não foram medidos: servem para planejar a cota.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Receita:
    nome: str
    base: str
    lr: float
    rank: int
    lote: int
    acumulacao: int
    quant4: bool
    minutos_estimados: int


# Taxas de aprendizado menores para modelos maiores, como é comum em LoRA.
# O 7B só cabe nos 16 GB da T4 quantizado em 4 bits (QLoRA).
RECEITAS = {
    "0.5b": Receita("qwen_0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 1e-3, 8, 8, 1, False, 10),
    "1.5b": Receita("qwen_1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 5e-4, 16, 8, 1, False, 25),
    "3b": Receita("qwen_3b", "Qwen/Qwen2.5-3B-Instruct", 3e-4, 16, 4, 2, False, 50),
    "7b": Receita("qwen_7b_qlora", "Qwen/Qwen2.5-7B-Instruct", 2e-4, 16, 4, 2, True, 150),
}


def escrever_config(r: Receita, epocas: int, saida: Path) -> Path:
    config = f"""# Gerado por pedido_lora.kaggle para {r.base}
base: {r.base}
task: sft

data:
  train: ./data/chat/treino.jsonl
  format: chatml
  val_split: 0.0
  max_length: 512

training:
  epochs: {epocas}
  lr: {r.lr}
  batch_size: {r.lote}
  gradient_accumulation_steps: {r.acumulacao}
  warmup_ratio: 0.06
  scheduler: linear
  seed: 3
  quantization: {"4bit" if r.quant4 else "none"}
  lora:
    r: {r.rank}
    alpha: {2 * r.rank}
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj]

output: ./outputs/{r.nome}
"""
    caminho = saida / f"{r.nome}.yaml"
    caminho.write_text(config)
    return caminho


def rodar(comando: list[str]) -> None:
    print("$", " ".join(comando), flush=True)
    subprocess.run(comando, check=True)


def limpar_cache(base: str) -> None:
    """Apaga os pesos baixados do modelo, para não lotar o disco do Kaggle."""
    pasta = Path.home() / ".cache/huggingface/hub" / f"models--{base.replace('/', '--')}"
    shutil.rmtree(pasta, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", choices=list(RECEITAS), default=list(RECEITAS))
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--teste-rapido", action="store_true",
                    help="valida configuração e avaliação sem treinar (roda em CPU)")
    ap.add_argument("--manter-cache", action="store_true")
    args = ap.parse_args()

    if not Path("data/chat/treino.jsonl").exists():
        raise SystemExit("O dataset ainda não foi montado (falta data/chat/treino.jsonl). "
                         "Rode ./scripts/montar_dataset_v2.sh antes de treinar.")

    receitas = [RECEITAS[m] for m in args.modelos]
    print("Plano (estimativa para uma T4):")
    for r in receitas:
        print(f"  {r.base}: cerca de {r.minutos_estimados} min de treino e avaliação")
    print(f"  total: cerca de {sum(r.minutos_estimados for r in receitas) / 60:.1f} h\n", flush=True)

    configs = Path("outputs/configs")
    configs.mkdir(parents=True, exist_ok=True)
    for r in receitas:
        if Path(f"results/{r.nome}.json").exists() and not args.teste_rapido:
            print(f"{r.nome}: já avaliado, pulando")
            continue
        config = escrever_config(r, args.epocas, configs)
        inicio = time.time()
        if args.teste_rapido:
            rodar(["soup", "train", "--config", str(config), "--dry-run"])
            continue
        rodar(["soup", "train", "--config", str(config), "--yes"])
        minutos_treino = (time.time() - inicio) / 60
        avaliar = [sys.executable, "-m", "pedido_lora.avaliar", "--nome", r.nome,
                   "--modelo", r.base, "--adaptador", f"outputs/{r.nome}"]
        rodar(avaliar + (["--quant4"] if r.quant4 else []))
        resultado = Path(f"results/{r.nome}.json")
        dados = json.loads(resultado.read_text())
        dados["minutos_treino"] = round(minutos_treino, 1)
        dados["receita"] = {"lr": r.lr, "rank": r.rank, "epocas": args.epocas, "quant4": r.quant4}
        resultado.write_text(json.dumps(dados, ensure_ascii=False, indent=2))
        print(f"{r.nome}: {dados['pedido_exato']}% de pedidos exatos, "
              f"{minutos_treino:.0f} min de treino\n", flush=True)
        if not args.manter_cache:
            limpar_cache(r.base)

    if not args.teste_rapido:
        rodar([sys.executable, "-m", "pedido_lora.tabela"])


if __name__ == "__main__":
    main()
