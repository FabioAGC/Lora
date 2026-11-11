#!/usr/bin/env bash
# Treina os ingredientes da sopa: mesmo modelo base e mesmos dados, mudando
# a taxa de aprendizado e a semente (ordem dos dados e inicialização do LoRA).
# Cada um leva cerca de 30 minutos em 4 núcleos de CPU. Pode ser interrompido
# e retomado: ingredientes já treinados são pulados.
set -euo pipefail
export PYTHONPATH=src
treinar() {  # nome, lr, seed
  if [ -f "outputs/$1/meta.json" ]; then echo "$1 já treinado"; return; fi
  python -m pedido_lora.treinar --saida "outputs/$1" --lr "$2" --seed "$3" --epocas 1
}
treinar ing_a 2e-4 1
treinar ing_b 5e-4 2
treinar ing_c 1e-3 3
