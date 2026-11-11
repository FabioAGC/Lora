#!/usr/bin/env bash
# Monta as sopas e avalia tudo no conjunto de teste. Pula o que já foi avaliado.
set -euo pipefail
export PYTHONPATH=src
[ -f results/sopa.json ] || python -m pedido_lora.sopa --ingredientes outputs/ing_a outputs/ing_b outputs/ing_c
avaliar() {  # nome, argumentos extras
  if [ -f "results/$1.json" ]; then echo "$1 já avaliado"; return; fi
  python -m pedido_lora.avaliar --nome "$@"
}
for nome in sopa_uniforme sopa_gulosa ing_a ing_b ing_c; do
  avaliar "$nome" --adaptador "outputs/$nome"
done
avaliar base_sem_ajuste --prompt longo
python -m pedido_lora.tabela
