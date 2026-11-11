#!/usr/bin/env bash
# Monta o dataset v2: filtra o v1 com o verificador e gera novos exemplos com
# casos de borda (também verificados). Precisa de GROQ_API_KEY. Pode ser
# interrompido e retomado.
set -euo pipefail
export PYTHONPATH=src
python -m pedido_lora.verificar data/v1/todos.jsonl \
  --saida data/v1_verificados.jsonl --rejeitados data/rejeitados_v1.jsonl
python -m pedido_lora.gerar_dados --total 1700 --borda 0.5 --seed 2 \
  --saida data/bruto_v2.jsonl --rejeitados data/rejeitados_v2.jsonl
python -m pedido_lora.dividir data/v1_verificados.jsonl data/bruto_v2.jsonl
python -m pedido_lora.exportar_chat
