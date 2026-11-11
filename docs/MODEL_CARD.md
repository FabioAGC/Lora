---
language: pt
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags: [lora, delivery, extracao-de-informacao, portugues]
---

# pedido-delivery-ptbr-lora

Adaptador LoRA para `Qwen/Qwen2.5-0.5B-Instruct` que extrai pedidos de delivery de
mensagens informais em português para JSON.

- **Entrada:** mensagem do cliente, com o prompt de sistema `Extraia o pedido de delivery da mensagem em JSON.`
- **Saída:** `{"itens":[{"item","qtd","obs"}],"pagamento","tipo","orcamento","troco_para"}`

## Treino

LoRA rank 8 em `q_proj`, `k_proj`, `v_proj`, `o_proj`; 1 época sobre 1.200 exemplos;
AdamW, taxa de aprendizado 1e-3 com aquecimento e decaimento linear; treinado em CPU
(16 minutos em 4 núcleos) com o [Soup CLI](https://github.com/MakazhanAlpamys/Soup).

## Dados

1.540 exemplos sintéticos. Os pedidos são sorteados de um catálogo fictício de 49 itens
e as mensagens são escritas por LLMs (gpt-oss-120b e qwen3.8-27b, via Groq).

## Resultados (200 exemplos de teste)

| Métrica | Valor |
|---|---:|
| Pedido exato | 84,5% |
| F1 de itens | 95,1% |
| JSON válido | 100% |

## Limitações

Dados sintéticos e catálogo fixo: o modelo só conhece os 49 itens do treino e não foi
avaliado em mensagens reais. Não use em produção sem avaliar com dados do seu domínio.
