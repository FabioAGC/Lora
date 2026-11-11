# Lora: extração de pedidos de delivery com LoRA em CPU

![testes](https://github.com/FabioAGC/Lora/actions/workflows/ci.yml/badge.svg)

Um modelo de linguagem pequeno (Qwen2.5 0.5B) ajustado **em CPU** para transformar
mensagens informais de delivery, em português, em pedidos estruturados:

```
"me ve 2 xburguer sem cebola e uma coca zero lata, pago no pix"
→ {"itens":[{"item":"x-burguer","qtd":2,"obs":["sem cebola"]},
            {"item":"coca-cola zero lata","qtd":1,"obs":[]}],
   "pagamento":"pix","tipo":null,"orcamento":null,"troco_para":null}
```

O projeto cobre o ciclo completo: geração de dados sintéticos, fine-tuning com LoRA e
avaliação contra o modelo base e contra um LLM grande via API. O treino pode ser feito
de dois jeitos, com resultados equivalentes:

- com o [Soup CLI](https://github.com/MakazhanAlpamys/Soup), a partir do `soup.yaml`;
- com um laço de treino escrito à mão (`treinar.py`), para quem quer ver cada passo.

Há também um experimento secundário de *model soup* (média de pesos de vários adaptadores).
Tudo roda sem GPU: os tempos abaixo são de uma máquina de 4 núcleos.

Para uma explicação passo a passo com diagramas, veja [docs/COMO_FUNCIONA.md](docs/COMO_FUNCIONA.md).

## Resultados

Conjunto de teste com 200 pedidos, nunca vistos no treino nem na escolha da sopa.

| Modelo | Pedido exato | F1 de itens | JSON válido | Tokens de prompt | Treino |
|---|---:|---:|---:|---:|---:|
| Modelo base, sem ajuste (prompt longo) | 0,0% | 0,0% | 0,0% | 883 | — |
| **Treino com Soup CLI (lr 1e-3)** | **84,5%** | 95,1% | 100% | 55 | 16 min |
| Laço manual, ingrediente A (lr 2e-4) | 54,5% | 80,0% | 99,5% | 55 | 30 min |
| Laço manual, ingrediente B (lr 5e-4) | 80,5% | 93,8% | 99,5% | 55 | 43 min |
| Laço manual, ingrediente C (lr 1e-3) | **85,0%** | 96,7% | 100% | 55 | 37 min |
| Sopa uniforme (A+B+C) | 83,0% | 94,7% | 100% | 55 | soma dos três |
| Sopa gulosa (escolheu só C) | 85,0% | 96,7% | 100% | 55 | soma dos três |
| LLM grande via API (gpt-oss-120b, prompt longo) | 90,0% | 97,8% | 100% | 848 | — |

"Pedido exato" exige acertar todos os itens, quantidades, observações e os quatro campos.
A inferência local leva cerca de 1,7 s por pedido em 4 núcleos de CPU; a API responde em 0,7 s.

### O que os números dizem

- **O fine-tuning é o que faz o modelo pequeno funcionar.** Sem ajuste, o modelo de
  0,5B não produz nenhum JSON no esquema, mesmo com o catálogo inteiro no prompt.
  Ajustado, chega a 85% de pedidos exatos com um prompt de uma frase.
- **O Soup CLI e o laço manual chegam ao mesmo lugar.** Com os mesmos hiperparâmetros,
  a diferença é de um pedido em 200 (84,5% contra 85,0%), dentro do ruído. O Soup treinou
  em menos da metade do tempo (16 contra 37 minutos); parte dessa diferença pode vir de
  o laço manual ter disputado CPU com outros processos durante a medição.
- **O modelo ajustado fica 5 pontos abaixo do LLM grande usando 15 vezes menos tokens
  de prompt** (55 contra 848), e roda sem GPU e sem API.
- **A sopa não superou o melhor ingrediente neste experimento.** A sopa uniforme ficou
  em 83,0%, entre os ingredientes B e C, e a sopa gulosa rejeitou A e B e ficou só com C.
  O motivo provável é que os ingredientes têm qualidade muito desigual: com uma única
  época de treino, as taxas de aprendizado menores ainda não convergiram (a perda de
  validação de A é três vezes a de C), e a média puxa o resultado para baixo. A
  literatura de model soups mostra ganho quando os ingredientes são comparáveis.
  O próximo experimento é treinar ingredientes com a mesma taxa de aprendizado (1e-3)
  e sementes diferentes.
- **A sopa uniforme ainda foi melhor que dois dos três ingredientes**, sem precisar de
  validação para escolher. Esse é o caso de uso dela: robustez quando não se sabe de
  antemão qual configuração é a melhor.

### Limitações

- O dataset é sintético e as mensagens seguem de perto os nomes do catálogo. Mensagens
  reais são mais variadas, e o desempenho nelas deve ser menor. Um exemplo testado à mão:
  "uma coquinha zero" vira um item inexistente, e "passo aí pra pegar" não é reconhecido
  como retirada, porque essas formas não aparecem no treino.
- Há ruído nos rótulos: em alguns exemplos o LLM gerador mencionou algo que não estava
  no pedido sorteado (por exemplo, "entrega"). Isso limita o teto de todos os modelos,
  inclusive o do LLM grande.
- O conjunto de teste tem 200 exemplos, então diferenças de 2 pontos (4 pedidos) estão
  dentro do ruído.
- O LLM grande avaliado é o mesmo que gerou a maior parte das mensagens, o que pode
  favorecê-lo.
- O catálogo é fixo (49 itens). Mudar o cardápio exige treinar de novo.

## Como funciona

| Etapa | Arquivo | O que faz |
|---|---|---|
| 1 | `amostrador.py` | Sorteia pedidos estruturados a partir do catálogo. O rótulo nasce correto. |
| 2 | `gerar_dados.py` | Um LLM grande (Groq) escreve a mensagem informal de cada pedido. |
| 3 | `dividir.py` | Separa 1.200 exemplos de treino, 140 de validação e 200 de teste. |
| 4 | `soup.yaml` ou `treinar.py` | LoRA (rank 8) nas projeções de atenção: 0,22% dos parâmetros. |
| 5 | `sopa.py` | Sopa uniforme e sopa gulosa dos adaptadores. |
| 6 | `avaliar.py`, `avaliar_api.py` | Métricas no conjunto de teste. |

### Experimento secundário: sopa de adaptadores LoRA

Cada ingrediente altera um peso `W` em `ΔW = (alpha/r)·B·A`. A sopa é a média dos `ΔW`.
Tirar a média de `A` e de `B` separadamente dá o resultado errado, porque
`média(B)·média(A) ≠ média(B·A)`. O `sopa.py` concatena as matrizes (`A` nas linhas,
`B` nas colunas, com a escala embutida em `B`), o que dá a média exata e mantém a sopa
como um adaptador LoRA de rank `r₁+r₂+r₃`. Os testes em `tests/test_sopa.py` verificam
as duas afirmações.

Uma ressalva: a sopa não reduz o custo de treino, ela custa um treino por ingrediente.
O que faz o projeto caber em hardware modesto é o modelo pequeno com LoRA. O ganho da
sopa é entregar um único modelo na inferência.

## Como rodar

```bash
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
```

Os dados já estão em `data/`. Para treinar com o Soup CLI (cerca de 16 minutos em
4 núcleos de CPU) e avaliar:

```bash
pip install "soup-cli[train]"
soup train --config soup.yaml --yes
python -m pedido_lora.avaliar --nome soup_cli --adaptador outputs/soup_cli
```

Para treinar um modelo maior numa GPU gratuita, veja [kaggle/README.md](kaggle/README.md).

Sem GPU, o Soup desliga a quantização e não usa o *layer streaming*, que é o recurso
dele para placas de vídeo com pouca memória; o `soup.yaml` já está configurado para CPU.

Para repetir o experimento de model soup com o laço manual (30 a 40 minutos por
ingrediente):

```bash
./scripts/treinar_ingredientes.sh
./scripts/avaliar_tudo.sh
```

Para testar uma mensagem:

```bash
python -m pedido_lora.inferir --adaptador outputs/soup_cli "2 xburguer sem cebola e uma coca zero, pago no pix"
```

Para gerar um dataset novo ou avaliar o LLM grande, é preciso uma chave da Groq na
variável `GROQ_API_KEY`:

```bash
python -m pedido_lora.gerar_dados --total 1300 --saida data/bruto.jsonl
python -m pedido_lora.dividir data/bruto.jsonl
python -m pedido_lora.avaliar_api
```

Para publicar o adaptador e o dataset no Hugging Face:

```bash
python -m pedido_lora.publicar --usuario SEU_USUARIO
```

## Referências

- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.*
- Wortsman et al., 2022. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.*
