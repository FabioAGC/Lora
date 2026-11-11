# Como o projeto funciona, passo a passo

Este documento explica o caminho completo, da criação dos dados até o modelo final.
O GitHub desenha os diagramas abaixo automaticamente.

## Visão geral

```mermaid
flowchart TD
    A["1. Sortear pedidos<br/>(amostrador.py)<br/>rótulo correto por construção"] --> B["2. LLM grande escreve a mensagem<br/>(gerar_dados.py, via Groq)<br/>gírias, erros de digitação, áudio"]
    B --> C["3. Dividir<br/>(dividir.py)<br/>1200 treino · 140 validação · 200 teste"]
    C --> D1["4a. Treinar ingrediente A<br/>LoRA, lr 2e-4"]
    C --> D2["4b. Treinar ingrediente B<br/>LoRA, lr 5e-4"]
    C --> D3["4c. Treinar ingrediente C<br/>LoRA, lr 1e-3"]
    D1 --> E["5. Sopa<br/>(sopa.py)<br/>média dos três em um adaptador só"]
    D2 --> E
    D3 --> E
    E --> F["6. Avaliar no teste<br/>(avaliar.py)<br/>pedido exato, F1, latência"]
    G["Modelo base sem ajuste"] --> F
    H["LLM grande via API"] --> F
    F --> I["7. Tabela de resultados<br/>(tabela.py)"]
```

## 1. De onde vêm os dados

O problema de pedir a um LLM que invente exemplos *e* rótulos é que ele erra os rótulos.
Aqui a ordem é invertida:

```mermaid
flowchart LR
    S["Sorteio<br/>2x x-burguer (sem cebola)<br/>1x coca-cola zero lata<br/>pagamento: pix"] --> L["LLM grande"]
    L --> M["'me ve 2 xburguer sem cebola<br/>e uma coca zero lata, pago no pix'"]
    S --> R["Rótulo (JSON)<br/>já conhecido"]
    M --> P["Par de treino"]
    R --> P
```

O computador sorteia o pedido, então o rótulo já está certo. O LLM grande só faz o que
sabe fazer bem: escrever como uma pessoa escreveria.

## 2. O que é LoRA

Um modelo de linguagem é uma pilha de matrizes grandes de números (os pesos).
Ajustar todas elas exige muita memória. O LoRA congela o modelo e treina, para cada
matriz `W`, duas matrizes pequenas `A` e `B`:

```mermaid
flowchart LR
    X["entrada x"] --> W["W (congelada)<br/>896 x 896"]
    X --> A["A (treinável)<br/>8 x 896"]
    A --> B["B (treinável)<br/>896 x 8"]
    W --> SOMA(("+"))
    B --> SOMA
    SOMA --> Y["saída"]
```

O efeito é `W + B·A`. Neste projeto isso significa treinar **0,22% dos parâmetros**
(1,1 milhão de 495 milhões), o que cabe em uma CPU comum.

## 3. O que o modelo vê no treino

```mermaid
flowchart LR
    subgraph ignorado ["Prompt (não entra na perda)"]
      P1["sistema: Extraia o pedido..."] --> P2["usuário: me ve 2 xburguer..."]
    end
    subgraph cobrado ["Resposta (entra na perda)"]
      R1["{&quot;itens&quot;:[{&quot;item&quot;:&quot;x-burguer&quot;,..."]
    end
    P2 --> R1
```

O modelo tenta prever cada token da resposta. O erro dessa previsão é a *perda*, e o
treino ajusta `A` e `B` para diminuí-la. Os tokens do prompt são mascarados: o modelo
não é cobrado por adivinhar a mensagem do cliente.

## 4. O que é a sopa (model soup)

O jeito tradicional é treinar várias versões com hiperparâmetros diferentes e ficar
só com a melhor. A sopa aproveita todas: tira a **média dos pesos**.

```mermaid
flowchart TD
    BASE["Modelo base (o mesmo para todos)"] --> IA["Ingrediente A<br/>mudança ΔW_A"]
    BASE --> IB["Ingrediente B<br/>mudança ΔW_B"]
    BASE --> IC["Ingrediente C<br/>mudança ΔW_C"]
    IA --> S["Sopa uniforme<br/>ΔW = (ΔW_A + ΔW_B + ΔW_C) / 3"]
    IB --> S
    IC --> S
    S --> UM["Um único modelo<br/>custo de inferência de um só"]
```

Dois detalhes importam:

- **Por que funciona:** todos partem do mesmo modelo base, então terminam em regiões
  próximas do espaço de pesos, e a média entre eles continua sendo um bom modelo.
- **A armadilha com LoRA:** não se pode tirar a média de `A` e de `B` separadamente,
  porque `média(B)·média(A)` é diferente de `média(B·A)`. O `sopa.py` empilha as
  matrizes (`A` nas linhas, `B` nas colunas), o que dá a média exata e mantém o
  resultado como um adaptador pequeno. O teste `tests/test_sopa.py` prova as duas coisas.

A **sopa gulosa** é uma variação: ordena os ingredientes do melhor para o pior na
validação e só acrescenta um ingrediente se ele não piorar o resultado.

```mermaid
flowchart TD
    O["Ordenar ingredientes pela perda de validação"] --> P["Começar com o melhor"]
    P --> Q{"Acrescentar o próximo<br/>melhora ou mantém a perda?"}
    Q -- sim --> K["Fica na sopa"]
    Q -- não --> D["Descarta"]
    K --> Q
    D --> Q
```

## 5. Como é medido

| Métrica | O que significa |
|---|---|
| Pedido exato | O pedido inteiro está certo: todos os itens, quantidades, observações e os quatro campos |
| F1 de itens | Acerto parcial: quantos itens (com quantidade e observações) batem |
| JSON válido | A saída pôde ser lida e segue o esquema |
| Tokens de prompt | Quanto texto o modelo precisa receber por pedido (custo) |

O conjunto de teste (200 pedidos) nunca é usado no treino nem na escolha da sopa;
a sopa gulosa decide com o conjunto de validação.

## 6. O que aconteceu neste experimento

Os resultados estão no [README](../README.md#resultados). Em resumo: o ajuste levou o
modelo de 0% para 85% de pedidos exatos, e a sopa não superou o melhor ingrediente
porque os três ficaram com qualidade muito desigual após uma única época de treino.
