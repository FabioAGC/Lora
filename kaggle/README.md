# Treinar no Kaggle (GPU gratuita)

O Kaggle oferece GPUs NVIDIA de graça em notebooks, com uma cota semanal de horas que
aparece no seu perfil. O notebook deste diretório treina o extrator em quatro tamanhos
de modelo numa T4 e compara todos no mesmo conjunto de teste.

| Modelo | Como treina | Estimativa na T4 |
|---|---|---:|
| Qwen2.5 0.5B | LoRA, float16 | ~10 min |
| Qwen2.5 1.5B | LoRA, float16 | ~25 min |
| Qwen2.5 3B | LoRA, float16 | ~50 min |
| Qwen2.5 7B | QLoRA, 4 bits | ~2,5 h |
| **Total** | | **~4 h** |

As estimativas incluem a avaliação e **não foram medidas**: confira o tempo real do
primeiro modelo no log e ajuste o plano. Quatro horas cabem numa sessão (o limite é de
12 horas) e usam uma parte da cota semanal.

## Uma vez só

1. Crie uma conta em kaggle.com.
2. Verifique o telefone em **Settings → Phone verification**. Sem isso, a GPU e a
   internet do notebook ficam bloqueadas.

## Para cada treino

1. Em **Create → New Notebook**, use **File → Import Notebook** e envie
   `kaggle/treinar_no_kaggle.ipynb` (ou cole a URL do arquivo no GitHub).
2. No painel lateral, em **Settings**:
   - **Accelerator**: `GPU T4 x2`
   - **Internet**: ligada
3. Use **Save Version → Save & Run All (Commit)** para rodar em segundo plano, com o
   navegador fechado. A sessão desliga sozinha ao fim.
4. Quando terminar, baixe os adaptadores, os resultados e as previsões na aba
   **Output** do notebook.

Para rodar só alguns tamanhos, troque o comando da etapa 3 do notebook:

```bash
python -m pedido_lora.kaggle --modelos 0.5b 1.5b
```

Modelos já avaliados são pulados, então dá para dividir o trabalho em várias sessões,
desde que você copie a pasta `results/` de uma para a outra.

## Para não gastar a cota à toa

- A cota conta enquanto a sessão com GPU estiver ligada, mesmo parada. Desligue com
  **Stop Session** quando terminar de usar.
- Escreva e teste o código com o acelerador desligado (`None`) e ligue a GPU só para treinar.
- Prefira a T4 à P100: versões recentes do PyTorch deixaram de suportar a arquitetura da P100.

## Problemas já encontrados

| Mensagem | Causa | Solução |
|---|---|---|
| `Hardware-fit gate ... exceeds 15.6 GB` | O Soup prevê a memória pelo `max_length` configurado | Já corrigido: o pipeline mede o comprimento nos dados |
| `peft refuses to load with the preinstalled torchao` | O Kaggle traz um `torchao` antigo | A célula de instalação já remove o `torchao` |
| Pasta `/kaggle/working/Lora/Lora` | Célula de instalação rodada duas vezes numa versão antiga do notebook | A célula atual pode ser repetida à vontade |

## Arquivos

| Arquivo | O que é |
|---|---|
| `treinar_no_kaggle.ipynb` | Notebook: instala o Soup, treina os quatro modelos, avalia e publica |
| `../src/pedido_lora/kaggle.py` | O pipeline: receitas de cada modelo, treino, avaliação e tabela |
