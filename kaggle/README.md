# Treinar no Kaggle (GPU gratuita)

O Kaggle oferece GPUs NVIDIA de graça em notebooks, com uma cota semanal de horas que
aparece no seu perfil. Este notebook treina o extrator com um modelo de 1,5B numa T4.

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
3. Rode célula por célula para acompanhar, ou use **Save Version → Save & Run All (Commit)**
   para rodar em segundo plano, com o navegador fechado.
4. Quando terminar, baixe o adaptador e os resultados na aba **Output** do notebook.

## Para não gastar a cota à toa

- A cota conta enquanto a sessão com GPU estiver ligada, mesmo parada. Desligue com
  **Stop Session** quando terminar de usar.
- Escreva e teste o código com o acelerador desligado (`None`) e ligue a GPU só para treinar.
- Prefira o commit (**Save & Run All**): a sessão desliga sozinha ao fim da execução.
- Prefira a T4 à P100: versões recentes do PyTorch deixaram de suportar a arquitetura da P100.

## Arquivos

| Arquivo | O que é |
|---|---|
| `treinar_no_kaggle.ipynb` | Notebook: instala o Soup, treina, avalia e publica |
| `../soup_gpu.yaml` | Configuração de treino para GPU (modelo de 1,5B) |
