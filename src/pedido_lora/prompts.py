"""Prompts usados no projeto.

Há dois prompts de extração, e a diferença entre eles é o argumento central
do projeto:

- PROMPT_CURTO: usado pelo modelo ajustado. Ele já aprendeu o esquema e o
  catálogo no fine-tuning, então basta uma frase.
- prompt_longo(): usado pelos modelos sem fine-tuning. Precisa descrever o
  esquema e listar o catálogo inteiro, o que custa centenas de tokens por pedido.
"""
from .catalogo import CATALOGO, OBS_POR_ITEM, OBSERVACOES

PROMPT_CURTO = "Extraia o pedido de delivery da mensagem em JSON."


def prompt_longo() -> str:
    itens = "\n".join(f"- {cat}: {', '.join(nomes)}" for cat, nomes in CATALOGO.items())
    obs = "\n".join(f"- {cat}: {', '.join(o)}" for cat, o in OBSERVACOES.items())
    obs += "\n" + "\n".join(f"- {item} (somente estas): {', '.join(o)}" for item, o in OBS_POR_ITEM.items())
    return f"""Extraia o pedido de delivery da mensagem do cliente e responda APENAS com um JSON, sem texto extra, neste formato:
{{"itens":[{{"item":"<nome do catálogo>","qtd":<inteiro>,"obs":["<observação>"]}}],"pagamento":"pix"|"cartao"|"dinheiro"|null,"tipo":"entrega"|"retirada"|null,"orcamento":<inteiro>|null,"troco_para":<inteiro>|null}}

Regras:
- "item" deve ser exatamente um nome do catálogo abaixo, mesmo que o cliente escreva de outro jeito.
- "obs" deve usar exatamente as observações permitidas abaixo; lista vazia se não houver.
- "orcamento" é o valor máximo que o cliente quer gastar; "troco_para" é a nota com que ele vai pagar em dinheiro.
- Use null para o que não for mencionado.

Catálogo:
{itens}

Observações permitidas por categoria (alguns itens têm lista própria):
{obs}"""


PROMPT_GERADOR = """Você escreve mensagens realistas de clientes brasileiros pedindo comida por aplicativo de mensagem.

Vou enviar uma lista numerada de pedidos, um por linha, no formato "QTDx item (observações); ... | campo: valor | estilo: ...". Para cada pedido, escreva UMA mensagem que um cliente real mandaria para fazer exatamente aquele pedido.

Regras obrigatórias:
1. A mensagem deve conter TODOS os itens, com as quantidades e observações de cada um. Não acrescente nem remova itens ou observações.
2. Deixe claro a qual item cada observação pertence.
3. Se a linha trouxer "pagamento", "tipo" (entrega ou retirada), "orcamento" (valor máximo em reais que quer gastar) ou "troco_para" (nota em reais com que vai pagar), mencione-os de forma natural. NÃO mencione nenhum desses que não esteja na linha.
4. Não copie os nomes do pedido ao pé da letra: varie como um cliente faria (ex.: "xburguer", "coquinha zero", "uma calabresa grande", "refri", números por extenso ou em dígitos), mas mantenha o que distingue o item (sabor, tamanho, lata ou 2 litros).
5. Siga o estilo pedido. Não invente endereço, nome ou telefone.

Responda APENAS com um JSON no formato {"mensagens": ["...", "..."]}, com uma mensagem por pedido, na mesma ordem."""
