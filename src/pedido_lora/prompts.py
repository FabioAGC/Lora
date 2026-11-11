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
6. Se a linha trouxer "complicação", siga-a. Ela muda a forma como o cliente se expressa, mas o pedido FINAL que se entende da mensagem continua exatamente o da linha.
7. Nunca comente o estilo nem a complicação dentro da mensagem: o cliente não sabe que existe uma instrução.

Responda APENAS com um JSON no formato {"mensagens": ["...", "..."]}, com uma mensagem por pedido, na mesma ordem."""


PROMPT_VERIFICADOR = """Você revisa um dataset de pedidos de delivery. Cada linha traz uma mensagem de cliente e o pedido final que deveria ser entendido dela.

Para cada linha, responda se a mensagem leva EXATAMENTE a esse pedido final. Confira:
- os mesmos itens, com as quantidades finais (correções e desistências na mensagem valem; o que vale é o que o cliente quer no fim);
- as observações de cada item, no item certo;
- pagamento, tipo (entrega ou retirada), orçamento e troco: se o pedido diz "não informado", a mensagem não pode escolher esse campo; se o pedido tem um valor, a mensagem precisa permitir entendê-lo. Perguntar sobre a entrega não é escolher entrega. Falar em troco permite entender pagamento em dinheiro.

Variações de escrita, gírias e erros de digitação são aceitáveis quando o item ainda é reconhecível.
Rejeite também mensagens que comentam a própria forma de escrever ou parecem seguir uma instrução (por exemplo, "vou escrever sem acento" ou "pode ficar simpático").

Responda APENAS com JSON listando só as linhas que NÃO batem: {"rejeitados": [{"n": 3, "motivo": "..."}]}. Em "motivo", explique em poucas palavras o que não bate. Se todas baterem, responda {"rejeitados": []}."""
