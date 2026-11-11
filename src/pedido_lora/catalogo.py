"""Catálogo fictício: os nomes canônicos que o modelo precisa aprender a produzir.

O modelo ajustado NÃO recebe este catálogo no prompt: ele aprende os nomes
canônicos durante o fine-tuning. Os modelos de comparação (sem fine-tuning)
recebem o catálogo inteiro no prompt, o que custa centenas de tokens por pedido.
"""

CATALOGO: dict[str, list[str]] = {
    "lanche": [
        "x-burguer", "x-salada", "x-bacon", "x-tudo", "x-egg",
        "hambúrguer artesanal", "cachorro-quente", "misto-quente", "sanduíche natural",
    ],
    "pizza": [
        "pizza de calabresa", "pizza de mussarela", "pizza margherita",
        "pizza portuguesa", "pizza de frango com catupiry", "pizza quatro queijos",
    ],
    "prato": [
        "marmita de frango", "marmita de carne", "feijoada", "strogonoff de frango",
        "parmegiana de frango", "yakisoba", "temaki de salmão", "combinado de sushi",
        "salada caesar",
    ],
    "porcao": [
        "batata frita", "porção de calabresa", "frango a passarinho", "anéis de cebola",
        "coxinha", "pastel de carne", "pastel de queijo", "esfiha de carne",
    ],
    "bebida": [
        "coca-cola lata", "coca-cola zero lata", "coca-cola 2 litros", "guaraná lata",
        "guaraná 2 litros", "suco de laranja", "suco de maracujá", "água mineral",
        "água com gás", "cerveja long neck",
    ],
    "sobremesa": [
        "açaí 300ml", "açaí 500ml", "pudim", "brigadeiro", "petit gâteau",
        "milkshake de chocolate", "sorvete de creme",
    ],
}

OBSERVACOES: dict[str, list[str]] = {
    "lanche": [
        "sem cebola", "sem tomate", "sem picles", "sem maionese", "sem salada",
        "bacon extra", "queijo extra", "bem passado", "mal passado",
    ],
    "pizza": ["borda recheada", "sem cebola", "sem azeitona", "massa fina", "bem assada"],
    "prato": ["sem pimenta", "sem salada", "molho à parte", "sem cebola"],
    "porcao": ["sem sal", "molho à parte", "bem crocante"],
    "bebida": ["bem gelada", "sem gelo", "com limão"],
    "sobremesa": ["calda à parte"],
}

# Observações que só fazem sentido para itens específicos.
OBS_POR_ITEM: dict[str, list[str]] = {
    "açaí 300ml": ["com granola", "sem leite condensado", "sem banana"],
    "açaí 500ml": ["com granola", "sem leite condensado", "sem banana"],
    "suco de laranja": ["sem açúcar", "sem gelo", "bem gelada"],
    "suco de maracujá": ["sem açúcar", "sem gelo", "bem gelada"],
    "salada caesar": ["molho à parte", "sem croutons"],
    "combinado de sushi": ["sem cream cheese", "molho à parte"],
    "temaki de salmão": ["sem cream cheese", "sem cebolinha"],
}


def obs_do_item(item: str) -> list[str]:
    return OBS_POR_ITEM.get(item) or OBSERVACOES[CATEGORIA_DO_ITEM[item]]

PAGAMENTOS = ["pix", "cartao", "dinheiro"]
TIPOS = ["entrega", "retirada"]

CATEGORIA_DO_ITEM: dict[str, str] = {
    item: cat for cat, itens in CATALOGO.items() for item in itens
}
TODOS_ITENS: list[str] = list(CATEGORIA_DO_ITEM)
TODAS_OBS: set[str] = {o for obs in [*OBSERVACOES.values(), *OBS_POR_ITEM.values()] for o in obs}
