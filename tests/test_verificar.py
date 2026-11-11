import json

from pedido_lora.verificar import descrever_para_revisao, problemas_por_regra

BASE = {"itens": [{"item": "x-burguer", "qtd": 1, "obs": []}],
        "pagamento": None, "tipo": None, "orcamento": None, "troco_para": None}


def _pedido(**campos):
    return {**BASE, **campos}


def test_rotulo_sem_pagamento_mas_mensagem_cita_pix():
    assert problemas_por_regra("1 x-burguer, pago no pix", _pedido())


def test_rotulo_com_pagamento_que_a_mensagem_nao_cita():
    assert problemas_por_regra("1 x-burguer", _pedido(pagamento="cartao"))


def test_troco_permite_entender_dinheiro():
    assert not problemas_por_regra("1 x-burguer, troco pra 50", _pedido(pagamento="dinheiro", troco_para=50))


def test_pergunta_sobre_entrega_nao_e_escolha():
    msg = "1 x-burguer, quanto tempo demora a entrega?"
    assert not problemas_por_regra(msg, _pedido(), "pergunta_entrega")


def test_retirada_sem_rotulo_e_rejeitada():
    assert problemas_por_regra("1 x-burguer, passo aí pra buscar", _pedido())


def test_vazamento_da_instrucao_de_estilo():
    assert problemas_por_regra("1 x-burguer sem acento e erros", _pedido())


def test_mensagem_consistente_passa():
    msg = "Oi! Um xburguer, pagamento no cartão, entrega por favor"
    assert not problemas_por_regra(msg, _pedido(pagamento="cartao", tipo="entrega"))


def test_descricao_marca_campos_vazios():
    texto = descrever_para_revisao(json.loads(json.dumps(_pedido(pagamento="pix"))))
    assert "pagamento: pix" in texto and "tipo: não informado" in texto
