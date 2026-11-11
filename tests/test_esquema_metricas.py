import json
import random

from pedido_lora.amostrador import sortear_pedido
from pedido_lora.catalogo import CATEGORIA_DO_ITEM, TODAS_OBS, obs_do_item
from pedido_lora.esquema import interpretar, serializar
from pedido_lora.metricas import agregar, comparar

PEDIDO = {"itens": [{"item": "x-burguer", "qtd": 2, "obs": ["sem tomate", "sem cebola"]},
                    {"item": "coca-cola lata", "qtd": 1, "obs": []}],
          "pagamento": "pix", "tipo": "entrega", "orcamento": 60, "troco_para": None}


def test_serializar_e_deterministico_e_ordena_obs():
    texto = serializar(PEDIDO)
    assert texto == serializar(json.loads(texto))
    assert json.loads(texto)["itens"][0]["obs"] == ["sem cebola", "sem tomate"]
    assert " " not in texto.replace("coca-cola lata", "").replace("sem cebola", "").replace("sem tomate", "")


def test_interpretar_aceita_json_embrulhado():
    assert interpretar("```json\n" + serializar(PEDIDO) + "\n```") == json.loads(serializar(PEDIDO))


def test_interpretar_rejeita_saidas_invalidas():
    for ruim in ["", "não sei", '{"itens": []}', '{"itens":[{"item":"x","qtd":"2"}]}',
                 '{"itens":[{"item":"x","qtd":1}],"pagamento":"boleto"}',
                 '{"itens":[{"item":"x","qtd":1}],"orcamento":"60"}', '{"itens":[{"item":"x"']:
        assert interpretar(ruim) is None, ruim


def test_comparar_ignora_ordem_dos_itens():
    trocado = {**PEDIDO, "itens": PEDIDO["itens"][::-1]}
    assert comparar(trocado, PEDIDO)["exato"]


def test_comparar_conta_erros_parciais():
    errado = json.loads(serializar(PEDIDO))
    errado["itens"][0]["qtd"] = 3
    errado["pagamento"] = "dinheiro"
    c = comparar(errado, PEDIDO)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 1)
    assert not c["exato"] and not c["pagamento"] and c["tipo"]


def test_agregar():
    r = agregar([comparar(PEDIDO, PEDIDO), comparar(None, PEDIDO)])
    assert r["json_valido"] == 50.0 and r["pedido_exato"] == 50.0
    assert r["itens_f1"] == round(100 * 2 * 1.0 * 0.5 / 1.5, 1)


def test_amostrador_gera_pedidos_validos_e_reprodutiveis():
    a = [sortear_pedido(random.Random(7)) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    rng = random.Random(0)
    for _ in range(500):
        p = sortear_pedido(rng)
        assert interpretar(serializar(p)) == json.loads(serializar(p))
        nomes = [i["item"] for i in p["itens"]]
        assert len(set(nomes)) == len(nomes)
        for i in p["itens"]:
            assert i["item"] in CATEGORIA_DO_ITEM
            assert set(i["obs"]) <= set(obs_do_item(i["item"])) <= TODAS_OBS
            assert not {"bem passado", "mal passado"} <= set(i["obs"])
        assert p["troco_para"] is None or p["pagamento"] == "dinheiro"
