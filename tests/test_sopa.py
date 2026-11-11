import pytest

torch = pytest.importorskip("torch")

from pedido_lora.sopa import misturar, sopa_gulosa  # noqa: E402

CHAVE_A = "base_model.model.camada.q_proj.lora_A.weight"
CHAVE_B = "base_model.model.camada.q_proj.lora_B.weight"


def _adaptador(r, alpha, seed, entrada=12, saida=10):
    g = torch.Generator().manual_seed(seed)
    pesos = {CHAVE_A: torch.randn(r, entrada, generator=g), CHAVE_B: torch.randn(saida, r, generator=g)}
    return pesos, {"r": r, "lora_alpha": alpha}


def _delta(pesos, config):
    return (config["lora_alpha"] / config["r"]) * pesos[CHAVE_B] @ pesos[CHAVE_A]


def test_sopa_uniforme_e_a_media_exata_dos_deltas():
    ads = [_adaptador(4, 8, 1), _adaptador(4, 8, 2), _adaptador(8, 4, 3)]
    pesos, config = misturar(ads)
    esperado = sum(_delta(*a) for a in ads) / 3
    assert config["r"] == 16 and config["lora_alpha"] == 16
    assert torch.allclose(_delta(pesos, config), esperado, atol=1e-5)


def test_media_ingenua_de_A_e_B_nao_e_a_sopa():
    ads = [_adaptador(4, 8, 1), _adaptador(4, 8, 2)]
    esperado = sum(_delta(*a) for a in ads) / 2
    ingenua = 2.0 * ((ads[0][0][CHAVE_B] + ads[1][0][CHAVE_B]) / 2) @ ((ads[0][0][CHAVE_A] + ads[1][0][CHAVE_A]) / 2)
    assert not torch.allclose(ingenua, esperado, atol=1e-2)


def test_proporcoes_ponderadas():
    ads = [_adaptador(4, 8, 1), _adaptador(4, 8, 2)]
    pesos, config = misturar(ads, [0.25, 0.75])
    assert torch.allclose(_delta(pesos, config), 0.25 * _delta(*ads[0]) + 0.75 * _delta(*ads[1]), atol=1e-5)


def test_sopa_gulosa_so_aceita_quem_nao_piora():
    notas = {"a": 0.30, "b": 0.20, "c": 0.25}
    perdas = {("b", "c"): 0.22, ("b", "a"): 0.18}
    escolhidos, registro = sopa_gulosa(list(notas), notas, lambda nomes: perdas[tuple(nomes)])
    assert escolhidos == ["b", "a"]
    assert [r["aceito"] for r in registro] == [True, False, True]
