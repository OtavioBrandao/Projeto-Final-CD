"""Testes da Pessoa 4 — ``pytest tests/test_explicabilidade_p4.py``.

Não exigem ``shap`` nem ``xgboost``: a agregação usa o pré-processamento real da Pessoa 3
(com estimador Dummy) e valores SHAP sintéticos de resposta conhecida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p3_modelagem.dados import carregar_base, selecionar_xy
from src.p3_modelagem.modelos import criar_pipeline
from src.p4_explicabilidade import explicabilidade as ex
from src.p4_explicabilidade import negocio as ng


@pytest.fixture(scope="module")
def base():
    return carregar_base()


@pytest.mark.parametrize("cenario", ["A", "B"])
def test_mapa_cobre_toda_a_matriz_transformada(base, cenario):
    X, y = selecionar_xy(base.loc[base.ano.eq(2024)].sample(4000, random_state=0), cenario)
    pipe = criar_pipeline(cenario, "dummy").fit(X, y)
    pre = pipe.named_steps["preprocessamento"]
    mapa = ex.mapa_colunas_originais(pre)
    assert len(mapa) == pre.transform(X).shape[1]
    assert set(mapa) == set(X.columns)  # toda feature aparece, nenhuma sobra


def test_agregacao_preserva_a_soma_por_acidente():
    rng = np.random.default_rng(0)
    mapa = ["a", "b", "b", "b", "c", "c"]
    shap = rng.normal(size=(50, len(mapa)))
    agg = ex.agregar_por_variavel(shap, mapa)
    assert list(agg.columns) == ["a", "b", "c"]
    np.testing.assert_allclose(agg.sum(axis=1), shap.sum(axis=1))
    np.testing.assert_allclose(agg["b"], shap[:, 1:4].sum(axis=1))


def test_agregacao_rejeita_mapa_de_tamanho_errado():
    with pytest.raises(ValueError):
        ex.agregar_por_variavel(np.zeros((3, 4)), ["a", "b"])


def test_importancia_ordena_e_soma_100pct():
    shap_var = pd.DataFrame({"x": [1.0, -1.0], "y": [0.1, 0.1], "z": [3.0, 3.0]})
    imp = ex.importancia(shap_var)
    assert imp["variavel"].tolist() == ["z", "x", "y"]
    assert imp["pct_do_total"].sum() == pytest.approx(1.0)


def test_confirmar_sinais_detecta_direcao_e_pula_coluna_ausente():
    X = pd.DataFrame({"envolve_pedestre": [1, 1, 0, 0], "envolve_moto": [1, 1, 0, 0]})
    shap_var = pd.DataFrame({"envolve_pedestre": [2.0, 2.0, -0.1, -0.1],
                             "envolve_moto": [-0.5, -0.5, 0.1, 0.1]})
    res = ex.confirmar_sinais(shap_var, X).set_index("sinal")
    assert res.loc["Pedestre envolvido", "confirma"]
    assert res.loc["Pedestre envolvido", "multiplicador_odds"] == pytest.approx(np.exp(2.1))
    assert res.loc["Motocicleta envolvida", "confirma"]        # esperado: reduz
    assert "Pista simples" not in res.index                    # coluna ausente (cenário B)


def test_confirmar_sinais_marca_quando_o_modelo_discorda():
    X = pd.DataFrame({"envolve_pedestre": [1, 0]})
    res = ex.confirmar_sinais(pd.DataFrame({"envolve_pedestre": [-1.0, 1.0]}), X)
    assert not res.loc[0, "confirma"]


def test_curva_ganho_de_modelo_perfeito_e_aleatorio():
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    perfeito = ng.curva_ganho(y, y.astype(float))
    assert perfeito.loc[perfeito.fracao_acidentes.round(2).eq(0.2), "fracao_mortes"].iloc[0] == 1.0
    assert perfeito["fracao_mortes"].iloc[-1] == 1.0


def test_concentracao_por_fracao_conta_mortes_e_lift():
    previsoes = pd.DataFrame({"modelo": "m", "y_real": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              "probabilidade": [.9, .8, .7, .6, .5, .4, .3, .2, .1, .0]})
    r = ng.concentracao_por_fracao(previsoes, fracoes=(0.2,)).iloc[0]
    assert r["mortes_cobertas"] == 2 and r["pct_mortes_cobertas"] == 1.0
    assert r["lift"] == pytest.approx(5.0)


def test_wilson_contem_a_proporcao():
    lo, hi = ng.intervalo_wilson(70, 1000)
    assert lo < 0.07 < hi


def test_segmentos_reproduzem_os_numeros_da_eda(base):
    seg = ng.tabela_segmentos(base).set_index("segmento")
    assert seg.loc["noite_madrugada", "pct_mortes"] == pytest.approx(0.572, abs=0.002)
    assert seg.loc["noite_madrugada", "risco_relativo_vs_resto"] == pytest.approx(1.94, abs=0.02)
    assert seg.loc["pedestre", "risco_relativo_vs_resto"] == pytest.approx(4.58, abs=0.03)
    assert seg.loc["frontal_pista_simples", "pct_mortes"] == pytest.approx(0.261, abs=0.002)
    assert seg.loc["moto", "risco_relativo_vs_resto"] < 1  # hipótese derrubada pela EDA
