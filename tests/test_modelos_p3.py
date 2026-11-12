"""Comportamento dos pipelines A/B com dados sintéticos."""

import builtins
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.dados import (
    gerar_cortes_temporais, selecionar_xy, separar_desenvolvimento_teste,
)
from src.p3_modelagem.modelos import ALGORITMOS, criar_estimador, criar_pipeline
from src.p3_modelagem.preprocessamento import criar_preprocessamento, identificar_tipos


@pytest.fixture
def base_sintetica():
    linhas = []
    for ano in (2024, 2025):
        for mes in range(1, 13):
            for i in range(5):
                linha = dict.fromkeys(cfg.COLUNAS_ESPERADAS, 0)
                linha.update(dict.fromkeys(cfg.COLUNAS_CATEGORICAS, "observada"))
                linha.update(
                    dia_semana=f"dia_{i % 2}", br=101,
                    km=float(mes * 10 + i), latitude=np.nan if i == 0 else -9. - i,
                    longitude=-35. - i, idade_media_veiculos=np.nan if i == 0 else float(i),
                    veiculos=i + 1, hora=i * 4, mes=mes, ano=ano,
                    acidente_fatal=int(i == 4), fim_de_semana=i % 2, via_reta=1,
                )
                linhas.append(linha)
    return pd.DataFrame(linhas, index=pd.Index(range(100, 940, 7), name="origem"))


def validar_saidas(pipeline, x):
    previsoes = pipeline.predict(x)
    probabilidades = pipeline.predict_proba(x)
    np.testing.assert_array_equal(pipeline.classes_, [0, 1])
    assert previsoes.shape == (len(x),)
    assert set(previsoes) <= {0, 1}
    assert probabilidades.shape == (len(x), 2)
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1., atol=1e-6)
    indice_positivo = np.flatnonzero(pipeline.classes_ == 1).item()
    assert indice_positivo == 1
    return probabilidades


@pytest.mark.parametrize("cenario,contagens", [("A", (7, 11, 16)), ("B", (5, 5, 2))])
def test_particao_exata_do_contrato(cenario, contagens):
    tipos = identificar_tipos(cenario)
    assert tuple(map(len, tipos.values())) == contagens
    todas = [c for grupo in tipos.values() for c in grupo]
    assert len(todas) == len(set(todas)) == len(cfg.FEATURES_POR_CENARIO[cenario])
    assert set(todas) == set(cfg.FEATURES_POR_CENARIO[cenario])
    assert not set(todas) & cfg.PREDITORES_PROIBIDOS
    assert "br" in tipos["categoricas"]
    assert {"hora", "mes"} <= set(tipos["numericas"])


@pytest.mark.parametrize("problema", ["desconhecida", "duplicada", "proibida", "ambigua"])
def test_contrato_invalido_e_recusado(monkeypatch, problema):
    if problema == "ambigua":
        monkeypatch.setattr(cfg, "COLUNAS_CATEGORICAS", (*cfg.COLUNAS_CATEGORICAS, "km"))
    else:
        coluna = {"desconhecida": "nova", "duplicada": "km", "proibida": "ano"}[problema]
        monkeypatch.setitem(cfg.FEATURES_POR_CENARIO, "B", (*cfg.FEATURES_B, coluna))
    with pytest.raises(ValueError):
        criar_preprocessamento("B")


@pytest.mark.parametrize("cenario", ["A", "B"])
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_fit_previsoes_determinismo_e_entradas_intactas(base_sintetica, cenario, algoritmo):
    antes = base_sintetica.copy(deep=True)
    dev, _ = separar_desenvolvimento_teste(base_sintetica)
    corte = gerar_cortes_temporais(dev)[0]
    x, y = selecionar_xy(dev, cenario)
    xt, yt = x.loc[corte.treino].copy(), y.loc[corte.treino].copy()
    xv = x.loc[corte.validacao].copy()
    xv["uf"] = "categoria_inedita"
    xv["br"] = "BR-INEDITA"
    xt_antes, xv_antes, yt_antes = xt.copy(deep=True), xv.copy(deep=True), yt.copy(deep=True)

    pipeline = criar_pipeline(cenario, algoritmo)
    pre = pipeline.named_steps["preprocessamento"]
    assert not hasattr(pre, "transformers_")
    assert not hasattr(pipeline.named_steps["modelo"], "classes_")
    with pytest.raises(NotFittedError):
        pipeline.predict(xv)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(xt, yt)
    probas = validar_saidas(pipeline, xv)
    assert tuple(pre.feature_names_in_) == cfg.FEATURES_POR_CENARIO[cenario]
    nomes = pre.get_feature_names_out()
    assert not any(c.split("__", 1)[1] in cfg.PREDITORES_PROIBIDOS for c in nomes)
    assert sparse.issparse(pre.transform(xv))
    assert ("escala" in pre.named_transformers_["numericas"].named_steps) == (
        algoritmo == "regressao_logistica"
    )
    if algoritmo == "dummy":
        # Caso conhecido: a coluna positiva é a prevalência do treino, 1/5.
        np.testing.assert_allclose(probas[:, 1], .2)
        np.testing.assert_allclose(pipeline.named_steps["modelo"].class_prior_, [.8, .2])
    else:
        # Sinal conhecido do sintético: hora=16 (i=4) identifica a classe 1.
        treino_probas = pipeline.predict_proba(xt)[:, 1]
        assert treino_probas[yt.eq(1)].mean() > treino_probas[yt.eq(0)].mean()
    outro = criar_pipeline(cenario, algoritmo).fit(xt, yt)
    # RF paralela pode somar as mesmas árvores em outra ordem (erro ~1e-16).
    np.testing.assert_allclose(probas, outro.predict_proba(xv), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(pipeline.predict(xv), outro.predict(xv))
    if algoritmo == "random_forest":
        for a, b in zip(pipeline[-1].estimators_, outro[-1].estimators_, strict=True):
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
            np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
    assert_frame_equal(xt, xt_antes)
    assert_frame_equal(xv, xv_antes)
    assert_series_equal(yt, yt_antes)
    assert_frame_equal(base_sintetica, antes)


@pytest.mark.parametrize("cenario", ["A", "B"])
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_estatisticas_exclusivas_do_treino(base_sintetica, cenario, algoritmo):
    dev, teste = separar_desenvolvimento_teste(base_sintetica)
    corte = gerar_cortes_temporais(dev)[0]
    x, y = selecionar_xy(dev, cenario)
    xt, xv = x.loc[corte.treino].copy(), x.loc[corte.validacao].copy()
    # Mediana conhecida de longitude = 3; distribuição diferente na validação.
    xt["longitude"] = [np.nan, 1., 3., 9., np.nan] * 6
    xv["longitude"] = 1e9
    xv["uf"] = "SO_VALIDACAO"
    pipeline = criar_pipeline(cenario, algoritmo).fit(xt, y.loc[xt.index])
    pre = pipeline.named_steps["preprocessamento"]
    num = pre.named_transformers_["numericas"]
    tipos = identificar_tipos(cenario)
    pos = tipos["numericas"].index("longitude")
    assert num.named_steps["imputacao"].statistics_[pos] == 3.
    if algoritmo == "regressao_logistica":
        np.testing.assert_allclose(num.named_steps["escala"].mean_[pos], 3.8)
        np.testing.assert_array_equal(num.named_steps["escala"].n_samples_seen_, len(xt))
    encoder = pre.named_transformers_["categoricas"].named_steps["onehot"]
    uf_pos = tipos["categoricas"].index("uf")
    np.testing.assert_array_equal(encoder.categories_[uf_pos], ["observada"])
    stats = num.named_steps["imputacao"].statistics_.copy()
    categorias = [c.copy() for c in encoder.categories_]
    pipeline.predict(xv)
    probabilidades = pipeline.predict_proba(xv)
    np.testing.assert_array_equal(num.named_steps["imputacao"].statistics_, stats)
    for antes, depois in zip(categorias, encoder.categories_, strict=True):
        np.testing.assert_array_equal(antes, depois)
    # Sentinelas em 2025 não alteram o desenvolvimento, o fit nem as probabilidades.
    alterada = base_sintetica.copy(deep=True)
    alterada.loc[teste.index, "longitude"] = -1e12
    alterada.loc[teste.index, "acidente_fatal"] = 1
    outro_dev, _ = separar_desenvolvimento_teste(alterada)
    assert_frame_equal(dev, outro_dev)
    outro_x, outro_y = selecionar_xy(outro_dev, cenario)
    outro_xt = outro_x.loc[corte.treino].copy()
    outro_xt["longitude"] = xt["longitude"]
    outro = criar_pipeline(cenario, algoritmo).fit(outro_xt, outro_y.loc[corte.treino])
    np.testing.assert_allclose(probabilidades, outro.predict_proba(xv), rtol=0, atol=1e-14)


@pytest.mark.parametrize("cenario", ["A", "B"])
def test_nulos_texto_binarios_e_numerica_totalmente_ausente(base_sintetica, cenario):
    x, _ = selecionar_xy(base_sintetica.loc[base_sintetica["ano"].eq(2024)], cenario)
    x.loc[x.index[:3], "uf"] = [None, pd.NA, np.nan]
    x["latitude"] = np.nan
    antes = x.copy(deep=True)
    pre = criar_preprocessamento(cenario)
    matriz = pre.fit_transform(x)
    assert sparse.issparse(matriz)
    assert np.isfinite(matriz.data).all()
    tipos = identificar_tipos(cenario)
    # A fatia binária é idêntica aos valores de entrada, sem One-Hot ou escala.
    binarios = matriz[:, pre.output_indices_["binarias"]].toarray()
    np.testing.assert_array_equal(binarios, x.loc[:, list(tipos["binarias"])])
    lat = pre.get_feature_names_out().tolist().index("numericas__latitude")
    np.testing.assert_array_equal(matriz[:, lat].toarray(), np.zeros((len(x), 1)))
    cat = pre.named_transformers_["categoricas"]
    encoder = cat.named_steps["onehot"]
    assert "__ausente__" in encoder.categories_[tipos["categoricas"].index("uf")]
    assert encoder.categories_[tipos["categoricas"].index("br")].tolist() == ["101"]
    inedita = x.iloc[:1].copy()
    inedita["br"] = "999"
    nomes = pre.get_feature_names_out().tolist()
    br_pos = [i for i, n in enumerate(nomes) if n.startswith("categoricas__br_")]
    assert pre.transform(inedita)[:, br_pos].nnz == 0
    assert pre.transform(inedita).shape[1] == matriz.shape[1]
    assert_frame_equal(x, antes)


@pytest.mark.parametrize("cenario", ["A", "B"])
def test_alvo_e_ano_extras_nunca_entram_na_matriz(base_sintetica, cenario):
    dev, _ = separar_desenvolvimento_teste(base_sintetica)
    x, y = selecionar_xy(dev, cenario)
    pre = criar_preprocessamento(cenario)
    esperado = pre.fit_transform(x)
    extra = x.assign(ano=2025, acidente_fatal=y)
    obtido = criar_preprocessamento(cenario).fit_transform(extra)
    np.testing.assert_array_equal(esperado.toarray(), obtido.toarray())
    with pytest.raises(ValueError):
        criar_preprocessamento(cenario).fit_transform(x.drop(columns="km"))


@pytest.mark.parametrize("cenario", ["A", "B"])
def test_categoria_inteiramente_ausente_e_matriz_pequena_densa(base_sintetica, cenario):
    dev, _ = separar_desenvolvimento_teste(base_sintetica)
    x, _ = selecionar_xy(dev, cenario)
    x["uf"] = pd.NA
    x["dia_semana"] = "constante"
    pre = criar_preprocessamento(cenario)
    matriz = pre.fit_transform(x)
    # Com uma única categoria por coluna, densidade estimada = 1 no sklearn.
    # Uma matriz pequena densa é válida; dados reais e categorias variadas são esparsos.
    assert not sparse.issparse(matriz)
    assert np.isfinite(matriz).all()
    novo = x.iloc[:2].copy()
    novo["uf"] = "inedita"
    assert pre.transform(novo).shape == (2, matriz.shape[1])


def test_erros_de_fabrica_e_instancias_independentes():
    with pytest.raises(ValueError, match="Cenário desconhecido"):
        criar_pipeline("C", "dummy")
    with pytest.raises(ValueError, match="Algoritmo desconhecido"):
        criar_pipeline("A", "inexistente")
    a, b = criar_pipeline("A", "random_forest"), criar_pipeline("A", "random_forest")
    a.named_steps["modelo"].set_params(max_depth=2)
    assert b.named_steps["modelo"].max_depth == 12


def test_xgboost_ausente_nao_impede_outros_algoritmos(monkeypatch):
    importar = builtins.__import__

    def sem_xgboost(nome, *args, **kwargs):
        if nome == "xgboost":
            raise ImportError("dependência ausente no teste")
        return importar(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sem_xgboost)
    with pytest.raises(ImportError, match="Instale xgboost>=2.0"):
        criar_estimador("xgboost")
    for algoritmo in ALGORITMOS[:-1]:
        assert criar_pipeline("A", algoritmo).named_steps["modelo"] is not None
