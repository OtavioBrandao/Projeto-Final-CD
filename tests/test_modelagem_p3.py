"""Contrato e cortes com exemplos conhecidos; integração com o CSV versionado."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.dados import (
    carregar_base, gerar_cortes_temporais, selecionar_xy,
    separar_desenvolvimento_teste, validar_esquema,
)


@pytest.fixture
def sintetica() -> pd.DataFrame:
    # Ordem intencionalmente não cronológica e índice não sequencial:
    # detecta ordenação, reset_index e confusão entre rótulo e posição.
    linhas = []
    for mes in (12, 1, 7, 6, 8, 2, 9, 3, 10, 4, 11, 5):
        for ano in (2025, 2024):
            for repeticao in (0, 1):
                linhas.append({
                    "dia_semana": "segunda-feira", "uf": "AL", "br": "BR-101",
                    "km": float(mes), "tipo_acidente": "Colisão traseira",
                    "fase_dia": "Pleno dia", "sentido_via": "Crescente",
                    "condicao_metereologica": "Céu Claro", "tipo_pista": "Simples",
                    "veiculos": 2, "latitude": -9.6, "longitude": -35.7,
                    "envolve_moto": 0, "envolve_caminhao": 0, "envolve_onibus": 0,
                    "envolve_bicicleta": 0, "envolve_pedestre": 0,
                    "idade_media_condutores": np.nan, "pct_condutores_homens": 100.,
                    "idade_media_veiculos": np.nan, "hora": 12,
                    "faixa_horaria": "Tarde", "mes": mes, "ano": ano,
                    "fim_de_semana": 0, "eh_feriado": 0, "regiao": "Nordeste",
                    "grupo_causa": "Outras", "via_reta": 1, "via_curva": 0,
                    "via_aclive": 0, "via_declive": 0, "via_intersecao": 0,
                    "via_rotatoria_retorno": 0, "via_obras_desvio": 0,
                    "via_ponte_viaduto_tunel": 0, "area_urbana": 0,
                    "veiculos_outlier": 0, "km_outlier": 0,
                    "idade_media_condutores_outlier": 0,
                    "idade_media_veiculos_outlier": 0,
                    "acidente_fatal": int(mes in (2, 7, 11) and repeticao == 0),
                })
    return pd.DataFrame(linhas, index=pd.Index(range(100, 436, 7), name="origem"))


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return carregar_base()


def test_csv_real_carrega_e_confirma_auditoria(real):
    assert real.shape == (145342, 42)
    assert real["ano"].value_counts().to_dict() == {2024: 73006, 2025: 72336}
    assert real["acidente_fatal"].value_counts().to_dict() == {0: 135102, 1: 10240}
    assert real["acidente_fatal"].dtype.kind in "iu"
    assert set(real["acidente_fatal"].unique()) == {0, 1}
    dicionario = pd.read_csv(cfg.CSV_MODELAGEM.with_name("dicionario_de_dados.csv"))
    assert set(real.columns) == set(dicionario["coluna"])
    assert real.isna().sum().loc[lambda s: s.gt(0)].to_dict() == {
        "idade_media_condutores": 7009, "pct_condutores_homens": 5037,
        "idade_media_veiculos": 5545,
    }


def test_listas_autorizadas_por_disponibilidade():
    # Classificação independente das listas da implementação.
    antecipadas = {
        "dia_semana", "uf", "br", "km", "latitude", "longitude", "hora",
        "faixa_horaria", "mes", "fim_de_semana", "eh_feriado", "regiao",
    }
    descritivas = {
        "tipo_acidente", "grupo_causa", "fase_dia", "sentido_via",
        "condicao_metereologica", "tipo_pista", "veiculos", "envolve_moto",
        "envolve_caminhao", "envolve_onibus", "envolve_bicicleta", "envolve_pedestre",
        "idade_media_veiculos", "via_reta", "via_curva", "via_aclive", "via_declive",
        "via_intersecao", "via_rotatoria_retorno", "via_obras_desvio",
        "via_ponte_viaduto_tunel", "area_urbana",
    }
    assert set(cfg.FEATURES_B) == antecipadas
    assert set(cfg.FEATURES_A) == antecipadas | descritivas
    assert len(cfg.FEATURES_A) == len(set(cfg.FEATURES_A)) == 34
    assert len(cfg.FEATURES_B) == len(set(cfg.FEATURES_B)) == 12
    for features in (cfg.FEATURES_A, cfg.FEATURES_B):
        assert not set(features) & {
            "acidente_fatal", "ano", "mortos", "feridos", "feridos_leves",
            "feridos_graves", "ilesos", "ignorados", "pessoas", "classificacao_acidente",
            "estado_fisico", "contagens_inconsistentes", "id", "id_veiculo",
            "regional", "delegacia", "uop", "ano_arquivo", "idade_media_condutores",
            "pct_condutores_homens", "veiculos_outlier", "km_outlier",
            "idade_media_condutores_outlier", "idade_media_veiculos_outlier",
        }
        assert not any(c.endswith("_outlier") for c in features)


def test_cortes_conhecidos_sem_sobreposicao_ou_reamostragem(sintetica):
    antes = sintetica.copy(deep=True)
    dev, teste = separar_desenvolvimento_teste(sintetica)
    assert len(dev) == len(teste) == 24
    assert dev["acidente_fatal"].value_counts().to_dict() == {0: 21, 1: 3}
    assert teste["acidente_fatal"].value_counts().to_dict() == {0: 21, 1: 3}
    assert dev.index.intersection(teste.index).empty
    assert set(dev.index) | set(teste.index) == set(sintetica.index)
    assert_frame_equal(dev, sintetica.loc[sintetica["ano"].eq(2024)])
    assert_frame_equal(teste, sintetica.loc[sintetica["ano"].eq(2025)])
    esperados = [
        (set(range(1, 7)), {7, 8}, 12, 4, 1, 1),
        (set(range(1, 9)), {9, 10}, 16, 4, 2, 0),
        (set(range(1, 11)), {11, 12}, 20, 4, 2, 1),
    ]
    cortes = gerar_cortes_temporais(dev)
    assert [c.rodada for c in cortes] == [1, 2, 3]
    for corte, (meses_t, meses_v, nt, nv, ft, fv) in zip(cortes, esperados, strict=True):
        treino, validacao = dev.loc[corte.treino], dev.loc[corte.validacao]
        assert set(treino["mes"]) == meses_t
        assert set(validacao["mes"]) == meses_v
        assert len(treino) == nt and len(validacao) == nv
        assert treino["acidente_fatal"].sum() == ft
        assert validacao["acidente_fatal"].sum() == fv
        assert corte.treino.intersection(corte.validacao).empty
        assert corte.treino.intersection(teste.index).empty
        assert corte.validacao.intersection(teste.index).empty
        assert treino["mes"].max() < validacao["mes"].min()
        assert_index_equal(corte.treino, dev.index[dev["mes"].isin(meses_t)])
        assert_index_equal(corte.validacao, dev.index[dev["mes"].isin(meses_v)])
        # Os mesmos rótulos selecionam X/y para todos os cenários.
        for cenario in ("A", "B"):
            x, y = selecionar_xy(dev, cenario)
            assert_index_equal(x.loc[corte.treino].index, treino.index)
            assert_series_equal(y.loc[corte.validacao], validacao["acidente_fatal"])
    assert_frame_equal(sintetica, antes)


def test_selecao_preserva_linhas_ordem_nulos_e_copias(sintetica):
    antes = sintetica.copy(deep=True)
    for cenario, features in (("A", cfg.FEATURES_A), ("B", cfg.FEATURES_B)):
        x, y = selecionar_xy(sintetica, cenario)
        assert tuple(x.columns) == features
        assert_index_equal(x.index, sintetica.index)
        assert_series_equal(y, sintetica["acidente_fatal"])
        for coluna in features:
            assert_series_equal(x[coluna], antes[coluna], check_dtype=coluna != "br")
        x.iloc[0, 0] = "alterado"
        y.iloc[0] = 1 - y.iloc[0]
    assert_frame_equal(sintetica, antes)


def test_determinismo_e_teste_final_independente(sintetica):
    dev, teste = separar_desenvolvimento_teste(sintetica)
    original = gerar_cortes_temporais(dev)
    alterada = sintetica.copy(deep=True)
    alterada.loc[teste.index, "acidente_fatal"] = 1
    alterada.loc[teste.index, "veiculos"] = 80
    outro_dev, _ = separar_desenvolvimento_teste(alterada)
    assert_frame_equal(dev, outro_dev)
    for a, b in zip(original, gerar_cortes_temporais(outro_dev), strict=True):
        assert_index_equal(a.treino, b.treino)
        assert_index_equal(a.validacao, b.validacao)
    for cenario in ("A", "B"):
        xa, ya = selecionar_xy(dev, cenario)
        xb, yb = selecionar_xy(outro_dev, cenario)
        assert_frame_equal(xa, xb)
        assert_series_equal(ya, yb)


def test_cortes_reais_e_alvos_preservados(real):
    dev, teste = separar_desenvolvimento_teste(real)
    assert teste["acidente_fatal"].value_counts().to_dict() == {0: 67232, 1: 5104}
    assert dev["acidente_fatal"].value_counts().to_dict() == {0: 67870, 1: 5136}
    for corte, (nt, nv, ft, fv) in zip(
        gerar_cortes_temporais(dev),
        [(35160, 12537, 2452, 909), (47697, 12599, 3361, 869), (60296, 12710, 4230, 906)],
        strict=True,
    ):
        assert (len(corte.treino), len(corte.validacao)) == (nt, nv)
        assert dev.loc[corte.treino, "acidente_fatal"].sum() == ft
        assert dev.loc[corte.validacao, "acidente_fatal"].sum() == fv
        assert corte.treino.intersection(corte.validacao).empty
        for cenario in ("A", "B"):
            x, y = selecionar_xy(dev, cenario)
            assert_index_equal(x.index, dev.index)
            assert_series_equal(y.loc[corte.validacao], dev.loc[corte.validacao, "acidente_fatal"])
            xt, yt = selecionar_xy(teste, cenario)
            assert_index_equal(xt.index, teste.index)
            assert_index_equal(yt.index, teste.index)
            assert x.index.intersection(xt.index).empty


@pytest.mark.parametrize("coluna", ["acidente_fatal", "ano", "mes", "br", "grupo_causa", "km_outlier"])
def test_coluna_ausente_erro_claro(sintetica, coluna):
    with pytest.raises(ValueError, match=f"obrigatórias ausentes=.*{coluna}"):
        validar_esquema(sintetica.drop(columns=coluna))


@pytest.mark.parametrize("extra", ["mortos", "id", "novo_proxy", "outro_outlier"])
def test_esquema_novo_ou_proxy_recusado(sintetica, extra):
    with pytest.raises(ValueError, match=f"extras=.*{extra}"):
        selecionar_xy(sintetica.assign(**{extra: 0}), "A")


@pytest.mark.parametrize("valor", [2, -1, 0.5, np.nan, "1"])
def test_alvo_invalido_recusado(sintetica, valor):
    sintetica["acidente_fatal"] = valor
    with pytest.raises(ValueError, match="acidente_fatal"):
        validar_esquema(sintetica)


@pytest.mark.parametrize("coluna,valor", [
    ("ano", 2026), ("ano", 2024.5), ("ano", np.nan),
    ("mes", 0), ("mes", 13), ("mes", 7.5), ("mes", np.nan),
    ("hora", 24), ("hora", -1), ("hora", 6.5), ("hora", np.nan),
    ("km", "cem"), ("km", np.inf), ("uf", 12), ("uf", None),
    ("envolve_moto", 2), ("via_curva", -1), ("veiculos_outlier", 3),
])
def test_dominios_e_tipos_invalidos(sintetica, coluna, valor):
    sintetica[coluna] = valor
    with pytest.raises(ValueError):
        validar_esquema(sintetica)


def test_indices_e_colunas_duplicados_recusados(sintetica):
    with pytest.raises(ValueError, match="índice.*único"):
        validar_esquema(pd.concat([sintetica, sintetica.iloc[:1]]))
    sintetica.index = [None, *sintetica.index[1:]]
    with pytest.raises(ValueError, match="índice.*sem nulos"):
        validar_esquema(sintetica)
    with pytest.raises(ValueError, match="Colunas duplicadas"):
        validar_esquema(pd.concat([sintetica, sintetica[["br"]]], axis=1))


def test_erros_de_uso_e_periodo_vazio(sintetica):
    with pytest.raises(ValueError, match="Cenário desconhecido"):
        selecionar_xy(sintetica, "C")
    with pytest.raises(ValueError, match="Base vazia"):
        validar_esquema(sintetica.iloc[:0])
    dev, _ = separar_desenvolvimento_teste(sintetica)
    with pytest.raises(ValueError, match="exige registros de 2024.*2025"):
        separar_desenvolvimento_teste(dev)
    with pytest.raises(ValueError, match="exclusivamente.*2024"):
        gerar_cortes_temporais(sintetica)
    with pytest.raises(ValueError, match="Rodada 1.*sem registros"):
        gerar_cortes_temporais(dev.loc[~dev["mes"].isin([7, 8])])


def test_br_numerica_continua_categorica(sintetica):
    sintetica["br"] = 101
    for cenario in ("A", "B"):
        x, _ = selecionar_xy(sintetica, cenario)
        assert x["br"].tolist() == ["101"] * len(sintetica)
        assert not pd.api.types.is_numeric_dtype(x["br"])
    assert sintetica["br"].eq(101).all()


def test_preditor_proibido_recusado_mesmo_se_configuracao_alterada(sintetica, monkeypatch):
    monkeypatch.setitem(cfg.FEATURES_POR_CENARIO, "A", (*cfg.FEATURES_A, "acidente_fatal"))
    with pytest.raises(ValueError, match="Preditores proibidos.*acidente_fatal"):
        selecionar_xy(sintetica, "A")


def test_csv_indice_estavel_deterministico_sem_escrita(tmp_path, sintetica):
    caminho = tmp_path / "base.csv"
    sintetica.to_csv(caminho, index=False)
    hash_antes = hashlib.sha256(caminho.read_bytes()).hexdigest()
    a, b = carregar_base(caminho), carregar_base(caminho)
    assert_frame_equal(a, b)
    assert_index_equal(a.index, pd.RangeIndex(48, name="linha_original"))
    assert a["mes"].tolist() == sintetica["mes"].tolist()
    assert "linha_original" not in selecionar_xy(a, "A")[0].columns
    assert hashlib.sha256(caminho.read_bytes()).hexdigest() == hash_antes


def test_csv_invalido_valida_antes_de_entregar(tmp_path, sintetica):
    caminho = tmp_path / "invalido.csv"
    sintetica.drop(columns="mes").to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="obrigatórias ausentes=.*mes"):
        carregar_base(caminho)
