"""Métricas e validação temporal com dados sintéticos."""

import hashlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem import executar_validacao as ev
from src.p3_modelagem.avaliacao import (
    CHAVES, analisar_previsoes, calcular_metricas, consolidar_resultados,
    extrair_probabilidade_positiva, selecionar_limiar, validar_alinhamento,
    validar_previsoes,
)
from src.p3_modelagem.dados import gerar_cortes_temporais
from src.p3_modelagem.modelos import criar_pipeline


def sha256(caminho):
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def series(y=(0, 0, 1, 1), p=(.1, .4, .35, .8)):
    indice = pd.Index([41, 9, 800, 2], name="linha_original")
    return pd.Series(y, index=indice), pd.Series(p, index=indice)


def test_metricas_analiticas_e_contagens():
    y, p = series()
    m = calcular_metricas(y, p)
    assert m["ap"] == pytest.approx(5 / 6)  # 1/2 * 1 + 1/2 * 2/3
    assert m["roc_auc"] == .75  # três dos quatro pares positivo/negativo
    assert [m[k] for k in ("tp", "fp", "tn", "fn")] == [1, 0, 2, 1]
    assert m["precision"] == 1 and m["recall"] == .5
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["brier"] == pytest.approx(.158125)
    assert m["log_loss"] == pytest.approx(-np.log(.9 * .6 * .35 * .8) / 4)
    assert m["prevalencia"] == .5 and m["indefinidas"] == ""
    outro = calcular_metricas(y, p, .35)
    assert [outro[k] for k in ("tp", "fp", "tn", "fn")] == [2, 1, 1, 0]
    assert outro["f1"] == .8  # igualdade ao limiar conta como positivo


def test_constantes_e_nenhum_positivo_previsto():
    y, p = series(p=(.07,) * 4)
    m = calcular_metricas(y, p)
    assert m["ap"] == .5 and m["roc_auc"] == .5
    assert np.isnan(m["precision"])
    assert m["recall"] == m["f1"] == 0
    assert [m[k] for k in ("tp", "fp", "tn", "fn")] == [0, 0, 2, 2]
    assert "nenhum positivo previsto" in m["indefinidas"]
    assert selecionar_limiar(y, p) == .07


@pytest.mark.parametrize("alvo", [(0,) * 4, (1,) * 4])
def test_classe_real_unica_explicita(alvo):
    y, p = series(y=alvo)
    m = calcular_metricas(y, p)
    assert np.isnan(m["roc_auc"])
    assert "apenas uma classe" in m["indefinidas"]
    assert np.isfinite(m["brier"]) and np.isfinite(m["log_loss"])
    if not sum(alvo):
        assert np.isnan(m["ap"]) and np.isnan(m["recall"])
        with pytest.raises(ValueError, match="sem positivos"):
            selecionar_limiar(y, p)
        assert np.isnan(calcular_metricas(y, p * 0)["f1"])
    else:
        assert m["ap"] == 1


def test_extracao_consulta_classes_e_preserva_indice():
    class Modelo:
        classes_ = np.array([1, 0])

        def predict_proba(self, x):
            return np.array([[.8, .2], [.1, .9]])

    x = pd.DataFrame({"x": [1, 2]}, index=[57, 3])
    p = extrair_probabilidade_positiva(Modelo(), x)
    assert p.index.equals(x.index)
    assert p.tolist() == [.8, .1]
    Modelo.classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match="classe positiva"):
        extrair_probabilidade_positiva(Modelo(), x)


@pytest.mark.parametrize("matriz", [[[.1, .2]], [[-1., 2.]], [[np.nan, .5]], [[.2, .8, 0]]])
def test_probabilidade_malformada(matriz):
    class Modelo:
        classes_ = np.array([0, 1])

        def predict_proba(self, x):
            return matriz

    with pytest.raises(ValueError):
        extrair_probabilidade_positiva(Modelo(), pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize("erro", ["ordem", "duplicado", "nulo", "infinito", "fora", "alvo"])
def test_alinhamento_rejeita_corrupcao(erro):
    y, p = series()
    if erro == "ordem":
        p = p.iloc[::-1]
    elif erro == "duplicado":
        y.index = p.index = [1, 1, 2, 3]
    elif erro == "nulo":
        y.index = p.index = [1, None, 2, 3]
    elif erro == "infinito":
        p.iloc[0] = np.inf
    elif erro == "fora":
        p.iloc[0] = 1.1
    else:
        y.iloc[0] = 2
    with pytest.raises(ValueError):
        validar_alinhamento(y, p)


@pytest.mark.parametrize("limiar", [-.1, 1.1, np.nan])
def test_limiar_invalido(limiar):
    with pytest.raises(ValueError):
        calcular_metricas(*series(), limiar)


def test_limiar_exato_pequeno_e_empate_deterministico():
    y, p = series(y=(1, 0, 0, 1), p=(.004, .003, .002, .001))
    # F1 = 2/3 tanto selecionando só o primeiro como selecionando todos.
    assert selecionar_limiar(y, p) == .004
    assert selecionar_limiar(y.iloc[::-1], p.iloc[::-1]) == .004
    assert selecionar_limiar(*series()) == .35


def test_pesos_so_do_treino_e_dummy_intacto():
    treino = pd.Series([0, 0, 0, 1])
    xgb = criar_pipeline("A", "xgboost")
    assert ev.configurar_variante(xgb, "xgboost", "ponderado", treino) == {"scale_pos_weight": 3.}
    assert xgb[-1].scale_pos_weight == 3.
    for algoritmo in ("regressao_logistica", "random_forest"):
        pipeline = criar_pipeline("B", algoritmo)
        ev.configurar_variante(pipeline, algoritmo, "ponderado", treino)
        assert pipeline[-1].class_weight == "balanced"
    dummy = criar_pipeline("B", "dummy")
    assert ev.configurar_variante(dummy, "dummy", "padrao", treino) == {}
    assert dummy[-1].strategy == "prior"
    with pytest.raises(ValueError, match="Dummy"):
        ev.configurar_variante(dummy, "dummy", "ponderado", treino)


@pytest.fixture
def desenvolvimento():
    linhas = []
    for mes in range(1, 13):
        for i in range(6):
            linha = dict.fromkeys(cfg.COLUNAS_ESPERADAS, 0)
            linha.update(dict.fromkeys(cfg.COLUNAS_CATEGORICAS, "categoria"))
            linha.update(ano=2024, mes=mes, br="101", hora=i * 3, km=float(i),
                         latitude=-9., longitude=-35., veiculos=1,
                         idade_media_veiculos=float(i), acidente_fatal=int(i == 5))
            linhas.append(linha)
    # Ordem não temporal e rótulos que não são posições.
    base = pd.DataFrame(linhas, index=pd.Index(range(200, 704, 7), name="linha_original"))
    return base.iloc[::-1].copy()


def test_executor_reutiliza_cortes_alinhamento_pesos_e_sem_vazamento(desenvolvimento, monkeypatch):
    antes = desenvolvimento.copy(deep=True)
    cortes = gerar_cortes_temporais(desenvolvimento)
    chamadas, modelos, ajustes = [], [], []
    original = ev.criar_pipeline

    def gerar(base):
        chamadas.append(base.index.copy())
        return cortes

    def fabrica(cenario, algoritmo):
        pipeline = original(cenario, algoritmo)
        # Registra os rótulos efetivamente recebidos pelo fit e predict_proba.
        fit, predict = pipeline.fit, pipeline.predict_proba

        def ajustar(x, y):
            assert x.index.equals(y.index)
            ajustes.append((cenario, algoritmo, x.index.copy(), y.copy(), pipeline[-1].get_params()))
            return fit(x, y)

        def prever(x):
            treino = ajustes[-1][2]
            assert treino.intersection(x.index).empty
            assert desenvolvimento.loc[treino, "mes"].max() < desenvolvimento.loc[x.index, "mes"].min()
            return predict(x)

        pipeline.fit, pipeline.predict_proba = ajustar, prever
        modelos.append(pipeline)  # Apenas sintético: permite verificar identidade.
        return pipeline

    monkeypatch.setattr(ev, "gerar_cortes_temporais", gerar)
    monkeypatch.setattr(ev, "criar_pipeline", fabrica)
    resultados, previsoes = ev.executar_experimentos(desenvolvimento)
    assert len(chamadas) == 1 and len(modelos) == len({id(m) for m in modelos}) == 42
    assert len(resultados) == 42 and len(previsoes) == 14 * 36
    assert resultados.query("variante == 'padrao'").shape[0] == 24
    assert resultados.query("variante == 'ponderado'").shape[0] == 18
    assert previsoes.ano.eq(2024).all()
    for experimento, ajuste in zip(resultados.to_dict("records"), ajustes, strict=True):
        corte = cortes[experimento["rodada"] - 1]
        assert ajuste[2].equals(corte.treino)
        parte = previsoes.loc[np.logical_and.reduce([previsoes[k].eq(experimento[k]) for k in [*CHAVES, "rodada"]])]
        assert parte.linha_original.tolist() == corte.validacao.tolist()
        assert parte.y_real.tolist() == desenvolvimento.loc[corte.validacao, "acidente_fatal"].tolist()
        if experimento["variante"] == "ponderado" and experimento["algoritmo"] == "xgboost":
            assert ajuste[4]["scale_pos_weight"] == 5.
    assert_frame_equal(antes, desenvolvimento)


def test_reprodutibilidade_com_semente_fixa(desenvolvimento, monkeypatch):
    monkeypatch.setattr(ev, "ALGORITMOS", ("dummy", "regressao_logistica", "xgboost", "random_forest"))
    a, pa = ev.executar_experimentos(desenvolvimento, ponderados=False)
    b, pb = ev.executar_experimentos(desenvolvimento, ponderados=False)
    assert_frame_equal(pa, pb, atol=1e-14, rtol=0)
    assert_frame_equal(a.drop(columns=["tempo_treino_s", "tempo_inferencia_s"]),
                       b.drop(columns=["tempo_treino_s", "tempo_inferencia_s"]), atol=1e-14, rtol=0)


def test_executor_rejeita_2025_antes_de_criar_modelo(desenvolvimento, monkeypatch):
    desenvolvimento.loc[desenvolvimento.index[0], "ano"] = 2025

    def proibido(*args):
        pytest.fail("A fábrica não pode ser chamada com dados de 2025.")

    monkeypatch.setattr(ev, "criar_pipeline", proibido)
    with pytest.raises(ValueError, match="2024"):
        ev.executar_experimentos(desenvolvimento)


def test_guardas_rejeitam_cortes_adulterados(desenvolvimento):
    from src.p3_modelagem.dados import CorteTemporal
    cortes = list(gerar_cortes_temporais(desenvolvimento))
    cortes[0] = CorteTemporal(1, cortes[0].treino.append(cortes[0].validacao[:1]), cortes[0].validacao)
    with pytest.raises(ValueError, match="vazamento"):
        ev.validar_cortes(desenvolvimento, cortes)
    cortes = list(gerar_cortes_temporais(desenvolvimento))
    cortes[0] = CorteTemporal(1, cortes[0].treino[:-1], cortes[0].validacao)
    with pytest.raises(ValueError, match="protocolo"):
        ev.validar_cortes(desenvolvimento, cortes)


def test_falhas_interrompem_o_experimento(desenvolvimento, monkeypatch):
    original = ev.criar_pipeline

    def fabrica(cenario, algoritmo):
        if cenario == "A":
            raise RuntimeError("falha sintética")
        return original(cenario, algoritmo)

    monkeypatch.setattr(ev, "ALGORITMOS", ("dummy",))
    monkeypatch.setattr(ev, "criar_pipeline", fabrica)
    with pytest.raises(RuntimeError, match="falha sintética"):
        ev.executar_experimentos(desenvolvimento, ponderados=False)


def test_resumo_distingue_media_e_global():
    # Cortes com AP conhecidos 1 e 1/3, tamanhos 2 e 3.
    linhas, metricas = [], []
    for rodada, ys, ps in ((1, [0, 1], [.1, .9]), (2, [1, 0, 0], [.2, .8, .7])):
        y, p = pd.Series(ys), pd.Series(ps)
        identidade = dict(cenario="A", algoritmo="dummy", variante="padrao")
        metricas.append({**identidade, "rodada": rodada, "tempo_treino_s": 2., "tempo_inferencia_s": 1., **calcular_metricas(y, p)})
        for i, (yy, pp) in enumerate(zip(ys, ps, strict=True)):
            linhas.append({**identidade, "rodada": rodada, "mes": 7 if rodada == 1 else 9,
                           "ano": 2024, "linha_original": rodada * 100 + i, "y_real": yy, "probabilidade": pp})
    resumo = consolidar_resultados(pd.DataFrame(metricas)).iloc[0]
    assert resumo.ap_media == pytest.approx(2 / 3)
    assert resumo.ap_dp == pytest.approx(1 / 3)
    assert resumo.tempo_treino_total_s == 4.
    globais, limiares, _ = analisar_previsoes(pd.DataFrame(linhas))
    assert globais.ap.iloc[0] == .75  # (1 + 2/4)/2
    assert globais.ap.iloc[0] != resumo.ap_media
    assert len(limiares) == 2


def test_arquivos_de_saida_e_protecao_p1_p2(desenvolvimento, monkeypatch, tmp_path):
    protegidos = [p for pasta in ("data", "src/p1_engenharia_dados", "src/p2_eda")
                  for p in (cfg.RAIZ_REPOSITORIO / pasta).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    antes = {str(p): sha256(p) for p in protegidos}
    monkeypatch.setattr(ev, "ALGORITMOS", ("dummy",))
    resultados, previsoes = ev.executar_experimentos(desenvolvimento, ponderados=False)
    ev.salvar_relatorios(resultados, previsoes, tmp_path)
    recarregado = pd.read_csv(tmp_path / "tabelas/p3_previsoes_validacao_2024.csv.gz")
    assert_frame_equal(recarregado, previsoes, check_dtype=False, atol=1e-14)
    assert not list(tmp_path.glob("*.json"))
    for nome in ("p3_comparacao_ap", "p3_precision_recall_validacao", "p3_calibracao_validacao"):
        assert (tmp_path / f"figuras/{nome}.png").stat().st_size > 1000
    for nome in ("p3_validacao_2024", "p3_resumo_validacao_2024", "p3_metricas_globais_2024",
                 "p3_comparacao_limiares", "p3_calibracao_2024"):
        tabela = pd.read_csv(tmp_path / f"tabelas/{nome}.csv")
        assert set(CHAVES) <= set(tabela)
        assert not tabela.empty
    assert antes == {str(p): sha256(p) for p in protegidos}
    for coluna, valor in (("ano", 2025), ("mes", 1), ("rodada", 4)):
        corrompida = previsoes.copy()
        corrompida.loc[0, coluna] = valor
        with pytest.raises(ValueError):
            validar_previsoes(corrompida)
    with pytest.raises(ValueError, match="duplicada"):
        validar_previsoes(pd.concat([previsoes, previsoes.iloc[:1]]))


def test_entrada_principal_isola_2024(desenvolvimento, monkeypatch, tmp_path):
    reservado = desenvolvimento.assign(ano=2025, acidente_fatal=1, km=1e12)
    reservado.index = reservado.index + 10000
    base = pd.concat([reservado, desenvolvimento])
    csv = tmp_path / "fonte.csv"
    base.to_csv(csv, index=False)
    monkeypatch.setattr(ev, "ALGORITMOS", ("dummy",))
    monkeypatch.setattr("sys.argv", ["validacao", "--csv", str(csv), "--saida", str(tmp_path / "reports")])
    ev.main()
    assert not list((tmp_path / "reports").glob("*.json"))
    p = pd.read_csv(tmp_path / "reports/tabelas/p3_previsoes_validacao_2024.csv.gz")
    # O carregador usa posição no CSV, preservando o deslocamento de 72 linhas.
    assert p.linha_original.between(72, 143).all()
    assert p.y_real.mean() == pytest.approx(1 / 6)
    assert p.probabilidade.eq(1 / 6).all() or np.allclose(p.probabilidade, 1 / 6)
    arquivos = list((tmp_path / "reports/tabelas").glob("*"))
    hashes_antes = {f.name: sha256(f) for f in arquivos}

    def proibido(*args):
        pytest.fail("Regenerar relatórios não pode ajustar pipelines.")

    monkeypatch.setattr(ev, "criar_pipeline", proibido)
    monkeypatch.setattr("sys.argv", ["validacao", "--somente-relatorios", "--saida", str(tmp_path / "reports")])
    ev.main()
    assert hashes_antes == {f.name: sha256(f) for f in arquivos}


@pytest.mark.parametrize("nome", ["p3_validacao_2024.csv", "p3_previsoes_validacao_2024.csv.gz"])
def test_validacao_recusa_sobrescrever_antes_de_carregar_dados(tmp_path, monkeypatch, nome):
    tabelas = tmp_path / "tabelas"
    tabelas.mkdir()
    existente = tabelas / nome
    existente.write_bytes(b"resultado existente")
    antes = sha256(existente)

    def proibido(*args):
        pytest.fail("A proteção deve ocorrer antes de carregar dados ou treinar.")

    monkeypatch.setattr(ev, "carregar_base", proibido)
    monkeypatch.setattr(ev, "executar_experimentos", proibido)
    monkeypatch.setattr("sys.argv", ["validacao", "--saida", str(tmp_path)])
    with pytest.raises(FileExistsError, match="--saida"):
        ev.main()
    assert sha256(existente) == antes
