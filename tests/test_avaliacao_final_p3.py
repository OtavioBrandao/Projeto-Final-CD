"""Treinamento final, persistência e relatórios com dados sintéticos."""

import subprocess
import sys

import hashlib

import joblib
import numpy as np
import pandas as pd
import pytest

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem import executar_avaliacao_final as ef
from src.p3_modelagem import executar_validacao as ev
from src.p3_modelagem.avaliacao import calcular_metricas, extrair_probabilidade_positiva
from src.p3_modelagem.dados import selecionar_xy


@pytest.fixture
def base():
    linhas = []
    for ano in (2025, 2024):
        for mes in range(1, 13):
            for i in range(6):
                r = dict.fromkeys(cfg.COLUNAS_ESPERADAS, 0)
                r.update(dict.fromkeys(cfg.COLUNAS_CATEGORICAS, "novo_2025" if ano == 2025 else "treino"))
                r.update(ano=ano, mes=mes, hora=i, km=10000. if ano == 2025 else float(i),
                         latitude=-9., longitude=-35., veiculos=1,
                         idade_media_veiculos=float(i), acidente_fatal=int(i == 5))
                linhas.append(r)
    b = pd.DataFrame(linhas, index=pd.Index(range(300, 1308, 7), name="linha_original"))
    return b.iloc[::-1].copy()


def test_temporal_preprocessamento_limiares_indices_e_persistencia(base, tmp_path, monkeypatch):
    chamadas, pipelines = [], []
    original = ef.criar_pipeline

    def fabrica(cenario, algoritmo):
        p = original(cenario, algoritmo)
        pipelines.append(p)
        fit = p.fit

        def ajustar(x, y):
            assert base.loc[x.index, "ano"].eq(2024).all()
            assert x.index.equals(y.index)
            assert len(x) == base.ano.eq(2024).sum()
            chamadas.append((cenario, algoritmo))
            return fit(x, y)

        # Usamos wrapper temporário que se remove antes de joblib.dump.
        def fit_unico(x, y):
            del p.fit
            return ajustar(x, y)
        p.fit = fit_unico
        return p

    monkeypatch.setattr(ef, "criar_pipeline", fabrica)
    m, previsoes = ef.executar_avaliacao(base, tmp_path)
    assert chamadas == [("A", "xgboost"), ("B", "xgboost"), ("B", "dummy")]
    assert len({id(p) for p in pipelines}) == 3
    assert ef.LIMIARES == {"A": .2066718339920044, "B": .101788230240345}
    with pytest.raises(TypeError):
        ef.LIMIARES["A"] = .5
    teste = base.loc[base.ano.eq(2025)]
    for (nome, cenario), pipeline in zip((("xgboost_A", "A"), ("xgboost_B", "B"), ("dummy", "B")), pipelines, strict=True):
        limiar = ef.LIMIARES[cenario] if nome != "dummy" else .5
        g = previsoes.loc[previsoes.modelo.eq(nome)]
        assert g.linha_original.tolist() == teste.index.tolist()
        assert g.y_real.tolist() == teste.acidente_fatal.tolist()
        assert g.previsao.tolist() == g.probabilidade.ge(limiar).astype(int).tolist()
        prep = pipeline[0]
        assert prep.named_transformers_["numericas"]["imputacao"].statistics_[0] == 2.5
        assert all("novo_2025" not in cats for cats in
                   prep.named_transformers_["categoricas"]["onehot"].categories_)
        if nome != "dummy":
            recarregado = ef.carregar_pipeline(cenario, tmp_path)
            x, _ = selecionar_xy(teste, cenario)
            np.testing.assert_allclose(ef.prever(recarregado, teste, cenario).probabilidade, g.probabilidade)
            novas = teste.loc[:, list(cfg.FEATURES_POR_CENARIO[cenario])[::-1]].copy()
            novas["coluna_extra"] = "ignorada"
            antes = novas.copy(deep=True)
            saida = ef.prever(recarregado, novas, cenario)
            np.testing.assert_allclose(saida.probabilidade, g.probabilidade)
            np.testing.assert_array_equal(saida.previsao, g.previsao)
            assert saida.index.equals(novas.index)
            pd.testing.assert_frame_equal(novas, antes)
            assert recarregado[-1].n_estimators == 100 and recarregado[-1].max_depth == 4
            assert recarregado[-1].learning_rate == .1
            assert recarregado[-1].scale_pos_weight is None
            assert len(prep.get_feature_names_out()) == prep.transform(x).shape[1]
    dummy = m.loc[m.modelo.eq("dummy")].iloc[0]
    assert dummy.ap == pytest.approx(1/6) and dummy.roc_auc == .5
    assert not (tmp_path / "p3_dummy.joblib").exists()
    assert len(list(tmp_path.glob("*.joblib"))) == 2
    assert len(m) == 5 and previsoes.ano.eq(2025).all()
    assert not previsoes.duplicated(["modelo", "linha_original"]).any()
    # Novo processo compatível consegue importar a função textual do pipeline.
    saida = subprocess.check_output([sys.executable, "-c",
        "import joblib,sys; p=joblib.load(sys.argv[1]); print(p.classes_.tolist())",
        str(tmp_path / "p3_xgboost_A.joblib")], cwd=cfg.RAIZ_REPOSITORIO, text=True)
    assert "[0, 1]" in saida
    with pytest.raises(FileExistsError):
        ef.executar_avaliacao(base, tmp_path)


def test_metricas_conhecidas_e_classe_positiva():
    y = pd.Series([0, 0, 1, 1], index=[8, 2, 9, 1])
    p = pd.Series([.1, .4, .35, .8], index=y.index)
    m = calcular_metricas(y, p, .35)
    assert m["ap"] == pytest.approx(5/6)
    assert m["roc_auc"] == .75 and m["f1"] == .8
    assert [m[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 0, 2]
    assert m["brier"] == pytest.approx(.158125)
    class Invertido:
        classes_ = np.array([1, 0])
        def predict_proba(self, x):
            return np.column_stack([p, 1-p])
    assert extrair_probabilidade_positiva(Invertido(), pd.DataFrame(index=y.index)).equals(p.rename("probabilidade"))


@pytest.fixture
def reports_2024(tmp_path):
    destino = tmp_path / "validacao"
    (destino / "tabelas").mkdir(parents=True)
    resumos, limiares = [], []
    for cenario in ("A", "B"):
        linha = dict(cenario=cenario, algoritmo="xgboost", variante="padrao")
        resumos.append({**linha, **{f"{m}_media": .25 for m in
                                    ("ap", "roc_auc", "brier", "log_loss", "precision", "recall", "f1")}})
        limiares.append({**linha, "tipo_limiar": "candidato_f1_ajuste",
                        "limiar": ef.LIMIARES[cenario], "precision": .3, "recall": .4, "f1": .35})
    pd.DataFrame(resumos).to_csv(destino / "tabelas/p3_resumo_validacao_2024.csv", index=False)
    pd.DataFrame(limiares).to_csv(destino / "tabelas/p3_comparacao_limiares.csv", index=False)
    return destino


def test_relatorios_leem_csvs_sem_treinar_ou_carregar_modelos(base, tmp_path, monkeypatch, reports_2024):
    saida = tmp_path / "final"
    saida.mkdir()
    m, p = ef.executar_avaliacao(base, tmp_path / "models")
    ef.gerar_relatorios(m, p, saida, reports_2024)
    m.to_csv(saida / "metricas_2025.csv", index=False)
    p.to_csv(saida / "previsoes_2025.csv.gz", index=False)
    arquivos = list(saida.glob("*.csv*"))
    hashes = {a: hashlib.sha256(a.read_bytes()).hexdigest() for a in arquivos}
    def proibido(*args):
        pytest.fail("Somente relatórios não pode treinar/carregar modelos.")
    monkeypatch.setattr(ef, "criar_pipeline", proibido)
    monkeypatch.setattr(ef.joblib, "load", proibido)
    monkeypatch.setattr(sys, "argv", ["final", "--somente-relatorios", "--saida", str(saida),
                                    "--reports-2024", str(reports_2024)])
    ef.main()
    assert hashes == {a: hashlib.sha256(a.read_bytes()).hexdigest() for a in arquivos}
    assert len(list((saida / "figuras").glob("*.png"))) == 5
    assert not list(saida.glob("*.json"))
    comp = pd.read_csv(saida / "comparacao_2024_2025.csv")
    assert np.allclose(comp.ap_delta_2025_menos_media_2024,
                       comp.ap_teste_2025 - comp.ap_media_validacao_2024_05)


def test_relatorios_2024_independem_de_metadados(base, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "ALGORITMOS", ("dummy",))
    m, p = ev.executar_experimentos(base.loc[base.ano.eq(2024)], ponderados=False)
    ev.salvar_relatorios(m, p, tmp_path)
    arquivos = list((tmp_path / "tabelas").glob("*"))
    hashes = {a: hashlib.sha256(a.read_bytes()).hexdigest() for a in arquivos}
    def proibido(*args):
        pytest.fail("Relatórios não podem ajustar modelos.")
    monkeypatch.setattr(ev, "criar_pipeline", proibido)
    monkeypatch.setattr(ev, "carregar_base", proibido)
    monkeypatch.setattr(sys, "argv", ["validacao", "--somente-relatorios", "--saida", str(tmp_path)])
    ev.main()
    assert hashes == {a: hashlib.sha256(a.read_bytes()).hexdigest() for a in arquivos}
    assert not list(tmp_path.glob("*.json"))


def test_rejeita_particoes_invalidas_antes_de_fit(base, tmp_path, monkeypatch):
    def proibido(*args):
        pytest.fail("Dados inválidos não podem criar modelo.")
    monkeypatch.setattr(ef, "criar_pipeline", proibido)
    with pytest.raises(ValueError):
        ef.executar_avaliacao(base.loc[base.ano.eq(2024)], tmp_path)
    base.loc[base.ano.eq(2024), "acidente_fatal"] = 0
    with pytest.raises(ValueError, match="ambas"):
        ef.executar_avaliacao(base, tmp_path)


@pytest.mark.parametrize("erro", ["cenario", "feature", "indice", "colunas"])
def test_prever_rejeita_entrada_incompleta_ou_ambigua(base, erro):
    novas = base.loc[:, list(cfg.FEATURES_B)].copy()
    cenario = "B"
    if erro == "cenario":
        cenario = "C"
    elif erro == "feature":
        novas = novas.drop(columns="km")
    elif erro == "indice":
        novas.index = [0] * len(novas)
    else:
        novas = pd.concat([novas, novas[["km"]]], axis=1)
    with pytest.raises(ValueError):
        ef.prever(None, novas, cenario)
