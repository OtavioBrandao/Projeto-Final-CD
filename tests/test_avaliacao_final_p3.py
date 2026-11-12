"""Etapa 04: nenhum teste comportamental acessa resultados reais de 2025."""

import json
from pathlib import Path
import subprocess
import sys

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
    m, previsoes, registros = ef.executar_avaliacao(base, tmp_path)
    assert chamadas == [("A", "xgboost"), ("B", "xgboost"), ("B", "dummy")]
    assert len({id(p) for p in pipelines}) == 3
    assert ef.LIMIARES == {"A": .2066718339920044, "B": .101788230240345}
    with pytest.raises(TypeError):
        ef.LIMIARES["A"] = .5
    teste = base.loc[base.ano.eq(2025)]
    for registro, pipeline in zip(registros, pipelines, strict=True):
        g = previsoes.loc[previsoes.modelo.eq(registro["modelo"])]
        assert g.linha_original.tolist() == teste.index.tolist()
        assert g.y_real.tolist() == teste.acidente_fatal.tolist()
        assert g.previsao.tolist() == g.probabilidade.ge(registro["limiar"]).astype(int).tolist()
        prep = pipeline[0]
        assert prep.named_transformers_["numericas"]["imputacao"].statistics_[0] == 2.5
        assert all("novo_2025" not in cats for cats in
                   prep.named_transformers_["categoricas"]["onehot"].categories_)
        recarregado = joblib.load(tmp_path / f"p3_{registro['modelo']}.joblib")
        x, _ = selecionar_xy(teste, registro["cenario"])
        np.testing.assert_allclose(extrair_probabilidade_positiva(recarregado, x), g.probabilidade)
        if registro["algoritmo"] == "xgboost":
            assert recarregado[-1].n_estimators == 100 and recarregado[-1].max_depth == 4
            assert recarregado[-1].learning_rate == .1
            assert recarregado[-1].scale_pos_weight is None
            assert len(prep.get_feature_names_out()) == prep.transform(x).shape[1]
    dummy = m.loc[m.modelo.eq("dummy")].iloc[0]
    assert dummy.ap == pytest.approx(1/6) and dummy.roc_auc == .5
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


def test_relatorios_sinteticos_e_referencias_publicas(base, tmp_path, monkeypatch):
    saida = tmp_path / "final"
    saida.mkdir()
    m, p, modelos = ef.executar_avaliacao(base, tmp_path / "models")
    meta = dict(csv=ev.referencia_publica(cfg.CSV_MODELAGEM), csv_sha256="sintetico",
                git_head="teste", modelos=modelos, status="concluido")
    ef.gerar_relatorios(m, p, meta, saida, cfg.RAIZ_REPOSITORIO / "reports")
    ef.gravar_json(saida / "manifesto.json", meta)
    for arquivo in ("README.md", "manifesto.json", "comparacao_2024_2025.csv", "calibracao_2025.csv"):
        assert (saida / arquivo).stat().st_size > 100
    assert len(list((saida / "figuras").glob("*.png"))) == 5
    texto = (saida / "manifesto.json").read_text(encoding="utf-8")
    assert "C:" not in texto and "Users" not in texto and "NaN" not in texto
    assert ev.referencia_publica(cfg.CSV_MODELAGEM) == "data/processed/acidentes_prf_modelagem.csv"
    assert ev.referencia_publica(tmp_path / "externo.csv") == "externo.csv"
    comp = pd.read_csv(saida / "comparacao_2024_2025.csv")
    assert np.allclose(comp.ap_delta_2025_menos_media_2024,
                       comp.ap_teste_2025 - comp.ap_media_validacao_2024_05)
    ef.gravar_json(tmp_path / "nan.json", {"indefinida": np.nan})
    assert json.loads((tmp_path / "nan.json").read_text()) == {"indefinida": None}
    m.to_csv(saida / "metricas_2025.csv", index=False)
    p.to_csv(saida / "previsoes_2025.csv.gz", index=False)
    hashes = {arquivo: ev.sha256(arquivo) for arquivo in
              (saida / "metricas_2025.csv", saida / "previsoes_2025.csv.gz")}
    def proibido(*args):
        pytest.fail("Somente relatórios não pode treinar/carregar modelos.")
    monkeypatch.setattr(ef, "criar_pipeline", proibido)
    monkeypatch.setattr(ef.joblib, "load", proibido)
    monkeypatch.setattr(sys, "argv", ["final", "--somente-relatorios", "--saida", str(saida)])
    ef.main()
    assert hashes == {arquivo: ev.sha256(arquivo) for arquivo in hashes}


def test_reproduzir_publicados_2024_sem_fit_e_sem_modificar_originais(tmp_path, monkeypatch):
    import shutil
    fonte = cfg.RAIZ_REPOSITORIO / "reports"
    hashes = {p: ev.sha256(p) for pasta in ("data", "src/p1_engenharia_dados", "src/p2_eda")
              for p in (cfg.RAIZ_REPOSITORIO / pasta).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    # Copia exclusivamente artefatos necessários; manifesto publicado sem chave csv.
    shutil.copytree(fonte / "tabelas", tmp_path / "tabelas")
    meta = json.loads((fonte / "p3_execucao_2024.json").read_text(encoding="utf-8"))
    meta.pop("csv", None)
    ef.gravar_json(tmp_path / "p3_execucao_2024.json", meta)
    tabelas = {p: ev.sha256(p) for p in (tmp_path / "tabelas").glob("*")}
    def proibido(*args):
        pytest.fail("Reprodução não pode ajustar modelo.")
    monkeypatch.setattr(ev, "criar_pipeline", proibido)
    monkeypatch.setattr(sys, "argv", ["validacao", "--somente-relatorios", "--saida", str(tmp_path)])
    ev.main()
    assert tabelas == {p: ev.sha256(p) for p in tabelas}
    assert hashes == {p: ev.sha256(p) for p in hashes}
    texto = (tmp_path / "README_modelagem.md").read_text(encoding="utf-8")
    assert "data/processed/acidentes_prf_modelagem.csv" in texto
    assert "C:" not in texto and "Users" not in texto


def test_rejeita_particoes_invalidas_antes_de_fit(base, tmp_path, monkeypatch):
    def proibido(*args):
        pytest.fail("Dados inválidos não podem criar modelo.")
    monkeypatch.setattr(ef, "criar_pipeline", proibido)
    with pytest.raises(ValueError):
        ef.executar_avaliacao(base.loc[base.ano.eq(2024)], tmp_path)
    base.loc[base.ano.eq(2024), "acidente_fatal"] = 0
    with pytest.raises(ValueError, match="ambas"):
        ef.executar_avaliacao(base, tmp_path)
