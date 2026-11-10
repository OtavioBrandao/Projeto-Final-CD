'''Testes da EDA e Estatística — ``pytest tests/test_eda_p2.py``.

Duas partes:

1. **Funções estatísticas** contra valores calculados à mão / de livro. Se alguém mexer numa fórmula, o teste quebra.
2. **Pipeline da EDA** rodado numa pasta temporária sobre a base versionada
   (``data/processed``): as saídas existem, são determinísticas e todo número citado na seção do artigo existe nas tabelas.

Ao contrário dos testes primários, estes **não** precisam dos dados brutos.
'''

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest

from src.p2_eda import configuracao as cfg
from src.p2_eda import estatistica as est
from src.p2_eda import executar_eda
from src.p2_eda.configuracao import Caminhos

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "data" / "processed" / "acidentes_prf_modelagem.csv"

# ---------------------------------------------------------------------------
# 1. Funções estatísticas
# ---------------------------------------------------------------------------


def test_ic_wilson_valor_de_referencia() -> None:
    inf, sup = est.ic_wilson(10, 100)
    assert inf == pytest.approx(0.0552, abs=1e-4)
    assert sup == pytest.approx(0.1744, abs=1e-4)


def test_ic_wilson_extremos() -> None:
    inf, sup = est.ic_wilson(0, 50)
    assert inf == 0.0 and 0 < sup < 0.1
    assert all(math.isnan(v) for v in est.ic_wilson(0, 0))


def test_risco_relativo_conta_a_mao() -> None:
    r = est.risco_relativo(20, 100, 10, 100)
    assert r.rr == pytest.approx(2.0)
    ep = math.sqrt(1 / 20 - 1 / 100 + 1 / 10 - 1 / 100)
    assert r.rr_inf == pytest.approx(2 * math.exp(-1.959964 * ep), rel=1e-5)
    assert r.rr_sup == pytest.approx(2 * math.exp(1.959964 * ep), rel=1e-5)
    assert r.dif_pp == pytest.approx(10.0)


def test_risco_relativo_sem_fatais_falha() -> None:
    with pytest.raises(ValueError):
        est.risco_relativo(0, 100, 10, 100)


def test_v_cramer_associacao_perfeita_e_nula() -> None:
    cat = pd.Series(["a"] * 50 + ["b"] * 50)
    perfeito = est.qui_quadrado(cat, pd.Series([1] * 50 + [0] * 50))
    nulo = est.qui_quadrado(cat, pd.Series([1, 0] * 50))
    assert perfeito.v_cramer == pytest.approx(1.0)
    assert nulo.v_cramer == pytest.approx(0.0, abs=1e-12)


def test_mann_whitney_rank_bisserial() -> None:
    assert est.mann_whitney(pd.Series([5, 6, 7]), pd.Series([1, 2, 3])).r_rank_bisserial == pytest.approx(1.0)
    assert est.mann_whitney(pd.Series([1, 2, 3]), pd.Series([5, 6, 7])).r_rank_bisserial == pytest.approx(-1.0)


def test_cmh_um_estrato_igual_ao_bruto() -> None:
    a, b, c, d = 30, 70, 15, 85
    r = est.cochran_mantel_haenszel([(a, b, c, d)])
    assert r.rr_mh == pytest.approx(r.rr_bruto)
    n = a + b + c + d
    qui2 = est.risco_relativo(a, a + b, c, c + d).qui2
    assert r.qui2_cmh == pytest.approx(qui2 * (n - 1) / n)


def test_cmh_detecta_confusao() -> None:
    estrato_risco_alto = (40, 60, 4, 6)  # expostos concentrados aqui: 40% e 40%
    estrato_risco_baixo = (1, 9, 10, 90)  # 10% e 10%
    r = est.cochran_mantel_haenszel([estrato_risco_alto, estrato_risco_baixo])
    assert r.rr_bruto > 2
    assert r.rr_mh == pytest.approx(1.0)
    assert r.p_valor > 0.5


def test_holm_contra_definicao() -> None:
    assert est.holm([0.01, 0.04, 0.03, 0.2, 0.001]) == pytest.approx([0.04, 0.09, 0.09, 0.2, 0.005])


def test_magnitude_efeito() -> None:
    assert [est.magnitude_efeito(v) for v in (0.05, -0.2, 0.35, 0.8)] == ["desprezível", "pequeno", "médio", "grande"]


def test_agregar_raras() -> None:
    s = pd.Series(["A"] * 300 + ["B"] * 5 + ["C"] * 3)
    out = est.agregar_raras(s, n_minimo=200)
    assert set(out) == {"A", "Outras"} and (out == "Outras").sum() == 8


# ---------------------------------------------------------------------------
# 2. Pipeline completo (pasta temporária)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def execucao(tmp_path_factory: pytest.TempPathFactory) -> tuple[Caminhos, executar_eda.ResultadosEDA]:
    if not BASE.exists():
        pytest.skip("base de modelagem ausente")
    caminhos = Caminhos(raiz=tmp_path_factory.mktemp("eda"), entrada=BASE)
    return caminhos, executar_eda.executar(caminhos)


def _hash_pasta(pasta: Path, padrao: str) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pasta.glob(padrao))}


def test_saidas_existem(execucao) -> None:  # noqa: ANN001
    caminhos, res = execucao
    assert len(list(caminhos.figuras.glob("p2_fig*.png"))) == 8
    for nome in res.tabelas:
        assert (caminhos.tabelas / f"{nome}.csv").exists()
    for arq in (caminhos.readme_eda, caminhos.secao_md, caminhos.secao_tex):
        assert arq.exists() and arq.stat().st_size > 1000


def test_base_nao_alterada_e_totais_batem(execucao) -> None:  # noqa: ANN001
    _, res = execucao
    df = pd.read_csv(BASE)
    r = res.tabelas["p2_resumo_geral"].set_index("recorte")
    assert int(r.loc["Total", "n"]) == len(df)
    assert int(r.loc["Total", "n_fatais"]) == int(df[cfg.COLUNA_ALVO].sum())
    assert int(r.loc["2024", "n"]) + int(r.loc["2025", "n"]) == len(df)


def test_taxas_por_grupo_somam_o_total(execucao) -> None:  # noqa: ANN001
    _, res = execucao
    g = res.tabelas["p2_taxa_por_grupo"]
    tot = res.tabelas["p2_resumo_geral"].iloc[0]
    for _, parte in g.groupby("dimensao"):
        assert parte["n"].sum() == tot["n"]
        assert parte["n_fatais"].sum() == tot["n_fatais"]
        assert parte["pct_dos_fatais"].sum() == pytest.approx(100)


def test_holm_aplicado_a_familia_inteira(execucao) -> None:  # noqa: ANN001
    _, res = execucao
    nomes = ["p2_fatores_risco", "p2_testes_qui_quadrado", "p2_testes_mann_whitney", "p2_estratificadas_resumo"]
    total = 0
    for n in nomes:
        t = res.tabelas[n]
        assert (t["p_holm"] >= t["p_valor"] - 1e-15).all()
        total += len(t)
    param = res.tabelas["p2_parametros"].set_index("parametro")
    assert int(param.loc["n_testes_familia_holm", "valor"]) == total


def test_qui_quadrado_respeita_suposicao(execucao) -> None:  # noqa: ANN001
    _, res = execucao
    assert (res.tabelas["p2_testes_qui_quadrado"]["pct_celulas_esperado_menor_5"] == 0).all()


def test_ranking_nao_tem_alvo(execucao) -> None:  # noqa: ANN001
    _, res = execucao
    assert cfg.COLUNA_ALVO not in set(res.tabelas["p2_ranking_atributos"]["atributo"])


def test_numeros_da_secao_vem_das_tabelas(execucao) -> None:  # noqa: ANN001
    caminhos, res = execucao
    texto = caminhos.secao_md.read_text(encoding="utf-8")
    r = res.tabelas["p2_resumo_geral"].iloc[0]
    f = res.tabelas["p2_fatores_risco"].set_index("fator")
    fmt = lambda v, c=2: f"{v:.{c}f}".replace(".", ",")  # noqa: E731
    assert f"{fmt(r['taxa_pct'])}%" in texto
    assert fmt(f.loc["Pedestre envolvido", "rr"]) in texto
    assert fmt(f.loc["Pista simples (vs. dupla/múltipla)", "rr"]) in texto
    assert "[[" not in texto, "referência cruzada não resolvida"


def test_reprodutivel(execucao, tmp_path: Path) -> None:  # noqa: ANN001
    caminhos, _ = execucao
    outra = Caminhos(raiz=tmp_path, entrada=BASE)
    executar_eda.executar(outra)
    assert _hash_pasta(caminhos.tabelas, "*.csv") == _hash_pasta(outra.tabelas, "*.csv")
    assert caminhos.secao_md.read_text(encoding="utf-8") == outra.secao_md.read_text(encoding="utf-8")


def test_tex_sem_caractere_problematico(execucao) -> None:  # noqa: ANN001
    caminhos, _ = execucao
    for linha in caminhos.secao_tex.read_text(encoding="utf-8").splitlines():
        if linha.startswith("%"):
            continue  # comentário
        corpo = linha.removesuffix("{%")  # o % final do \resizebox é proposital
        for ruim in ("α", "χ", "×", "≥", "→"):
            assert ruim not in corpo, linha
        if not any(cmd in corpo for cmd in ("\\ref", "\\label", "\\includegraphics")):
            for i, c in enumerate(corpo):
                if c in "%_":
                    assert i > 0 and corpo[i - 1] == "\\", linha
