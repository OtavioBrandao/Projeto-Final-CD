"""Funções estatísticas da EDA (puras: recebem números/tabelas, devolvem números).

Ficam separadas do resto para serem **testadas isoladamente** contra valores de
livro (``tests/test_eda_p2.py``). Nada aqui lê arquivo ou desenha gráfico.

Escolhas e porquês (resumo para a banca):

* **Proporção com IC de Wilson**, não o IC "normal" (Wald): o Wald erra para
  proporções pequenas (nossa taxa é 7%) e para grupos pequenos (pedestre, UF).
* **Risco relativo (RR)** como efeito principal das comparações 2x2: é a medida
  que um gestor entende ("pista simples tem 2,1 vezes o risco de morte").
  Odds ratio só aparece onde é necessário (estratificação).
* **Qui-quadrado de independência + V de Cramér**: o teste diz *se* há
  associação; o V diz *quanto*. Com 145 mil linhas quase tudo é "significativo",
  então quem decide é o V.
* **Mann-Whitney** (e não teste t) para numéricas: as distribuições são
  assimétricas e com cauda longa (``veiculos`` vai até 82). Efeito: correlação
  rank-bisserial.
* **Cochran-Mantel-Haenszel**: testa o fator *dentro* de estratos (ex.: noite vs.
  dia só entre pistas simples, depois só entre duplas...). Separa causa raiz de
  confusão (o "efeito noite" poderia ser só "à noite há mais pista simples").
* **Holm**: corrige o p-valor para muitos testes ao mesmo tempo, controlando a
  chance de pelo menos um falso positivo na família inteira.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.p2_eda import configuracao as cfg


def _z(confianca: float) -> float:
    return float(stats.norm.ppf(1 - (1 - confianca) / 2))


# ---------------------------------------------------------------------------
# Proporções
# ---------------------------------------------------------------------------


def ic_wilson(k: int, n: int, confianca: float = cfg.CONFIANCA) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = _z(confianca)
    p = k / n
    denominador = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denominador
    margem = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominador
    # Nos extremos (k = 0 ou k = n) o limite é exatamente 0 ou 1; evita resíduo de ponto flutuante.
    inferior = 0.0 if k == 0 else max(0.0, centro - margem)
    superior = 1.0 if k == n else min(1.0, centro + margem)
    return (inferior, superior)


def taxa_por_grupo(df: pd.DataFrame, coluna: str | list[str], alvo: str = cfg.COLUNA_ALVO) -> pd.DataFrame:
    g = df.groupby(coluna, observed=True)[alvo].agg(n="size", n_fatais="sum").reset_index()
    ics = [ic_wilson(int(k), int(n)) for k, n in zip(g["n_fatais"], g["n"], strict=True)]
    g["taxa_pct"] = 100 * g["n_fatais"] / g["n"]
    g["ic_inf_pct"] = [100 * a for a, _ in ics]
    g["ic_sup_pct"] = [100 * b for _, b in ics]
    g["pct_dos_acidentes"] = 100 * g["n"] / g["n"].sum()
    g["pct_dos_fatais"] = 100 * g["n_fatais"] / g["n_fatais"].sum()
    return g


# ---------------------------------------------------------------------------
# Comparações 2x2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoRR:

    n_exp: int
    fatais_exp: int
    n_ref: int
    fatais_ref: int
    taxa_exp: float
    taxa_ref: float
    rr: float
    rr_inf: float
    rr_sup: float
    dif_pp: float
    dif_inf_pp: float
    dif_sup_pp: float
    qui2: float
    p_valor: float
    phi: float


def risco_relativo(
    fatais_exp: int, n_exp: int, fatais_ref: int, n_ref: int, confianca: float = cfg.CONFIANCA
) -> ResultadoRR:
    a, n1, c, n0 = int(fatais_exp), int(n_exp), int(fatais_ref), int(n_ref)
    if min(a, c) == 0 or min(n1, n0) == 0:
        raise ValueError("RR indefinido: grupo vazio ou sem acidentes fatais.")
    z = _z(confianca)
    p1, p0 = a / n1, c / n0
    rr = p1 / p0
    ep_log = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    dif = p1 - p0
    ep_dif = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    tabela = np.array([[a, n1 - a], [c, n0 - c]])
    qui2, p, _, _ = stats.chi2_contingency(tabela, correction=False)
    n = n1 + n0
    return ResultadoRR(
        n_exp=n1, fatais_exp=a, n_ref=n0, fatais_ref=c, taxa_exp=p1, taxa_ref=p0,
        rr=rr, rr_inf=math.exp(math.log(rr) - z * ep_log), rr_sup=math.exp(math.log(rr) + z * ep_log),
        dif_pp=100 * dif, dif_inf_pp=100 * (dif - z * ep_dif), dif_sup_pp=100 * (dif + z * ep_dif),
        qui2=float(qui2), p_valor=float(p), phi=math.sqrt(float(qui2) / n),
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Tabelas r x c
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoQui2:
    qui2: float
    gl: int
    p_valor: float
    v_cramer: float
    n: int
    n_categorias: int
    pct_celulas_esperado_menor_5: float


def agregar_raras(serie: pd.Series, n_minimo: int = cfg.N_MINIMO_GRUPO, rotulo: str = "Outras") -> pd.Series:
    contagem = serie.value_counts()
    raras = set(contagem[contagem < n_minimo].index)
    if not raras:
        return serie
    return serie.astype(object).where(~serie.isin(raras), rotulo)


def qui_quadrado(categoria: pd.Series, alvo: pd.Series) -> ResultadoQui2:
    tabela = pd.crosstab(categoria, alvo)
    qui2, p, gl, esperado = stats.chi2_contingency(tabela.to_numpy(), correction=False)
    n = int(tabela.to_numpy().sum())
    k = min(tabela.shape) - 1
    return ResultadoQui2(
        qui2=float(qui2), gl=int(gl), p_valor=float(p), v_cramer=math.sqrt(float(qui2) / (n * k)), n=n,
        n_categorias=int(tabela.shape[0]),
        pct_celulas_esperado_menor_5=float(100 * (esperado < 5).mean()),
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Numéricas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoMannWhitney:
    n_fatal: int
    n_nao_fatal: int
    mediana_fatal: float
    mediana_nao_fatal: float
    media_fatal: float
    media_nao_fatal: float
    u: float
    p_valor: float
    r_rank_bisserial: float


def mann_whitney(x_fatal: pd.Series, x_nao_fatal: pd.Series) -> ResultadoMannWhitney:
    a = x_fatal.dropna().to_numpy(dtype=float)
    b = x_nao_fatal.dropna().to_numpy(dtype=float)
    res = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    u = float(res.statistic)
    return ResultadoMannWhitney(
        n_fatal=len(a), n_nao_fatal=len(b),
        mediana_fatal=float(np.median(a)), mediana_nao_fatal=float(np.median(b)),
        media_fatal=float(a.mean()), media_nao_fatal=float(b.mean()),
        u=u, p_valor=float(res.pvalue), r_rank_bisserial=2 * u / (len(a) * len(b)) - 1,
    )  # fmt: skip


def normalidade(serie: pd.Series, n_amostra: int = cfg.TAMANHO_AMOSTRA_NORMALIDADE) -> dict[str, float]:
    x = serie.dropna()
    amostra = x.sample(n=min(n_amostra, len(x)), random_state=cfg.SEMENTE_ALEATORIA)
    return {
        "assimetria": float(stats.skew(x)),
        "curtose": float(stats.kurtosis(x)),
        "p_normalidade": float(stats.normaltest(amostra).pvalue),
    }


# ---------------------------------------------------------------------------
# Estratificação (Cochran-Mantel-Haenszel)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultadoCMH:

    rr_bruto: float
    rr_mh: float
    rr_mh_inf: float
    rr_mh_sup: float
    or_mh: float
    qui2_cmh: float
    p_valor: float
    n_estratos: int


def cochran_mantel_haenszel(
    tabelas: Sequence[tuple[int, int, int, int]], confianca: float = cfg.CONFIANCA
) -> ResultadoCMH:
    t = np.array([x for x in tabelas if (x[0] + x[1]) > 0 and (x[2] + x[3]) > 0], dtype=float)
    a, b, c, d = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    n1, n0 = a + b, c + d
    m1, m0 = a + c, b + d
    n = n1 + n0

    num_rr, den_rr = (a * n0 / n).sum(), (c * n1 / n).sum()
    rr_mh = num_rr / den_rr
    var_log_rr = ((n1 * n0 * m1 - a * c * n) / n**2).sum() / (num_rr * den_rr)
    z = _z(confianca)

    or_mh = (a * d / n).sum() / (b * c / n).sum()
    esperado = n1 * m1 / n
    variancia = n1 * n0 * m1 * m0 / (n**2 * (n - 1))
    qui2 = float((a - esperado).sum() ** 2 / variancia.sum())

    rr_bruto = (a.sum() / n1.sum()) / (c.sum() / n0.sum())
    return ResultadoCMH(
        rr_bruto=float(rr_bruto), rr_mh=float(rr_mh),
        rr_mh_inf=float(math.exp(math.log(rr_mh) - z * math.sqrt(var_log_rr))),
        rr_mh_sup=float(math.exp(math.log(rr_mh) + z * math.sqrt(var_log_rr))),
        or_mh=float(or_mh), qui2_cmh=qui2, p_valor=float(stats.chi2.sf(qui2, 1)), n_estratos=len(t),
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Múltiplos testes e interpretação
# ---------------------------------------------------------------------------


def holm(p_valores: Sequence[float]) -> list[float]:
    p = np.asarray(p_valores, dtype=float)
    m = len(p)
    ordem = np.argsort(p, kind="stable")
    ajustado = np.empty(m)
    maximo = 0.0
    for posicao, i in enumerate(ordem):
        maximo = max(maximo, (m - posicao) * p[i])
        ajustado[i] = min(1.0, maximo)
    return ajustado.tolist()


def magnitude_efeito(valor: float, limiares: tuple[float, float, float] = cfg.LIMIARES_EFEITO) -> str:
    v = abs(valor)
    if v < limiares[0]:
        return "desprezível"
    if v < limiares[1]:
        return "pequeno"
    if v < limiares[2]:
        return "médio"
    return "grande"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")
