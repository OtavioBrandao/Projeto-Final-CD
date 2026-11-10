"""Análises da EDA: monta todas as tabelas (descritivas e de inferência).

Cada função recebe a base tratada e devolve um ``DataFrame`` pronto para ser
gravado em ``reports/tabelas``. Figuras e textos leem **estas** tabelas, então
nenhum número do artigo é digitado à mão.

Ordem lógica:

1. Descrever: tamanho, alvo, numéricas (tendência, dispersão, forma), categóricas.
2. Quando (H1, H5, H9): hora, dia da semana, mês, replicando 2024 x 2025.
3. Onde e em que via (H2, H4): pista, trecho, curva, região e UF.
4. Quem (H3): pedestre, ciclista, moto, caminhão, ônibus.
5. Por quê (H6, H7): clima, causa e tipo de acidente, em volume x gravidade.
6. Numéricas (H8): fatais x não fatais (Mann-Whitney).
7. Causa raiz x confusão: análises estratificadas (CMH).
8. Correção para múltiplos testes (Holm) e veredito por hipótese.
9. Para a Pessoa 3: ranking de associação de cada atributo com o alvo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.p2_eda import configuracao as cfg
from src.p2_eda import estatistica as est

logger = logging.getLogger(__name__)

ALVO = cfg.COLUNA_ALVO


def carregar_base(caminho) -> pd.DataFrame:  # noqa: ANN001 (Path ou str)
    df = pd.read_csv(caminho)
    usadas = {ALVO, cfg.COLUNA_ANO, "hora", "mes", "dia_semana", *cfg.COLUNAS_NUMERICAS}
    usadas |= {f[2] for f in cfg.FATORES_RISCO} | {c for _, c in cfg.TESTES_QUI_QUADRADO}
    faltando = sorted(usadas - set(df.columns))
    if faltando:
        raise ValueError(f"Colunas ausentes na base de modelagem: {faltando}")
    if not set(df[ALVO].unique()) <= {0, 1}:
        raise ValueError("O alvo deve ser binário (0/1).")
    logger.info("Base carregada: %d linhas x %d colunas", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 1. Descritiva
# ---------------------------------------------------------------------------


def resumo_geral(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for rotulo, parte in [("Total", df), *((str(a), g) for a, g in df.groupby(cfg.COLUNA_ANO))]:
        n, k = len(parte), int(parte[ALVO].sum())
        inf, sup = est.ic_wilson(k, n)
        linhas.append({"recorte": rotulo, "n": n, "n_fatais": k, "taxa_pct": 100 * k / n,
                       "ic_inf_pct": 100 * inf, "ic_sup_pct": 100 * sup})  # fmt: skip
    return pd.DataFrame(linhas)


def descritiva_numericas(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for c in cfg.COLUNAS_NUMERICAS:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        forma = est.normalidade(s)
        linhas.append({
            "variavel": c, "n_validos": int(s.notna().sum()), "n_nulos": int(s.isna().sum()),
            "pct_nulos": 100 * s.isna().mean(), "media": s.mean(), "mediana": s.median(),
            "desvio_padrao": s.std(), "cv_pct": 100 * s.std() / s.mean(), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "minimo": s.min(), "maximo": s.max(), **forma,
            "media_fatal": s[df[ALVO] == 1].mean(), "media_nao_fatal": s[df[ALVO] == 0].mean(),
            "mediana_fatal": s[df[ALVO] == 1].median(), "mediana_nao_fatal": s[df[ALVO] == 0].median(),
        })  # fmt: skip
    return pd.DataFrame(linhas)


def taxas_por_dimensao(df: pd.DataFrame) -> pd.DataFrame:
    dimensoes = [
        "faixa_horaria", "fase_dia", "dia_semana", "mes", "hora", "fim_de_semana", "eh_feriado",
        "tipo_pista", "area_urbana", "via_curva", "regiao", "uf", "grupo_causa", "tipo_acidente",
        "condicao_metereologica", "envolve_pedestre", "envolve_bicicleta", "envolve_moto",
        "envolve_caminhao", "envolve_onibus", "ano",
    ]  # fmt: skip
    partes = []
    for d in dimensoes:
        t = est.taxa_por_grupo(df, d).rename(columns={d: "grupo"})
        t.insert(0, "dimensao", d)
        t["grupo"] = t["grupo"].astype(str)
        t["n_pequeno"] = t["n"] < cfg.N_MINIMO_GRUPO
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def correlacoes_spearman(df: pd.DataFrame) -> pd.DataFrame:
    colunas = [*cfg.COLUNAS_NUMERICAS, "hora", ALVO]
    m = df[colunas].corr(method="spearman")
    return m.reset_index(names="variavel")


# ---------------------------------------------------------------------------
# 2 a 5. Fatores de risco (2x2), qui-quadrado e tempo
# ---------------------------------------------------------------------------


def _mascaras(df: pd.DataFrame, coluna: str, expostos: tuple, referencia: tuple | None) -> tuple[pd.Series, pd.Series]:
    #Máscaras booleanas de expostos e referência para um fator.
    exp = df[coluna].isin(expostos)
    ref = ~exp if referencia is None else df[coluna].isin(referencia)
    return exp, ref


def _rr(df: pd.DataFrame, exp: pd.Series, ref: pd.Series) -> est.ResultadoRR:
    #Atalho: RR de duas máscaras.
    return est.risco_relativo(int(df.loc[exp, ALVO].sum()), int(exp.sum()),
                              int(df.loc[ref, ALVO].sum()), int(ref.sum()))  # fmt: skip


def fatores_de_risco(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for hip, nome, coluna, expostos, referencia in cfg.FATORES_RISCO:
        exp, ref = _mascaras(df, coluna, expostos, referencia)
        r = _rr(df, exp, ref)
        por_ano = {}
        for ano, g in df.groupby(cfg.COLUNA_ANO):
            e, f = _mascaras(g, coluna, expostos, referencia)
            por_ano[int(ano)] = _rr(g, e, f)
        sentido = np.sign(np.log(r.rr))
        replica = all(
            np.sign(np.log(x.rr)) == sentido and (x.rr_inf > 1 or x.rr_sup < 1) for x in por_ano.values()
        )
        linha = {
            "hipotese": hip, "fator": nome, "coluna": coluna,
            "n_expostos": r.n_exp, "fatais_expostos": r.fatais_exp, "taxa_expostos_pct": 100 * r.taxa_exp,
            "n_referencia": r.n_ref, "fatais_referencia": r.fatais_ref, "taxa_referencia_pct": 100 * r.taxa_ref,
            "rr": r.rr, "rr_ic_inf": r.rr_inf, "rr_ic_sup": r.rr_sup,
            "dif_pp": r.dif_pp, "dif_ic_inf_pp": r.dif_inf_pp, "dif_ic_sup_pp": r.dif_sup_pp,
            "qui2": r.qui2, "p_valor": r.p_valor, "phi": r.phi, "magnitude_phi": est.magnitude_efeito(r.phi),
            "pct_dos_fatais_expostos": 100 * r.fatais_exp / int(df[ALVO].sum()),
        }  # fmt: skip
        for ano, x in por_ano.items():
            linha[f"rr_{ano}"] = x.rr
            linha[f"rr_{ano}_ic_inf"] = x.rr_inf
            linha[f"rr_{ano}_ic_sup"] = x.rr_sup
        linha["replica_nos_dois_anos"] = replica
        linhas.append(linha)
    return pd.DataFrame(linhas)


def testes_qui_quadrado(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for hip, coluna in cfg.TESTES_QUI_QUADRADO:
        serie = est.agregar_raras(df[coluna])
        # Se nem juntas as raras chegam ao mínimo (ex.: Neve + Granizo = 4 acidentes), o grupo
        # "Outras" sai do teste: a frequência esperada ficaria < 5 e o qui-quadrado não vale.
        contagem = serie.value_counts()
        fora = serie.isin(contagem[contagem < cfg.N_MINIMO_GRUPO].index)
        serie, alvo = serie[~fora], df.loc[~fora, ALVO]
        r = est.qui_quadrado(serie, alvo)
        taxas = est.taxa_por_grupo(df.loc[~fora].assign(**{coluna: serie}), coluna)
        maior = taxas.loc[taxas["taxa_pct"].idxmax()]
        menor = taxas.loc[taxas["taxa_pct"].idxmin()]
        linhas.append({
            "hipotese": hip, "variavel": coluna, "n": r.n, "n_excluidos_raros": int(fora.sum()),
            "n_categorias": r.n_categorias,
            "qui2": r.qui2, "gl": r.gl, "p_valor": r.p_valor, "v_cramer": r.v_cramer,
            "magnitude": est.magnitude_efeito(r.v_cramer),
            "pct_celulas_esperado_menor_5": r.pct_celulas_esperado_menor_5,
            "grupo_maior_taxa": str(maior[coluna]), "maior_taxa_pct": maior["taxa_pct"],
            "grupo_menor_taxa": str(menor[coluna]), "menor_taxa_pct": menor["taxa_pct"],
        })  # fmt: skip
    return pd.DataFrame(linhas)


def tempo_por_ano(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    por_ano = est.taxa_por_grupo(df, [coluna, cfg.COLUNA_ANO])
    total = est.taxa_por_grupo(df, coluna).assign(**{cfg.COLUNA_ANO: "Total"})
    por_ano[cfg.COLUNA_ANO] = por_ano[cfg.COLUNA_ANO].astype(str)
    return pd.concat([por_ano, total], ignore_index=True)


def semana_x_faixa(df: pd.DataFrame) -> pd.DataFrame:
    t = est.taxa_por_grupo(df, ["dia_semana", "faixa_horaria"])
    t["ordem_dia"] = t["dia_semana"].map({d: i for i, d in enumerate(cfg.ORDEM_DIA_SEMANA)})
    t["ordem_faixa"] = t["faixa_horaria"].map({f: i for i, f in enumerate(cfg.ORDEM_FAIXA_HORARIA)})
    return t.sort_values(["ordem_dia", "ordem_faixa"]).drop(columns=["ordem_dia", "ordem_faixa"])


def tipo_acidente_por_pista(df: pd.DataFrame) -> pd.DataFrame:
    t = est.taxa_por_grupo(df, ["tipo_pista", "tipo_acidente"])
    t["pct_dos_acidentes_da_pista"] = 100 * t["n"] / t.groupby("tipo_pista")["n"].transform("sum")
    t["pct_dos_fatais_da_pista"] = 100 * t["n_fatais"] / t.groupby("tipo_pista")["n_fatais"].transform("sum")
    return t.sort_values(["tipo_pista", "n_fatais"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Numéricas
# ---------------------------------------------------------------------------


def testes_mann_whitney(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    fatal = df[ALVO] == 1
    for c in cfg.COLUNAS_NUMERICAS_TESTE:
        r = est.mann_whitney(df.loc[fatal, c], df.loc[~fatal, c])
        linhas.append({
            "hipotese": "H8", "variavel": c, "n_fatal": r.n_fatal, "n_nao_fatal": r.n_nao_fatal,
            "mediana_fatal": r.mediana_fatal, "mediana_nao_fatal": r.mediana_nao_fatal,
            "media_fatal": r.media_fatal, "media_nao_fatal": r.media_nao_fatal,
            "u": r.u, "p_valor": r.p_valor, "r_rank_bisserial": r.r_rank_bisserial,
            "magnitude": est.magnitude_efeito(r.r_rank_bisserial),
        })  # fmt: skip
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 7. Estratificação
# ---------------------------------------------------------------------------


def analises_estratificadas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resumo, detalhe = [], []
    for hip, nome, coluna, expostos, referencia, estrato in cfg.ANALISES_ESTRATIFICADAS:
        exp, ref = _mascaras(df, coluna, expostos, referencia)
        tabelas = []
        for valor, g in df[exp | ref].groupby(estrato):
            e, f = exp.loc[g.index], ref.loc[g.index]
            a, n1 = int(g.loc[e, ALVO].sum()), int(e.sum())
            c, n0 = int(g.loc[f, ALVO].sum()), int(f.sum())
            if min(n1, n0) == 0:
                continue
            tabelas.append((a, n1 - a, c, n0 - c))
            if min(a, c) > 0:
                r = est.risco_relativo(a, n1, c, n0)
                detalhe.append({"hipotese": hip, "analise": nome, "estrato": f"{estrato}={valor}",
                                "n_expostos": n1, "taxa_expostos_pct": 100 * r.taxa_exp,
                                "n_referencia": n0, "taxa_referencia_pct": 100 * r.taxa_ref,
                                "rr": r.rr, "rr_ic_inf": r.rr_inf, "rr_ic_sup": r.rr_sup})  # fmt: skip
        r = est.cochran_mantel_haenszel(tabelas)
        resumo.append({
            "hipotese": hip, "analise": nome, "fator": coluna, "estrato": estrato, "n_estratos": r.n_estratos,
            "rr_bruto": r.rr_bruto, "rr_ajustado_mh": r.rr_mh, "rr_mh_ic_inf": r.rr_mh_inf,
            "rr_mh_ic_sup": r.rr_mh_sup, "or_mh": r.or_mh, "qui2_cmh": r.qui2_cmh, "p_valor": r.p_valor,
            "pct_efeito_mantido": 100 * np.log(r.rr_mh) / np.log(r.rr_bruto),
        })  # fmt: skip
    return pd.DataFrame(resumo), pd.DataFrame(detalhe)


# ---------------------------------------------------------------------------
# 8. Holm + veredito
# ---------------------------------------------------------------------------


def aplicar_holm(tabelas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    todos = np.concatenate([t["p_valor"].to_numpy() for t in tabelas.values()])
    ajustados = est.holm(todos)
    saida, i = {}, 0
    for nome, t in tabelas.items():
        t = t.copy()
        t["p_holm"] = ajustados[i : i + len(t)]
        t["significativo"] = t["p_holm"] < cfg.ALFA
        saida[nome] = t
        i += len(t)
    return saida


def veredito_hipoteses(fatores: pd.DataFrame, qui2: pd.DataFrame, mw: pd.DataFrame,
                       estrat: pd.DataFrame) -> pd.DataFrame:  # fmt: skip
    linhas = []
    for hip, texto in cfg.HIPOTESES.items():
        evid: list[str] = []
        f = fatores[fatores["hipotese"] == hip]
        if hip in {"H7", "H9"}:
            q = qui2[qui2["hipotese"] == hip]
            relevantes = q[q["significativo"] & (q["v_cramer"] >= cfg.LIMIARES_EFEITO[0])]
            sig = q[q["significativo"]]
            if len(relevantes) == len(q):
                v = "confirmada"
            elif len(sig) > 0 and len(relevantes) == 0:
                v = "significativa, efeito desprezível"
            elif len(relevantes) > 0:
                v = "parcial"
            else:
                v = "rejeitada"
            evid = [f"{r.variavel}: V={r.v_cramer:.3f}" for r in q.itertuples()]
        elif hip == "H8":
            relevantes = mw[mw["significativo"] & (mw["r_rank_bisserial"].abs() >= cfg.LIMIARES_EFEITO[0])]
            v = "confirmada" if len(relevantes) == len(mw) else (
                "parcial" if len(relevantes) else "significativa, efeito desprezível")
            evid = [f"{r.variavel}: r={r.r_rank_bisserial:+.3f}" for r in mw.itertuples()]
        else:
            apoia = f["significativo"] & (f["rr"] > 1) & f["replica_nos_dois_anos"]
            oposto = f["significativo"] & (f["rr"] < 1) & f["replica_nos_dois_anos"]
            if apoia.all():
                v = "confirmada"
            elif apoia.any():
                v = "parcial"
            elif oposto.all():
                v = "rejeitada (efeito oposto)"
            else:
                v = "rejeitada"
            evid = [f"{r.fator}: RR={r.rr:.2f}" for r in f.itertuples()]
        e = estrat[estrat["hipotese"] == hip]
        evid += [f"{r.analise}: RR ajustado={r.rr_ajustado_mh:.2f}" for r in e.itertuples()]
        linhas.append({"hipotese": hip, "enunciado": texto, "veredito": v, "evidencias": "; ".join(evid)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 9. Ranking de atributos 
# ---------------------------------------------------------------------------

#: Observação por atributo para a modelagem (o ranking é estatístico; isto é contexto).
_NOTAS_ATRIBUTOS: dict[str, str] = {
    "tipo_acidente": "Pós-evento (definido pela PRF no atendimento): ótimo para explicar, não para alertar antes.",
    "grupo_causa": "Pós-evento (definido pela PRF no atendimento): ótimo para explicar, não para alertar antes.",
    "ano": "Não usar como atributo: reservado para validação temporal (2024 treino, 2025 teste).",
    "idade_media_condutores": "Sensível: usar com cautela e auditar equidade. Imputar no Pipeline.",
    "pct_condutores_homens": "Sensível: usar com cautela e auditar equidade. Imputar no Pipeline.",
    "idade_media_veiculos": "Imputar no Pipeline (nulos).",
    "uf": "Alta cardinalidade (27): considerar target encoding dentro do CV ou usar 'regiao'.",
    "br": "Alta cardinalidade: target encoding dentro do CV, ou deixar de fora.",
    "latitude": "Associação vem da geografia (Norte/Nordeste); redundante com uf/regiao.",
    "longitude": "Associação vem da geografia (Norte/Nordeste); redundante com uf/regiao.",
    "fase_dia": "Redundante com faixa_horaria/hora; escolher uma das três.",
    "hora": "Relação não linear (pico na madrugada): usar faixa_horaria ou codificação cíclica.",
}


def ranking_atributos(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for c in df.columns:
        if c == ALVO:
            continue
        s = df[c]
        if pd.api.types.is_float_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.nunique() > 30):
            faixas = pd.qcut(s, cfg.QUANTIS_RANKING, duplicates="drop")
            if faixas.nunique() < 2:  # massa num valor só (ex.: 100% homens): compara "no máximo" x "abaixo"
                faixas = pd.Series(np.where(s >= s.max(), "maximo", "abaixo_do_maximo"), index=df.index)
                faixas = faixas.where(s.notna())
            s = pd.Series(faixas, index=df.index).astype(object)
        s = s.astype(object).where(s.notna(), "Nulo").astype(str)  # nulo vira categoria (não some do teste)
        s = est.agregar_raras(s)
        r = est.qui_quadrado(s, df[ALVO])
        linhas.append({"atributo": c, "v_cramer": r.v_cramer, "magnitude": est.magnitude_efeito(r.v_cramer),
                       "n_categorias_teste": r.n_categorias, "observacao": _NOTAS_ATRIBUTOS.get(c, "")})  # fmt: skip
    t = pd.DataFrame(linhas).sort_values("v_cramer", ascending=False).reset_index(drop=True)
    t.insert(0, "posicao", range(1, len(t) + 1))
    return t
