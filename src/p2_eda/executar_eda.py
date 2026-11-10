"""Roda a EDA inteira: tabelas -> figuras -> textos (README da EDA e seção do artigo).

Uso (na raiz do repositório)::

    python -m src.p2_eda.executar_eda

Lê ``data/processed/acidentes_prf_modelagem.csv`` (Pessoa 1) e escreve em
``reports/``. É determinístico: rodar duas vezes gera as mesmas tabelas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.p2_eda import analises as an
from src.p2_eda import configuracao as cfg
from src.p2_eda import documentacao, graficos
from src.p2_eda.configuracao import Caminhos

logger = logging.getLogger(__name__)


@dataclass
class ResultadosEDA:
    tabelas: dict[str, pd.DataFrame]
    figuras: list[str]


def calcular_tabelas(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fatores = an.fatores_de_risco(df)
    qui2 = an.testes_qui_quadrado(df)
    mw = an.testes_mann_whitney(df)
    estrat, estrat_det = an.analises_estratificadas(df)
    corrigidas = an.aplicar_holm({"fatores": fatores, "qui2": qui2, "mw": mw, "estrat": estrat})
    fatores, qui2, mw, estrat = (corrigidas[k] for k in ("fatores", "qui2", "mw", "estrat"))
    n_testes = sum(len(t) for t in corrigidas.values())
    return {
        "p2_resumo_geral": an.resumo_geral(df),
        "p2_descritiva_numericas": an.descritiva_numericas(df),
        "p2_taxa_por_grupo": an.taxas_por_dimensao(df),
        "p2_correlacoes_spearman": an.correlacoes_spearman(df),
        "p2_fatores_risco": fatores,
        "p2_testes_qui_quadrado": qui2,
        "p2_testes_mann_whitney": mw,
        "p2_estratificadas_resumo": estrat,
        "p2_estratificadas_por_estrato": estrat_det,
        "p2_hora_por_ano": an.tempo_por_ano(df, "hora"),
        "p2_mes_por_ano": an.tempo_por_ano(df, "mes"),
        "p2_semana_x_faixa_horaria": an.semana_x_faixa(df),
        "p2_tipo_acidente_por_pista": an.tipo_acidente_por_pista(df),
        "p2_veredito_hipoteses": an.veredito_hipoteses(fatores, qui2, mw, estrat),
        "p2_ranking_atributos": an.ranking_atributos(df),
        "p2_parametros": pd.DataFrame([
            {"parametro": "alfa", "valor": cfg.ALFA},
            {"parametro": "confianca", "valor": cfg.CONFIANCA},
            {"parametro": "n_testes_familia_holm", "valor": n_testes},
            {"parametro": "n_minimo_grupo", "valor": cfg.N_MINIMO_GRUPO},
        ]),  # fmt: skip
    }


def gerar_figuras(df: pd.DataFrame, t: dict[str, pd.DataFrame], caminhos: Caminhos) -> list[str]:
    pasta = caminhos.figuras
    taxa = float(t["p2_resumo_geral"].iloc[0]["taxa_pct"])
    grupos = t["p2_taxa_por_grupo"]
    causas = grupos[grupos["dimensao"] == "grupo_causa"].rename(columns={"grupo": "grupo_causa"})
    uf = grupos[grupos["dimensao"] == "uf"].rename(columns={"grupo": "uf"})
    v_mes = float(t["p2_testes_qui_quadrado"].set_index("variavel").loc["mes", "v_cramer"])
    feitas = [
        graficos.fig_risco_relativo(t["p2_fatores_risco"], len(df), pasta),
        graficos.fig_hora_do_dia(t["p2_hora_por_ano"], taxa, pasta),
        graficos.fig_volume_vs_gravidade(causas, taxa, pasta),
        graficos.fig_estratificada(t["p2_estratificadas_por_estrato"], t["p2_estratificadas_resumo"], pasta),
        graficos.fig_sazonalidade_mensal(t["p2_mes_por_ano"], v_mes, taxa, pasta),
        graficos.fig_semana_x_faixa(t["p2_semana_x_faixa_horaria"], pasta),
        graficos.fig_uf(uf, taxa, pasta),
        graficos.fig_boxplots(df, t["p2_testes_mann_whitney"], pasta),
    ]
    return [p.stem for p in feitas]


def salvar_tabelas(t: dict[str, pd.DataFrame], caminhos: Caminhos) -> None:
    caminhos.tabelas.mkdir(parents=True, exist_ok=True)
    for nome, tabela in t.items():
        tabela.to_csv(caminhos.tabelas / f"{nome}.csv", index=False, encoding="utf-8",
                      lineterminator="\n", float_format="%.6f")  # fmt: skip


def executar(caminhos: Caminhos | None = None) -> ResultadosEDA:
    caminhos = caminhos or Caminhos()
    df = an.carregar_base(caminhos.csv_modelagem)
    tabelas = calcular_tabelas(df)
    salvar_tabelas(tabelas, caminhos)
    figuras = gerar_figuras(df, tabelas, caminhos)
    documentacao.gerar_textos(tabelas, caminhos)
    logger.info("EDA concluída: %d tabelas, %d figuras.", len(tabelas), len(figuras))
    return ResultadosEDA(tabelas=tabelas, figuras=figuras)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    executar()
