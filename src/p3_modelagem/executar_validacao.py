"""Validação temporal dos classificadores usando exclusivamente 2024."""

import argparse
from pathlib import Path
import time

import pandas as pd

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.avaliacao import (
    analisar_previsoes, calcular_metricas, consolidar_resultados,
    extrair_probabilidade_positiva,
)
from src.p3_modelagem.dados import carregar_base, gerar_cortes_temporais, selecionar_xy
from src.p3_modelagem.modelos import ALGORITMOS, criar_pipeline


def configurar_variante(pipeline, algoritmo: str, variante: str, y_treino: pd.Series) -> dict:
    if variante not in ("padrao", "ponderado"):
        raise ValueError("Variante desconhecida.")
    if set(y_treino.unique()) != {0, 1}:
        raise ValueError("O treino precisa conter ambas as classes.")
    parametros = {}
    if variante == "ponderado":
        if algoritmo in ("regressao_logistica", "random_forest"):
            parametros["class_weight"] = "balanced"
        elif algoritmo == "xgboost":
            parametros["scale_pos_weight"] = float(y_treino.eq(0).sum() / y_treino.eq(1).sum())
        else:
            raise ValueError("Dummy não recebe ponderação.")
    pipeline.set_params(**{f"modelo__{k}": v for k, v in parametros.items()})
    return parametros


def validar_cortes(desenvolvimento, cortes):
    if not desenvolvimento.ano.eq(2024).all():
        raise ValueError("Os experimentos aceitam exclusivamente 2024.")
    if tuple(c.rodada for c in cortes) != (1, 2, 3):
        raise ValueError("São necessárias as três rodadas temporais.")
    vistas = pd.Index([])
    for corte, (_, fim_treino, inicio, fim) in zip(cortes, cfg.CORTES_TEMPORAIS, strict=True):
        if (not corte.treino.is_unique or not corte.validacao.is_unique
                or corte.treino.intersection(corte.validacao).size
                or vistas.intersection(corte.validacao).size):
            raise ValueError("Índices duplicados ou vazamento entre treino e validação.")
        esperado_t = desenvolvimento.index[desenvolvimento.mes.between(1, fim_treino)]
        esperado_v = desenvolvimento.index[desenvolvimento.mes.between(inicio, fim)]
        if not corte.treino.equals(esperado_t) or not corte.validacao.equals(esperado_v):
            raise ValueError("Corte não corresponde ao protocolo temporal completo.")
        vistas = vistas.append(corte.validacao)


def executar_experimentos(desenvolvimento, *, ponderados=True):
    """Retorna métricas por rodada e previsões fora do treino."""
    cortes = gerar_cortes_temporais(desenvolvimento)
    validar_cortes(desenvolvimento, cortes)
    resultados, previsoes = [], []
    for variante in (("padrao", "ponderado") if ponderados else ("padrao",)):
        for cenario in ("A", "B"):
            x, y = selecionar_xy(desenvolvimento, cenario)
            for algoritmo in ALGORITMOS:
                if variante == "ponderado" and algoritmo == "dummy":
                    continue
                for corte in cortes:
                    identidade = dict(cenario=cenario, algoritmo=algoritmo,
                                      variante=variante, rodada=corte.rodada)
                    xt, yt = x.loc[corte.treino], y.loc[corte.treino]
                    xv, yv = x.loc[corte.validacao], y.loc[corte.validacao]
                    pipeline = criar_pipeline(cenario, algoritmo)
                    configurar_variante(pipeline, algoritmo, variante, yt)
                    inicio = time.perf_counter()
                    pipeline.fit(xt, yt)
                    treino_s = time.perf_counter() - inicio
                    inicio = time.perf_counter()
                    p = extrair_probabilidade_positiva(pipeline, xv)
                    inferencia_s = time.perf_counter() - inicio
                    resultados.append({**identidade, "n_treino": len(yt),
                        "n_positivos_treino": int(yt.sum()),
                        "tempo_treino_s": treino_s, "tempo_inferencia_s": inferencia_s,
                        **calcular_metricas(yv, p)})
                    previsoes.append(pd.DataFrame({
                        **identidade, "linha_original": xv.index, "y_real": yv.to_numpy(),
                        "probabilidade": p.to_numpy(), "previsao_05": p.ge(.5).astype(int).to_numpy(),
                        "ano": 2024, "mes": desenvolvimento.loc[xv.index, "mes"].to_numpy(),
                    }))
    return pd.DataFrame(resultados), pd.concat(previsoes, ignore_index=True)


def gerar_figuras(resumo, previsoes, calibracao, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), sharex=True, constrained_layout=True)
    for cenario, ax in zip(("A", "B"), eixos, strict=True):
        parte = resumo.loc[resumo.cenario.eq(cenario)].sort_values("ap_media")
        rotulos = parte.algoritmo + " / " + parte.variante
        ax.barh(rotulos, parte.ap_media, xerr=parte.ap_dp, color="#246b86", capsize=3)
        ax.axvline(parte.prevalencia_media.iloc[0], color="#b74b32", ls="--", label="Prevalência média")
        ax.set(title=f"Cenário {cenario}", xlabel="Average Precision: média ± DP descritivo")
        ax.legend(loc="lower right")
    fig.savefig(destino / "p3_comparacao_ap.png", dpi=160)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for cenario, ax in zip(("A", "B"), eixos, strict=True):
        for (algoritmo, variante), grupo in previsoes.loc[previsoes.cenario.eq(cenario)].groupby(["algoritmo", "variante"]):
            precision, recall, _ = precision_recall_curve(grupo.y_real, grupo.probabilidade)
            ax.step(recall, precision, where="post", lw=1.2,
                    linestyle="--" if variante == "ponderado" else "-",
                    label=f"{algoritmo} / {variante}")
        ax.axhline(previsoes.y_real.mean(), color="grey", ls=":", label="Prevalência")
        ax.set(title=f"Cenário {cenario} — validações reunidas", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.savefig(destino / "p3_precision_recall_validacao.png", dpi=160)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for cenario, ax in zip(("A", "B"), eixos, strict=True):
        for (algoritmo, variante), grupo in calibracao.loc[calibracao.cenario.eq(cenario)].groupby(["algoritmo", "variante"]):
            ax.plot(grupo.probabilidade_media, grupo.fracao_fatal, marker=".", lw=1,
                    ls="--" if variante == "ponderado" else "-", label=f"{algoritmo} / {variante}")
        ax.plot([0, 1], [0, 1], color="grey", ls=":")
        ax.set(title=f"Cenário {cenario} — calibração aproximada", xlabel="Probabilidade média por faixa", ylabel="Fração fatal", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.savefig(destino / "p3_calibracao_validacao.png", dpi=160)
    plt.close(fig)


def salvar_relatorios(resultados, previsoes, destino, *, salvar_tabelas=True):
    """Salva tabelas do experimento ou recria gráficos dos CSVs existentes."""
    tabelas, figuras = destino / "tabelas", destino / "figuras"
    tabelas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    if salvar_tabelas:
        resumo = consolidar_resultados(resultados)
        globais, limiares, calibracao = analisar_previsoes(previsoes)
        for nome, tabela in (("p3_validacao_2024", resultados), ("p3_resumo_validacao_2024", resumo),
                             ("p3_metricas_globais_2024", globais), ("p3_comparacao_limiares", limiares),
                             ("p3_calibracao_2024", calibracao)):
            tabela.to_csv(tabelas / f"{nome}.csv", index=False)
        previsoes.to_csv(tabelas / "p3_previsoes_validacao_2024.csv.gz", index=False,
                         compression={"method": "gzip", "mtime": 0})
    else:
        resumo = pd.read_csv(tabelas / "p3_resumo_validacao_2024.csv", float_precision="round_trip")
        calibracao = pd.read_csv(tabelas / "p3_calibracao_2024.csv", float_precision="round_trip")
    gerar_figuras(resumo, previsoes, calibracao, figuras)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports")
    parser.add_argument("--csv", type=Path, default=cfg.CSV_MODELAGEM)
    parser.add_argument("--somente-relatorios", action="store_true")
    args = parser.parse_args()
    if args.somente_relatorios:
        resultados = pd.read_csv(args.saida / "tabelas/p3_validacao_2024.csv", float_precision="round_trip")
        previsoes = pd.read_csv(args.saida / "tabelas/p3_previsoes_validacao_2024.csv.gz", float_precision="round_trip")
    else:
        if any((args.saida / "tabelas").glob("p3_*.csv*")):
            raise FileExistsError(
                "Resultados de validação já existem; escolha outra pasta com --saida "
                "ou use --somente-relatorios."
            )
        base = carregar_base(args.csv)
        desenvolvimento = base.loc[base.ano.eq(cfg.ANO_DESENVOLVIMENTO)]
        resultados, previsoes = executar_experimentos(desenvolvimento)
    salvar_relatorios(resultados, previsoes, args.saida, salvar_tabelas=not args.somente_relatorios)
    print("Tabelas e gráficos da validação de 2024 disponíveis.")


if __name__ == "__main__":
    main()
