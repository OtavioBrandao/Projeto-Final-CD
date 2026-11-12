"""Treinamento em todo 2024 e avaliação temporal final em 2025."""

import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.configuracao import LIMIARES
from src.p3_modelagem.avaliacao import calcular_metricas, extrair_probabilidade_positiva
from src.p3_modelagem.dados import carregar_base, selecionar_xy, separar_desenvolvimento_teste
from src.p3_modelagem.modelos import criar_pipeline


def executar_avaliacao(base, modelos_dir):
    """Treina em 2024, avalia em 2025 e salva somente os dois XGBoost."""
    treino, teste = separar_desenvolvimento_teste(base)
    if set(treino[cfg.COLUNA_ALVO].unique()) != {0, 1}:
        raise ValueError("Treino de 2024 exige ambas as classes.")
    modelos_dir = Path(modelos_dir)
    modelos_dir.mkdir(parents=True, exist_ok=True)
    if any((modelos_dir / f"p3_{nome}.joblib").exists()
           for nome in ("xgboost_A", "xgboost_B")):
        raise FileExistsError("Modelos definitivos já existem; não sobrescrever.")
    metricas, previsoes = [], []
    for nome, cenario, algoritmo, limiar in (
        ("xgboost_A", "A", "xgboost", LIMIARES["A"]),
        ("xgboost_B", "B", "xgboost", LIMIARES["B"]),
        ("dummy", "B", "dummy", .5),
    ):
        xt, yt = selecionar_xy(treino, cenario)
        xv, yv = selecionar_xy(teste, cenario)
        pipeline = criar_pipeline(cenario, algoritmo)
        inicio = time.perf_counter()
        pipeline.fit(xt, yt)
        tempo_fit = time.perf_counter() - inicio
        if algoritmo == "xgboost":
            joblib.dump(pipeline, modelos_dir / f"p3_{nome}.joblib")
        inicio = time.perf_counter()
        p = extrair_probabilidade_positiva(pipeline, xv)
        tempo_inf = time.perf_counter() - inicio
        identidade = dict(modelo=nome, cenario=cenario, algoritmo=algoritmo, variante="padrao")
        for tipo, t in (("congelado_2024" if algoritmo == "xgboost" else "baseline_0.5", limiar),
                        ("descritivo_0.5", .5)):
            if algoritmo == "dummy" and tipo == "descritivo_0.5":
                continue
            metricas.append({**identidade, "ano": 2025, "tipo_limiar": tipo,
                             "n_treino": len(yt), "tempo_treino_s": tempo_fit,
                             "tempo_inferencia_s": tempo_inf, **calcular_metricas(yv, p, t)})
        previsoes.append(pd.DataFrame({"modelo": nome, "cenario": cenario,
            "linha_original": xv.index, "ano": 2025, "mes": teste.mes.to_numpy(),
            "y_real": yv.to_numpy(), "probabilidade": p.to_numpy(),
            "limiar": limiar, "previsao": p.ge(limiar).astype(int).to_numpy(),
            "previsao_05": p.ge(.5).astype(int).to_numpy()}))
    return pd.DataFrame(metricas), pd.concat(previsoes, ignore_index=True)


def carregar_pipeline(cenario, modelos_dir=cfg.RAIZ_REPOSITORIO / "models"):
    """Carrega somente artefato local confiável; joblib pode executar código."""
    if cenario not in LIMIARES:
        raise ValueError("Cenário deve ser A ou B.")
    return joblib.load(Path(modelos_dir) / f"p3_xgboost_{cenario}.joblib")


def prever(pipeline, base, cenario):
    """Probabilidades e classes pelo limiar congelado, com índice preservado."""
    x, _ = selecionar_xy(base, cenario)
    p = extrair_probabilidade_positiva(pipeline, x)
    return pd.DataFrame({"probabilidade": p, "previsao": p.ge(LIMIARES[cenario]).astype(int)})


def comparar_2024(metricas, reports_2024):
    resumo = pd.read_csv(reports_2024 / "tabelas/p3_resumo_validacao_2024.csv")
    limiares = pd.read_csv(reports_2024 / "tabelas/p3_comparacao_limiares.csv",
                          float_precision="round_trip")
    linhas = []
    for cenario in LIMIARES:
        antigo = resumo.query("cenario == @cenario and algoritmo == 'xgboost' and variante == 'padrao'").iloc[0]
        ajuste = limiares.query("cenario == @cenario and algoritmo == 'xgboost' and variante == 'padrao' and tipo_limiar == 'candidato_f1_ajuste'").iloc[0]
        if not np.isclose(ajuste.limiar, LIMIARES[cenario], rtol=0, atol=1e-14):
            raise ValueError("Limiar publicado de 2024 diverge da decisão congelada.")
        final = metricas.query("cenario == @cenario and tipo_limiar == 'congelado_2024'").iloc[0]
        linha = {"cenario": cenario, "limiar": LIMIARES[cenario]}
        dummy = metricas.loc[metricas.modelo.eq("dummy"), "ap"] if "modelo" in metricas else pd.Series(dtype=float)
        if not dummy.empty:
            linha.update(ap_dummy_2025=dummy.iloc[0], prevalencia_2025=final.prevalencia,
                         ap_sobre_dummy=final.ap / dummy.iloc[0],
                         ap_menos_prevalencia_2025=final.ap - final.prevalencia)
        for metrica in ("ap", "roc_auc", "brier", "log_loss", "precision", "recall", "f1"):
            linha[f"{metrica}_media_validacao_2024_05"] = antigo[f"{metrica}_media"]
            linha[f"{metrica}_teste_2025"] = final[metrica]
            linha[f"{metrica}_delta_2025_menos_media_2024"] = final[metrica] - antigo[f"{metrica}_media"]
        for metrica in ("precision", "recall", "f1"):
            linha[f"{metrica}_ajuste_reunido_2024_otimista"] = ajuste[metrica]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def gerar_relatorios(metricas, previsoes, destino, reports_2024, *, salvar_tabelas=True):
    """Produz gráficos e tabelas derivadas, sem treinar nem carregar modelos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    comparacao = comparar_2024(metricas, reports_2024)
    destino.mkdir(parents=True, exist_ok=True)
    if salvar_tabelas:
        comparacao.to_csv(destino / "comparacao_2024_2025.csv", index=False)
    figuras = destino / "figuras"
    figuras.mkdir(exist_ok=True)
    finais = metricas.loc[~metricas.tipo_limiar.eq("descritivo_0.5")]
    cal = []
    for nome, g in previsoes.groupby("modelo"):
        faixas = pd.cut(g.probabilidade, np.linspace(0, 1, 11), include_lowest=True)
        for faixa, parte in g.groupby(faixas, observed=True):
            cal.append(dict(modelo=nome, ano=2025, faixa=str(faixa), n=len(parte),
                            probabilidade_media=parte.probabilidade.mean(), fracao_fatal=parte.y_real.mean()))
    calibracao = pd.DataFrame(cal)
    if salvar_tabelas:
        calibracao.to_csv(destino / "calibracao_2025.csv", index=False)
    for tipo in ("precision_recall", "roc", "calibracao"):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for nome, g in previsoes.groupby("modelo"):
            if tipo == "precision_recall":
                precision, recall, _ = precision_recall_curve(g.y_real, g.probabilidade)
                ax.step(recall, precision, where="post", label=nome)
            elif tipo == "roc":
                fpr, tpr, _ = roc_curve(g.y_real, g.probabilidade)
                ax.plot(fpr, tpr, label=nome)
            else:
                c = calibracao.loc[calibracao.modelo.eq(nome)]
                ax.plot(c.probabilidade_media, c.fracao_fatal, ".-", label=nome)
        if tipo == "precision_recall":
            ax.axhline(previsoes.y_real.mean(), ls=":", color="gray", label="Prevalência 2025")
            ax.set(xlabel="Recall", ylabel="Precision")
        else:
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set(xlabel="FPR" if tipo == "roc" else "Probabilidade média",
                   ylabel="TPR / recall" if tipo == "roc" else "Fração fatal")
        ax.set(title=f"Teste temporal 2025 — {tipo}", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        fig.savefig(figuras / f"{tipo}_2025.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (_, m) in zip(axes, finais.iterrows(), strict=True):
        matriz = np.array([[m.tn, m.fp], [m.fn, m.tp]], dtype=int)
        ax.imshow(matriz, cmap="Blues")
        for (i, j), n in np.ndenumerate(matriz):
            ax.text(j, i, str(n), ha="center", va="center", color="red")
        ax.set(title=f"2025 — {m.modelo}\nt={m.limiar:.6f}", xlabel="Previsto", ylabel="Real",
               xticks=[0, 1], yticks=[0, 1])
    fig.savefig(figuras / "confusao_2025.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    x = np.arange(2)
    ax.bar(x - .18, comparacao.ap_media_validacao_2024_05, .36, label="Validações 2024 (média)")
    ax.bar(x + .18, comparacao.ap_teste_2025, .36, label="Teste 2025")
    ax.axhline(finais.loc[finais.modelo.eq("dummy"), "ap"].iloc[0], ls=":", label="Dummy / prevalência 2025")
    ax.set(title="Average Precision — validação 2024 e teste 2025; A versus B",
           ylabel="Average Precision", xticks=x, xticklabels=["A", "B"])
    ax.legend()
    fig.savefig(figuras / "ap_2024_2025.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=cfg.CSV_MODELAGEM)
    parser.add_argument("--saida", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports/p3_final")
    parser.add_argument("--modelos", type=Path, default=cfg.RAIZ_REPOSITORIO / "models")
    parser.add_argument("--reports-2024", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports")
    parser.add_argument("--somente-relatorios", action="store_true")
    args = parser.parse_args()
    if args.somente_relatorios:
        metricas = pd.read_csv(args.saida / "metricas_2025.csv", float_precision="round_trip")
        previsoes = pd.read_csv(args.saida / "previsoes_2025.csv.gz", float_precision="round_trip")
    else:
        if any((args.saida / nome).exists() for nome in ("metricas_2025.csv", "previsoes_2025.csv.gz")):
            raise FileExistsError("Resultados finais já existem; use --somente-relatorios ou outra --saida.")
        base = carregar_base(args.csv)
        metricas, previsoes = executar_avaliacao(base, args.modelos)
        args.saida.mkdir(parents=True, exist_ok=True)
        metricas.to_csv(args.saida / "metricas_2025.csv", index=False)
        previsoes.to_csv(args.saida / "previsoes_2025.csv.gz", index=False,
                         compression={"method": "gzip", "mtime": 0})
    gerar_relatorios(metricas, previsoes, args.saida, args.reports_2024,
                     salvar_tabelas=not args.somente_relatorios)
    print("Métricas, previsões e gráficos finais disponíveis.")


if __name__ == "__main__":
    main()
