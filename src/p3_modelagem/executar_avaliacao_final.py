"""Etapa 04: decisões de 2024 congeladas, um treino e teste temporal final.

Executar uma vez: python -m src.p3_modelagem.executar_avaliacao_final
Reproduzir saídas sem treino/inferência: acrescentar --somente-relatorios.
"""

import argparse
import hashlib
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from types import MappingProxyType

import joblib
import numpy as np
import pandas as pd

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.avaliacao import calcular_metricas, extrair_probabilidade_positiva
from src.p3_modelagem.dados import carregar_base, selecionar_xy, separar_desenvolvimento_teste
from src.p3_modelagem.executar_validacao import referencia_publica, sha256, tabela_md
from src.p3_modelagem.modelos import criar_pipeline

LIMIARES = MappingProxyType({"A": 0.2066718339920044, "B": 0.101788230240345})
LIMITACOES = """O teste de 2025 foi reservado à avaliação final da Pessoa 3, mas já foi utilizado
em auditorias e análises exploratórias anteriores: não é completamente cego.
A engenharia anterior usou a base conjunta 2024/2025; indicadores globais de outliers
foram excluídos dos preditores, mas a dependência do processamento anterior permanece.
As métricas descrevem acidentes registrados pela PRF, sem exposição ao tráfego ou
observações sem acidente. A utiliza informações observadas após o acidente; B usa
localização/calendário antecipadamente especificáveis. Ambos estimam fatalidade
condicional à ocorrência de um acidente registrado, nunca probabilidade de acidente.
B sozinho não permite inferir risco absoluto por trecho nem demanda de fiscalização.
A seleção de limiares nas próprias validações de 2024 pode apresentar otimismo;
2025 avalia esses limiares independentemente. Não houve generalização espacial
independente nem análise causal. As validações usaram treinos de tamanhos diferentes;
o treino definitivo usa todo 2024. Mudanças observadas não identificam suas causas.
AP é Average Precision sobre probabilidades, distinta de acurácia e de precision
em um limiar. Nenhuma escolha ou recalibração foi feita com resultados de 2025.
"""


def json_publico(obj):
    """JSON estrito, com métricas indefinidas como null."""
    if isinstance(obj, dict):
        return {k: json_publico(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [json_publico(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def gravar_json(caminho, obj):
    caminho.write_text(json.dumps(json_publico(obj), indent=2, ensure_ascii=False,
                                 allow_nan=False), encoding="utf-8")


def executar_avaliacao(base, modelos_dir, *, ao_concluir=None):
    """Três fits novos em 2024; uma predict_proba em 2025 por modelo.

    Reload/roundtrip usa só amostra de treino, sem segunda inferência de teste.
    O baseline único usa B e limiar 0,5; suas probabilidades independem de X.
    """
    treino, teste = separar_desenvolvimento_teste(base)
    if set(treino[cfg.COLUNA_ALVO].unique()) != {0, 1}:
        raise ValueError("Treino de 2024 exige ambas as classes.")
    modelos_dir = Path(modelos_dir)
    modelos_dir.mkdir(parents=True, exist_ok=True)
    if any((modelos_dir / f"p3_{nome}.joblib").exists()
           for nome in ("xgboost_A", "xgboost_B", "dummy")):
        raise FileExistsError("Modelos definitivos já existem; não sobrescrever.")
    metricas, previsoes, registros = [], [], []
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
        arquivo = modelos_dir / f"p3_{nome}.joblib"
        joblib.dump(pipeline, arquivo)
        amostra = xt.iloc[:32]
        np.testing.assert_allclose(
            extrair_probabilidade_positiva(pipeline, amostra),
            extrair_probabilidade_positiva(joblib.load(arquivo), amostra), rtol=1e-7, atol=1e-8)
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
        registros.append({**identidade, "limiar": limiar, "features": list(xt.columns),
            "arquivo": referencia_publica(arquivo), "sha256": sha256(arquivo),
            "parametros_modelo": pipeline[-1].get_params(),
            "pipeline": repr(pipeline),
            "parametros_preprocessamento": {k: repr(v) for k, v in pipeline[0].get_params(deep=True).items()},
            "features_transformadas": pipeline[0].get_feature_names_out().tolist(),
            "roundtrip_treino": True})
        if ao_concluir:
            ao_concluir(pd.DataFrame(metricas), pd.concat(previsoes, ignore_index=True), registros)
    return pd.DataFrame(metricas), pd.concat(previsoes, ignore_index=True), registros


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


def gerar_relatorios(metricas, previsoes, manifesto, destino, reports_2024):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    comparacao = comparar_2024(metricas, reports_2024)
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
    texto = ["# Pessoa 3 — avaliação temporal final 2025", "",
        "Treino: todos os registros de 2024. XGBoost padrão e limiares congelados antes da inferência final. "
        "Baseline prior treinado uma vez; probabilidade constante comum aos cenários.", "",
        "## Métricas de 2025", "", tabela_md(metricas), "",
        "Precision indefinida sem positivos previstos é registrada como campo vazio no CSV/null no JSON.", "",
        "## Comparação 2024/2025", "", tabela_md(comparacao), "",
        "As médias de 2024 usam limiar 0,5 para precision/recall/F1. As colunas ajuste_reunido_2024_otimista "
        "usam os limiares candidatos nas próprias previsões que os escolheram; não são avaliação independente. "
        "Deltas são diferenças assinadas em unidades da métrica, sem normalização percentual.", ""]
    for _, r in comparacao.iterrows():
        texto.append(f"Cenário {r.cenario}: ΔAP={r.ap_delta_2025_menos_media_2024:+.6f}; "
                     "a diferença não prova uma causa nem compara condições idênticas.")
    texto += ["", "## Calibração aproximada", "", tabela_md(calibracao), "",
        "Faixas fixas de 0,1, sem reajuste. Compare probabilidade média e fração fatal considerando n; "
        "faixas pequenas são instáveis. Brier e Log Loss menores representam melhor qualidade probabilística, "
        "mas não provam calibração perfeita.", "", "## Gráficos", ""]
    texto += [f"![{p.stem}](figuras/{p.name})" for p in sorted(figuras.glob("*.png"))]
    texto += ["", "## Consumo pelas Pessoas 4 e 5", "",
        "Carregue apenas os joblib confiáveis produzidos aqui, em ambiente Python compatível. "
        "As versões, features originais/transformadas, parâmetros completos, caminhos e hashes estão em [manifesto.json](manifesto.json).", "",
        "```python\nfrom src.p3_modelagem.executar_avaliacao_final import carregar_pipeline, prever, LIMIARES\n"
        "from src.p3_modelagem.dados import carregar_base, selecionar_xy\n"
        "modelo = carregar_pipeline('A')\nbase = carregar_base()\nX, y = selecionar_xy(base, 'A')\n"
        "X_transformado = modelo.named_steps['preprocessamento'].transform(X)\n"
        "nomes = modelo.named_steps['preprocessamento'].get_feature_names_out()\n"
        "estimador = modelo.named_steps['modelo']\n# SHAP deve alinhar X_transformado e nomes.\n"
        "# Para a avaliação de 2025, reutilize as previsões salvas, sem nova inferência.\n```", "",
        "`prever(modelo, base, cenario)` preserva índice e aplica o limiar congelado; "
        "`pipeline.predict` usa a decisão padrão do estimador e não substitui essa interface. "
        "Nas tabelas de previsões, linha_original é o índice posicional original do CSV (base zero); "
        "modelo + linha_original é chave única. Não há SHAP ou dashboard implementado.", "",
        "[Métricas](metricas_2025.csv), [previsões](previsoes_2025.csv.gz), "
        "[comparação](comparacao_2024_2025.csv), [calibração](calibracao_2025.csv).", "",
        "## Limitações metodológicas", "", LIMITACOES, "",
        "## Rastreabilidade e reprodução", "",
        f"Dataset: `{manifesto['csv']}`; SHA-256: `{manifesto['csv_sha256']}`. "
        f"HEAD: `{manifesto['git_head']}`. Fontes e artefatos têm hashes no manifesto. "
        "O estado local é registrado: o HEAD sozinho não inclui alterações não commitadas.", "",
        "`python -m src.p3_modelagem.executar_avaliacao_final --somente-relatorios` "
        "reconstrói tabelas derivadas e figuras sem fit nem predict. A execução recusa saída preexistente "
        "e mantém checkpoint/estado em caso de falha técnica; não retome treinamentos para melhorar métricas.", ""]
    (destino / "README.md").write_text("\n\n".join(texto), encoding="utf-8")


def git_info(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={cfg.RAIZ_REPOSITORIO.as_posix()}",
        *args], cwd=cfg.RAIZ_REPOSITORIO, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=cfg.CSV_MODELAGEM)
    parser.add_argument("--saida", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports/p3_final")
    parser.add_argument("--modelos", type=Path, default=cfg.RAIZ_REPOSITORIO / "models")
    parser.add_argument("--reports-2024", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports")
    parser.add_argument("--somente-relatorios", action="store_true")
    args = parser.parse_args()
    manifesto_path = args.saida / "manifesto.json"
    if args.somente_relatorios:
        meta = json.loads(manifesto_path.read_text(encoding="utf-8"))
        metricas = pd.read_csv(args.saida / "metricas_2025.csv", float_precision="round_trip")
        previsoes = pd.read_csv(args.saida / "previsoes_2025.csv.gz", float_precision="round_trip")
        if meta["status"] != "concluido":
            raise ValueError("Avaliação incompleta; preserve checkpoint para diagnóstico técnico.")
    else:
        # mkdir exclusivo impede repetição silenciosa, mesmo após falha técnica.
        args.saida.mkdir(parents=True, exist_ok=False)
        meta = {"status": "iniciado", "inicio_utc": datetime.now(timezone.utc).isoformat(),
            "csv": referencia_publica(args.csv), "csv_sha256": sha256(args.csv),
            "git_head": git_info("rev-parse", "HEAD"), "git_branch": git_info("branch", "--show-current"),
            "git_status_inicial": git_info("status", "--porcelain", "-uno"),
            "python": platform.python_version(),
            "versoes": {p: importlib.metadata.version(p) for p in
                        ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib")},
            "fontes_sha256": {referencia_publica(p): sha256(p) for p in
                              Path(__file__).parent.glob("*.py")},
            "artefatos_2024_sha256": {referencia_publica(p): sha256(p) for p in
                                    (args.reports_2024 / "tabelas").glob("*")},
            "limiares_congelados": dict(LIMIARES), "ano_treino": 2024, "ano_teste": 2025,
            "limitacoes": LIMITACOES}
        preparacao = cfg.RAIZ_REPOSITORIO / "reports/p3_preparacao_etapa04.json"
        if preparacao.is_file():
            meta["verificacao_previa"] = json.loads(preparacao.read_text(encoding="utf-8"))
        gravar_json(manifesto_path, meta)

        def checkpoint(m, p, modelos):
            m.to_csv(args.saida / "metricas_2025.csv", index=False)
            p.to_csv(args.saida / "previsoes_2025.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
            meta.update(modelos=modelos, metricas=m.to_dict("records"))
            gravar_json(manifesto_path, meta)

        try:
            base = carregar_base(args.csv)
            treino, teste = separar_desenvolvimento_teste(base)
            meta["particoes"] = {str(ano): {"n": len(b), "classes": b[cfg.COLUNA_ALVO].value_counts().to_dict(),
                "indices_sha256": hashlib.sha256(b.index.to_numpy(dtype="int64").tobytes()).hexdigest()}
                for ano, b in ((2024, treino), (2025, teste))}
            # Checa os limiares publicados antes de qualquer fit/inferência final.
            comparar_2024(pd.DataFrame([{"cenario": c, "tipo_limiar": "congelado_2024",
                **{k: 0. for k in ("ap", "roc_auc", "brier", "log_loss", "precision", "recall", "f1")}}
                for c in LIMIARES]), args.reports_2024)
            metricas, previsoes, _ = executar_avaliacao(base, args.modelos, ao_concluir=checkpoint)
            if sha256(args.csv) != meta["csv_sha256"]:
                raise RuntimeError("Dataset mudou durante a execução.")
            meta["status"] = "concluido"
        except Exception as erro:
            # Não publicar mensagens que possam incluir caminhos pessoais.
            meta.update(status="falha_tecnica", tipo_erro=type(erro).__name__)
            gravar_json(manifesto_path, meta)
            raise
    try:
        gerar_relatorios(metricas, previsoes, meta, args.saida, args.reports_2024)
        meta["status_relatorios"] = "concluido"
    except Exception as erro:
        meta.update(status_relatorios="falha_tecnica", tipo_erro_relatorios=type(erro).__name__)
        gravar_json(manifesto_path, meta)
        raise
    meta["artefatos_sha256"] = {referencia_publica(p): sha256(p) for p in args.saida.rglob("*")
                               if p.is_file() and p != manifesto_path}
    gravar_json(manifesto_path, meta)
    print("Avaliação final concluída; relatórios e pipelines disponíveis.")


if __name__ == "__main__":
    main()
