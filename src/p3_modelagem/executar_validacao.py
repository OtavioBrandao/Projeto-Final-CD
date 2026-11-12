"""Etapa 03: experimentos sequenciais e relatórios reproduzíveis, somente 2024.

Executar: .venv/Scripts/python -m src.p3_modelagem.executar_validacao
Não treina modelo definitivo nem persiste estimadores ajustados.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.avaliacao import (
    CHAVES, analisar_previsoes, calcular_metricas, consolidar_resultados,
    extrair_probabilidade_positiva, validar_previsoes,
)
from src.p3_modelagem.dados import (
    carregar_base, gerar_cortes_temporais, selecionar_xy, separar_desenvolvimento_teste,
)
from src.p3_modelagem.modelos import ALGORITMOS, SEMENTE, criar_pipeline


def sha256(caminho: Path) -> str:
    with caminho.open("rb") as arquivo:
        return hashlib.file_digest(arquivo, "sha256").hexdigest()


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


def executar_experimentos(desenvolvimento: pd.DataFrame, *, ponderados=True,
                         ao_concluir=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reutiliza cortes/fábrica, guarda apenas previsões e descarta cada modelo."""
    cortes = gerar_cortes_temporais(desenvolvimento)
    validar_cortes(desenvolvimento, cortes)
    resultados, previsoes, execucoes = [], [], []
    # Todas as 24 configurações principais precedem as variantes ponderadas.
    for variante in (("padrao", "ponderado") if ponderados else ("padrao",)):
        for cenario in ("A", "B"):
            x, y = selecionar_xy(desenvolvimento, cenario)
            for algoritmo in ALGORITMOS:
                if variante == "ponderado" and algoritmo == "dummy":
                    continue
                for corte in cortes:
                    identidade = dict(cenario=cenario, algoritmo=algoritmo,
                                      variante=variante, rodada=corte.rodada)
                    registro = {**identidade, "status": "iniciado"}
                    pipeline = None
                    inicio = time.perf_counter()
                    print(f"INICIO {identidade}", flush=True)
                    try:
                        xt, yt = x.loc[corte.treino], y.loc[corte.treino]
                        xv, yv = x.loc[corte.validacao], y.loc[corte.validacao]
                        pipeline = criar_pipeline(cenario, algoritmo)
                        registro["ponderacao"] = configurar_variante(pipeline, algoritmo, variante, yt)
                        registro["parametros_modelo"] = {
                            k: "NaN" if isinstance(v, float) and np.isnan(v) else v
                            for k, v in pipeline[-1].get_params().items()
                        }
                        with warnings.catch_warnings(record=True) as avisos:
                            warnings.simplefilter("always")
                            inicio_fit = time.perf_counter()
                            pipeline.fit(xt, yt)
                            treino_s = time.perf_counter() - inicio_fit
                            inicio_inf = time.perf_counter()
                            p = extrair_probabilidade_positiva(pipeline, xv)
                            inferencia_s = time.perf_counter() - inicio_inf
                        registro["avisos"] = [f"{a.category.__name__}: {a.message}" for a in avisos]
                        for aviso in avisos:
                            warnings.warn_explicit(str(aviso.message), aviso.category, aviso.filename, aviso.lineno)
                        metricas = calcular_metricas(yv, p)
                        resultados.append({**identidade, "n_treino": len(yt),
                                           "n_positivos_treino": int(yt.sum()),
                                           "tempo_treino_s": treino_s,
                                           "tempo_inferencia_s": inferencia_s, **metricas})
                        tabela = pd.DataFrame({"linha_original": xv.index, "y_real": yv.to_numpy(),
                                               "probabilidade": p.to_numpy(),
                                               "previsao_05": p.ge(.5).astype(int).to_numpy(),
                                               "ano": 2024,
                                               "mes": desenvolvimento.loc[xv.index, "mes"].to_numpy()})
                        for nome, valor in identidade.items():
                            tabela[nome] = valor
                        previsoes.append(tabela)
                        registro["status"] = "concluido"
                        print(f"FIM AP={metricas['ap']:.6f}; treino={treino_s:.2f}s; inferencia={inferencia_s:.2f}s", flush=True)
                    except Exception as erro:
                        registro.update(status="falha", erro=f"{type(erro).__name__}: {erro}")
                        print(f"FALHA {registro['erro']}", flush=True)
                    finally:
                        registro["tempo_total_s"] = time.perf_counter() - inicio
                        execucoes.append(registro)
                        del pipeline
                        gc.collect()
                        if ao_concluir is not None:
                            ao_concluir(resultados, previsoes, execucoes)
    if not resultados:
        raise RuntimeError(f"Nenhum experimento concluído: {execucoes}")
    tabela_previsoes = pd.concat(previsoes, ignore_index=True)
    validar_previsoes(tabela_previsoes)
    protocolo = []
    for c in cortes:
        protocolo.append({"rodada": c.rodada, "n_treino": len(c.treino),
                          "n_validacao": len(c.validacao),
                          "positivos_treino": int(desenvolvimento.loc[c.treino, cfg.COLUNA_ALVO].sum()),
                          "positivos_validacao": int(desenvolvimento.loc[c.validacao, cfg.COLUNA_ALVO].sum()),
                          "sha256_indices_treino": hashlib.sha256(c.treino.to_numpy(dtype="int64").tobytes()).hexdigest(),
                          "sha256_indices_validacao": hashlib.sha256(c.validacao.to_numpy(dtype="int64").tobytes()).hexdigest()})
    return pd.DataFrame(resultados), tabela_previsoes, {"cortes": protocolo, "execucoes": execucoes}


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


def tabela_md(tabela):
    """Markdown sem dependência adicional de tabulate."""
    def formato(valor):
        if isinstance(valor, (float, np.floating)):
            return "indefinida" if pd.isna(valor) else f"{valor:.6f}"
        return str(valor).replace("|", "/")
    return "\n".join([
        "| " + " | ".join(tabela.columns) + " |",
        "| " + " | ".join(["---"] * len(tabela.columns)) + " |",
        *["| " + " | ".join(formato(v) for v in linha) + " |" for linha in tabela.itertuples(index=False, name=None)],
    ])


def gerar_relatorio(resultados, resumo, globais, limiares, metadados, destino):
    candidatas = []
    for cenario in ("A", "B"):
        elegiveis = resumo.loc[resumo.cenario.eq(cenario) & resumo.n_rodadas.eq(3)]
        if not elegiveis.empty:
            # Recomendação preliminar por AP médio; desempate por menor DP,
            # Brier e nome, jamais pelo F1 ajustado.
            candidatas.append(elegiveis.sort_values(
                ["ap_media", "ap_dp", "brier_media", "algoritmo", "variante"],
                ascending=[False, True, True, True, True]).iloc[0])
    linhas = [
        "# Pessoa 3 — Etapa 03: validação temporal de 2024", "",
        "Relatório gerado por `python -m src.p3_modelagem.executar_validacao`.", "",
        "## Protocolo e rastreabilidade", "",
        f"CSV: `{metadados['csv']}`; SHA-256: `{metadados['csv_sha256']}`.",
        f"Desenvolvimento completo: {metadados['n_desenvolvimento']} registros de 2024. Semente: {SEMENTE}.",
        "O carregador existente lê e valida o contrato do CSV completo; a separação existente isola 2024. "
        "O objeto reservado de 2025 é descartado antes dos experimentos, sem inferência, métricas, seleção ou inspeção de resultados de 2025.",
        "Treino janeiro–junho / validação julho–agosto; janeiro–agosto / setembro–outubro; janeiro–outubro / novembro–dezembro. "
        "Cortes gerados uma única vez e reutilizados nos dois cenários e em todas as variantes. "
        "Pipeline novo por ajuste, pré-processamento ajustado apenas no treino; estimadores descartados após inferência. "
        "Tempo de inferência inclui transformação e predict_proba; tempo de treino inclui todo o fit.",
        tabela_md(pd.DataFrame(metadados["cortes"])[["rodada", "n_treino", "n_validacao", "positivos_treino", "positivos_validacao"]]), "",
        f"Execuções concluídas: {sum(e['status'] == 'concluido' for e in metadados['execucoes'])}; "
        f"falhas: {sum(e['status'] == 'falha' for e in metadados['execucoes'])}.",
        "Configurações e pesos efetivos de cada treino, versões, avisos, falhas e hashes dos índices estão em `p3_execucao_2024.json`.", "",
        "## Comparação no limiar 0,5", "",
        "Médias não ponderadas entre três cortes. DP populacional (ddof=0), apenas descritivo, sem interpretação de intervalo de confiança. "
        "Campos NaN/CSV vazio indicam métricas indefinidas; a coluna `indefinidas` explica os motivos. "
        "Médias e DP ignoram valores indefinidos; os campos *_n_definidos no resumo informam quantos cortes contribuíram. "
        "Precision é indefinida quando não há positivos previstos; recall é indefinido sem positivos reais; "
        "F1 é zero quando há positivos reais mas nenhum previsto; AP é indefinida sem positivos reais e ROC-AUC exige ambas as classes. "
        "Brier e Log Loss são calculados com as probabilidades originais, sem recalibração. AP usa average_precision_score.",
        tabela_md(resumo[[*CHAVES, "ap_media", "ap_dp", "roc_auc_media", "precision_media", "precision_n_definidos", "recall_media", "f1_media", "brier_media", "log_loss_media", "tempo_treino_total_s"]]), "",
        "## Rodadas individuais", "",
        tabela_md(resultados[[*CHAVES, "rodada", "prevalencia", "ap", "roc_auc", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]]), "",
        "## Previsões reunidas e limiares", "",
        "As validações julho–dezembro são disjuntas. As métricas abaixo usam todas as previsões reunidas; "
        "não são médias das métricas dos cortes. O baseline constante do Dummy por rodada tem AP igual à prevalência da respectiva validação; "
        "ao reunir rodadas, suas constantes podem variar e o AP global pode diferir da prevalência global.",
        tabela_md(globais[[*CHAVES, "n_validacao", "prevalencia", "ap", "roc_auc", "brier", "log_loss"]]), "",
        "Busca exata nos escores únicos observados (inclui probabilidades pequenas). Maximiza F1; empates com atol=1e-12 e rtol=0 "
        "favorecem o maior limiar. A decisão é p >= limiar. O limiar e seu F1 são ajustados nas mesmas validações: "
        "esses valores são otimistas e não constituem avaliação independente. Também existe otimismo de seleção ao comparar modelos em 2024.",
        tabela_md(limiares[[*CHAVES, "tipo_limiar", "limiar", "precision", "recall", "f1", "fp", "tp", "fn"]]), "",
        "## Ponderação e calibração aproximada", "",
    ]
    for cenario in ("A", "B"):
        linhas += [f"### Cenário {cenario}", ""]
        parte = resumo.loc[resumo.cenario.eq(cenario)]
        for algoritmo in ALGORITMOS[1:]:
            pares = parte.loc[parte.algoritmo.eq(algoritmo)].set_index("variante")
            if {"padrao", "ponderado"} <= set(pares.index):
                a, b = pares.loc["padrao"], pares.loc["ponderado"]
                diferenca_precision = b.precision_media - a.precision_media
                delta_precision = f"{diferenca_precision:+.6f}" if pd.notna(diferenca_precision) else "indefinida"
                linhas += [f"- {algoritmo}: ΔAP={b.ap_media-a.ap_media:+.6f}, Δrecall={b.recall_media-a.recall_media:+.6f}, "
                           f"Δprecision={delta_precision}, ΔF1={b.f1_media-a.f1_media:+.6f}, "
                           f"ΔBrier={b.brier_media-a.brier_media:+.6f}, ΔLogLoss={b.log_loss_media-a.log_loss_media:+.6f}; "
                           f"DP(AP) padrão={a.ap_dp:.6f}, ponderado={b.ap_dp:.6f}."]
        linhas += [""]
    linhas += ["Deltas são ponderado menos padrão; Brier/Log Loss menores são melhores. "
               "Pesos balanced em LR/RF e scale_pos_weight=negativos/positivos do treino em XGBoost. "
               "Sem reamostragem. As faixas fixas de largura 0,1 no CSV/gráfico de calibração permitem observar "
               "discrepâncias entre probabilidade média e frequência fatal; faixas pouco povoadas são instáveis. "
               "Brier/Log Loss avaliam qualidade probabilística e não demonstram sozinhos calibração perfeita.", "",
               "## Candidatas preliminares para revisão", ""]
    for candidata in candidatas:
        mascara = np.logical_and.reduce([limiares[k].eq(candidata[k]) for k in CHAVES])
        ajuste = limiares.loc[mascara & limiares.tipo_limiar.eq("candidato_f1_ajuste")].iloc[0]
        dummy = resumo.loc[resumo.cenario.eq(candidata.cenario) & resumo.algoritmo.eq("dummy")].iloc[0]
        linhas += [f"- Cenário {candidata.cenario}: **{candidata.algoritmo} / {candidata.variante}**, "
                   f"AP médio={candidata.ap_media:.6f} ± {candidata.ap_dp:.6f} "
                   f"(Dummy={dummy.ap_media:.6f}; razão={candidata.ap_media/dummy.ap_media:.2f}×). "
                   f"Brier={candidata.brier_media:.6f}; Log Loss={candidata.log_loss_media:.6f}. "
                   f"Limiar candidato={ajuste.limiar:.12g}; precision de ajuste={ajuste.precision:.6f}, "
                   f"recall={ajuste.recall:.6f}, F1={ajuste.f1:.6f}, FP={int(ajuste.fp)}. "
                   "Recomendação preliminar pelo maior AP médio completo; estabilidade e probabilidades devem ser revistas "
                   "nas tabelas acima. O F1 ajustado não participa da escolha do modelo."]
        individuais = resultados.loc[np.logical_and.reduce(
            [resultados[k].eq(candidata[k]) for k in CHAVES])].sort_values("rodada")
        linhas += ["AP nas rodadas 1/2/3: " + ", ".join(f"{v:.6f}" for v in individuais.ap) + "."]
        concorrentes = resumo.loc[resumo.cenario.eq(candidata.cenario)
                                  & resumo.n_rodadas.eq(3)].sort_values("ap_media", ascending=False)
        if len(concorrentes) > 1:
            segunda = concorrentes.iloc[1]
            linhas += [f"A segunda configuração por AP médio foi {segunda.algoritmo} / {segunda.variante} "
                       f"({segunda.ap_media:.6f} ± {segunda.ap_dp:.6f}); diferença observada "
                       f"de {candidata.ap_media-segunda.ap_media:.6f}. Esses três cortes não demonstram "
                       "superioridade estatística rigorosa. A decisão permanece preliminar."]
    if len(candidatas) == 2:
        a, b = candidatas
        linhas += ["", f"A diferença observada entre as candidatas A e B é ΔAP={a.ap_media-b.ap_media:.6f} "
                   f"(razão A/B={a.ap_media/b.ap_media:.2f}×). "
                   "A usa informações retrospectivas do acidente; B estima fatalidade condicional a um acidente registrado "
                   "com atributos disponíveis antecipadamente. B não prevê a ocorrência de acidentes. "
                   "A diferença de AP entre cenários sugere dependência de informações posteriores à ocorrência, "
                   "mas não é uma decomposição causal nem isola o efeito de cada atributo."]
    if "verificacao" in metadados:
        linhas += ["", "## Verificação da entrega", "",
                   tabela_md(pd.DataFrame([{"item": k, "resultado": v}
                                           for k, v in metadados["verificacao"].items()])), ""]
    linhas += ["", "## Artefatos e reprodução", "",
               "- `tabelas/p3_validacao_2024.csv`: 42 linhas esperadas, métricas individuais e tempos.",
               "- `tabelas/p3_resumo_validacao_2024.csv`: médias, DP, contagens de valores definidos e tempo total.",
               "- `tabelas/p3_metricas_globais_2024.csv`: métricas das validações reunidas.",
               "- `tabelas/p3_previsoes_validacao_2024.csv.gz`: índice original, ano/mês, rodada, alvo, probabilidade e classe a 0,5; CSV comprimido para evitar duplicação volumosa.",
               "- `tabelas/p3_comparacao_limiares.csv` e `tabelas/p3_calibracao_2024.csv`.",
               "- `figuras/p3_comparacao_ap.png`, `figuras/p3_precision_recall_validacao.png`, `figuras/p3_calibracao_validacao.png`.",
               "- `p3_execucao_2024.json`: proveniência, parâmetros, avisos, falhas e hashes.", "",
               "Reexecutar com `.venv/Scripts/python -m src.p3_modelagem.executar_validacao`; "
               "`--somente-relatorios` regenera gráficos e relatório dos resultados salvos sem novos ajustes. "
               "Sementes fixas garantem repetibilidade numérica dentro das tolerâncias de ponto flutuante; tempos dependem da máquina. "
               "Versões do ambiente estão no JSON. Testes: `.venv/Scripts/python -m pytest -q`.", "",
               "## Limitações e próxima decisão", "",
               "Três cortes consecutivos de um único ano não estabelecem generalização para outros anos. "
               "Pesos e limiares alteram o compromisso entre falsos positivos e perdas de fatais; não foi informado um custo operacional. "
               "O limiar por F1 é apenas candidato para revisão. Não há recalibração, SHAP, dashboard, busca extensa "
               "ou treinamento definitivo. A etapa encerra com decisões preliminares: após a revisão, congelar "
               "cenário, algoritmo, variante e limiar antes de uma única avaliação futura em 2025. Nenhuma avaliação de 2025 foi realizada.", ""]
    (destino / "README_modelagem.md").write_text("\n\n".join(linha for linha in linhas if linha), encoding="utf-8")


def salvar_relatorios(resultados, previsoes, metadados, destino, *, salvar_tabelas=True):
    tabelas, figuras = destino / "tabelas", destino / "figuras"
    tabelas.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)
    resumo = consolidar_resultados(resultados)
    globais, limiares, calibracao = analisar_previsoes(previsoes)
    if salvar_tabelas:
        for nome, tabela in (("p3_validacao_2024", resultados), ("p3_resumo_validacao_2024", resumo),
                             ("p3_metricas_globais_2024", globais), ("p3_comparacao_limiares", limiares),
                             ("p3_calibracao_2024", calibracao)):
            tabela.to_csv(tabelas / f"{nome}.csv", index=False)
        previsoes.to_csv(tabelas / "p3_previsoes_validacao_2024.csv.gz", index=False,
                         compression={"method": "gzip", "mtime": 0})
    gerar_figuras(resumo, previsoes, calibracao, figuras)
    gerar_relatorio(resultados, resumo, globais, limiares, metadados, destino)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", type=Path, default=cfg.RAIZ_REPOSITORIO / "reports")
    parser.add_argument("--csv", type=Path, default=cfg.CSV_MODELAGEM)
    parser.add_argument("--somente-relatorios", action="store_true")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    manifesto = args.saida / "p3_execucao_2024.json"
    if args.somente_relatorios:
        metadados = json.loads(manifesto.read_text(encoding="utf-8"))
        resultados = pd.read_csv(args.saida / "tabelas/p3_validacao_2024.csv", float_precision="round_trip")
        previsoes = pd.read_csv(args.saida / "tabelas/p3_previsoes_validacao_2024.csv.gz", float_precision="round_trip")
    else:
        hash_antes = sha256(args.csv)
        base = carregar_base(args.csv)
        desenvolvimento, reservado = separar_desenvolvimento_teste(base)
        del reservado, base
        metadados = {"csv": str(args.csv.resolve()), "csv_sha256": hash_antes,
                     "ano_experimentos": 2024, "teste_final_avaliado": False,
                     "n_desenvolvimento": len(desenvolvimento), "semente": SEMENTE,
                     "python": platform.python_version(),
                     "versoes": {p: importlib.metadata.version(p) for p in
                                 ("pandas", "numpy", "scikit-learn", "xgboost", "matplotlib")},
                     "fontes_sha256": {p.name: sha256(p) for p in
                                        Path(__file__).parent.glob("*.py")}}

        def checkpoint(resultados, previsoes, execucoes):
            # Um manifesto incremental preserva falhas/avisos mesmo em interrupção.
            manifesto.write_text(json.dumps({**metadados, "execucoes": execucoes},
                                             indent=2, ensure_ascii=False), encoding="utf-8")

        resultados, previsoes, protocolo = executar_experimentos(desenvolvimento, ao_concluir=checkpoint)
        if sha256(args.csv) != hash_antes:
            raise RuntimeError("O CSV foi alterado durante a execução.")
        metadados.update(protocolo)
        manifesto.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
    salvar_relatorios(resultados, previsoes, metadados, args.saida,
                     salvar_tabelas=not args.somente_relatorios)
    print(f"Resultados de 2024 salvos em {args.saida}", flush=True)


if __name__ == "__main__":
    main()
