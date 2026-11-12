"""Métricas e limiares da validação temporal; nenhuma seleção usa 2025."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, log_loss,
    precision_recall_curve, roc_auc_score,
)

CHAVES = ["cenario", "algoritmo", "variante"]
METRICAS = [
    "ap", "roc_auc", "precision", "recall", "f1", "brier", "log_loss",
    "prevalencia", "tp", "fp", "tn", "fn", "tempo_treino_s", "tempo_inferencia_s",
]


def validar_alinhamento(y: pd.Series, probabilidades: pd.Series) -> None:
    """Exige rótulos únicos e a mesma ordem, sem realinhamento silencioso."""
    if not isinstance(y, pd.Series) or not isinstance(probabilidades, pd.Series):
        raise TypeError("Rótulos e probabilidades devem ser Series com índice original.")
    if y.empty or not y.index.is_unique or y.index.hasnans:
        raise ValueError("Índice vazio, duplicado ou nulo.")
    if not y.index.equals(probabilidades.index):
        raise ValueError("Índices de rótulos e probabilidades desalinhados.")
    if not y.isin([0, 1]).all():
        raise ValueError("Alvo deve ser binário e sem ausentes.")
    p = probabilidades.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilidades devem ser finitas entre 0 e 1.")


def extrair_probabilidade_positiva(pipeline, x: pd.DataFrame) -> pd.Series:
    """Consulta classes_ explicitamente; não pressupõe a posição da classe 1."""
    classes = np.asarray(pipeline.classes_)
    posicao = np.flatnonzero(classes == 1)
    if len(posicao) != 1:
        raise ValueError("A classe positiva 1 está ausente ou duplicada em classes_.")
    matriz = np.asarray(pipeline.predict_proba(x))
    if matriz.shape != (len(x), len(classes)):
        raise ValueError("predict_proba retornou dimensões incompatíveis.")
    if (not np.isfinite(matriz).all() or (matriz < 0).any()
            or (matriz > 1).any() or not np.allclose(matriz.sum(axis=1), 1)):
        raise ValueError("predict_proba retornou probabilidades inválidas.")
    return pd.Series(matriz[:, posicao.item()], index=x.index, name="probabilidade")


def calcular_metricas(y: pd.Series, probabilidades: pd.Series, limiar=.5) -> dict:
    """p >= limiar é positivo; razões indefinidas são NaN e têm motivo explícito."""
    validar_alinhamento(y, probabilidades)
    if not np.isfinite(limiar) or not 0 <= limiar <= 1:
        raise ValueError("Limiar deve estar entre 0 e 1.")
    tn, fp, fn, tp = confusion_matrix(y, probabilidades.ge(limiar), labels=[0, 1]).ravel()
    positivos, negativos = int(y.sum()), int(len(y) - y.sum())
    motivos = []
    if not tp + fp:
        motivos.append("precision: nenhum positivo previsto")
    if not positivos:
        motivos.append("ap/recall: nenhum positivo real")
    if not positivos or not negativos:
        motivos.append("roc_auc: apenas uma classe real")
    if not 2 * tp + fp + fn:
        motivos.append("f1: denominador nulo")
    return {
        "n_validacao": len(y), "n_positivos": positivos, "limiar": float(limiar),
        "ap": float(average_precision_score(y, probabilidades)) if positivos else np.nan,
        "roc_auc": float(roc_auc_score(y, probabilidades)) if positivos and negativos else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / positivos if positivos else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "brier": float(brier_score_loss(y, probabilidades)),
        "log_loss": float(log_loss(y, probabilidades, labels=[0, 1])),
        "prevalencia": positivos / len(y), "indefinidas": "; ".join(motivos),
    }


def selecionar_limiar(y: pd.Series, probabilidades: pd.Series) -> float:
    """Maximiza F1 em todos os escores únicos, incluindo os menores que 0,01.

    Empates numéricos (atol=1e-12, rtol=0) favorecem o maior limiar,
    reduzindo alertas. O F1 resultante é de ajuste, não avaliação independente.
    """
    validar_alinhamento(y, probabilidades)
    if not y.sum():
        raise ValueError("Não é possível ajustar F1 sem positivos reais.")
    precision, recall, limiares = precision_recall_curve(y, probabilidades)
    numerador = 2 * precision[:-1] * recall[:-1]
    denominador = precision[:-1] + recall[:-1]
    f1 = np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador > 0)
    return float(limiares[np.isclose(f1, f1.max(), rtol=0, atol=1e-12)].max())


def validar_previsoes(previsoes: pd.DataFrame) -> None:
    obrigatorias = {*CHAVES, "rodada", "linha_original", "ano", "mes", "y_real", "probabilidade"}
    if not obrigatorias <= set(previsoes) or previsoes.empty:
        raise ValueError("Tabela de previsões vazia ou com esquema incompleto.")
    if not previsoes["ano"].eq(2024).all():
        raise ValueError("Previsões devem conter exclusivamente 2024.")
    if previsoes.duplicated([*CHAVES, "linha_original"]).any():
        raise ValueError("Observação duplicada entre validações da mesma configuração.")
    for rodada, inicio, fim in ((1, 7, 8), (2, 9, 10), (3, 11, 12)):
        parte = previsoes.loc[previsoes.rodada.eq(rodada)]
        if not parte.mes.between(inicio, fim).all():
            raise ValueError("Mês incompatível com a rodada.")
    if not previsoes.rodada.isin([1, 2, 3]).all():
        raise ValueError("Rodada desconhecida.")
    for _, grupo in previsoes.groupby(CHAVES, sort=True):
        g = grupo.set_index("linha_original")
        validar_alinhamento(g.y_real, g.probabilidade)


def consolidar_resultados(resultados: pd.DataFrame) -> pd.DataFrame:
    """Média não ponderada e DP populacional (ddof=0), sem interpretação de IC."""
    linhas = []
    for chave, grupo in resultados.groupby(CHAVES, sort=True):
        linha = dict(zip(CHAVES, chave))
        linha["n_rodadas"] = len(grupo)
        for metrica in METRICAS:
            valores = grupo[metrica].dropna()
            linha[f"{metrica}_media"] = valores.mean()
            linha[f"{metrica}_dp"] = valores.std(ddof=0)
            linha[f"{metrica}_n_definidos"] = len(valores)
        linha["tempo_treino_total_s"] = grupo.tempo_treino_s.sum()
        linhas.append(linha)
    return pd.DataFrame(linhas)


def analisar_previsoes(previsoes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Métricas reunidas, comparação de limiares e calibração por faixas fixas."""
    validar_previsoes(previsoes)
    globais, limiares, calibracao = [], [], []
    for chave, grupo in previsoes.groupby(CHAVES, sort=True):
        identificacao = dict(zip(CHAVES, chave))
        g = grupo.set_index("linha_original")
        globais.append({**identificacao, **calcular_metricas(g.y_real, g.probabilidade)})
        candidato = selecionar_limiar(g.y_real, g.probabilidade) if g.y_real.sum() else None
        for tipo, limiar in (("padrao_0.5", .5), ("candidato_f1_ajuste", candidato)):
            if limiar is not None:
                limiares.append({**identificacao, "tipo_limiar": tipo,
                                 "uso": "validacao_reunida_2024; ajuste_nao_independente",
                                 **calcular_metricas(g.y_real, g.probabilidade, limiar)})
        faixas = pd.cut(g.probabilidade, np.linspace(0, 1, 11), include_lowest=True)
        for faixa, parte in g.groupby(faixas, observed=True):
            calibracao.append({**identificacao, "faixa": str(faixa), "n": len(parte),
                               "probabilidade_media": parte.probabilidade.mean(),
                               "fracao_fatal": parte.y_real.mean()})
    return pd.DataFrame(globais), pd.DataFrame(limiares), pd.DataFrame(calibracao)
